"""Validation driver: HEG VMC with plane-wave + Jastrow trial.

Trains the HEG trial wavefunction with Adam-VMC, then evaluates the
converged energy and compares against:

  * the finite-cell Hartree–Fock energy (AFQMC trial energy), and
  * the thermodynamic-limit HF energy + Perdew–Zunger/Ceperley–Alder
    correlation.

Defaults reproduce a small N=14 unpolarised HEG at rs=2.  Larger
systems are supported via ``--N`` and ``--rs``; be mindful of the
MCMC equilibration scale and Adam iteration count when doing so.

Example::

    python scripts/run_heg_psiformer.py --rs 2 --N 14 --iters 500

Output: a checkpoint file ``<prefix>.chk.h5`` plus a printed
summary table.  The printed kinetic/potential decomposition and
correlation-energy recovery fraction are the key observables —
they tell you whether the trial wavefunction is sensible and how
much beyond-HF correlation the Jastrow captures.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

import numpy as np
import jax

from OmegaQMC.afqmc_3deg import (
    build_3deg_system,
    get_afqmc_3deg_func,
    pz_correlation_energy,
)
from OmegaQMC.psi.nn.heg_wf import HEGConfig, HEGPsiFormerConfig
from OmegaQMC.vmcopt_nn_heg import get_vmcopt_nn_heg_func
from OmegaQMC.vmcopt_nn_heg_sr import get_vmcopt_nn_heg_sr_func
from OmegaQMC.pretrain_heg import pretrain_heg_psiformer
from OmegaQMC.vmc_nn_heg import (
    get_vmc_nn_heg_func,
    run_twist_averaged_heg,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rs', type=float, default=2.0,
                   help='Wigner-Seitz density parameter')
    p.add_argument('--N', type=int, default=14,
                   help='Electron count (closed-shell: 14, 38, 54)')
    p.add_argument(
        '--polarization', type=str, default='unpolarized',
        choices=['unpolarized', 'polarized'],
    )
    p.add_argument('--iters', type=int, default=200,
                   help='Adam optimisation iterations')
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--opt-walkers', type=int, default=128)
    p.add_argument('--eval-walkers', type=int, default=256)
    p.add_argument('--eval-blocks', type=int, default=40)
    p.add_argument('--eval-equil-blocks', type=int, default=20)
    p.add_argument('--steps-per-block', type=int, default=30)
    p.add_argument('--mc-timestep', type=float, default=0.1)
    p.add_argument('--n-det', type=int, default=1)
    p.add_argument('--no-jastrow', action='store_true',
                   help='Skip Jastrow (pure HF ansatz)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prefix', type=str, default=None)
    p.add_argument('--ewald-n-real', type=int, default=3)
    p.add_argument('--ewald-n-recip', type=int, default=6)
    p.add_argument('--skip-opt', action='store_true',
                   help='Skip training; evaluate ansatz at init')
    p.add_argument('--twists', type=int, default=0,
                   help=('If > 0, run TABC with this many Halton '
                         'twists after the eval pass (transfers the '
                         'trained Jastrow to a complex ansatz at each '
                         'twist).'))
    p.add_argument('--twist-blocks', type=int, default=None,
                   help='Override --eval-blocks for per-twist runs')
    p.add_argument('--twist-equil-blocks', type=int, default=None,
                   help='Override --eval-equil-blocks for per-twist runs')
    p.add_argument('--twist-walkers', type=int, default=None,
                   help='Override --eval-walkers for per-twist runs')
    # PsiFormer ansatz flags (all off → minimal Slater-Jastrow).
    p.add_argument('--psiformer', action='store_true',
                   help=('Use the full PsiFormer-class HEG ansatz '
                         '(GNN + attention + backflow) instead of '
                         'the minimal Slater-Jastrow.'))
    p.add_argument('--pf-embedding-dim', type=int, default=64,
                   help='PsiFormer one-particle embedding dim')
    p.add_argument('--pf-layers', type=int, default=2,
                   help='PsiFormer attention/GNN layers')
    p.add_argument('--pf-two-particle-dim', type=int, default=16,
                   help='PsiFormer two-particle stream dim')
    p.add_argument('--pf-heads', type=int, default=2,
                   help='PsiFormer attention heads per layer')
    p.add_argument('--pf-full-determinant', action='store_true',
                   help='Full-determinant PsiFormer (wider orbitals)')
    p.add_argument('--pf-no-cusp', action='store_true',
                   help=('Disable the e-e Kato cusp correction. '
                         'Cusp is ON by default — the '
                         'PeriodicElectronicCusp implementation '
                         '(softplus α, smooth cell-face cutoff) '
                         'plus the eps-consistent Ewald origin '
                         'regularization together give a '
                         'variationally-valid Kato-cusped trial '
                         'from iter 0.  Use this flag for '
                         'diagnostic comparisons against a '
                         'cusp-less ansatz.'))
    p.add_argument('--pf-deep-jastrow', action='store_true',
                   help='Enable the GNN-side deep Jastrow head')
    p.add_argument('--pf-pair-jastrow', action='store_true',
                   help='Add an extra explicit pair Jastrow')
    p.add_argument('--pf-jas-activation', type=str, default='tanh',
                   choices=['tanh', 'ssp', 'silu', 'none'],
                   help=('Deep Jastrow MLP activation (default tanh). '
                         '"none" makes the head linear — '
                         'historical buggy behaviour, kept for '
                         'reproducibility only.'))
    p.add_argument('--pf-jas-no-bias', action='store_true',
                   help='Disable bias terms in the deep Jastrow MLP.')
    p.add_argument('--pf-jas-no-zero-init', action='store_true',
                   help=('Skip zero-init of the deep Jastrow '
                         'last layer. By default the Jastrow '
                         'contributes 0 at iter 0, so walkers '
                         'start from the envelope prior.'))
    p.add_argument('--pf-n-virt-pw', type=int, default=12,
                   help=('Plane-wave basis functions beyond the '
                         'Fermi shell.  Adds virtual orbitals the '
                         'multi-det expansion can specialise into. '
                         'Default 12 covers one extra shell for '
                         'cubic HEG at N=14.  Set to 0 to disable '
                         '(reverts to the old basis=Fermi-shell '
                         'behaviour where n_det>1 is degenerate).'))
    p.add_argument('--pf-det-jitter', type=float, default=0.02,
                   help=('Gaussian perturbation magnitude applied '
                         'to envelope coefficients of dets d≥1 at '
                         'init.  Det 0 stays at pure Fermi-sea '
                         'HF.  Without this, all dets are '
                         'identical and n_det>1 is a waste.  '
                         'Default 0.02 is conservative; try 0.05 '
                         'if the multi-det expansion is sluggish.'))
    p.add_argument('--pf-no-ghost-atom', action='store_true',
                   help=('Disable the ghost-atom positional '
                         'fingerprint in the electron embedding. '
                         'Default is ON — matches FermiNet HEG '
                         'convention and breaks the same-spin GNN '
                         'embedding degeneracy that stalls HEG '
                         'training.  Use this flag only for '
                         'diagnostic comparisons.'))
    # Mixed-objective training.
    p.add_argument('--var-weight', type=float, default=0.0,
                   help=('Weight β of Var(E_L) in the Umrigar-style '
                         'mixed objective L = ⟨E⟩ + β · Var(E_L). '
                         'Default 0.0 (pure energy). Typical: '
                         '0.01-0.1 for stabilising Adam training on '
                         'large networks.'))
    # Optimizer choice.
    p.add_argument('--optimizer', type=str, default='adam',
                   choices=['adam', 'sr'],
                   help=('Parameter optimiser.  "adam" is the '
                         'default diagonally-rescaled gradient '
                         'descent; "sr" is natural-gradient '
                         'stochastic reconfiguration (Fisher-'
                         'rescaled step via CG).  SR typically '
                         'needs larger --opt-walkers to beat the '
                         'gradient-noise floor.'))
    p.add_argument('--sr-damping', type=float, default=1e-3,
                   help='SR Fisher-matrix damping ε')
    p.add_argument('--sr-n-cg', type=int, default=30,
                   help='Conjugate-gradient iterations per SR step')
    p.add_argument('--mcmc-decorr-steps', type=int, default=20,
                   help='MCMC steps between optimiser updates')
    # Supervised pre-training (PsiFormer only).
    p.add_argument('--pretrain-iters', type=int, default=0,
                   help=('If > 0, run supervised Hartree–Fock '
                         'pre-training for this many iterations '
                         'before the energy VMC stage.  Strongly '
                         'recommended for PsiFormer — without it '
                         'the network stalls at the HF plateau.  '
                         'Typical value: 300-1000.'))
    p.add_argument('--pretrain-walkers', type=int, default=256,
                   help='MCMC walkers for pre-training')
    p.add_argument('--pretrain-lr', type=float, default=1e-3,
                   help='Adam learning rate during pre-training')
    args = p.parse_args()

    if args.prefix is None:
        tag = (f"heg_{args.polarization[:5]}_N{args.N}"
               f"_rs{args.rs:.2f}".replace('.', 'p'))
        args.prefix = tag

    # Build HEG system (used for L, AFQMC-HF reference, PZ correlation).
    sys = build_3deg_system(
        args.rs, N_elec=args.N, N_pw=args.N // 2,
        polarization=args.polarization,
    )
    L = sys['L']
    n_up = sys['nup']
    n_down = sys['ndown']

    print("=" * 70)
    print(f"HEG VMC run: rs={args.rs}  N={args.N}  "
          f"pol={args.polarization}")
    print(f"  Cell L={L:.4f} Bohr   V={sys['volume']:.4f} Bohr³")
    print(f"  n_up={n_up}  n_down={n_down}")
    print(f"  Ansatz: PW envelope + "
          f"{'Jastrow' if not args.no_jastrow else 'no Jastrow'}")
    print(f"  Prefix: {args.prefix}")
    print("=" * 70)

    # Finite-cell HF reference.
    afqmc = get_afqmc_3deg_func(
        sys, dt=0.005, include_coulomb=True, verbose=False,
    )
    e_hf_ha = float(afqmc.e_trial) / args.N
    print(f"\n[ref] Finite-cell HF (AFQMC trial): "
          f"{e_hf_ha:.6f} Ha/elec = "
          f"{e_hf_ha * 2:.6f} Ry/elec")
    # Thermodynamic-limit correlation.
    e_corr_pz_ha = pz_correlation_energy(
        args.rs, args.polarization,
    )
    print(f"[ref] Perdew-Zunger correlation (∞-limit): "
          f"{e_corr_pz_ha:.6f} Ha/elec = "
          f"{e_corr_pz_ha * 2:.6f} Ry/elec")

    if args.psiformer:
        config = HEGPsiFormerConfig(
            n_up=n_up, n_down=n_down, L=L,
            n_det=args.n_det,
            full_determinant=args.pf_full_determinant,
            embedding_dim=args.pf_embedding_dim,
            n_interactions=args.pf_layers,
            two_particle_stream_dim=args.pf_two_particle_dim,
            n_attention_heads=args.pf_heads,
            use_cusp=(not args.pf_no_cusp),
            use_deep_jastrow=args.pf_deep_jastrow,
            use_pair_jastrow=args.pf_pair_jastrow,
            jas_mlp_activation=(
                None if args.pf_jas_activation == 'none'
                else args.pf_jas_activation
            ),
            jas_mlp_bias=(not args.pf_jas_no_bias),
            jas_mlp_zero_init_last=(not args.pf_jas_no_zero_init),
            n_virt_pw=args.pf_n_virt_pw,
            det_jitter=args.pf_det_jitter,
            use_ghost_atom=(not args.pf_no_ghost_atom),
        )
        print(
            f"  Ansatz: PsiFormer — emb={args.pf_embedding_dim}, "
            f"layers={args.pf_layers}, "
            f"tp_dim={args.pf_two_particle_dim}, "
            f"heads={args.pf_heads}, "
            f"n_det={args.n_det}"
        )
    else:
        config = HEGConfig(
            n_up=n_up, n_down=n_down, L=L,
            n_det=args.n_det,
            use_jastrow=(not args.no_jastrow),
        )

    rng = jax.random.key(args.seed)
    init_key, opt_key, eval_key, pretrain_key = jax.random.split(rng, 4)

    # ------- [0/2] Supervised Hartree–Fock pre-training -------
    # Applies only when a PsiFormer config is in use.  Drives the
    # randomly-initialised backflow / GNN to reproduce the HF
    # Slater determinant via MSE regression.  Leaves the weights
    # in an "engaged" state from which subsequent energy VMC can
    # actually descend.  Without it, both Adam and SR plateau at
    # HF for the small (emb=64) PsiFormer ansatz on N=14 rs=2.
    pretrained_params = None
    if args.pretrain_iters > 0:
        if not args.psiformer:
            print(
                "[warn] --pretrain-iters > 0 ignored (pre-training "
                "is only implemented for PsiFormer ansätze)"
            )
        else:
            print(
                f"\n[0/2] Supervised HF pre-training: "
                f"{args.pretrain_iters} iters, "
                f"{args.pretrain_walkers} walkers, "
                f"lr={args.pretrain_lr}"
            )
            pretrain_result = pretrain_heg_psiformer(
                config, init_key,
                num_iters=args.pretrain_iters,
                num_walkers=args.pretrain_walkers,
                lr=args.pretrain_lr,
                verbose=1,
            )
            pretrained_params = pretrain_result['params']
            print(
                f"  Pre-training MSE: "
                f"{pretrain_result['loss_history'][0]:.4e} → "
                f"{pretrain_result['final_loss']:.4e}"
            )

    # ------- Training -------
    if not args.skip_opt:
        obj_tag = (
            f"L = ⟨E⟩ + {args.var_weight:.3g}·Var"
            if args.var_weight > 0 else "⟨E⟩"
        )
        opt_tag = args.optimizer.upper()
        print(f"\n[1/2] {opt_tag}-VMC training: "
              f"{args.iters} iters, {args.opt_walkers} walkers, "
              f"lr={args.lr}, objective = {obj_tag}")
        if args.optimizer == 'sr':
            opt = get_vmcopt_nn_heg_sr_func(
                config, init_key,
                lr=args.lr,
                damping=args.sr_damping,
                n_cg=args.sr_n_cg,
                var_weight=args.var_weight,
                ewald_n_real=args.ewald_n_real,
                ewald_n_recip=args.ewald_n_recip,
            )
        else:
            opt = get_vmcopt_nn_heg_func(
                config, init_key,
                prefix=args.prefix, lr=args.lr,
                var_weight=args.var_weight,
                ewald_n_real=args.ewald_n_real,
                ewald_n_recip=args.ewald_n_recip,
            )

        # Inject the pre-trained parameters into the optimizer.
        # Both optimizers build a network via the same factory
        # (build_heg_psiformer_wf) with the same config/init_key,
        # so the pytree structure is identical — only the numerical
        # values (post-pre-training) change.
        if pretrained_params is not None:
            if args.optimizer == 'sr':
                from jax.flatten_util import ravel_pytree
                opt.params_flat = ravel_pytree(pretrained_params)[0]
            else:
                opt.params = pretrained_params

        result_opt = opt(
            opt_key,
            num_iters=args.iters,
            num_walkers=args.opt_walkers,
            mcmc_decorr_steps=args.mcmc_decorr_steps,
            num_equil_steps=400,
            mc_timestep=args.mc_timestep,
            verbose=1,
        )
        trained_params = result_opt['params']
        print(f"  Final training E/N: "
              f"{result_opt['E_final_ha']:.6f} Ha/elec")
    else:
        print("\n[1/2] Skipping training (--skip-opt).")
        trained_params = None

    # ------- Evaluation -------
    print(f"\n[2/2] Evaluation VMC: "
          f"{args.eval_blocks} blocks × {args.steps_per_block} steps "
          f"× {args.eval_walkers} walkers")
    driver = get_vmc_nn_heg_func(
        config, init_key,
        prefix=args.prefix,
        ewald_n_real=args.ewald_n_real,
        ewald_n_recip=args.ewald_n_recip,
    )
    if trained_params is not None:
        driver.params = trained_params

    result = driver(
        eval_key,
        num_walkers=args.eval_walkers,
        num_steps_per_block=args.steps_per_block,
        num_blocks=args.eval_blocks,
        num_blocks_equil=args.eval_equil_blocks,
        mc_timestep=args.mc_timestep,
        verbose=1,
    )
    e_vmc_ha = result['E_per_elec_ha']
    e_serr_ha = result['E_serr'] / args.N

    # ------- Summary -------
    e_corr_ha = e_vmc_ha - e_hf_ha
    recovered_frac = (
        100.0 * e_corr_ha / e_corr_pz_ha
        if e_corr_pz_ha != 0 else float('nan')
    )

    print("\n" + "=" * 70)
    print("SUMMARY (Ha/elec unless noted)")
    print("=" * 70)
    print(f"  VMC Γ-point             = {e_vmc_ha:+.6f} ± "
          f"{e_serr_ha:.6f} Ha")
    print(f"  VMC Γ-point             = {e_vmc_ha * 2:+.6f} ± "
          f"{e_serr_ha * 2:.6f} Ry")
    print(f"  Finite-cell HF (Γ)      = {e_hf_ha:+.6f} Ha "
          f"(= {e_hf_ha * 2:+.6f} Ry)")
    print(f"  Correlation (VMC - HF)  = {e_corr_ha:+.6f} Ha")
    print(f"  Perdew-Zunger corr (∞)  = {e_corr_pz_ha:+.6f} Ha")
    print(f"  Correlation recovered   = {recovered_frac:.1f}%")
    print("=" * 70)

    # ------- Twist averaging -------
    if args.twists > 0:
        print(f"\n[TABC] {args.twists} Halton twists, "
              f"injecting trained Jastrow into complex ansatz.")
        tw_blocks = args.twist_blocks or args.eval_blocks
        tw_equil = args.twist_equil_blocks or args.eval_equil_blocks
        tw_walkers = args.twist_walkers or args.eval_walkers
        tabc = run_twist_averaged_heg(
            config, init_key,
            trained_params_real=trained_params,
            n_twists=args.twists,
            ewald_n_real=args.ewald_n_real,
            ewald_n_recip=args.ewald_n_recip,
            num_walkers=tw_walkers,
            num_steps_per_block=args.steps_per_block,
            num_blocks=tw_blocks,
            num_blocks_equil=tw_equil,
            mc_timestep=args.mc_timestep,
            eval_seed=args.seed + 1000,
            verbose=1,
        )

        e_tabc_ha = tabc['E_per_elec_ha']
        e_tabc_err_ha = tabc['E_serr_ha'] / args.N
        e_tabc_corr_ha = e_tabc_ha - e_hf_ha
        recovered_tabc = (
            100.0 * e_tabc_corr_ha / e_corr_pz_ha
            if e_corr_pz_ha != 0 else float('nan')
        )

        print("\n" + "=" * 70)
        print("TABC SUMMARY (Ha/elec unless noted)")
        print("=" * 70)
        print(f"  VMC TABC                = {e_tabc_ha:+.6f} ± "
              f"{e_tabc_err_ha:.6f} Ha")
        print(f"  VMC TABC                = {e_tabc_ha * 2:+.6f} ± "
              f"{e_tabc_err_ha * 2:.6f} Ry")
        print(f"  Correlation (TABC - HF) = {e_tabc_corr_ha:+.6f} Ha")
        print(f"  Correlation recovered   = {recovered_tabc:.1f}%")
        print(f"  Twist-to-twist scatter  = "
              f"{np.std(tabc['energies_per_twist'])/args.N:.6f} Ha/elec")
        print("=" * 70)

    return {
        'e_vmc_ha': e_vmc_ha,
        'e_serr_ha': e_serr_ha,
        'e_hf_ha': e_hf_ha,
        'e_corr_pz_ha': e_corr_pz_ha,
        'e_corr_vmc_ha': e_corr_ha,
        'recovered_frac': recovered_frac,
    }


if __name__ == '__main__':
    main()
