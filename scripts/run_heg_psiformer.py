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
from OmegaQMC.psi.nn.heg_wf import HEGConfig
from OmegaQMC.vmcopt_nn_heg import get_vmcopt_nn_heg_func
from OmegaQMC.vmc_nn_heg import get_vmc_nn_heg_func


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

    config = HEGConfig(
        n_up=n_up, n_down=n_down, L=L,
        n_det=args.n_det,
        use_jastrow=(not args.no_jastrow),
    )

    rng = jax.random.key(args.seed)
    init_key, opt_key, eval_key = jax.random.split(rng, 3)

    # ------- Training -------
    if not args.skip_opt:
        print(f"\n[1/2] Adam-VMC training: "
              f"{args.iters} iters, {args.opt_walkers} walkers, "
              f"lr={args.lr}")
        opt = get_vmcopt_nn_heg_func(
            config, init_key,
            prefix=args.prefix, lr=args.lr,
            ewald_n_real=args.ewald_n_real,
            ewald_n_recip=args.ewald_n_recip,
        )
        result_opt = opt(
            opt_key,
            num_iters=args.iters,
            num_walkers=args.opt_walkers,
            mcmc_decorr_steps=10,
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
    print(f"  VMC total               = {e_vmc_ha:+.6f} ± "
          f"{e_serr_ha:.6f} Ha")
    print(f"  VMC total               = {e_vmc_ha * 2:+.6f} ± "
          f"{e_serr_ha * 2:.6f} Ry")
    print(f"  Finite-cell HF          = {e_hf_ha:+.6f} Ha "
          f"(= {e_hf_ha * 2:+.6f} Ry)")
    print(f"  Correlation (VMC - HF)  = {e_corr_ha:+.6f} Ha")
    print(f"  Perdew-Zunger corr (∞)  = {e_corr_pz_ha:+.6f} Ha")
    print(f"  Correlation recovered   = {recovered_frac:.1f}%")
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
