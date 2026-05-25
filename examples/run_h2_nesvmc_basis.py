"""H2 NES-VMC training with the basis-resolved penalty variant.

Companion to ``run_h2_nesvmc.py`` --- same setup but uses the
basis-resolved variant of NES-VMC (penalty against
Psi_synth = Sum_I c_hat^(0)_I D_I, not against the raw ground-state NN).

Workflow:
  1. Train ground state Psi_NN^(0) (SR) or load
  2. Sample Psi_NN^(0) walker bank
  3. CS-recover c_hat^(0) in the natural-orbital basis
  4. Train Psi_NN^(1) with the basis-resolved penalty
  5. Sample Psi_NN^(1) walker bank
  6. CS-recover c_hat^(1) and compute <c_hat^(0), c_hat^(1)>

If the basis-resolved penalty works as designed, the CI-space
overlap <c_hat^(0)|c_hat^(1)> should be much smaller than the 0.999
observed for the real-space penalty variant.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import numpy as np

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmc_nn import get_vmc_nn_func
from OmegaQMC.vmcopt_nn_sr import get_vmcopt_nn_func as get_sr_func
from OmegaQMC.vmcopt_nn_nes import get_vmcopt_nn_nes_basis_func
from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import (
    evaluate_orbitals_on_walkers, f_I_matrix, normalize_and_align,
)
from OmegaQMC.psi.nn.adapter import make_nn_log_psi

sys.path.insert(0, str(Path(__file__).parent))
from run_cs_h4_scaling import evaluate_signed_psi  # noqa: E402


def build_h2(R_bohr, basis):
    mol = Mole_custom()
    mol.build(
        atom=[("H", [0, 0, 0]), ("H", [0, 0, R_bohr])],
        basis=basis, spin=0, charge=0, unit="Bohr", verbose=0,
    )
    return mol


def recover_c_hat(walker_bank_path, ckpt_path, mol, fci_ref, ansatz, key_seed):
    walkers, _, _ = load_walker_bank(walker_bank_path)
    init_key = jax.random.split(jax.random.key(key_seed), 4)[1]
    driver = get_vmc_nn_func(mol, ansatz, init_key,
                              prefix=ckpt_path.replace(".chk.h5", ""))
    driver.load_checkpoint(ckpt_path)
    log_psi, _, _, _ = make_nn_log_psi(ansatz, mol, init_key)
    psi_vals = evaluate_signed_psi(walkers, np.asarray(mol.atom_coords()),
                                    driver.params, log_psi)
    n_alpha, n_beta = fci_ref["nelec"]
    orb = evaluate_orbitals_on_walkers(
        mol, walkers, fci_ref["no_coeff_ao"],
        convention="interleaved", n_alpha=n_alpha, n_beta=n_beta,
    )
    f_I = f_I_matrix(orb, fci_ref["candidate_set"], psi_vals,
                     n_alpha=n_alpha, n_beta=n_beta)
    c_raw = f_I.mean(axis=1)
    c_true = np.array([fci_ref["ci_dict"][k]
                       for k in fci_ref["candidate_set"]])
    c_hat, proj_mass = normalize_and_align(
        c_raw, float(np.sign(c_true[0])),
    )
    return c_hat, proj_mass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=2.5)
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--ansatz",
                   default=str(Path(__file__).parent / "inputs"
                               / "psiformer_small.yaml"))
    p.add_argument("--out-dir", default="cs_h2_nesvmc_basis_results")
    p.add_argument("--gs-source-dir", default="cs_h2_nesvmc_results",
                   help="reuse ground-state checkpoint + bank from this dir")
    p.add_argument("--seed-es", type=int, default=33)
    p.add_argument("--es-iters", type=int, default=300)
    p.add_argument("--es-walkers", type=int, default=128)
    p.add_argument("--num-sample-blocks", type=int, default=2)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--mc-timestep", type=float, default=0.1)
    p.add_argument("--lambda-penalty", type=float, default=1.0)
    p.add_argument("--sample-blocks", type=int, default=50)
    p.add_argument("--sample-walkers", type=int, default=256)
    p.add_argument("--candidate-tol", type=float, default=1e-6)
    p.add_argument("--init-from-ground", action="store_true",
                   help="initialise Psi_1 from the trained ground-state "
                        "params so the penalty has appreciable gradient")
    args = p.parse_args()

    R_str = f"R{args.R:.3f}".replace(".", "p")
    basis_tag = args.basis.replace("*", "s")
    src_dir = Path(args.gs_source_dir)
    gs_prefix = src_dir / f"h2_gs_{R_str}_{basis_tag}"
    chk_gs = gs_prefix.with_suffix(".chk.h5")
    bank_gs = src_dir / (gs_prefix.name + "_walkers.h5")
    if not chk_gs.exists() or not bank_gs.exists():
        sys.exit(f"missing ground state files in {src_dir}; run "
                 f"examples/run_h2_nesvmc.py first")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_es = out_dir / f"h2_es_basis_{R_str}_{basis_tag}"
    chk_es = prefix_es.with_suffix(".chk.h5")
    bank_es = out_dir / (prefix_es.name + "_walkers.h5")

    print(f"=== H2 basis-resolved NES-VMC at R={args.R} Bohr, {args.basis} ===")
    mol = build_h2(args.R, args.basis)
    fci_ref = compute_fci_reference(
        mol, n_alpha=1, n_beta=1,
        candidate_tol=args.candidate_tol,
    )
    n_det = len(fci_ref["candidate_set"])
    print(f"  mol: {mol.nao} AOs, |candidate set|={n_det}")
    print(f"  E_FCI = {fci_ref['E_FCI']:.6f} Ha")

    # ----- Recover c_hat^(0) from existing ground-state bank -----
    print(f"\n[1/4] Recovering c_hat^(0) from {bank_gs.name}")
    c_hat_gs, pm_gs = recover_c_hat(
        str(bank_gs), str(chk_gs), mol, fci_ref, args.ansatz, key_seed=11,
    )
    print(f"  c_hat^(0)[ref] = {c_hat_gs[0]:+.5f}, proj_mass = {pm_gs:.3e}")
    print(f"  |c_hat^(0)| top 3: {[f'{c:+.4f}' for c in c_hat_gs[:3]]}")

    # ----- Train Psi_NN^(1) with basis-resolved penalty -----
    if not chk_es.exists():
        print(f"\n[2/4] Training Psi_NN^(1) with basis-resolved penalty"
              f" (lambda={args.lambda_penalty}, {args.es_iters} iters)")
        rng_es = jax.random.key(args.seed_es)
        init_key, opt_key = jax.random.split(rng_es)
        nes_b = get_vmcopt_nn_nes_basis_func(
            mol, args.ansatz, init_key,
            c_hat_ground=c_hat_gs, fci_ref=fci_ref,
            lambda_penalty=args.lambda_penalty,
            init_from_ground_checkpoint=(str(chk_gs)
                                          if args.init_from_ground
                                          else None),
        )
        nes_b(
            opt_key, num_iters=args.es_iters,
            num_walkers=args.es_walkers,
            num_epochs=args.num_epochs,
            num_sample_blocks=args.num_sample_blocks,
            batch_size=args.batch_size,
            mc_timestep=args.mc_timestep, lr=args.lr,
            prefix=str(prefix_es), verbose=1,
        )
    else:
        print(f"\n[2/4] Reusing excited-state checkpoint {chk_es.name}")

    # ----- Sample excited-state walker bank -----
    if not bank_es.exists():
        print(f"\n[3/4] Sampling Psi_NN^(1) walker bank")
        rng_smp = jax.random.key(args.seed_es + 1000)
        init_key, smp_key = jax.random.split(rng_smp)
        driver_es = get_vmc_nn_func(mol, args.ansatz, init_key,
                                     prefix=str(prefix_es))
        driver_es.load_checkpoint(str(chk_es))
        driver_es(
            smp_key,
            num_walkers=args.sample_walkers,
            num_steps_per_block=20,
            num_blocks=args.sample_blocks,
            num_blocks_equil=5,
            mc_timestep=args.mc_timestep,
            compute_gradients=False,
            dump_walkers_path=str(bank_es),
            verbose=1,
        )
    else:
        print(f"\n[3/4] Reusing excited-state bank {bank_es.name}")

    # ----- Recover c_hat^(1) and compare to c_hat^(0) -----
    print(f"\n[4/4] Recovering c_hat^(1) and comparing to c_hat^(0)")
    c_hat_es, pm_es = recover_c_hat(
        str(bank_es), str(chk_es), mol, fci_ref, args.ansatz,
        key_seed=args.seed_es,
    )
    print(f"  c_hat^(1)[ref] = {c_hat_es[0]:+.5f}, proj_mass = {pm_es:.3e}")
    print(f"  |c_hat^(1)| top 3: {[f'{c:+.4f}' for c in c_hat_es[:3]]}")

    print(f"\n=== CI-space orthogonality ===")
    ovl = float(np.dot(c_hat_gs, c_hat_es))
    print(f"  <c_hat^(0) | c_hat^(1)> = {ovl:+.5f}")
    print(f"  ||c_hat^(0) - c_hat^(1)||_2 = {np.linalg.norm(c_hat_gs - c_hat_es):.4f}")
    print(f"  (for comparison: real-space NES gave +0.99925)")
    if abs(ovl) < 0.1:
        print(f"  basis-resolved penalty DROVE CI-space orthogonality")
    else:
        print(f"  basis-resolved penalty did not fully reach CI orthogonality")
        print(f"  (try larger --lambda-penalty or more --es-iters)")


if __name__ == "__main__":
    main()
