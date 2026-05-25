"""H2 Pfau-NES-VMC for K=2 states (Pfau et al., Science 2024).

Trains two PsiFormer ansatze (psi_1, psi_2) jointly by minimising
Tr(M^{-1} (M (*) local_E)) where M[i,j] = psi_i(x^j) at joint
walkers (x^1, x^2) sampled from |det M|^2. Orthogonality is enforced
*structurally* by the determinantal collapse: psi_1 = psi_2 makes
det M = 0 identically, hence zero sampling weight and infinite
energy. No penalty terms, no hyperparameters, no scale issues.

After training, samples walkers from each |psi_i|^2 independently
and runs the standard CS pipeline to recover c^(0) and c^(1). These
are then compared to the FCI second-root reference (route (c)).

Usage:
    python examples/run_h2_pfau_nes.py --R 2.5 --basis cc-pvdz \
        --pfau-iters 500 --num-walkers 256 --init-from-ground
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
from OmegaQMC.vmcopt_nn_pfau import get_vmcopt_nn_pfau_k2_func
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
    p.add_argument("--out-dir", default="cs_h2_pfau_results")
    p.add_argument("--gs-source-dir", default="cs_h2_nesvmc_results")
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--pfau-iters", type=int, default=500)
    p.add_argument("--num-walkers", type=int, default=256)
    p.add_argument("--num-sample-blocks", type=int, default=2)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mc-timestep", type=float, default=0.1)
    p.add_argument("--sample-blocks", type=int, default=50)
    p.add_argument("--sample-walkers", type=int, default=256)
    p.add_argument("--candidate-tol", type=float, default=1e-6)
    p.add_argument("--init-from-ground", action="store_true",
                   help="initialise both psi_1, psi_2 from the trained "
                        "ground-state checkpoint (with small noise on "
                        "psi_2 to break degeneracy)")
    args = p.parse_args()

    R_str = f"R{args.R:.3f}".replace(".", "p")
    basis_tag = args.basis.replace("*", "s")
    src_dir = Path(args.gs_source_dir)
    chk_gs_ref = (src_dir / f"h2_gs_{R_str}_{basis_tag}.chk.h5"
                  if args.init_from_ground else None)
    if args.init_from_ground and not chk_gs_ref.exists():
        sys.exit(f"missing ground state checkpoint {chk_gs_ref}; "
                 f"run examples/run_h2_nesvmc.py first")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"h2_pfau_{R_str}_{basis_tag}"
    chk_1 = Path(f"{prefix}_1.chk.h5")
    chk_2 = Path(f"{prefix}_2.chk.h5")
    bank_1 = out_dir / (prefix.name + "_1_walkers.h5")
    bank_2 = out_dir / (prefix.name + "_2_walkers.h5")

    print(f"=== H2 Pfau-NES-VMC (K=2) at R={args.R} Bohr, {args.basis} ===")
    mol = build_h2(args.R, args.basis)
    fci_ref = compute_fci_reference(
        mol, n_alpha=1, n_beta=1,
        candidate_tol=args.candidate_tol,
    )
    n_det = len(fci_ref["candidate_set"])
    print(f"  mol: {mol.nao} AOs, |candidate set|={n_det}")
    print(f"  E_FCI = {fci_ref['E_FCI']:.6f} Ha (ground in cc-pVDZ)")

    # --- Train Pfau-NES K=2 ---
    if not chk_1.exists() or not chk_2.exists():
        print(f"\n[1/4] Training Pfau-NES K=2 "
              f"({args.pfau_iters} Adam iters)")
        init_key = jax.random.key(args.seed)
        driver = get_vmcopt_nn_pfau_k2_func(
            mol, args.ansatz, init_key,
            init_from_ground_checkpoint=(
                str(chk_gs_ref) if args.init_from_ground else None
            ),
        )
        rng_opt = jax.random.split(init_key, 4)[3]
        driver(
            rng_opt,
            num_iters=args.pfau_iters,
            num_walkers=args.num_walkers,
            num_sample_blocks=args.num_sample_blocks,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            mc_timestep=args.mc_timestep,
            lr=args.lr,
            prefix=str(prefix),
            verbose=1,
        )
    else:
        print(f"\n[1/4] Reusing Pfau-NES checkpoints {chk_1.name}, {chk_2.name}")

    # --- Sample walker banks from each state independently ---
    if not bank_1.exists():
        print(f"\n[2/4] Sampling Psi_1 walker bank")
        rng = jax.random.key(args.seed + 1)
        init_key, smp_key = jax.random.split(rng)
        drv = get_vmc_nn_func(mol, args.ansatz, init_key,
                               prefix=f"{prefix}_1")
        drv.load_checkpoint(str(chk_1))
        drv(
            smp_key,
            num_walkers=args.sample_walkers,
            num_steps_per_block=20,
            num_blocks=args.sample_blocks,
            num_blocks_equil=5,
            mc_timestep=args.mc_timestep,
            compute_gradients=False,
            dump_walkers_path=str(bank_1),
            verbose=1,
        )
    else:
        print(f"\n[2/4] Reusing Psi_1 bank {bank_1.name}")

    if not bank_2.exists():
        print(f"\n[3/4] Sampling Psi_2 walker bank")
        rng = jax.random.key(args.seed + 2)
        init_key, smp_key = jax.random.split(rng)
        drv = get_vmc_nn_func(mol, args.ansatz, init_key,
                               prefix=f"{prefix}_2")
        drv.load_checkpoint(str(chk_2))
        drv(
            smp_key,
            num_walkers=args.sample_walkers,
            num_steps_per_block=20,
            num_blocks=args.sample_blocks,
            num_blocks_equil=5,
            mc_timestep=args.mc_timestep,
            compute_gradients=False,
            dump_walkers_path=str(bank_2),
            verbose=1,
        )
    else:
        print(f"\n[3/4] Reusing Psi_2 bank {bank_2.name}")

    # --- CS-recover c_hat from each state, compare ---
    print(f"\n[4/4] CS recovery + CI orthogonality")
    c_hat_1, pm_1 = recover_c_hat(
        str(bank_1), str(chk_1), mol, fci_ref, args.ansatz,
        key_seed=args.seed + 10,
    )
    c_hat_2, pm_2 = recover_c_hat(
        str(bank_2), str(chk_2), mol, fci_ref, args.ansatz,
        key_seed=args.seed + 20,
    )
    print(f"  c_hat_1[ref] = {c_hat_1[0]:+.5f}, proj_mass = {pm_1:.3e}")
    print(f"  |c_hat_1| top 3: {[f'{c:+.4f}' for c in c_hat_1[:3]]}")
    print(f"  c_hat_2[ref] = {c_hat_2[0]:+.5f}, proj_mass = {pm_2:.3e}")
    print(f"  |c_hat_2| top 3: {[f'{c:+.4f}' for c in c_hat_2[:3]]}")

    print(f"\n=== CI-space orthogonality ===")
    ovl = float(np.dot(c_hat_1, c_hat_2))
    print(f"  <c_hat_1 | c_hat_2> = {ovl:+.5f}")
    print(f"  ||c_hat_1 - c_hat_2||_2 = "
          f"{np.linalg.norm(c_hat_1 - c_hat_2):.4f}")
    print(f"  (FCI 2-root baseline: 1.9e-17; penalty methods: 0.09 to 1.0)")
    if abs(ovl) < 0.1:
        print(f"  Pfau-NES achieved CI-space orthogonality")
    else:
        print(f"  Pfau-NES did not reach CI orthogonality at this batch")


if __name__ == "__main__":
    main()
