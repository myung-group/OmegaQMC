"""CS pipeline validation on stretched H2/STO-3G.

H2 at 2.5 Bohr is a textbook two-configuration system: the FCI ground
state is dominated by ``c_HF^2 ~ 0.9`` and ``c_doubly~ -0.3`` with the
two singly-excited determinants forbidden by spin symmetry. STO-3G has
only four singlet determinants total, so FCI is exact in microseconds
and the candidate set is the full determinant space.

This script trains a PsiFormer to FCI quality, dumps a walker bank,
recovers the CI vector via the f_I sample-mean estimator, and reports
per-coefficient agreement against PySCF FCI. A successful run is the
pre-flight check before launching the H4 scaling experiment.

Usage:
    python examples/validate_cs_h2_stretched.py
    python examples/validate_cs_h2_stretched.py --R 2.5 --train-iters 500
    python examples/validate_cs_h2_stretched.py --skip-train
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import numpy as np

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmc_nn import get_vmc_nn_func
from OmegaQMC.vmcopt_nn_sr import get_vmcopt_nn_func
from OmegaQMC.psi.nn.adapter import make_nn_log_psi

from OmegaQMC.cs.estimators import (
    evaluate_orbitals_on_walkers,
    f_I_matrix,
    normalize_and_align,
)
from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.walkers import load_walker_bank


def build_h2_mol(R_bohr: float, basis: str) -> Mole_custom:
    mol = Mole_custom()
    mol.build(
        atom=[("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, R_bohr])],
        basis=basis, spin=0, charge=0, unit="Bohr", verbose=0,
    )
    return mol


def evaluate_signed_psi(
    walkers: np.ndarray,
    nuc_crds: np.ndarray,
    params,
    log_psi,
    batch_size: int = 4096,
) -> np.ndarray:
    """Evaluate signed Psi at each walker using log_psi.signed."""
    signed_v = jax.vmap(log_psi.signed, in_axes=(0, None, None))
    try:
        signed_fn = jax.jit(signed_v)
    except Exception:
        signed_fn = signed_v
    K_s = walkers.shape[0]
    nuc_j = jnp.asarray(nuc_crds)
    out = np.empty(K_s, dtype=np.float64)
    for start in range(0, K_s, batch_size):
        end = min(start + batch_size, K_s)
        batch = jnp.asarray(walkers[start:end])
        sign_b, log_amp_b = signed_fn(batch, nuc_j, params)
        psi_b = sign_b * jnp.exp(log_amp_b)
        out[start:end] = np.asarray(psi_b, dtype=np.float64)
    return out


def align_global_sign(c_hat: np.ndarray, c_true: np.ndarray) -> np.ndarray:
    """Flip the global sign of ``c_hat`` so the dominant determinant
    agrees in sign with ``c_true``. Overall phase is unobservable."""
    ref = int(np.argmax(np.abs(c_true)))
    if c_hat[ref] * c_true[ref] < 0:
        return -c_hat
    return c_hat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=2.5,
                   help="H-H bond length in Bohr (default 2.5)")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--ansatz", default=str(
        Path(__file__).parent / "inputs" / "psiformer_small.yaml"))
    p.add_argument("--out-dir", default="cs_h2_validation")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--train-iters", type=int, default=400)
    p.add_argument("--train-walkers", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--mc-timestep", type=float, default=0.1)
    p.add_argument("--jac-batch", type=int, default=4)
    p.add_argument("--sample-blocks", type=int, default=50)
    p.add_argument("--sample-walkers", type=int, default=256)
    p.add_argument("--sample-steps-per-block", type=int, default=20)
    p.add_argument("--sample-equil-blocks", type=int, default=5)
    p.add_argument("--psi-batch", type=int, default=2048)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-sample", action="store_true")
    p.add_argument("--energy-gate-mE_h", type=float, default=1.0)
    p.add_argument("--recovery-tol", type=float, default=0.05,
                   help="L_inf threshold for PASS verdict")
    args = p.parse_args()

    print(f"== JAX backend: {jax.default_backend()}, "
          f"devices: {jax.devices()} ==")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"h2_R{args.R:.3f}_{args.basis}".replace(".", "p")
    work = out_dir / prefix
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / f"{prefix}.chk.h5"
    bank_path = work / f"{prefix}_walkers.h5"

    # [1] FCI reference
    print(f"\n[1] FCI reference: R={args.R} Bohr, basis={args.basis}")
    mol = build_h2_mol(args.R, args.basis)
    fci_ref = compute_fci_reference(mol, n_alpha=1, n_beta=1, candidate_tol=0.0)
    candidate = fci_ref["candidate_set"]
    c_true = np.array([fci_ref["ci_dict"][k] for k in candidate])
    print(f"    E_HF  = {fci_ref['E_HF']: .8f} Ha")
    print(f"    E_FCI = {fci_ref['E_FCI']: .8f} Ha")
    print(f"    |candidate set| = {len(candidate)}")
    for k, c in zip(candidate, c_true):
        print(f"      {k}  c_FCI = {c:+.6f}")

    # [2] Train PsiFormer
    rng = jax.random.key(args.seed)
    rng, init_key, opt_key, smp_key = jax.random.split(rng, 4)
    if args.skip_train or checkpoint.exists():
        if checkpoint.exists():
            print(f"\n[2] reusing checkpoint {checkpoint.name}")
        else:
            print(f"\n[2] --skip-train but no checkpoint exists")
    else:
        print(f"\n[2] training PsiFormer: {args.train_iters} iters, "
              f"{args.train_walkers} walkers")
        opt = get_vmcopt_nn_func(mol, args.ansatz, init_key)
        opt(
            opt_key,
            num_iters=args.train_iters,
            num_walkers=args.train_walkers,
            lr=args.lr,
            mc_timestep=args.mc_timestep,
            jac_batch_size=args.jac_batch,
            prefix=str(work / prefix),
            verbose=1,
        )

    # [3] Load checkpoint + sample
    driver = get_vmc_nn_func(mol, args.ansatz, init_key, prefix=str(work / prefix))
    if checkpoint.exists():
        meta = driver.load_checkpoint(str(checkpoint))
        e_train = float(meta.get("energy", float("nan")))
        print(f"    epoch={int(meta.get('epoch', -1))}  "
              f"train E={e_train:.6f} Ha")
    else:
        e_train = float("nan")
        print(f"    no checkpoint; running with random init "
              f"(recovery will not converge)")
    delta_E_mEh = (e_train - fci_ref["E_FCI"]) * 1000.0
    print(f"    Delta E (train vs FCI) = {delta_E_mEh:+.3f} mE_h "
          f"(gate: <= {args.energy_gate_mE_h:.2f})")

    if args.skip_sample and bank_path.exists():
        print(f"\n[4] reusing walker bank {bank_path.name}")
    else:
        print(f"\n[4] VMC sampling: {args.sample_blocks} blocks x "
              f"{args.sample_walkers} walkers")
        driver(
            smp_key,
            num_walkers=args.sample_walkers,
            num_steps_per_block=args.sample_steps_per_block,
            num_blocks=args.sample_blocks,
            num_blocks_equil=args.sample_equil_blocks,
            mc_timestep=args.mc_timestep,
            compute_gradients=False,
            dump_walkers_path=str(bank_path),
            verbose=1,
        )

    # [5] Signed Psi at walker bank
    walker_bank, _log_psi_bank, bank_meta = load_walker_bank(str(bank_path))
    print(f"\n[5] walker bank: shape={walker_bank.shape}  "
          f"schema_version={bank_meta.get('schema_version', '?')}")
    log_psi, _params0, _gdef, _ = make_nn_log_psi(args.ansatz, mol, init_key)
    psi_vals = evaluate_signed_psi(
        walker_bank, np.asarray(mol.atom_coords()),
        driver.params, log_psi,
        batch_size=args.psi_batch,
    )
    psi_mag = np.abs(psi_vals)
    print(f"    |Psi| range [{psi_mag.min():.3e}, {psi_mag.max():.3e}]")
    print(f"    sign mean  {np.mean(np.sign(psi_vals)):+.3f}")

    # [6] f_I sample-mean estimator (via normalize_and_align)
    print(f"\n[6] f_I sample-mean estimator")
    orb = evaluate_orbitals_on_walkers(
        mol, walker_bank, fci_ref["no_coeff_ao"],
        convention="interleaved", n_alpha=1, n_beta=1,
    )
    f_I = f_I_matrix(orb, candidate, psi_vals, n_alpha=1, n_beta=1)
    c_raw = f_I.mean(axis=1)
    c_se = f_I.std(axis=1) / np.sqrt(f_I.shape[1])
    # Shared with run_sweep: normalize by L_2 mass (removes the 1/sqrt(Z)
    # factor from an unnormalized trial), then align the global sign to
    # the reference determinant of c_true.
    ref_sign = float(np.sign(c_true[0])) if c_true[0] != 0 else 0.0
    c_hat, proj_mass = normalize_and_align(c_raw, ref_sign)
    Z_est = 1.0 / proj_mass if proj_mass > 0 else float("inf")
    print(f"    K_s = {f_I.shape[1]}, n_det = {f_I.shape[0]}")
    print(f"    Sum |c_raw|^2 (basis proj mass) = {proj_mass:.5f}")
    print(f"    => Z_est = integral |Psi_NN|^2 / basis-mass ~ {Z_est:.2f}")

    # [7] Per-determinant comparison and verdict
    print(f"\n[7] Per-determinant recovery (after L_2 normalization)")
    print(f"    {'det':>32s} {'c_hat':>11s} {'c_FCI':>11s} {'diff':>11s}")
    for k, ch, ct in zip(candidate, c_hat, c_true):
        flag = "*" if abs(ch - ct) > args.recovery_tol else " "
        print(f"  {flag} {str(k):>32s} {ch:+11.5f} "
              f"{ct:+11.5f} {ch-ct:+11.5f}")

    L_inf = float(np.max(np.abs(c_hat - c_true)))
    L_2 = float(np.linalg.norm(c_hat - c_true))
    rec_ok = L_inf <= args.recovery_tol
    # Basis-completeness diagnostic: if the NN converged into the basis,
    # the basis projection mass before normalization should be close to 1
    # (the recovered CI vector exhausts the wavefunction). When it is far
    # below 1, the NN has significant amplitude *outside* the FCI basis
    # and the per-coefficient comparison is not apples-to-apples.
    basis_complete = proj_mass > 0.95

    print(f"\n=== validation summary ===")
    print(f"  E_NN - E_FCI         = {delta_E_mEh:+.3f} mE_h")
    print(f"     (NN can be lower than basis-FCI when it captures "
          f"correlation outside the basis)")
    print(f"  basis projection mass = {proj_mass:.5f}  "
          f"({'PASS' if basis_complete else 'NN extends beyond basis'})")
    print(f"  L_inf(c_hat, c_FCI)  = {L_inf:.5f}  "
          f"(tol {args.recovery_tol:.3f}, "
          f"{'PASS' if rec_ok else 'FAIL'})")
    print(f"  L_2  (c_hat, c_FCI)  = {L_2:.5f}")

    summary = dict(
        R=args.R, basis=args.basis,
        E_FCI=fci_ref["E_FCI"], E_HF=fci_ref["E_HF"],
        E_NN=e_train, delta_E_mEh=delta_E_mEh,
        c_true=[float(x) for x in c_true],
        c_raw=[float(x) for x in c_raw],
        c_hat=[float(x) for x in c_hat],
        c_se=[float(x) for x in c_se],
        candidate=[str(k) for k in candidate],
        K_s=int(f_I.shape[1]), n_det=int(f_I.shape[0]),
        proj_mass=proj_mass, Z_est=Z_est,
        L_inf=L_inf, L_2=L_2,
        basis_complete=bool(basis_complete),
        recovery_pass=bool(rec_ok),
    )
    summary_path = out_dir / f"{prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {summary_path}")

    sys.exit(0 if rec_ok else 1)


if __name__ == "__main__":
    main()
