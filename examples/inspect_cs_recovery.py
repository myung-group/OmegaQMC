"""Head-to-head c_hat vs c_FCI inspector for a CS pilot cell.

Re-loads the FCI reference, walker bank, and trained NN params for a given
``(geometry, R, basis)`` cell, recomputes the recovered CI vector, and
prints a structured diagnostic. Useful for sanity-checking a run without
having to interpret the K_s scaling regression in isolation.

Usage:
    python examples/inspect_cs_recovery.py
    python examples/inspect_cs_recovery.py --R 1.0 --geometry linear
    python examples/inspect_cs_recovery.py --R 2.5 --geometry square --basis cc-pvdz
    python examples/inspect_cs_recovery.py --cell-dir cs_h4_results/h4_linear_..._cc-pvdz
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import jax

from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import (
    evaluate_orbitals_on_walkers,
    f_I_matrix,
    normalize_and_align,
)
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.vmc_nn import get_vmc_nn_func

# Reuse helpers from the pilot driver.
sys.path.insert(0, str(Path(__file__).parent))
from run_cs_h4_scaling import (  # noqa: E402
    build_h4_mol, cell_prefix, evaluate_signed_psi,
)


def inspect(cell_dir: Path, geometry: str, R: float, basis: str,
            ansatz: str, n_alpha: int = 2, n_beta: int = 2,
            psi_batch: int = 4096):
    prefix = cell_dir.name
    walkers_path = cell_dir / f"{prefix}_walkers.h5"
    checkpoint = cell_dir / f"{prefix}.chk.h5"
    if not walkers_path.exists():
        raise FileNotFoundError(f"walker bank not found: {walkers_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    mol = build_h4_mol(R, basis, geometry)
    fci = compute_fci_reference(mol, n_alpha=n_alpha, n_beta=n_beta)
    candidate = fci["candidate_set"]
    c_true = np.array([fci["ci_dict"][k] for k in candidate])

    print(f"=== cell: {prefix} ===")
    print(f"  geometry={geometry}  R={R} A  basis={basis}")
    print(f"  E_HF  = {fci['E_HF']: .6f} Ha")
    print(f"  E_FCI = {fci['E_FCI']: .6f} Ha")

    walkers, _, bank_meta = load_walker_bank(str(walkers_path))
    print(f"  walker bank: K_s = {walkers.shape[0]}, "
          f"n_det in candidate = {len(candidate)}")

    init_key = jax.random.split(jax.random.key(42), 4)[1]
    driver = get_vmc_nn_func(mol, ansatz, init_key, prefix=str(cell_dir / prefix))
    driver.load_checkpoint(str(checkpoint))
    log_psi, _, _, _ = make_nn_log_psi(ansatz, mol, init_key)

    psi_vals = evaluate_signed_psi(
        walkers, np.asarray(mol.atom_coords()),
        driver.params, log_psi, batch_size=psi_batch,
    )
    orb = evaluate_orbitals_on_walkers(
        mol, walkers, fci["no_coeff_ao"],
        convention="interleaved", n_alpha=n_alpha, n_beta=n_beta,
    )
    f_I = f_I_matrix(orb, candidate, psi_vals, n_alpha=n_alpha, n_beta=n_beta)
    c_raw = f_I.mean(axis=1)
    c_se = f_I.std(axis=1) / np.sqrt(f_I.shape[1])
    c_hat, proj_mass = normalize_and_align(c_raw, float(np.sign(c_true[0])))
    c_se_norm = c_se / np.sqrt(proj_mass)

    L_inf = float(np.max(np.abs(c_hat - c_true)))
    L_2 = float(np.linalg.norm(c_hat - c_true))
    overlap = float(np.dot(c_hat, c_true))
    pearson = float(np.corrcoef(c_hat, c_true)[0, 1])

    print(f"\n--- wavefunction-level diagnostics ---")
    print(f"  proj_mass = {proj_mass:.4e}  (Z ~ {1/proj_mass:.2e})")
    print(f"  <c_hat | c_FCI> overlap = {overlap:+.5f}")
    print(f"  Pearson(c_hat, c_FCI)   = {pearson:+.5f}")
    print(f"  L_inf = {L_inf:.5f}   L_2 = {L_2:.5f}   "
          f"over n_det = {len(candidate)}")

    print(f"\n--- top 12 determinants by |c_FCI| ---")
    print(f"  {'rank':>4} {'det':>22s} {'c_hat':>10s} {'±SE':>8s} "
          f"{'c_FCI':>10s} {'diff':>10s} {'rel':>8s}")
    order_fci = np.argsort(-np.abs(c_true))[:12]
    for r, i in enumerate(order_fci):
        rel = ((c_hat[i] - c_true[i]) / c_true[i] * 100
               if abs(c_true[i]) > 1e-4 else 0.0)
        print(f"  {r:>4} {str(candidate[i]):>22s} {c_hat[i]:+10.5f}"
              f" {c_se_norm[i]:8.4f} {c_true[i]:+10.5f}"
              f" {c_hat[i]-c_true[i]:+10.5f} {rel:+7.1f}%")

    print(f"\n--- top 8 by |c_hat| (recovered ranking) ---")
    print(f"  {'rank':>4} {'det':>22s} {'c_hat':>10s} {'c_FCI':>10s}")
    for r, i in enumerate(np.argsort(-np.abs(c_hat))[:8]):
        print(f"  {r:>4} {str(candidate[i]):>22s} {c_hat[i]:+10.5f}"
              f" {c_true[i]:+10.5f}")

    print(f"\n--- spurious mass (recovered where |c_FCI|<1e-3) ---")
    mask = np.abs(c_true) < 1e-3
    if mask.any():
        spurious = np.argsort(-np.abs(c_hat) * mask)[:8]
        for i in spurious:
            print(f"  {str(candidate[i]):>22s} {c_hat[i]:+10.5f}"
                  f" {c_true[i]:+10.5f}")
        spurious_mass = float(np.sum(c_hat[mask] ** 2))
        true_mass_in_mask = float(np.sum(c_true[mask] ** 2))
        print(f"  total spurious Σ|c_hat|² = {spurious_mass:.5f}  "
              f"(true mass in same set: {true_mass_in_mask:.5f})")

    print(f"\n--- singlet-symmetry pairs (should have c[(a,b),(c,d)]=c[(c,d),(a,b)]) ---")
    print(f"  {'pair':>40s} {'c_hat':>10s} {'c_hat_sym':>10s} {'asym':>10s}")
    det_map = {k: i for i, k in enumerate(candidate)}
    shown = 0
    for i, (oa, ob) in enumerate(candidate):
        if shown >= 6:
            break
        sym_key = (ob, oa)
        if sym_key in det_map and det_map[sym_key] != i \
                and abs(c_true[i]) > 5e-3:
            j = det_map[sym_key]
            if i < j:
                asym = c_hat[i] - c_hat[j]
                print(f"  {str((oa, ob)) + ' vs ' + str(sym_key):>40s}"
                      f" {c_hat[i]:+10.5f} {c_hat[j]:+10.5f} {asym:+10.5f}")
                shown += 1

    print(f"\n--- error histogram ---")
    err = np.abs(c_hat - c_true)
    for thr in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4]:
        n = int(np.sum(err > thr))
        print(f"  |err| > {thr:.0e}: {n:5d}  ({100*n/len(err):5.2f}%)")

    print(f"\n--- cumulative mass on FCI-ranked top-K ---")
    print(f"  {'K':>6} {'Σ|c_FCI|²':>14} {'Σ|c_hat|²':>14} {'ratio':>10}")
    fci_order = np.argsort(-np.abs(c_true))
    for K in [1, 4, 10, 50, 100, 500, 1000, 2000, len(candidate)]:
        if K > len(candidate):
            continue
        top = fci_order[:K]
        fci_mass = float(np.sum(c_true[top] ** 2))
        hat_mass = float(np.sum(c_hat[top] ** 2))
        ratio = hat_mass / fci_mass if fci_mass > 0 else float("nan")
        print(f"  {K:>6d} {fci_mass:>14.5f} {hat_mass:>14.5f} {ratio:>10.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-dir", default=None,
                   help="Explicit cell directory. If omitted, built from "
                        "--geometry, --R, --basis, --ansatz-tag, --out-dir.")
    p.add_argument("--out-dir", default="cs_h4_results")
    p.add_argument("--geometry", default="linear", choices=["square", "linear"])
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--ansatz-tag", default="ferminet_jastrow")
    p.add_argument("--ansatz", default=str(
        Path(__file__).parents[1] / "OmegaQMC" / "psi" / "nn" / "conf"
        / "ferminet_jastrow.yaml"))
    p.add_argument("--n-alpha", type=int, default=2)
    p.add_argument("--n-beta", type=int, default=2)
    p.add_argument("--psi-batch", type=int, default=4096)
    args = p.parse_args()

    if args.cell_dir is None:
        prefix = cell_prefix(args.R, args.basis, args.ansatz_tag, args.geometry)
        cell_dir = Path(args.out_dir) / prefix
    else:
        cell_dir = Path(args.cell_dir)

    if not cell_dir.exists():
        sys.exit(f"cell directory does not exist: {cell_dir}")

    inspect(cell_dir, args.geometry, args.R, args.basis, args.ansatz,
            n_alpha=args.n_alpha, n_beta=args.n_beta,
            psi_batch=args.psi_batch)


if __name__ == "__main__":
    main()
