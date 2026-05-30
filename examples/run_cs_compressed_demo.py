"""Test the random-rotation CS decoder against the identity-design baseline.

Validates the genuine compressed-sensing variant on H₂/cc-pVDZ
(100 dets, where m << N_det is tight; we can still test correctness).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import jax

sys.path.insert(0, str(Path(__file__).parent))

from OmegaQMC.utils import Mole_custom
from OmegaQMC.cs.reference import compute_fci_reference, run_rhf
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import evaluate_orbitals_on_walkers, f_I_matrix
from OmegaQMC.cs.compressed import compressed_sensing_decode
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.vmc_nn import get_vmc_nn_func
from run_cs_h4_scaling import evaluate_signed_psi  # noqa: E402


def build_mol_dispatch(geometry: str, R: float, basis: str, unit: str):
    """Same atom convention as examples/run_cs_properties.py (essential
    for trial wavefunction compatibility)."""
    if geometry == "h2":
        atoms = [("H", [0, 0, 0]), ("H", [0, 0, R])]
    elif geometry == "beh2":
        atoms = [("Be", [0, 0, 0]),
                 ("H", [0, 0, +R]), ("H", [0, 0, -R])]
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    mol = Mole_custom()
    mol.build(atom=atoms, basis=basis, spin=0, charge=0,
              unit=unit, verbose=0)
    return mol


def enumerate_all(n_orb, n_alpha, n_beta):
    a = list(combinations(range(n_orb), n_alpha))
    b = list(combinations(range(n_orb), n_beta))
    return [(x, y) for x in a for y in b]


def baseline_identity_decode(mol, walkers, psi_vals, no_coeff,
                              candidate_set, n_alpha, n_beta):
    orb_vals = evaluate_orbitals_on_walkers(
        mol, walkers, no_coeff, convention="interleaved",
        n_alpha=n_alpha, n_beta=n_beta,
    )
    f_I = f_I_matrix(orb_vals, candidate_set, psi_vals, n_alpha, n_beta)
    c_raw = np.asarray(f_I).mean(axis=1)
    return c_raw / np.linalg.norm(c_raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["h2", "beh2"], default="h2")
    ap.add_argument("--R", type=float, default=2.5)
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--unit", default="Bohr")
    ap.add_argument("--cell-dir", type=str,
                    default="cs_h2_validation/h2_R2p500_cc-pvdz")
    ap.add_argument("--ansatz", type=str,
                    default="examples/inputs/psiformer_small.yaml")
    ap.add_argument("--n-alpha", type=int, default=None)
    ap.add_argument("--n-beta", type=int, default=None)
    ap.add_argument("--candidate-tol", type=float, default=1e-10)
    ap.add_argument("--use-filtered-candidates", action="store_true",
                    help="Use FCI |c| > candidate-tol filter instead of "
                         "full enumeration (for large systems).")
    ap.add_argument("--m-values", type=str, default="20,30,50,80",
                    help="comma-separated list of m values to sweep")
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=str,
                    default="cs_h2_compressed_demo.json")
    args = ap.parse_args()

    cell_dir = Path(args.cell_dir)
    prefix = cell_dir.name

    mol = build_mol_dispatch(args.geometry, args.R, args.basis, args.unit)
    defaults = {"h2": (1, 1), "beh2": (3, 3)}
    n_alpha, n_beta = args.n_alpha, args.n_beta
    if n_alpha is None or n_beta is None:
        n_alpha, n_beta = defaults[args.geometry]
    n_orb = int(mol.nao)

    fci_ref = compute_fci_reference(mol, n_alpha=n_alpha, n_beta=n_beta,
                                    candidate_tol=args.candidate_tol)
    if args.use_filtered_candidates:
        candidate_set = fci_ref["candidate_set"]
    else:
        candidate_set = enumerate_all(n_orb, n_alpha, n_beta)
    n_det = len(candidate_set)
    print(f"=== {args.geometry.upper()} R={args.R}, n_orb={n_orb}, "
          f"n_det={n_det} ({'filtered' if args.use_filtered_candidates else 'full'}) ===")
    print(f"  E_HF = {fci_ref['E_HF']:.6f}, E_FCI = {fci_ref['E_FCI']:.6f}")

    fci_ord_cs = []
    for I in candidate_set:
        fci_ord_cs.append(fci_ref["ci_dict"].get(I, 0.0))
    c_fci = np.array(fci_ord_cs)
    c_fci = c_fci / np.linalg.norm(c_fci)

    walkers, _, _ = load_walker_bank(
        str(cell_dir / f"{prefix}_walkers.h5"))
    init_key = jax.random.split(jax.random.key(42), 4)[1]
    ansatz = args.ansatz
    driver = get_vmc_nn_func(mol, ansatz, init_key,
                              prefix=str(cell_dir / prefix))
    driver.load_checkpoint(str(cell_dir / f"{prefix}.chk.h5"))
    log_psi, _, _, _ = make_nn_log_psi(ansatz, mol, init_key)
    psi_vals = evaluate_signed_psi(walkers, np.asarray(mol.atom_coords()),
                                    driver.params, log_psi, batch_size=4096)
    print(f"  walkers: K_s = {walkers.shape[0]}")

    # 1. Baseline (identity-design) recovery
    no_coeff = fci_ref["no_coeff_ao"]
    c_hat_id = baseline_identity_decode(
        mol, walkers, psi_vals, no_coeff, candidate_set, n_alpha, n_beta,
    )
    if c_hat_id[0] * c_fci[0] < 0:
        c_hat_id = -c_hat_id
    err_id = float(np.linalg.norm(c_hat_id - c_fci))
    print(f"  baseline (identity, K_s={walkers.shape[0]}): "
          f"‖ĉ-c_FCI‖₂ = {err_id:.4f}, c[ref]={c_hat_id[0]:+.4f}")

    # 2. CS (random-rotation) recovery sweep over m
    print(f"\n  CS sweep over m values (λ={args.lam}):")
    results = []
    m_list = [int(x) for x in args.m_values.split(",")]
    for m in m_list:
        if m > n_det:
            print(f"  m={m} > n_det={n_det}, skipping")
            continue
        errs = []
        for s in range(args.seeds):
            c_hat_cs, diag = compressed_sensing_decode(
                mol, walkers, psi_vals, no_coeff, candidate_set,
                n_alpha, n_beta, m=m, lam=args.lam, seed=s,
                return_diagnostics=True,
            )
            if c_hat_cs[0] * c_fci[0] < 0:
                c_hat_cs = -c_hat_cs
            errs.append(float(np.linalg.norm(c_hat_cs - c_fci)))
        results.append({"m": m, "errs": errs,
                        "mean_err": float(np.mean(errs)),
                        "std_err": float(np.std(errs))})
        print(f"    m={m:>3} ({m/n_det:>5.1%}): "
              f"‖ĉ_CS-c_FCI‖₂ = {np.mean(errs):.4f} ± {np.std(errs):.4f}  "
              f"(vs baseline {err_id:.4f})")

    out = dict(
        molecule="H2", R=args.R, basis="cc-pvdz",
        n_orb=n_orb, n_det=n_det,
        K_s=int(walkers.shape[0]),
        baseline_err=err_id,
        baseline_C0=float(c_hat_id[0]),
        FCI_C0=float(c_fci[0]),
        m_sweep=results,
    )
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
