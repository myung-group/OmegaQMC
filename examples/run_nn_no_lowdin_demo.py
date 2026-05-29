"""Generalised Löwdin-AO bootstrap CS recovery demo (Option 1).

Drop-in for any molecule with an existing trained NN-VMC checkpoint
and walker bank. Workflow:

  Pass 1: Löwdin-orthogonalised AOs (S^{-1/2} on raw AOs; no SCF)
          -> CS sweep -> c_hat^Löwdin in this basis
  1-RDM   built from c_hat^Löwdin in Löwdin basis
  Diag    eigenvectors U give NN-NOs as mixtures of Löwdin AOs
  Pass 2: NN-NO basis -> CS sweep -> c_hat^NN-NO

Compares NN-NO frame against FCI-NO frame for validation (the FCI
reference is for VALIDATION ONLY -- the framework itself does not
require FCI).
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import jax
import scipy.linalg as sla

from OmegaQMC.utils import Mole_custom
from OmegaQMC.cs.reference import compute_fci_reference, run_rhf
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import evaluate_orbitals_on_walkers, f_I_matrix
from OmegaQMC.cs.properties import compute_1rdm
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.vmc_nn import get_vmc_nn_func

import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_cs_h4_scaling import evaluate_signed_psi  # noqa: E402


def build_mol(geometry: str, R: float, basis: str,
              unit: str = "Angstrom") -> Mole_custom:
    """Matches the atom-ordering convention of examples/run_cs_properties.py
    so the walker banks dumped by training stay aligned with the NN's
    nuclei embedding."""
    if geometry == "h2":
        atoms = [("H", [0, 0, 0]), ("H", [0, 0, R])]
    elif geometry == "beh2":
        atoms = [("Be", [0, 0, 0]),
                 ("H", [0, 0, +R]), ("H", [0, 0, -R])]
    elif geometry == "lih":
        atoms = [("Li", [0, 0, 0]), ("H", [0, 0, R])]
    elif geometry == "n2":
        atoms = [("N", [0, 0, 0]), ("N", [0, 0, R])]
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    mol = Mole_custom()
    mol.build(atom=atoms, basis=basis, spin=0, charge=0,
              unit=unit, verbose=0)
    return mol


def lowdin_orbitals(mol):
    S = mol.intor("int1e_ovlp")
    return np.asarray(sla.fractional_matrix_power(S, -0.5).real), S


def cs_sweep_one_basis(mol, walkers, psi_vals, no_coeff_ao,
                       candidate_set, n_alpha, n_beta,
                       walker_convention="interleaved"):
    orb_vals = evaluate_orbitals_on_walkers(
        mol, walkers, no_coeff_ao,
        convention=walker_convention, n_alpha=n_alpha, n_beta=n_beta,
    )
    f_I = f_I_matrix(orb_vals, candidate_set, psi_vals, n_alpha, n_beta)
    c_raw = np.asarray(f_I).mean(axis=1)
    norm = float(np.linalg.norm(c_raw))
    if norm == 0.0:
        raise RuntimeError("zero-norm c_raw")
    return c_raw / norm, c_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-dir", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--R", type=float, required=True)
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--unit", default="Angstrom")
    ap.add_argument("--n-alpha", type=int, required=True)
    ap.add_argument("--n-beta", type=int, required=True)
    ap.add_argument("--ansatz", required=True)
    ap.add_argument("--candidate-tol", type=float, default=1e-4)
    ap.add_argument("--pass1-basis", choices=["hf", "lowdin"], default="hf",
                    help="Pass-1 orbital frame. 'hf' is robust for filtered "
                         "candidate sets (chemistry-aligned). 'lowdin' "
                         "requires full enumeration to be equivalent.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--psi-batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cell_dir = Path(args.cell_dir)
    prefix = cell_dir.name
    walkers_path = cell_dir / f"{prefix}_walkers.h5"
    checkpoint = cell_dir / f"{prefix}.chk.h5"
    for p in (walkers_path, checkpoint):
        if not p.exists():
            raise FileNotFoundError(p)

    print(f"=== Löwdin-bootstrap NN-NO recovery  ({prefix}) ===")
    mol = build_mol(args.geometry, args.R, args.basis, args.unit)
    n_alpha, n_beta = args.n_alpha, args.n_beta
    n_orb = int(mol.nao)
    print(f"  mol: {n_orb} AOs, nelec=({n_alpha},{n_beta}), basis={args.basis}")

    # 1) Pass-1 basis: HF (one SCF cycle) or Löwdin (S^{-1/2}, no SCF)
    _, S = lowdin_orbitals(mol)  # S always needed for alignment
    if args.pass1_basis == "hf":
        E_HF, pass1_coeff = run_rhf(mol)
        print(f"  Pass-1 basis: HF orbitals  (E_HF = {E_HF: .6f} Ha)")
    else:
        pass1_coeff, _ = lowdin_orbitals(mol)
        ortho_err = float(np.max(np.abs(
            pass1_coeff.T @ S @ pass1_coeff - np.eye(n_orb)
        )))
        print(f"  Pass-1 basis: Löwdin AOs  "
              f"(no SCF; ortho residual = {ortho_err:.2e})")

    # 2) FCI reference (VALIDATION ONLY -- not used by framework)
    print("  computing FCI reference (validation only)...")
    fci_ref = compute_fci_reference(
        mol, n_alpha=n_alpha, n_beta=n_beta,
        candidate_tol=args.candidate_tol,
    )
    candidate_set = fci_ref["candidate_set"]
    fci_no_coeff_ao = np.asarray(fci_ref["no_coeff_ao"])
    fci_occ_no = np.asarray(fci_ref["occ_no"])
    print(f"  E_FCI({args.basis}) = {fci_ref['E_FCI']:.6f} Ha; "
          f"|cand set| = {len(candidate_set)}")

    # 3) Load walker bank + signed Psi
    walkers, _, _ = load_walker_bank(str(walkers_path))
    init_key = jax.random.split(jax.random.key(args.seed), 4)[1]
    driver = get_vmc_nn_func(mol, args.ansatz, init_key,
                              prefix=str(cell_dir / prefix))
    driver.load_checkpoint(str(checkpoint))
    log_psi, _, _, _ = make_nn_log_psi(args.ansatz, mol, init_key)
    psi_vals = evaluate_signed_psi(
        walkers, np.asarray(mol.atom_coords()),
        driver.params, log_psi, batch_size=args.psi_batch,
    )
    K_s = walkers.shape[0]
    print(f"  K_s = {K_s} walkers")

    # 4) Pass 1: Löwdin basis
    c_hat_lowdin, c_raw_lowdin = cs_sweep_one_basis(
        mol, walkers, psi_vals, pass1_coeff,
        candidate_set, n_alpha, n_beta,
    )
    print(f"  Pass 1 (Löwdin basis):"
          f"  ||c_hat||_2 = 1, ||c_raw||_2 = {np.linalg.norm(c_raw_lowdin):.4e}")

    # 5) 1-RDM in Löwdin basis -> NN-NOs
    gamma_lowdin = compute_1rdm(c_hat_lowdin, candidate_set, n_orb,
                                (n_alpha, n_beta))
    gamma_lowdin = 0.5 * (gamma_lowdin + gamma_lowdin.T)
    n_nn, U = np.linalg.eigh(gamma_lowdin)
    order = np.argsort(-n_nn)
    n_nn = n_nn[order]
    U = U[:, order]
    nn_no_coeff_ao = np.asarray(pass1_coeff) @ U
    print(f"  NN-NO occupations (top 6): "
          + ", ".join(f"{x:.4f}" for x in n_nn[:6]))
    print(f"  FCI-NO occupations (top 6): "
          + ", ".join(f"{x:.4f}" for x in fci_occ_no[:6]))

    # 6) Pass 2: NN-NO basis
    c_hat_nn, c_raw_nn = cs_sweep_one_basis(
        mol, walkers, psi_vals, nn_no_coeff_ao,
        candidate_set, n_alpha, n_beta,
    )

    # 7) NN-NO / FCI-NO orbital alignment (top occupied)
    nn_overlap_fci = nn_no_coeff_ao.T @ S @ fci_no_coeff_ao
    diag = np.abs(np.diag(nn_overlap_fci))
    print(f"  NN-NO vs FCI-NO orbital alignment (top 6): "
          + ", ".join(f"{x:.4f}" for x in diag[:6]))

    # 8) Reference determinant amplitude (sign-aligned to FCI's C0)
    ref_det = fci_ref["reference_det"]
    ref_idx = candidate_set.index(ref_det)
    c_fci_ref = list(fci_ref["ci_dict"].values())[
        list(fci_ref["ci_dict"].keys()).index(ref_det)
    ]
    if c_hat_nn[ref_idx] * c_fci_ref < 0:
        c_hat_nn = -c_hat_nn
    print(f"  C0 (reference determinant amplitude):"
          f"  c_hat^NN-NO = {c_hat_nn[ref_idx]:+.6f},  "
          f"c_FCI = {c_fci_ref:+.6f}")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = dict(
        molecule=args.geometry, R=args.R, basis=args.basis,
        n_orb=n_orb, nelec=[n_alpha, n_beta],
        K_s=int(K_s),
        candidate_set_size=len(candidate_set),
        E_FCI=float(fci_ref["E_FCI"]),
        c_hat_NN_NO_top10=c_hat_nn[:10].tolist(),
        c_hat_Lowdin_top10=c_hat_lowdin[:10].tolist(),
        nn_no_occupations=n_nn.tolist(),
        fci_no_occupations=fci_occ_no.tolist(),
        nn_no_vs_fci_no_diag_alignment=diag.tolist(),
        c0_NN_NO=float(c_hat_nn[ref_idx]),
        c0_FCI=float(c_fci_ref),
        framework_inputs_used="AO basis + Lowdin S^(-1/2); no HF, CASCI, FCI",
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
