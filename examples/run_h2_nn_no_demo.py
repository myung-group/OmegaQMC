"""H2/cc-pVDZ self-consistent NN-NO demonstration (Option 1).

Replaces the FCI-derived natural-orbital frame with one derived
purely from the NN-VMC 1-RDM, leaving the rest of the CS pipeline
unchanged. Two-pass workflow:

  Pass 1: HF orbitals -> c_hat^HF -> 1-RDM in HF basis -> diagonalise
          gives NN-NOs as the rotation U from HF to NN-NO basis.
  Pass 2: NN-NO orbitals (pass1_coeff @ U) -> c_hat^NN-NO.

For H2 at R=2.5 a0 in cc-pVDZ the candidate set is enumerated as the
full (n_alpha=1, n_beta=1) combinatorics in the 10-AO basis (100
determinants). No FCI or CASCI calculation is invoked.

The result is compared against the existing FCI-NO calibration:
  - HF-coefficient recovery in NN-NO frame
  - NN-NO occupations vs FCI-NO occupations
  - Vector overlap of c_hat^NN-NO with c^FCI after rotating to NN-NO frame
"""

from __future__ import annotations
import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import jax

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


def build_h2_mol(R_bohr: float, basis: str = "cc-pvdz") -> Mole_custom:
    mol = Mole_custom()
    mol.atom = f"H 0.0 0.0 0.0; H 0.0 0.0 {R_bohr}"
    mol.unit = "Bohr"
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.build()
    return mol


def enumerate_all_determinants(n_orb: int, n_alpha: int, n_beta: int):
    occs_a = list(combinations(range(n_orb), n_alpha))
    occs_b = list(combinations(range(n_orb), n_beta))
    return [(a, b) for a in occs_a for b in occs_b]


def cs_sweep_one_basis(mol, walkers, psi_vals, no_coeff_ao,
                       candidate_set, n_alpha, n_beta,
                       walker_convention: str = "interleaved") -> np.ndarray:
    """One CS pass: orb_vals -> f_I -> sample mean -> normalised c_hat."""
    orb_vals = evaluate_orbitals_on_walkers(
        mol, walkers, no_coeff_ao,
        convention=walker_convention,
        n_alpha=n_alpha, n_beta=n_beta,
    )
    f_I = f_I_matrix(orb_vals, candidate_set, psi_vals, n_alpha, n_beta)
    c_raw = np.asarray(f_I).mean(axis=1)
    norm = float(np.linalg.norm(c_raw))
    if norm == 0.0:
        raise RuntimeError("zero-norm c_raw; check walker bank / psi_vals")
    return c_raw / norm, c_raw


def align_sign(c_hat: np.ndarray, ref_idx: int, ref_sign: float) -> np.ndarray:
    if c_hat[ref_idx] * ref_sign < 0:
        return -c_hat
    return c_hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=2.5)
    ap.add_argument("--basis", type=str, default="cc-pvdz")
    ap.add_argument("--cell-dir", type=str,
                    default="cs_h2_validation/h2_R2p500_cc-pvdz")
    ap.add_argument("--ansatz", type=str,
                    default="examples/inputs/psiformer_small.yaml")
    ap.add_argument("--out", type=str,
                    default="cs_h2_nn_no_demo/h2_R2p500_cc-pvdz_nn_no.json")
    ap.add_argument("--psi-batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pass1-basis", choices=["hf", "lowdin"], default="hf",
                    help="Phase-1 orbital frame; 'hf' runs RHF, 'lowdin' "
                         "uses S^(-1/2) on the raw AOs (no SCF iteration).")
    args = ap.parse_args()

    cell_dir = Path(args.cell_dir)
    prefix = cell_dir.name
    walkers_path = cell_dir / f"{prefix}_walkers.h5"
    checkpoint = cell_dir / f"{prefix}.chk.h5"
    for p in (walkers_path, checkpoint):
        if not p.exists():
            raise FileNotFoundError(p)

    print(f"=== H2 NN-NO bootstrap demo  ({prefix}) ===")
    mol = build_h2_mol(args.R, args.basis)
    n_alpha, n_beta = 1, 1
    n_orb = int(mol.nao)
    print(f"  mol: {n_orb} AOs, nelec=({n_alpha},{n_beta})")

    # 1) Pass-1 orthonormal one-particle basis (no FCI / CASCI)
    if args.pass1_basis == "hf":
        E_HF, pass1_coeff = run_rhf(mol)
        print(f"  Pass-1 basis: HF orbitals  (E_HF = {E_HF: .6f} Ha)")
    elif args.pass1_basis == "lowdin":
        import scipy.linalg as sla
        S = mol.intor("int1e_ovlp")
        pass1_coeff = np.asarray(sla.fractional_matrix_power(S, -0.5).real)
        E_HF = float("nan")  # not computed
        # Sanity-check orthonormality: pass1_coeff^T S pass1_coeff = I
        ortho_err = float(np.max(np.abs(
            pass1_coeff.T @ S @ pass1_coeff - np.eye(n_orb)
        )))
        print(f"  Pass-1 basis: Löwdin AOs  (no SCF; "
              f"orthonormality residual = {ortho_err:.2e})")

    # 2) Candidate set = all determinants in the 10-AO basis (100 dets)
    candidate_set = enumerate_all_determinants(n_orb, n_alpha, n_beta)
    print(f"  |candidate set| = {len(candidate_set)} "
          f"(full enumeration, no FCI / CASCI filter)")
    ref_det_idx = candidate_set.index(((0,), (0,)))

    # 3) Load walker bank + signed NN-VMC trial
    walkers, _, _ = load_walker_bank(str(walkers_path))
    init_key = jax.random.split(jax.random.key(args.seed), 4)[1]
    driver = get_vmc_nn_func(mol, args.ansatz, init_key, prefix=str(cell_dir / prefix))
    driver.load_checkpoint(str(checkpoint))
    log_psi, _, _, _ = make_nn_log_psi(args.ansatz, mol, init_key)
    psi_vals = evaluate_signed_psi(
        walkers, np.asarray(mol.atom_coords()),
        driver.params, log_psi, batch_size=args.psi_batch,
    )
    K_s = walkers.shape[0]
    print(f"  K_s = {K_s} walkers")

    # 4) Pass 1: CS sweep in HF basis
    c_hat_hf, c_raw_hf = cs_sweep_one_basis(
        mol, walkers, psi_vals, pass1_coeff,
        candidate_set, n_alpha, n_beta,
    )
    # sign-align to make c_HF[(0)(0)] > 0
    c_hat_hf = align_sign(c_hat_hf, ref_det_idx, +1.0)
    print(f"  Pass 1 (HF basis):")
    print(f"    HF coefficient c_hat[(0)(0)] = {c_hat_hf[ref_det_idx]:+.6f}")

    # 5) NN-VMC 1-RDM in HF basis -> NN-NOs
    gamma_hf = compute_1rdm(c_hat_hf, candidate_set, n_orb, (n_alpha, n_beta))
    gamma_hf = 0.5 * (gamma_hf + gamma_hf.T)
    n_nn, U = np.linalg.eigh(gamma_hf)
    order = np.argsort(-n_nn)
    n_nn = n_nn[order]
    U = U[:, order]
    # NN-NO orbital coefficients in AO basis
    nn_no_coeff_ao = np.asarray(pass1_coeff) @ U
    print(f"  NN-NO occupations (top 4): "
          f"{n_nn[0]:.4f}, {n_nn[1]:.4f}, {n_nn[2]:.4f}, {n_nn[3]:.4f}")

    # 6) Pass 2: CS sweep in NN-NO basis
    c_hat_nn, c_raw_nn = cs_sweep_one_basis(
        mol, walkers, psi_vals, nn_no_coeff_ao,
        candidate_set, n_alpha, n_beta,
    )
    c_hat_nn = align_sign(c_hat_nn, ref_det_idx, +1.0)
    print(f"  Pass 2 (NN-NO basis):")
    print(f"    HF coefficient c_hat[(0)(0)] = {c_hat_nn[ref_det_idx]:+.6f}")

    # 7) Calibration: FCI in FCI-NO basis (one-time, only for comparison)
    fci_ref = compute_fci_reference(mol, n_alpha=n_alpha, n_beta=n_beta,
                                    candidate_tol=1e-10)
    c_fci_dict = fci_ref["ci_dict"]
    # Align FCI candidate ordering to ours
    fci_cand_set = list(c_fci_dict.keys())
    fci_no_coeff_ao = np.asarray(fci_ref["no_coeff_ao"])
    fci_occ_no = np.asarray(fci_ref["occ_no"])
    print(f"  FCI-NO occupations (top 4): "
          f"{fci_occ_no[0]:.4f}, {fci_occ_no[1]:.4f}, "
          f"{fci_occ_no[2]:.4f}, {fci_occ_no[3]:.4f}")

    # 8) Frame-difference report: NN-NO vs FCI-NO orbitals
    # cos(angle) between corresponding orbitals after AO overlap-weighted
    # inner product
    S_ao = mol.intor("int1e_ovlp")
    nn_overlap_fci = nn_no_coeff_ao.T @ S_ao @ fci_no_coeff_ao  # (n,n)
    diag = np.abs(np.diag(nn_overlap_fci))
    print(f"  NN-NO/FCI-NO frame alignment (first 4 diag(|U^T S V|)): "
          f"{diag[0]:.4f}, {diag[1]:.4f}, {diag[2]:.4f}, {diag[3]:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = dict(
        molecule="H2", R_bohr=args.R, basis=args.basis,
        n_orb=n_orb, nelec=[n_alpha, n_beta],
        candidate_set_size=len(candidate_set),
        K_s=int(K_s),
        E_HF=float(E_HF),
        E_FCI=float(fci_ref["E_FCI"]),
        c_hat_HF_basis=c_hat_hf.tolist(),
        c_hat_NN_NO_basis=c_hat_nn.tolist(),
        nn_no_occupations=n_nn.tolist(),
        fci_no_occupations=fci_occ_no.tolist(),
        nn_no_vs_fci_no_diag_alignment=diag.tolist(),
        hf_coeff_recovery_HF_basis=float(c_hat_hf[ref_det_idx]),
        hf_coeff_recovery_NN_NO_basis=float(c_hat_nn[ref_det_idx]),
        hf_coeff_FCI=float(list(c_fci_dict.values())[
            fci_cand_set.index(((0,), (0,)))
        ]),
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
