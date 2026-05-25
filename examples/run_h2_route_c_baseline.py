"""Route (c) baseline: FCI second-root in cc-pVDZ as the gold standard.

The orthogonality-physicality trade-off documented for routes (1),
(2), (3), and (a) is a consequence of the NN ansatz extending
beyond the chosen Gaussian basis. A *basis-restricted* ansatz
eliminates that pathway: the wavefunction lives entirely in
span{D_I}, and CI orthogonality is enforced exactly by the eigen-
decomposition of the Hamiltonian in that subspace.

The simplest basis-restricted ansatz is the FCI/CASCI ansatz itself
(no NN, no Jastrow). For H2/cc-pVDZ at R = 2.5 Bohr this is just
the 10-AO FCI matrix's eigendecomposition. The ground state is the
first eigenvector; the first excited (Sigma_g+) singlet is the
second. Orthogonality is exact to machine precision.

The cost is the absence of any out-of-basis correlation: the
ground-state energy reaches only the cc-pVDZ FCI limit
(~-1.087 Ha) and the excited-state energy is the cc-pVDZ basis-set
truncation of the true excited state. The NN-VMC ansatz would, if
trained without the NES constraint, recover more correlation
energy --- but at the cost of being non-orthogonal to the
ground-state CI projection, as documented in routes (1)-(a).

The right paper-grade route forward is a *Slater-Jastrow CI* hybrid:
Psi(R; theta_NN, c) = J(R; theta_NN) * Sum_I c_I D_I(R), where the c
vector is an explicit learnable parameter set (so orthogonality can
be enforced exactly on the c's) and the neural Jastrow J captures
the out-of-basis dynamic correlation as a multiplicative correction.
This script demonstrates the c-vector half of that architecture in
its purest form.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from pyscf import fci

from OmegaQMC.utils import Mole_custom
from OmegaQMC.cs.reference import (
    run_rhf, run_fci, natural_orbitals, ci_to_dict, reference_determinant,
    ordered_candidate_set,
)


def build_h2(R_bohr, basis):
    mol = Mole_custom()
    mol.build(
        atom=[("H", [0, 0, 0]), ("H", [0, 0, R_bohr])],
        basis=basis, spin=0, charge=0, unit="Bohr", verbose=0,
    )
    return mol


def fci_two_roots(mol, mo_coeff):
    """Run FCI with nroots=2 in the basis spanned by ``mo_coeff``.

    Returns (energies[2], ci_matrices[2], cisolver).
    """
    cisolver = fci.FCI(mol, mo_coeff)
    cisolver.verbose = 0
    cisolver.nroots = 2
    E_list, ci_list = cisolver.kernel()
    return (
        [float(E_list[0]), float(E_list[1])],
        [np.asarray(ci_list[0]), np.asarray(ci_list[1])],
        cisolver,
    )


def ci_matrix_to_dict_aligned(ci, norb, n_alpha, n_beta, candidate_set):
    """Project a CI matrix onto the supplied ordered candidate set.

    Returns a (n_det,) numpy array of CI coefficients in
    ``candidate_set`` order, padded with zeros for determinants not
    present in the matrix's support.
    """
    full_dict = ci_to_dict(ci, norb, n_alpha, n_beta, tol=0.0)
    return np.array(
        [full_dict.get(k, 0.0) for k in candidate_set],
        dtype=np.float64,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=2.5)
    p.add_argument("--basis", default="cc-pvdz")
    args = p.parse_args()

    print(f"=== Route (c) baseline: H2/{args.basis} at R={args.R} Bohr ===")
    mol = build_h2(args.R, args.basis)
    n_alpha, n_beta = 1, 1
    nelec = (n_alpha, n_beta)
    print(f"  mol: {mol.nao} AOs, nelec={nelec}")

    # Step 1: RHF
    E_HF, mo_coeff = run_rhf(mol)
    print(f"\n[1/4] RHF: E = {E_HF:.6f} Ha")

    # Step 2: Single-root FCI for natural orbitals (canonical CS pipeline)
    E_FCI_hf, ci_hf, cisolver_hf = run_fci(mol, mo_coeff)
    occ_no, no_coeff = natural_orbitals(cisolver_hf, ci_hf, mo_coeff, nelec)
    print(f"\n[2/4] FCI in HF basis: E_FCI = {E_FCI_hf:.6f} Ha")
    print(f"      NO occupations: {occ_no}")

    # Step 3: Two-root FCI in the natural-orbital basis
    norb = int(no_coeff.shape[1])
    print(f"\n[3/4] Two-root FCI in NO basis (nroots=2)")
    E_list, ci_list, _ = fci_two_roots(mol, no_coeff)
    print(f"  Ground state: E^(0) = {E_list[0]:.6f} Ha")
    print(f"  1st excited:  E^(1) = {E_list[1]:.6f} Ha")
    print(f"  Excitation:   dE     = "
          f"{(E_list[1] - E_list[0]) * 27.2114:.4f} eV "
          f"({(E_list[1] - E_list[0]) * 1e3:.2f} mE_h)")

    # Step 4: Compute CI vectors on a unified candidate set + orthogonality
    print(f"\n[4/4] CI-vector orthogonality")
    ci_dict_0 = ci_to_dict(ci_list[0], norb, n_alpha, n_beta, tol=1e-10)
    ci_dict_1 = ci_to_dict(ci_list[1], norb, n_alpha, n_beta, tol=1e-10)
    # Union candidate set
    all_keys = sorted(set(ci_dict_0) | set(ci_dict_1))
    ref = reference_determinant(ci_dict_0)
    candidate_set = ordered_candidate_set(
        {k: ci_dict_0.get(k, 0.0) + ci_dict_1.get(k, 0.0)
         for k in all_keys},
        ref,
    )
    n_det = len(candidate_set)
    print(f"  |union candidate set| = {n_det}")

    c0 = ci_matrix_to_dict_aligned(
        ci_list[0], norb, n_alpha, n_beta, candidate_set,
    )
    c1 = ci_matrix_to_dict_aligned(
        ci_list[1], norb, n_alpha, n_beta, candidate_set,
    )
    # Align c0 sign so the reference coefficient is positive (matches
    # the convention used throughout the CS pipeline)
    if c0[0] < 0:
        c0 = -c0
    if c1[0] < 0 and abs(c1[0]) > 1e-6:
        c1 = -c1

    n0 = float(np.linalg.norm(c0))
    n1 = float(np.linalg.norm(c1))
    ovl = float(np.dot(c0, c1)) / (n0 * n1)
    diff = float(np.linalg.norm(c0 / n0 - c1 / n1))

    print(f"  ||c^(0)||  = {n0:.6f}")
    print(f"  ||c^(1)||  = {n1:.6f}")
    print(f"  <c^(0) | c^(1)> (normalised) = {ovl:+.2e}")
    print(f"  ||c^(0) - c^(1)||_2 (normalised) = {diff:.4f}")
    if abs(ovl) < 1e-10:
        print(f"  CI orthogonality EXACT (machine precision)")
    elif abs(ovl) < 1e-4:
        print(f"  CI orthogonality near-machine-precision")
    else:
        print(f"  WARNING: orthogonality larger than expected")

    print(f"\n  Top |c^(0)| coefficients:")
    sorted_idx_0 = np.argsort(-np.abs(c0))[:5]
    for i in sorted_idx_0:
        print(f"    {candidate_set[i]}  c = {c0[i]:+.4f}")
    print(f"\n  Top |c^(1)| coefficients:")
    sorted_idx_1 = np.argsort(-np.abs(c1))[:5]
    for i in sorted_idx_1:
        print(f"    {candidate_set[i]}  c = {c1[i]:+.4f}")

    print(f"\n=== Comparison to NN-VMC NES variants ===")
    print(f"  Route (c) (FCI 2-root):     "
          f"E^(1) = {E_list[1]:.4f} Ha, overlap = {ovl:+.2e}")
    print(f"  Route (a) lambda=1   (NES): "
          f"E^(1) = -0.590  Ha, overlap = +0.870  (physical, NOT orthogonal)")
    print(f"  Route (a) lambda=10  (NES): "
          f"E^(1) = +0.209  Ha, overlap = +0.088  (orthogonal, UNPHYSICAL)")
    print(f"\nRoute (c) achieves BOTH simultaneously, at the cost of")
    print(f"basis-set truncation (no out-of-basis correlation).")
    print(f"A Slater-Jastrow CI hybrid would close that gap while")
    print(f"preserving exact orthogonality on the explicit c vector.")


if __name__ == "__main__":
    main()
