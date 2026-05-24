"""
FCI reference, natural orbitals, and candidate set for the H4-square
compressed-sensing scaling experiment.

This module is the data-collection side's source of ground truth. It computes:

- the FCI ground-state energy and CI vector,
- natural orbitals from the FCI 1-RDM,
- the FCI CI vector re-expressed in the natural-orbital basis,
- the candidate set ordered with the reference determinant at index 0,
- summary statistics consumed by :mod:`OmegaQMC.cs.analysis`.

Stable PySCF API only (gto, scf.RHF, fci.FCI, fci.cistring).
"""

from typing import Mapping, Optional, Sequence

import numpy as np

from pyscf import gto, scf, fci
from pyscf.fci import cistring


def build_h4_square(R: float, basis: str = "cc-pVDZ", unit: str = "Angstrom"):
    """H4 with the four atoms at the corners of a square of side ``R``."""
    atoms = (
        f"H  0.0  0.0  0.0\n"
        f"H  {R}  0.0  0.0\n"
        f"H  {R}  {R}  0.0\n"
        f"H  0.0  {R}  0.0\n"
    )
    return gto.M(atom=atoms, basis=basis, unit=unit, symmetry=False, verbose=0)


def run_rhf(mol):
    """RHF; returns ``(E_HF, mo_coeff)``."""
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    return float(mf.e_tot), np.asarray(mf.mo_coeff)


def run_fci(mol, mo_coeff):
    """FCI in the basis spanned by ``mo_coeff``.

    Returns ``(E_FCI, ci_matrix, cisolver)``. The CI matrix is indexed by
    (alpha-string-index, beta-string-index).
    """
    cisolver = fci.FCI(mol, mo_coeff)
    cisolver.verbose = 0
    E, ci = cisolver.kernel()
    return float(E), np.asarray(ci), cisolver


def natural_orbitals(cisolver, ci, mo_coeff, nelec):
    """Diagonalize the spin-summed 1-RDM produced by ``cisolver``.

    Returns ``(occ_numbers_desc, no_coeff_in_ao_basis)``.
    """
    norb = mo_coeff.shape[1]
    dm1 = cisolver.make_rdm1(ci, norb, nelec)
    dm1 = 0.5 * (dm1 + dm1.T)
    occ, U = np.linalg.eigh(dm1)
    order = np.argsort(-occ)
    occ = occ[order]
    U = U[:, order]
    no_coeff = np.asarray(mo_coeff) @ U
    return occ, no_coeff


def _string_to_occ(s: int, norb: int):
    return tuple(k for k in range(norb) if (s >> k) & 1)


def ci_to_dict(
    ci: np.ndarray,
    norb: int,
    n_alpha: int,
    n_beta: int,
    tol: float = 0.0,
) -> dict:
    """Materialize the CI matrix as ``{(occ_alpha, occ_beta): coeff}``.

    Determinants with ``|coeff| <= tol`` are dropped.
    """
    strings_a = cistring.make_strings(range(norb), n_alpha)
    strings_b = cistring.make_strings(range(norb), n_beta)
    out: dict = {}
    for i, sa in enumerate(strings_a):
        occ_a = _string_to_occ(int(sa), norb)
        row = ci[i]
        for j, sb in enumerate(strings_b):
            c = float(row[j])
            if abs(c) > tol:
                occ_b = _string_to_occ(int(sb), norb)
                out[(occ_a, occ_b)] = c
    return out


def reference_determinant(ci_dict: Mapping):
    """The determinant with the largest |coefficient|."""
    return max(ci_dict, key=lambda k: abs(ci_dict[k]))


def ordered_candidate_set(ci_dict: Mapping, ref_det) -> list:
    """List of (occ_a, occ_b) tuples; reference at index 0, then descending |c|."""
    others = sorted(
        (k for k in ci_dict if k != ref_det),
        key=lambda k: -abs(ci_dict[k]),
    )
    return [ref_det] + others


def K_eff(ci_dict: Mapping, eta: float) -> int:
    """Number of CI coefficients strictly above ``eta``."""
    return sum(1 for v in ci_dict.values() if abs(v) > eta)


def max_c_corr(ci_dict: Mapping, ref_det) -> float:
    """Largest |coefficient| over determinants other than the reference."""
    return max((abs(v) for k, v in ci_dict.items() if k != ref_det), default=0.0)


def K_eff_table(ci_dict: Mapping, etas: Sequence[float] = (1e-2, 1e-3, 1e-4)) -> dict:
    return {float(eta): K_eff(ci_dict, eta) for eta in etas}


def compute_fci_reference(
    mol,
    n_alpha: Optional[int] = None,
    n_beta: Optional[int] = None,
    candidate_tol: float = 1e-10,
    energy_consistency_tol: float = 1e-6,
) -> dict:
    """End-to-end ground-truth pipeline.

    Steps: RHF, FCI in HF basis, natural orbitals from the FCI 1-RDM,
    FCI in the NO basis, candidate-set materialization. Verifies that the
    two FCI calculations agree on energy.
    """
    if n_alpha is None or n_beta is None:
        ne = int(mol.nelectron)
        if ne % 2 != 0 and (n_alpha is None and n_beta is None):
            raise ValueError(
                "odd electron count; pass n_alpha and n_beta explicitly"
            )
        n_alpha = ne // 2
        n_beta = ne // 2
    nelec = (int(n_alpha), int(n_beta))

    E_HF, mo_coeff = run_rhf(mol)
    E_FCI_hf, ci_hf, cisolver_hf = run_fci(mol, mo_coeff)
    occ_no, no_coeff = natural_orbitals(cisolver_hf, ci_hf, mo_coeff, nelec)
    E_FCI_no, ci_no, _ = run_fci(mol, no_coeff)

    if abs(E_FCI_hf - E_FCI_no) > energy_consistency_tol:
        raise RuntimeError(
            f"FCI energy inconsistent across bases: "
            f"HF={E_FCI_hf:.10f}, NO={E_FCI_no:.10f}"
        )

    norb = int(no_coeff.shape[1])
    ci_dict = ci_to_dict(ci_no, norb, nelec[0], nelec[1], tol=candidate_tol)
    if not ci_dict:
        raise RuntimeError("FCI vector vanished under candidate_tol filter")
    ref = reference_determinant(ci_dict)

    return dict(
        E_HF=float(E_HF),
        E_FCI=float(E_FCI_no),
        mo_coeff_hf=mo_coeff,
        no_coeff_ao=no_coeff,
        occ_no=occ_no,
        ci_dict=ci_dict,
        reference_det=ref,
        candidate_set=ordered_candidate_set(ci_dict, ref),
        n_orb=norb,
        nelec=nelec,
    )
