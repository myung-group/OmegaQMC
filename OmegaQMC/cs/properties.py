"""
Reduced density matrices and one-electron properties from a recovered
CI vector.

The pipeline takes a normalized CI vector ``c_hat`` in the natural-orbital
basis (output of :func:`OmegaQMC.cs.estimators.normalize_and_align`),
reshapes it into the PySCF FCI matrix layout, and produces:

- 1-RDM ``gamma_pq`` in the NO basis (via PySCF's ``direct_spin1.make_rdm1``)
- natural occupation numbers (eigenvalues of ``gamma``; reordered NO
  occupations sorted descending)
- electric dipole moment ``mu`` in Debye and atomic units
- electric quadrupole moment ``Q`` (traceless, atomic units)
- multireference diagnostic estimates: the natural-orbital occupation
  gap and the "C0" (reference-determinant amplitude squared) measure

All quantities are computed analytically once ``gamma`` is in hand; no
additional Monte Carlo sampling is required beyond what produced
``c_hat``.
"""

from typing import Mapping, Sequence, Tuple

import numpy as np

# Atomic-unit dipole to debye conversion factor
AU_TO_DEBYE = 2.541746229


def _string_to_occ(s: int, norb: int):
    return tuple(k for k in range(norb) if (s >> k) & 1)


def reshape_chat_to_pyscf_matrix(
    c_hat: np.ndarray,
    candidate_set: Sequence,
    n_orb: int,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    """Convert ``(n_det,)`` ``c_hat`` indexed by ``candidate_set`` into the
    PySCF FCI 2D matrix ``ci[i_alpha, j_beta]``.

    Determinants present in ``candidate_set`` are placed at their PySCF
    string indices; all other entries are zero. The resulting matrix is
    suitable for ``pyscf.fci.direct_spin1.make_rdm1`` and related helpers.
    """
    from pyscf.fci import cistring
    strings_a = cistring.make_strings(range(n_orb), n_alpha)
    strings_b = cistring.make_strings(range(n_orb), n_beta)
    a_idx = {_string_to_occ(int(s), n_orb): i for i, s in enumerate(strings_a)}
    b_idx = {_string_to_occ(int(s), n_orb): j for j, s in enumerate(strings_b)}
    ci_matrix = np.zeros((len(strings_a), len(strings_b)), dtype=float)
    for c, (occ_a, occ_b) in zip(np.asarray(c_hat), candidate_set):
        i = a_idx[tuple(occ_a)]
        j = b_idx[tuple(occ_b)]
        ci_matrix[i, j] = float(c)
    return ci_matrix


def compute_1rdm(
    c_hat: np.ndarray,
    candidate_set: Sequence,
    n_orb: int,
    nelec: Tuple[int, int],
) -> np.ndarray:
    """One-particle reduced density matrix in the NO basis.

    ``gamma_pq = <Psi | a_p^dagger a_q | Psi>`` (spin-summed). Shape
    ``(n_orb, n_orb)``.
    """
    from pyscf import fci as pyscf_fci
    ci_matrix = reshape_chat_to_pyscf_matrix(
        c_hat, candidate_set, n_orb, nelec[0], nelec[1],
    )
    return pyscf_fci.direct_spin1.make_rdm1(
        ci_matrix, n_orb, nelec,
    )


def natural_occupations_from_rdm(gamma: np.ndarray) -> np.ndarray:
    """Eigenvalues of the 1-RDM, sorted descending."""
    g = 0.5 * (gamma + gamma.T)
    occ = np.linalg.eigvalsh(g)
    return np.sort(occ)[::-1]


def electric_dipole(
    mol,
    no_coeff_ao: np.ndarray,
    gamma_no: np.ndarray,
    origin: np.ndarray = None,
) -> dict:
    """Electric dipole moment from the 1-RDM in NO basis.

    Returns a dict with ``mu_au`` (3-vector in atomic units),
    ``mu_debye`` (3-vector in Debye), and ``mu_magnitude_debye``.
    """
    if origin is None:
        # Center of nuclear charge as the dipole origin
        charges = np.asarray(mol.atom_charges(), dtype=float)
        coords = np.asarray(mol.atom_coords())
        origin = np.sum(charges[:, None] * coords, axis=0) / np.sum(charges)
    origin = np.asarray(origin, dtype=float)

    with mol.with_common_origin(origin):
        ao_dip = mol.intor("int1e_r", comp=3)  # (3, nao, nao)
    no_dip = np.einsum(
        "xij,ip,jq->xpq",
        ao_dip,
        np.asarray(no_coeff_ao),
        np.asarray(no_coeff_ao),
    )
    elec_dip = -np.einsum("xpq,qp->x", no_dip, gamma_no)
    charges = np.asarray(mol.atom_charges(), dtype=float)
    coords = np.asarray(mol.atom_coords())
    nuc_dip = np.einsum("a,ax->x", charges, coords - origin)
    mu_au = nuc_dip + elec_dip
    return dict(
        mu_au=mu_au,
        mu_debye=mu_au * AU_TO_DEBYE,
        mu_magnitude_au=float(np.linalg.norm(mu_au)),
        mu_magnitude_debye=float(np.linalg.norm(mu_au) * AU_TO_DEBYE),
        origin_bohr=origin,
    )


def electric_quadrupole(
    mol,
    no_coeff_ao: np.ndarray,
    gamma_no: np.ndarray,
    origin: np.ndarray = None,
) -> np.ndarray:
    """Electric quadrupole tensor in atomic units, ``(3, 3)``.

    Computed as the second moment of charge,
    ``Q_ij = sum_a Z_a (r_a)_i (r_a)_j  -  Tr(gamma . r_i r_j)``.
    Not made traceless; the trace can be subtracted by the caller if a
    traceless form is preferred.
    """
    if origin is None:
        charges = np.asarray(mol.atom_charges(), dtype=float)
        coords = np.asarray(mol.atom_coords())
        origin = np.sum(charges[:, None] * coords, axis=0) / np.sum(charges)
    origin = np.asarray(origin, dtype=float)

    with mol.with_common_origin(origin):
        ao_rr = mol.intor("int1e_rr", comp=9).reshape(3, 3, mol.nao, mol.nao)
    no_rr = np.einsum(
        "xyij,ip,jq->xypq",
        ao_rr,
        np.asarray(no_coeff_ao),
        np.asarray(no_coeff_ao),
    )
    elec_q = -np.einsum("xypq,qp->xy", no_rr, gamma_no)
    charges = np.asarray(mol.atom_charges(), dtype=float)
    coords = np.asarray(mol.atom_coords()) - origin
    nuc_q = np.einsum("a,ai,aj->ij", charges, coords, coords)
    return nuc_q + elec_q


def multireference_diagnostics(
    c_hat: np.ndarray,
    candidate_set: Sequence,
    nelec: Tuple[int, int],
) -> dict:
    """Quick multireference diagnostics from the recovered CI vector.

    - ``C0``: amplitude of the dominant (reference) determinant
    - ``C0_sq``: ``C0**2`` (the "percentage of mass on HF-like determinant")
    - ``no_occupation_gap``: requires an externally-computed 1-RDM and is
      reported via :func:`natural_occupations_from_rdm`; included here for
      easy aggregation
    - ``effective_unpaired``: ``sum_p n_p (2 - n_p) / 2`` from NOs;
      requires NO occupations (computed elsewhere and passed in via
      ``no_occupations`` for the full report)
    """
    c_abs = np.abs(np.asarray(c_hat))
    ref_idx = int(np.argmax(c_abs))
    C0 = float(c_hat[ref_idx])
    return dict(
        reference_index=ref_idx,
        reference_det=candidate_set[ref_idx],
        C0=C0,
        C0_sq=C0 * C0,
        n_det_above_1pct=int(np.sum(c_abs > 0.01)),
        n_det_above_5pct=int(np.sum(c_abs > 0.05)),
    )


def effective_unpaired_electrons(occupations: np.ndarray) -> float:
    """Head--Gordon's effective number of unpaired electrons.

    ``N_unp = sum_p n_p (2 - n_p) / 2``. Zero for closed-shell single
    reference; rises with multireference character.
    """
    occ = np.asarray(occupations)
    return float(np.sum(occ * (2.0 - occ)) / 2.0)


def report_properties(
    c_hat: np.ndarray,
    fci_ref: Mapping,
    mol,
) -> dict:
    """Aggregate properties from a recovered CI vector + FCI reference.

    Convenience wrapper returning everything needed for a paper table.
    """
    candidate = fci_ref["candidate_set"]
    n_orb = int(fci_ref["n_orb"])
    nelec = tuple(fci_ref["nelec"])
    no_coeff = fci_ref["no_coeff_ao"]

    gamma = compute_1rdm(c_hat, candidate, n_orb, nelec)
    occupations = natural_occupations_from_rdm(gamma)
    dipole = electric_dipole(mol, no_coeff, gamma)
    quadrupole = electric_quadrupole(mol, no_coeff, gamma)
    diags = multireference_diagnostics(c_hat, candidate, nelec)
    diags["effective_unpaired"] = effective_unpaired_electrons(occupations)

    return dict(
        gamma=gamma,
        occupations=occupations,
        dipole=dipole,
        quadrupole=quadrupole,
        diagnostics=diags,
        trace_gamma=float(np.trace(gamma)),
    )


def compare_to_fci(
    chat_report: dict,
    fci_ref: Mapping,
    mol,
) -> dict:
    """Compute the same properties from the exact PySCF FCI vector and
    report side-by-side differences."""
    ci_dict = fci_ref["ci_dict"]
    candidate = fci_ref["candidate_set"]
    c_fci = np.array([ci_dict[k] for k in candidate])
    fci_report = report_properties(c_fci, fci_ref, mol)

    delta = {
        "dipole_mu_au_diff": chat_report["dipole"]["mu_au"]
                              - fci_report["dipole"]["mu_au"],
        "dipole_magnitude_debye_diff":
            chat_report["dipole"]["mu_magnitude_debye"]
            - fci_report["dipole"]["mu_magnitude_debye"],
        "occupations_L_inf":
            float(np.max(np.abs(
                chat_report["occupations"] - fci_report["occupations"]
            ))),
        "trace_gamma_diff":
            chat_report["trace_gamma"] - fci_report["trace_gamma"],
        "C0_diff": chat_report["diagnostics"]["C0"]
                   - fci_report["diagnostics"]["C0"],
        "effective_unpaired_diff":
            chat_report["diagnostics"]["effective_unpaired"]
            - fci_report["diagnostics"]["effective_unpaired"],
    }
    return dict(chat=chat_report, fci=fci_report, delta=delta)
