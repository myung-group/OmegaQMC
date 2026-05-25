"""Transition density matrices, transition dipoles, oscillator strengths,
and natural transition orbitals (NTOs) from two CS-recovered CI vectors.

Given two CI vectors c_hat^(0) and c_hat^(k) for two electronic states
(both in the same NO basis, both produced by the standard CS-recovery
pipeline), this module produces all spectroscopically relevant
quantities for the 0->k electronic transition:

- 1-particle transition density matrix
    gamma^(0k)_pq = <Psi_0 | a_p^dagger a_q | Psi_k>
  via PySCF's direct_spin1.trans_rdm1 on the reshaped CI matrices.

- Transition dipole moment
    mu_{0k,x} = - sum_{pq} <p|r_x|q> gamma^(0k)_qp
  (no nuclear contribution; nuclei don't contribute to 0->k transitions
  for k != 0).

- Oscillator strength (length gauge)
    f_{0k} = (2/3) (E_k - E_0) |mu_{0k}|^2
  with energies in atomic units, mu in atomic units; dimensionless.

- Natural Transition Orbitals (NTOs) via the SVD of gamma^(0k):
    gamma^(0k) = U S V^T  (T because gamma is real but non-symmetric)
  The columns of U are the "hole" orbitals (occupied character in Psi_0),
  the columns of V are the "particle" orbitals (virtual character in Psi_k),
  and the singular values S give the participation amplitudes. The
  dominant (largest S) pair gives the principal one-electron picture of
  the transition, regardless of how many configurations contribute to
  c_hat^(0) or c_hat^(k). Reference: Martin, JCP 118, 4775 (2003).

All routines accept the FCI reference dict produced by
:func:`OmegaQMC.cs.reference.compute_fci_reference` (used for the
candidate set, n_orb, nelec, no_coeff_ao). The CI vectors should be
the c_hat output of :func:`OmegaQMC.cs.estimators.normalize_and_align`
(unit-norm, sign-aligned to the chosen reference convention).
"""

from typing import Mapping, Sequence, Tuple

import numpy as np

from .properties import reshape_chat_to_pyscf_matrix, AU_TO_DEBYE


def compute_1tdm(
    c_hat_bra: np.ndarray,
    c_hat_ket: np.ndarray,
    candidate_set: Sequence,
    n_orb: int,
    nelec: Tuple[int, int],
) -> np.ndarray:
    """One-particle transition density matrix in the NO basis.

    Returns gamma_pq = <Psi_bra | a_p^dagger a_q | Psi_ket> with shape
    ``(n_orb, n_orb)``. Both CI vectors must be expressed in the same
    candidate set / NO basis (the standard output of the CS pipeline).
    """
    from pyscf import fci as pyscf_fci

    ci_bra = reshape_chat_to_pyscf_matrix(
        c_hat_bra, candidate_set, n_orb, nelec[0], nelec[1],
    )
    ci_ket = reshape_chat_to_pyscf_matrix(
        c_hat_ket, candidate_set, n_orb, nelec[0], nelec[1],
    )
    # PySCF returns (dm_a, dm_b) — spin-up and spin-down components.
    # We need the spin-summed transition density.
    dm_a, dm_b = pyscf_fci.direct_spin1.trans_rdm1s(
        ci_bra, ci_ket, n_orb, nelec,
    )
    return np.asarray(dm_a) + np.asarray(dm_b)


def transition_dipole(
    mol,
    no_coeff_ao: np.ndarray,
    gamma_01: np.ndarray,
    origin: np.ndarray = None,
) -> dict:
    """Transition dipole moment from a 1-TDM in NO basis.

    For electronic transitions there is no nuclear contribution; only
    the electronic term ``-Tr(gamma . r)`` survives. The origin choice
    is irrelevant for the magnitude of the transition dipole when the
    two states are exact eigenstates (because <Psi_0|Psi_k> = 0); for
    approximate CI vectors the choice can matter slightly, so we
    default to the center of nuclear charge to match the dipole
    convention used by :func:`OmegaQMC.cs.properties.electric_dipole`.

    Returns a dict with ``mu_au`` (3-vector), ``mu_debye`` (3-vector),
    ``mu_magnitude_au``, ``mu_magnitude_debye``.
    """
    if origin is None:
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
    mu_au = -np.einsum("xpq,qp->x", no_dip, gamma_01)
    return dict(
        mu_au=mu_au,
        mu_debye=mu_au * AU_TO_DEBYE,
        mu_magnitude_au=float(np.linalg.norm(mu_au)),
        mu_magnitude_debye=float(np.linalg.norm(mu_au) * AU_TO_DEBYE),
        origin_bohr=origin,
    )


def oscillator_strength(
    mu_au: np.ndarray,
    delta_E_au: float,
) -> float:
    """Length-gauge oscillator strength.

    f_{0k} = (2/3) (E_k - E_0) |mu_{0k}|^2 in atomic units. Dimensionless.
    """
    return float((2.0 / 3.0) * float(delta_E_au)
                 * float(np.dot(mu_au, mu_au)))


def natural_transition_orbitals(
    gamma_01: np.ndarray,
    no_coeff_ao: np.ndarray,
    keep_threshold: float = 1e-6,
) -> dict:
    """NTO decomposition of a 1-TDM (Martin, JCP 118, 4775).

    Performs an SVD ``gamma = U S V^T`` and returns:
    - ``singular_values``: sorted descending (the "participation amplitudes")
    - ``participation_ratios``: ``S^2 / sum(S^2)`` (the "weights")
    - ``n_dominant``: number of pairs above the keep_threshold
    - ``hole_nto_ao``: ``no_coeff_ao @ U`` (AO-basis NTOs for hole/occupied side)
    - ``particle_nto_ao``: ``no_coeff_ao @ V`` (AO-basis NTOs for particle side)
    - ``hole_nto_no``: ``U`` (NO-basis hole NTOs)
    - ``particle_nto_no``: ``V`` (NO-basis particle NTOs)

    The dominant (S, hole, particle) triplet gives the principal
    one-electron orbital picture of the transition, regardless of the
    number of determinants in c_hat^(0) and c_hat^(k).
    """
    gamma = np.asarray(gamma_01)
    U, S, Vt = np.linalg.svd(gamma, full_matrices=False)
    V = Vt.T
    total = float(np.sum(S ** 2)) + 1e-30
    participation = (S ** 2) / total
    n_dom = int(np.sum(S > keep_threshold))

    no_coeff = np.asarray(no_coeff_ao)
    hole_nto_ao = no_coeff @ U
    particle_nto_ao = no_coeff @ V
    return dict(
        singular_values=S,
        participation_ratios=participation,
        n_dominant=n_dom,
        hole_nto_ao=hole_nto_ao,
        particle_nto_ao=particle_nto_ao,
        hole_nto_no=U,
        particle_nto_no=V,
    )


def report_transition_properties(
    c_hat_bra: np.ndarray,
    c_hat_ket: np.ndarray,
    fci_ref: Mapping,
    mol,
    delta_E_au: float,
) -> dict:
    """Aggregate spectroscopic properties for the bra->ket transition.

    Convenience wrapper used by the H2 / H2O demo scripts. Returns a
    dict with the 1-TDM, transition dipole (au and Debye), oscillator
    strength, NTO decomposition, and the static CI-vector inner
    product (useful as an orthogonality diagnostic).
    """
    candidate = fci_ref["candidate_set"]
    n_orb = int(fci_ref["n_orb"])
    nelec = tuple(fci_ref["nelec"])
    no_coeff = fci_ref["no_coeff_ao"]

    gamma_01 = compute_1tdm(
        c_hat_bra, c_hat_ket, candidate, n_orb, nelec,
    )
    mu = transition_dipole(mol, no_coeff, gamma_01)
    f = oscillator_strength(mu["mu_au"], delta_E_au)
    nto = natural_transition_orbitals(gamma_01, no_coeff)
    ci_overlap = float(np.dot(np.asarray(c_hat_bra), np.asarray(c_hat_ket)))

    return dict(
        gamma_01=gamma_01,
        transition_dipole=mu,
        oscillator_strength=f,
        nto=nto,
        ci_overlap=ci_overlap,
        delta_E_au=float(delta_E_au),
    )


def print_transition_summary(report: dict, label: str = "0->1") -> None:
    """One-paragraph summary for stdout / log files."""
    mu_d = report["transition_dipole"]["mu_debye"]
    mu_mag = report["transition_dipole"]["mu_magnitude_debye"]
    f = report["oscillator_strength"]
    dE_ev = report["delta_E_au"] * 27.2114
    dE_nm = (1239.84 / dE_ev) if dE_ev > 1e-6 else float("inf")
    nto_s = report["nto"]["singular_values"][:3]
    nto_p = report["nto"]["participation_ratios"][:3]
    print(f"=== {label} transition ===")
    print(f"  Delta E       = {report['delta_E_au']:.6f} Ha "
          f"({dE_ev:.4f} eV, {dE_nm:.1f} nm)")
    print(f"  mu (au)       = ({mu_d[0]/AU_TO_DEBYE:+.4f}, "
          f"{mu_d[1]/AU_TO_DEBYE:+.4f}, {mu_d[2]/AU_TO_DEBYE:+.4f})")
    print(f"  mu (Debye)    = ({mu_d[0]:+.4f}, {mu_d[1]:+.4f}, "
          f"{mu_d[2]:+.4f}); |mu| = {mu_mag:.4f}")
    print(f"  Oscillator f  = {f:.6f}")
    print(f"  NTO singvals  = "
          + ", ".join(f"{s:.4f}" for s in nto_s))
    print(f"  NTO weights   = "
          + ", ".join(f"{p*100:.2f}%" for p in nto_p))
    print(f"  CI overlap    = {report['ci_overlap']:+.5e}")
