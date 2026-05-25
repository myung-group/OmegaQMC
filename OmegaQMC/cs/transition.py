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
    casci_meta: dict = None,
) -> np.ndarray:
    """One-particle transition density matrix in the NO basis.

    Returns gamma_pq = <Psi_bra | a_p^dagger a_q | Psi_ket> with shape
    ``(n_orb, n_orb)``. Both CI vectors must be expressed in the same
    candidate set / NO basis (the standard output of the CS pipeline).

    For active-space CASCI references where n_orb >> ncas, pass
    ``casci_meta = {"ncore": ..., "ncas": ..., "nelecas": (na, nb)}``
    to do the trans_rdm1s contraction in the (much smaller) active-
    space FCI; otherwise allocating the full
    ``(n_strings_a, n_strings_b)`` zero matrix can OOM
    (e.g.\ aug-cc-pVDZ H2O has 749k alpha strings -> 4 TiB array).
    The returned gamma is embedded back into the full-orbital
    ``(n_orb, n_orb)`` shape with core entries set to identity (core
    contributes diagonally to the 1-RDM for both bra and ket, so it
    cancels in the transition density).
    """
    from pyscf import fci as pyscf_fci

    if casci_meta is not None:
        ncore = int(casci_meta["ncore"])
        ncas = int(casci_meta["ncas"])
        nelecas = tuple(casci_meta["nelecas"])
        # Convert candidate set to active-space indexing
        active_cand = []
        for (occ_a, occ_b) in candidate_set:
            act_a = tuple(o - ncore for o in occ_a[ncore:])
            act_b = tuple(o - ncore for o in occ_b[ncore:])
            active_cand.append((act_a, act_b))
        ci_bra = reshape_chat_to_pyscf_matrix(
            c_hat_bra, active_cand, ncas, nelecas[0], nelecas[1],
        )
        ci_ket = reshape_chat_to_pyscf_matrix(
            c_hat_ket, active_cand, ncas, nelecas[0], nelecas[1],
        )
        dm_a_act, dm_b_act = pyscf_fci.direct_spin1.trans_rdm1s(
            ci_bra, ci_ket, ncas, nelecas,
        )
        gamma_active = (np.asarray(dm_a_act) + np.asarray(dm_b_act))
        # Embed into full-orbital matrix: core block has the static
        # core 1-TDM which equals zero for orthogonal bra/ket (the
        # frozen core is identical in both states, so the core block
        # only contributes when bra=ket; for transitions it cancels).
        gamma_full = np.zeros((n_orb, n_orb), dtype=np.float64)
        gamma_full[ncore:ncore + ncas, ncore:ncore + ncas] = gamma_active
        return gamma_full

    ci_bra = reshape_chat_to_pyscf_matrix(
        c_hat_bra, candidate_set, n_orb, nelec[0], nelec[1],
    )
    ci_ket = reshape_chat_to_pyscf_matrix(
        c_hat_ket, candidate_set, n_orb, nelec[0], nelec[1],
    )
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

    casci_meta = None
    if "ncore" in fci_ref and "ncas" in fci_ref:
        casci_meta = dict(
            ncore=int(fci_ref["ncore"]),
            ncas=int(fci_ref["ncas"]),
            nelecas=tuple(fci_ref["nelecas_active"]),
        )
    gamma_01 = compute_1tdm(
        c_hat_bra, c_hat_ket, candidate, n_orb, nelec,
        casci_meta=casci_meta,
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


def subspace_rotate_to_eigenstates(
    c_hats: Sequence[np.ndarray],
    fci_ref: Mapping,
    mol,
) -> dict:
    """Diagonalise the K-state Hamiltonian within the span of the
    Pfau-NES-recovered CI vectors to extract energy eigenstates.

    Pfau et al.'s determinantal loss
    L(theta) = Tr(M^{-1}(M (*) H_loc)) is invariant under unitary
    mixing within the span of the K trial wavefunctions: the network
    learns *the lowest-K-eigenstate subspace* but the individual
    psi_i's can be any basis of that subspace. In practice, when
    initialised from the ground checkpoint with a small perturbation,
    both psi_i's converge to nearly-parallel CI vectors (CI overlap
    ~ 1) even though the trace Tr(E_Psi) reaches the correct sum
    sum_k E_k of eigenvalues.

    To recover the actual energy eigenstates we solve the
    generalised K x K eigenproblem
        H v = E S v,    H_ij = <Psi_i | H | Psi_j>,
                        S_ij = <Psi_i | Psi_j>
    in the basis of CS-recovered CI vectors. The eigenvectors v give
    the unitary rotation; the rotated CI vectors
        c^(k)_eig = sum_i v_ki c_hat^(i)
    are pure energy eigenstates of H restricted to the K-state span.

    Returns a dict with:
      ``c_eig``           : list of K rotated CI vectors (each unit-norm,
                            sign-aligned)
      ``E_eig``           : numpy array of K eigenvalues (Ha)
      ``rotation``        : the K x K eigenvector matrix V
      ``overlap_matrix``  : the K x K input overlap matrix S
      ``H_matrix``        : the K x K input Hamiltonian matrix H
      ``input_ci_overlap``: max_{i != j} |S_ij| / sqrt(S_ii S_jj),
                            the worst nonorthogonality of the input
                            (1 -> very stuck states; 0 -> already
                            orthogonal)

    Notes
    -----
    * The H matrix is computed via PySCF's
      ``direct_spin1.contract_2e`` after absorbing h1 into h2, so the
      cost is one CI-vector contraction per state pair. Negligible
      compared to the Pfau-NES training step.
    * Sign alignment: each rotated vector is multiplied by -1 if its
      leading (reference-determinant) coefficient is negative, matching
      the CS-pipeline convention.
    * The K-state span is a SUBSET of the full Hilbert space. The
      eigenvalues returned are eigenvalues of P_K H P_K, where P_K is
      the projector onto the span; they may be higher than the true
      lowest-K eigenvalues of H if the network did not capture the
      correct subspace (e.g. if both states ended up in a single
      symmetry sector and the lowest excited state was excluded).
    """
    from pyscf import scf, ao2mo, mcscf
    from pyscf import fci as pyscf_fci
    from pyscf.fci import cistring

    candidate = fci_ref["candidate_set"]
    n_orb = int(fci_ref["n_orb"])
    nelec = tuple(fci_ref["nelec"])
    no_coeff = np.asarray(fci_ref["no_coeff_ao"])

    K = len(c_hats)
    if K < 2:
        raise ValueError("need at least K=2 CI vectors to rotate")

    is_casci = ("ncore" in fci_ref and "ncas" in fci_ref
                and "nelecas_active" in fci_ref)

    if is_casci:
        # CASCI reference: contract in the (much smaller) active-space
        # FCI to avoid the intractable full-basis contract_2e.
        ncore = int(fci_ref["ncore"])
        ncas = int(fci_ref["ncas"])
        nelecas = tuple(fci_ref["nelecas_active"])

        # Convert each candidate's full-orbital tuple
        # (0,...,ncore-1, ncore+a_0, ncore+a_1, ...) to active-space
        # tuple (a_0, a_1, ...)
        active_cand = []
        for (occ_a, occ_b) in candidate:
            act_a = tuple(o - ncore for o in occ_a[ncore:])
            act_b = tuple(o - ncore for o in occ_b[ncore:])
            active_cand.append((act_a, act_b))

        # Build active-space CI matrices
        ci_matrices = [
            reshape_chat_to_pyscf_matrix(
                c, active_cand, ncas, nelecas[0], nelecas[1],
            )
            for c in c_hats
        ]

        # Build CASCI effective h1, h2, ecore using PySCF
        mf = scf.RHF(mol).run(verbose=0)
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.ncore = ncore
        mc.mo_coeff = no_coeff
        mc.verbose = 0
        h1eff, ecore_eff = mc.get_h1eff()
        h2eff = mc.get_h2eff()
        h2eff = ao2mo.restore(1, h2eff, ncas)

        h2_eff_absorbed = pyscf_fci.direct_spin1.absorb_h1e(
            h1eff, h2eff, ncas, nelecas, 0.5,
        )
        H_ci = [
            pyscf_fci.direct_spin1.contract_2e(
                h2_eff_absorbed, cj, ncas, nelecas,
            )
            for cj in ci_matrices
        ]
        ecore = float(ecore_eff)
    else:
        # Full-FCI reference: contract in the full basis (small system)
        ci_matrices = [
            reshape_chat_to_pyscf_matrix(
                c, candidate, n_orb, nelec[0], nelec[1],
            )
            for c in c_hats
        ]
        mf = scf.RHF(mol).run(verbose=0)
        h1_ao = mf.get_hcore()
        h1 = no_coeff.T @ h1_ao @ no_coeff
        h2 = ao2mo.restore(1, ao2mo.kernel(mol, no_coeff), n_orb)
        ecore = float(mol.energy_nuc())
        h2_eff = pyscf_fci.direct_spin1.absorb_h1e(
            h1, h2, n_orb, nelec, 0.5,
        )
        H_ci = [
            pyscf_fci.direct_spin1.contract_2e(h2_eff, cj, n_orb, nelec)
            for cj in ci_matrices
        ]
    H_mat = np.zeros((K, K), dtype=np.float64)
    S_mat = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            ovl = float(np.sum(ci_matrices[i] * ci_matrices[j]))
            H_mat[i, j] = float(np.sum(ci_matrices[i] * H_ci[j])) + ecore * ovl
            S_mat[i, j] = ovl
    H_mat = 0.5 * (H_mat + H_mat.T)
    S_mat = 0.5 * (S_mat + S_mat.T)

    # Generalised eigenproblem H V = S V E
    from scipy.linalg import eigh
    E_eig, V = eigh(H_mat, S_mat)

    # Form rotated CI vectors and normalise/sign-align in basis
    c_eig = []
    for k in range(K):
        c_k = sum(float(V[i, k]) * np.asarray(c_hats[i])
                  for i in range(K))
        nrm = float(np.linalg.norm(c_k))
        if nrm < 1e-30:
            c_k_n = c_k
        else:
            c_k_n = c_k / nrm
        if c_k_n[0] < 0:
            c_k_n = -c_k_n
        c_eig.append(c_k_n)

    # Diagnostic: worst input nonorthogonality
    worst = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            denom = (S_mat[i, i] * S_mat[j, j]) ** 0.5
            if denom > 1e-30:
                worst = max(worst, abs(S_mat[i, j]) / denom)

    return dict(
        c_eig=c_eig,
        E_eig=E_eig,
        rotation=V,
        overlap_matrix=S_mat,
        H_matrix=H_mat,
        input_ci_overlap=float(worst),
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
