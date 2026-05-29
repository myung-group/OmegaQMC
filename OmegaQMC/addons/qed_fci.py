"""
QED-FCI: Exact diagonalization of the Pauli-Fierz Hamiltonian.

Constructs the full Hamiltonian in the product basis:
    |FCI_electronic⟩ ⊗ |n_photon⟩

and diagonalizes using scipy.linalg.eigh.

Supports the dipole gauge Pauli-Fierz Hamiltonian:

    H = H_elec + DSE + √(Ω/2)·(â+â†)·D + Ω·â†â

where:
    DSE = (1/2)·λ²·(ε·∑_i r̂_i)²    (dipole self-energy, electronic)
    D   = λ·∑_pq ⟨p|ε·r̂|q⟩·Ê_pq    (dipole operator, second-quantized)

Reference: arXiv:2410.18838

Reference wavefunction
----------------------
The QED-FCI eigenvalue itself is invariant under unitary rotations of
the one-particle MO basis, so any orthonormal orbital set spanning the
AO basis yields the same ``e_qed_fci``. The *correlation energy*,
however, is referenced to a mean field. To make
    E_corr(QED) = E_QED-FCI − E_QED-HF
the natural cavity-aware analogue of the usual FCI correlation
energy, this module runs a dipole-gauge QED-HF on ``(mol, omega,
coupling_vec)`` (closed-shell only — see :mod:`OmegaQMC.addons.qed_hf`)
and uses those orbitals to build the integrals. The QED-HF energy and
the resulting correlation energy are returned in the output dict.

For open-shell systems QED-HF is not currently implemented; the code
falls back to ``mf.mo_coeff`` and reports ``e_qed_hf = None``.

Photon basis (coherent_state flag)
----------------------------------
By default (``coherent_state=True``) the photon Fock states are built
in the coherent-state (displaced) frame b = a + z, z = ⟨D̂⟩/√(2Ω),
where ⟨D̂⟩ is the reference-determinant dipole expectation. This is the
CS-QED-FCI of Vu et al., J. Chem. Theory Comput. 20, 1214 (2024). The
displacement is an exact unitary transform of the Pauli-Fierz
Hamiltonian, so the energy is unchanged at convergence in ``nph_max``,
but for polar molecules (e.g. LiH) the displaced basis converges at
``nph_max=1`` whereas the raw photon-number basis needs many Fock
states. Set ``coherent_state=False`` for the raw PN basis (a=0).

Note on DSE basis-set treatment (proper_dse flag):
  In a truncated basis, the operator-squared form D̂² is NOT the same
  as the true (∑_i ε·r̂_i)². They agree only in the complete-basis
  limit via resolution of identity. Specifically, the 1-body part of
  the DSE involves the quadrupole matrix Q_pq = ⟨p|(ε·r̂)²|q⟩, which
  in a truncated basis differs from the matrix product (d²)_pq =
  ∑_q ⟨p|ε·r̂|r⟩⟨r|ε·r̂|q⟩ by the basis-incompleteness residual.

  The default ``proper_dse=True`` adds a one-body correction
  (1/2)·λ²·(Q − d̃²) to h1e so the implemented Hamiltonian matches the
  true Pauli-Fierz form in any basis. Set ``proper_dse=False`` to
  reproduce the legacy (operator-squared) behaviour.
"""

import warnings

import numpy as np
from scipy.linalg import eigh
from pyscf import ao2mo, fci
from pyscf.fci import direct_spin1, cistring

from OmegaQMC.addons.qed_hf import run_qed_hf


def _build_fci_matrices(h1e, eri, dip_mo, norb, nelec, enuc):
    """Build electronic H and dipole operator matrices in the FCI basis.

    Args:
        h1e: one-body integrals (norb, norb).
        eri: two-electron integrals with DSE, chemist notation
            (norb, norb, norb, norb).
        dip_mo: dipole matrix in MO basis (norb, norb).
        norb: number of orbitals.
        nelec: (nalpha, nbeta).
        enuc: nuclear repulsion energy.

    Returns:
        H_mat: FCI Hamiltonian matrix (ndim, ndim).
        D_mat: Dipole operator matrix (ndim, ndim).
        ndim: FCI space dimension.
    """
    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ndim = na * nb

    # Absorb h1e into eri for efficient contraction
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac=0.5)

    H_mat = np.zeros((ndim, ndim))
    D_mat = np.zeros((ndim, ndim))

    for i in range(ndim):
        # Basis vector e_i
        ci_i = np.zeros((na, nb))
        ia, ib = divmod(i, nb)
        ci_i[ia, ib] = 1.0

        # H_elec + DSE (including nuclear repulsion)
        Hci = direct_spin1.contract_2e(h2e, ci_i, norb, nelec)
        H_mat[:, i] = Hci.ravel()

        # Dipole operator D = Σ_pq d_pq c†_pσ c_qσ
        Dci = direct_spin1.contract_1e(dip_mo, ci_i, norb, nelec)
        D_mat[:, i] = Dci.ravel()

    # Add nuclear repulsion to diagonal
    H_mat += enuc * np.eye(ndim)

    return H_mat, D_mat, ndim


def run_qed_fci(mf, omega, coupling_vec, nph_max=10, proper_dse=True,
                use_qed_hf_reference=True, coherent_state=True):
    """QED-FCI: exact diagonalization of the Pauli-Fierz Hamiltonian.

    Builds the full Hamiltonian in the product basis |FCI⟩ ⊗ |n_ph⟩
    and finds the ground state energy.

    Dipole gauge:
        H = H_elec + DSE + √(Ω/2)·(â+â†)·D + Ω·â†â

    The photon Fock space is truncated at nph_max photons.

    The reference orbitals used to build the integrals are taken from a
    dipole-gauge QED-HF run on ``(mol, omega, coupling_vec)`` whenever
    ``use_qed_hf_reference=True`` (default) and the molecule is
    closed-shell. The QED-FCI eigenvalue is orbital-invariant, but
    using QED-HF orbitals makes the reported correlation energy
    ``e_corr_qed = e_qed_fci − e_qed_hf`` the natural cavity-aware
    analogue of the FCI correlation energy. For open-shell systems
    QED-HF is not available; the code falls back to ``mf.mo_coeff`` and
    reports ``e_qed_hf = None``.

    Args:
        mf: PySCF mean-field object (must have run kernel()).
        omega: Photon frequency in Hartree.
        coupling_vec: Light-matter coupling vector (3,). Direction gives
            polarization ε, magnitude gives coupling strength λ.
        nph_max: Maximum number of photons in Fock space truncation.
        proper_dse: If True (default), add the one-body correction
            (1/2)·λ²·(Q − d̃²) to h1e where Q is the true quadrupole
            matrix ⟨p|(ε·r̂)²|q⟩ and d̃² is the matrix product of
            dipole matrices. This makes the DSE exact in any basis.
            If False, use the legacy operator-squared form (D̂²) which
            is exact only in the complete-basis limit.
        use_qed_hf_reference: If True (default), use QED-HF orbitals
            and report the correlation energy relative to QED-HF. If
            False, use ``mf.mo_coeff`` directly (legacy behaviour).
        coherent_state: If True (default), work in the coherent-state
            (displaced) photon basis b = a + z, with z = ⟨D̂⟩/√(2Ω)
            and ⟨D̂⟩ the dipole expectation in the reference
            determinant. This is the "CS-QED-FCI" of Vu et al., J.
            Chem. Theory Comput. 20, 1214 (2024); the displacement is
            an exact unitary transform, so the energy is unchanged at
            convergence, but the Fock truncation converges far faster
            for polar molecules (e.g. LiH already converges at
            nph_max=1). If False, use the raw photon-number (Fock)
            basis centred at a=0 (PN-QED-FCI), which converges slowly
            for polar systems.

    Returns:
        dict with:
            'e_qed_fci': QED-FCI ground state energy.
            'e_fci': standard FCI energy (no cavity).
            'e_qed_hf': QED-HF reference energy (None if not used).
            'e_hf': bare HF energy of the supplied ``mf`` (``mf.e_tot``).
            'e_corr_qed': QED correlation energy
                e_qed_fci − e_qed_hf (None if QED-HF was not run).
            'e_corr': bare correlation energy e_fci − e_hf.
            'reference': 'QED-HF' if QED-HF orbitals were used, else 'HF'.
            'eigenvalues': all eigenvalues of the product Hamiltonian.
            'eigenvectors': corresponding eigenvectors.
            'nph_max': photon truncation used.
            'ndim_elec': electronic FCI dimension.
            'ndim_total': total product space dimension.
            'n_photon': expectation value of the photon number in the
                ground state. With ``coherent_state=True`` this is the
                displaced-frame number ⟨b†b⟩ (the photon fluctuation
                about the coherent mean), not the bare ⟨a†a⟩.
            'coherent_state': whether the coherent-state photon basis
                was used.
            'cs_displacement': the coherent-state displacement z =
                ⟨D̂⟩/√(2Ω) (0.0 when ``coherent_state=False`` or λ=0).
            'proper_dse': whether the proper-DSE 1-body correction was
                applied.
            'dse_correction_norm': Frobenius norm of the 1-body correction
                (Ha) — non-zero only in incomplete bases; gauges the
                basis-set residual.
    """
    mol = mf.mol
    norb = mol.nao_nr()
    nelec = mol.nelec
    enuc = mol.energy_nuc()

    # --- Coupling vector → magnitude λ and polarization ε ---
    coupling_vec = np.asarray(coupling_vec, dtype=np.float64)
    lam = np.linalg.norm(coupling_vec)
    if lam > 1e-15:
        epsilon = coupling_vec / lam
    else:
        epsilon = np.array([0.0, 0.0, 1.0])
        lam = 0.0

    # --- Reference wavefunction: QED-HF (closed shell) or fall back ---
    # QED-HF is implemented for closed-shell molecules only. For
    # open-shell inputs we transparently fall back to ``mf.mo_coeff``.
    # FCI is orbital-invariant in the full one-particle basis, so the
    # eigenvalue ``e_qed_fci`` is independent of this choice; the
    # orbital set only fixes which mean field defines the correlation
    # energy.
    closed_shell = (nelec[0] == nelec[1])
    e_qed_hf = None
    reference = 'HF'
    if use_qed_hf_reference and closed_shell:
        qedhf = run_qed_hf(
            mol, omega, lambda_cav=tuple(coupling_vec.tolist()),
        )
        mo_coeff = np.asarray(qedhf['C'])
        e_qed_hf = float(qedhf['E_qed_hf'])
        reference = 'QED-HF'
    else:
        if use_qed_hf_reference and not closed_shell:
            warnings.warn(
                "qed_fci: QED-HF reference requested but molecule is "
                "open-shell; falling back to mf.mo_coeff. The QED-FCI "
                "energy is unaffected, but e_corr_qed will be None.",
                RuntimeWarning,
                stacklevel=2,
            )
        mo_coeff = np.asarray(mf.mo_coeff)

    # --- Electronic integrals in the chosen MO basis ---
    h1e = mo_coeff.T @ np.asarray(mf.get_hcore()) @ mo_coeff
    eri_mo = ao2mo.full(mol, mo_coeff)
    eri_mo_full = ao2mo.restore(1, eri_mo, norb)

    # --- Dipole matrix in the chosen MO basis ---
    dip_ao = mol.intor('int1e_r', comp=3)
    dip_ao_proj = lam * np.einsum('k,kpq->pq', epsilon, dip_ao)
    dip_mo = mo_coeff.T @ dip_ao_proj @ mo_coeff

    # --- Standard FCI (no cavity) ---
    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()

    # --- Add DSE to two-electron integrals ---
    # DSE: (pq|rs) += d_pq * d_rs  (chemist notation)
    # In the chemist 2-body convention (1/2)·∑(pq|rs)·Ê_pq Ê_rs, this
    # contributes (1/2)·(d²)·Ê + (1/2)·∑ d_pq d_rs · Ω_{pq,rs} where
    # (d²)_ps = ∑_q d_pq d_qs is the *matrix product* — exact in CBS but
    # truncated in finite basis.
    eri_mo_dse = eri_mo_full + np.einsum('pq,rs->pqrs', dip_mo, dip_mo)

    # --- Proper-DSE 1-body correction ---
    # The true Pauli-Fierz DSE has a 1-body part (1/2)·λ²·Q_pq·Ê_pq
    # where Q_pq = ⟨p|(ε·r̂)²|q⟩ is the quadrupole matrix element.
    # The chemist 2-body convention above instead delivers
    # (1/2)·(d²)_pq·Ê_pq = (1/2)·λ²·(d̃²)_pq·Ê_pq where d̃²=d̃·d̃
    # is the matrix product (= Q only in the complete-basis limit).
    # We add the difference to h1e so the implemented Hamiltonian
    # matches the true PF form regardless of basis.
    h1e_used = h1e
    dse_correction_norm = 0.0
    if proper_dse and lam > 0:
        # ⟨p|r̂_α r̂_β|q⟩ — comp=9 returns shape (9, nao, nao);
        # reshape to (3, 3, nao, nao) row-major so axes match (α, β).
        quad_ao = mol.intor('int1e_rr', comp=9).reshape(
            3, 3, norb, norb,
        )
        # Project onto (ε·r̂)² = ε_α ε_β r̂_α r̂_β
        quad_ao_proj = np.einsum('a,b,abpq->pq', epsilon, epsilon, quad_ao)
        quad_mo = mo_coeff.T @ quad_ao_proj @ mo_coeff

        # The true Pauli-Fierz DSE in 2nd quant has 1-body part
        #     (1/2) λ² Q_pq E_pq   (Q is exact quadrupole matrix element)
        # and 2-body part = (1/2) × [2-body part of D̂²] (cross-particle).
        # PySCF's chemist 2-body convention with the δ_qr Wick subtraction
        # already supplies exactly the (1/2) × [2-body of D̂²] when we
        # augment ERI with d⊗d. So the missing piece is just the proper
        # 1-body (1/2) λ² Q. (Subtracting d̃² here was a bug — that piece
        # is already removed by the chemist convention.)
        dse_correction_h1e = 0.5 * (lam ** 2) * quad_mo
        h1e_used = h1e + dse_correction_h1e
        dse_correction_norm = float(
            np.linalg.norm(dse_correction_h1e),
        )

    # --- Build FCI matrices ---
    H_elec, D_elec, ndim_elec = _build_fci_matrices(
        h1e_used, eri_mo_dse, dip_mo, norb, nelec, enuc)

    # Symmetrize (should already be symmetric, but ensure numerical stability)
    H_elec = 0.5 * (H_elec + H_elec.T)
    D_elec = 0.5 * (D_elec + D_elec.T)

    # --- Coherent-state (displaced) photon basis ---
    # Displace b = a + z with z = d0/√(2Ω), d0 = ⟨D̂⟩ in the reference
    # determinant. This is an exact unitary transform of the Pauli-Fierz
    # Hamiltonian (eigenvalues unchanged at convergence):
    #   diagonal block:  H_elec → H_elec − d0·D + ½·d0²·I
    #   photon coupling: D      → D − d0·I        (fluctuation dipole)
    # The displaced Fock truncation converges far faster for polar
    # molecules — this reproduces the CS-QED-FCI of Vu et al. 2024.
    cs_displacement = 0.0
    if coherent_state and lam > 0:
        na_ref, nb_ref = nelec
        d0 = (sum(dip_mo[i, i] for i in range(na_ref))
              + sum(dip_mo[i, i] for i in range(nb_ref)))
        eye = np.eye(ndim_elec)
        H_elec = H_elec - d0 * D_elec + 0.5 * d0 ** 2 * eye
        D_elec = D_elec - d0 * eye
        cs_displacement = float(d0 / np.sqrt(2.0 * omega))

    # --- Build product space Hamiltonian ---
    nph = nph_max + 1  # photon Fock states: |0⟩, |1⟩, ..., |nph_max⟩
    ndim_total = ndim_elec * nph

    H_total = np.zeros((ndim_total, ndim_total))

    for n in range(nph):
        # Diagonal block: H_elec + DSE + Ω*n
        r0 = n * ndim_elec
        r1 = (n + 1) * ndim_elec
        H_total[r0:r1, r0:r1] = H_elec + omega * n * np.eye(ndim_elec)

        # Off-diagonal: √(Ω/2) * D * <m|a+a†|n>
        # <n+1|a†|n> = √(n+1) → couples block n to block n+1
        if n + 1 < nph:
            m = n + 1
            m0 = m * ndim_elec
            m1 = (m + 1) * ndim_elec
            coupling_matrix = np.sqrt(omega / 2.0) * np.sqrt(n + 1) * D_elec
            H_total[m0:m1, r0:r1] += coupling_matrix
            H_total[r0:r1, m0:m1] += coupling_matrix.T

    # Symmetrize
    H_total = 0.5 * (H_total + H_total.T)

    # --- Diagonalize ---
    eigenvalues, eigenvectors = eigh(H_total)

    # Ground state
    e_gs = eigenvalues[0]
    psi_gs = eigenvectors[:, 0]

    # Compute photon number expectation value
    # <n_ph> = Σ_n n * |c_n|²  where c_n = norm of component in photon sector n
    n_photon = 0.0
    for n in range(nph):
        r0 = n * ndim_elec
        r1 = (n + 1) * ndim_elec
        weight_n = np.sum(psi_gs[r0:r1] ** 2)
        n_photon += n * weight_n

    # --- Correlation energies ---
    # The bare ``e_hf`` is taken from the supplied mean-field object so
    # ``e_corr = e_fci − e_hf`` is always defined. The QED correlation
    # energy is defined only when a QED-HF reference was actually
    # computed (closed-shell, use_qed_hf_reference=True).
    e_hf = float(mf.e_tot) if getattr(mf, 'e_tot', None) is not None else None
    e_corr = (e_fci - e_hf) if e_hf is not None else None
    e_corr_qed = (e_gs - e_qed_hf) if e_qed_hf is not None else None

    return {
        'e_qed_fci': e_gs,
        'e_fci': e_fci,
        'e_qed_hf': e_qed_hf,
        'e_hf': e_hf,
        'e_corr_qed': e_corr_qed,
        'e_corr': e_corr,
        'reference': reference,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'nph_max': nph_max,
        'ndim_elec': ndim_elec,
        'ndim_total': ndim_total,
        'n_photon': n_photon,
        'coherent_state': bool(coherent_state and lam > 0),
        'cs_displacement': cs_displacement,
        'proper_dse': bool(proper_dse and lam > 0),
        'dse_correction_norm': dse_correction_norm,
    }
