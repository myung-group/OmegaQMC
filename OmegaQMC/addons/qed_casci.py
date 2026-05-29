"""
QED-CASCI: Exact diagonalization of the Pauli-Fierz Hamiltonian in an
electronic complete active space, tensored with a truncated photon Fock
space.

This is the active-space counterpart of :mod:`OmegaQMC.addons.qed_fci`.
All Hamiltonian conventions match qed_fci.py exactly, so the two
methods coincide in the limit where the active space spans the full
one-particle MO basis (``ncas = nmo - ncore`` with all valence
electrons active).

Following Vu et al., J. Chem. Theory Comput. 20, 1214 (2024)
[doi:10.1021/acs.jctc.3c01207], this implementation is a **CS-QED-CASCI**:
the QED-HF reference fixes the orbital choice, and (with the default
``coherent_state=True``) the photon Fock states are built in the
coherent-state frame Û_CS = exp[z(b†-b)], z = ⟨D̂⟩/√(2Ω). The
displacement is an exact unitary transform of the Pauli-Fierz
Hamiltonian, so the energy is unchanged at convergence in ``nph_max``;
the coherent-state basis just converges far faster (e.g. LiH already
converges at nph_max=1, whereas the raw photon-number basis needs many
Fock states). Set ``coherent_state=False`` to recover the raw
photon-number (PN) basis centred at a=0.

The total dipole μ̂ = μ̂_e + μ_n is approximated by the electronic
part μ̂_e only (matching qed_fci.py and qed_hf.py). For neutral
molecules this is equivalent to fixing the origin at the nuclear
center of charge; results for other origin choices include an
origin-dependent shift that cancels between QED-HF and QED-CASCI in
the correlation energy.

Pauli-Fierz Hamiltonian (dipole gauge):

    H = H_elec + (1/2)(λ·d̂)² + √(Ω/2)(â+â†)(λ·d̂) + Ω â†â

where d̂ = ∑_pq ⟨p|ε·r̂|q⟩ E_pq is the electronic dipole operator and
ε is the cavity polarization direction.

Notes on the active-space projection:
  • The DSE-augmented one- and two-electron integrals are built in
    the *full* MO basis (h1e + ½λ²Q quadrupole correction, ERI +
    λ²·d⊗d in chemist convention). Standard frozen-core
    (J − K) dressing then folds the core's contribution into
    ``h1eff_act`` and ``e_core``. The augmented-ERI machinery
    automatically captures all DSE cross terms between core and
    active orbitals (the d_pq d_rs tensor includes pq ∈ core,
    rs ∈ active blocks, which become a 1-body contribution to
    ``h1eff_act`` after the 2J − K contraction).
  • The bilinear photon coupling needs the *total* electronic dipole
    operator, which in CAS|core⟩ acts as
    D̂|core⟩|act⟩ ≈ (d_core + d̂_act)|core⟩|act⟩,
    where d_core = 2 Σ_{i∈core} d_ii is the c-number contribution
    from the doubly-occupied frozen core and d̂_act is the dipole
    restricted to the active orbitals. The constant ``d_core`` is
    added back as a scalar shift of the off-diagonal photon-coupling
    block.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import eigh
from pyscf import ao2mo, mcscf

from OmegaQMC.addons.qed_fci import _build_fci_matrices
from OmegaQMC.addons.qed_hf import run_qed_hf


def _parse_active_space(mol, ncas, nelecas):
    """Parse and validate an active-space specification.

    Returns
    -------
    ncore : int
        Number of doubly-occupied frozen-core orbitals.
    nelecas_a, nelecas_b : int
        Number of α/β electrons in the active space.
    """
    nelec_total = mol.nelectron
    if isinstance(nelecas, (int, np.integer)):
        nelecas_int = int(nelecas)
        if nelecas_int % 2 != 0:
            raise ValueError(
                "When ``nelecas`` is an int it must be even; for "
                "open-shell active spaces pass a (na, nb) tuple."
            )
        nelecas_a = nelecas_b = nelecas_int // 2
    else:
        nelecas_a, nelecas_b = int(nelecas[0]), int(nelecas[1])

    nelec_act_total = nelecas_a + nelecas_b
    if nelec_act_total > nelec_total:
        raise ValueError(
            f"nelecas={nelec_act_total} exceeds molecular electron "
            f"count {nelec_total}."
        )

    ncore_total = nelec_total - nelec_act_total
    if ncore_total % 2 != 0:
        raise ValueError(
            f"Frozen core must be doubly occupied; "
            f"nelec_total - nelecas = {ncore_total} is odd. "
            "Use a closed-shell core."
        )
    ncore = ncore_total // 2

    norb = mol.nao_nr()
    if ncas + ncore > norb:
        raise ValueError(
            f"ncas + ncore = {ncas + ncore} exceeds nmo = {norb}."
        )

    return ncore, nelecas_a, nelecas_b


def run_qed_casci(mf, ncas, nelecas, omega, coupling_vec,
                  nph_max=10, proper_dse=True,
                  use_qed_hf_reference=True, coherent_state=True):
    """QED-CASCI: exact diagonalization of the Pauli-Fierz Hamiltonian
    in an electronic active space tensored with a truncated photon
    Fock space.

    Conventions match :func:`OmegaQMC.addons.qed_fci.run_qed_fci`
    exactly; ``run_qed_casci`` reduces to ``run_qed_fci`` when the
    active space is the full MO basis (``ncas = nmo`` and
    ``nelecas = mol.nelec``).

    Args:
        mf: PySCF mean-field object (must have run kernel()).
        ncas: Number of active spatial orbitals.
        nelecas: Number of active electrons. Either an int (total,
            must be even) or a (n_alpha, n_beta) tuple.
        omega: Photon frequency in Hartree.
        coupling_vec: Light-matter coupling vector λ·ε of length 3.
            Direction gives polarization ε; magnitude gives λ.
        nph_max: Maximum photon number in the Fock truncation. The
            photon space has ``nph_max + 1`` basis states |0⟩…|nph_max⟩.
        proper_dse: If True (default), add ½λ²·q_pq with the true
            quadrupole integral q_pq = ⟨p|(ε·r̂)²|q⟩ to the one-body
            Hamiltonian, so the DSE matches the operator-form
            ½(λ·d̂)² in any basis (matches Eq. 20 of the paper). If
            False, only the d⊗d augmentation of the ERI is applied
            (operator-squared form, exact only in the CBS limit).
        use_qed_hf_reference: If True (default), use cavity-relaxed
            QED-HF orbitals (matches the CS-QED-CASCI orbital choice
            of Vu et al. 2024). Falls back to ``mf.mo_coeff`` for
            open-shell systems with a RuntimeWarning. Set to False
            to force HF orbitals (PN-QED-CASCI orbital choice).
        coherent_state: If True (default), use the coherent-state
            (displaced) photon basis b = a + z with z = ⟨D̂⟩/√(2Ω),
            matching the CS-QED-CASCI photon basis of Vu et al. 2024.
            The displacement is an exact unitary transform (energy
            unchanged at convergence in ``nph_max``) but converges far
            faster for polar molecules. With the same reference dipole,
            this stays bit-identical to ``run_qed_fci`` at full active
            space. If False, use the raw photon-number (Fock) basis.

    Returns:
        dict with:
            'e_qed_casci'        : QED-CASCI ground-state energy.
            'e_casci'            : Standard CASCI energy (no cavity)
                                   in the same reference orbitals.
            'e_qed_hf'           : QED-HF reference energy (None if
                                   QED-HF was not run).
            'e_hf'               : Bare RHF total energy from ``mf``.
            'e_corr_qed'         : e_qed_casci − e_qed_hf, the QED
                                   correlation energy in the active
                                   space (None if QED-HF unavailable).
            'e_corr'             : e_casci − e_hf, bare CASCI
                                   correlation energy.
            'reference'          : 'QED-HF' or 'HF'.
            'ncas'               : Number of active orbitals.
            'ncore'              : Number of frozen-core orbitals.
            'nelecas'            : (n_alpha_act, n_beta_act).
            'eigenvalues'        : All product-space eigenvalues.
            'eigenvectors'       : Corresponding eigenvectors.
            'nph_max'            : Photon truncation used.
            'ndim_elec'          : Electronic CASCI dimension.
            'ndim_total'         : Total product-space dimension.
            'n_photon'           : <n_ph> in the ground state. With
                                   ``coherent_state=True`` this is the
                                   displaced-frame ⟨b†b⟩, not ⟨a†a⟩.
            'coherent_state'     : Whether the coherent-state photon
                                   basis was used.
            'cs_displacement'    : Coherent-state displacement z =
                                   ⟨D̂_total⟩/√(2Ω) (0.0 if disabled).
            'd_core_const'       : Constant electronic dipole shift
                                   from the frozen core
                                   (= 2 Σ_{i∈core} d_ii).
            'proper_dse'         : Whether the ½λ²q correction was
                                   applied.
            'dse_correction_norm': Frobenius norm of the proper-DSE
                                   1-body correction (full MO basis).
    """
    mol = mf.mol
    norb = mol.nao_nr()
    enuc = mol.energy_nuc()

    # --- Active-space partitioning ---
    ncore, nelecas_a, nelecas_b = _parse_active_space(mol, ncas, nelecas)
    nelec_act = (nelecas_a, nelecas_b)

    # --- Coupling vector → magnitude λ and polarization ε ---
    coupling_vec = np.asarray(coupling_vec, dtype=np.float64)
    lam = float(np.linalg.norm(coupling_vec))
    if lam > 1e-15:
        epsilon = coupling_vec / lam
    else:
        epsilon = np.array([0.0, 0.0, 1.0])
        lam = 0.0

    # --- Reference orbitals: QED-HF (closed shell) or fall back ---
    closed_shell = (mol.nelec[0] == mol.nelec[1])
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
                "qed_casci: QED-HF reference requested but molecule is "
                "open-shell; falling back to mf.mo_coeff. e_corr_qed "
                "will be None.",
                RuntimeWarning,
                stacklevel=2,
            )
        mo_coeff = np.asarray(mf.mo_coeff)

    # --- Electronic integrals (full MO basis) ---
    h1e_mo = mo_coeff.T @ np.asarray(mf.get_hcore()) @ mo_coeff
    eri_mo = ao2mo.full(mol, mo_coeff)
    eri_mo_full = ao2mo.restore(1, eri_mo, norb)

    # --- Dipole / quadrupole integrals (full MO basis) ---
    dip_ao = mol.intor('int1e_r', comp=3)
    dip_ao_proj = lam * np.einsum('k,kpq->pq', epsilon, dip_ao)
    dip_mo = mo_coeff.T @ dip_ao_proj @ mo_coeff

    # --- DSE 2-body augmentation: (pq|rs) += d_pq d_rs (chemist) ---
    eri_mo_dse = eri_mo_full + np.einsum('pq,rs->pqrs', dip_mo, dip_mo)

    # --- DSE 1-body proper quadrupole correction (full MO) ---
    h1e_dse = h1e_mo
    dse_correction_norm = 0.0
    if proper_dse and lam > 0:
        quad_ao = mol.intor('int1e_rr', comp=9).reshape(
            3, 3, norb, norb,
        )
        quad_ao_proj = np.einsum('a,b,abpq->pq', epsilon, epsilon, quad_ao)
        quad_mo = mo_coeff.T @ quad_ao_proj @ mo_coeff
        dse_correction_h1e = 0.5 * (lam ** 2) * quad_mo
        h1e_dse = h1e_mo + dse_correction_h1e
        dse_correction_norm = float(np.linalg.norm(dse_correction_h1e))

    # --- Active-space integral projection ---
    # Standard CASCI frozen-core dressing applied to the
    # DSE-augmented integrals. The augmented ERI's d⊗d structure
    # automatically yields the right core-active DSE cross terms via
    # the J - K contraction.
    cas = slice(ncore, ncore + ncas)
    core = slice(0, ncore)

    if ncore > 0:
        # Effective 1-body in active space: h1eff[p,q] =
        #   h1e_dse[p,q] + Σ_{i∈core} (2·eri_dse[p,q,i,i]
        #                              − eri_dse[p,i,i,q])
        J_core_act = 2.0 * np.einsum(
            'pqii->pq', eri_mo_dse[cas, cas, core, core])
        K_core_act = np.einsum(
            'piiq->pq', eri_mo_dse[cas, core, core, cas])
        h1eff_act = h1e_dse[cas, cas] + J_core_act - K_core_act

        # Frozen-core energy: enuc + 2·Tr(h1e_dse_core)
        #                          + Σ_{ij∈core} (2(ii|jj) − (ij|ji))
        e_core_1e = 2.0 * np.trace(h1e_dse[core, core])
        eri_cc = eri_mo_dse[core, core, core, core]
        e_core_2e = (
            2.0 * np.einsum('iijj->', eri_cc)
            - np.einsum('ijji->', eri_cc)
        )
        e_core = float(enuc + e_core_1e + e_core_2e)
    else:
        h1eff_act = h1e_dse[cas, cas].copy()
        e_core = float(enuc)

    eri_eff_act = np.ascontiguousarray(eri_mo_dse[cas, cas, cas, cas])

    # --- Dipole in active space + constant core shift ---
    # Restricted to CAS|core⟩, D̂ acts as (d_core_const + D̂_act):
    #   d_core_const = 2 Σ_{i∈core} d_ii  (closed-shell core)
    #   D̂_act       = Σ_{pq∈act} d_pq E_pq
    dip_act = np.ascontiguousarray(dip_mo[cas, cas])
    if ncore > 0:
        d_core_const = 2.0 * float(np.trace(dip_mo[core, core]))
    else:
        d_core_const = 0.0

    # --- Build active-space FCI matrices (electronic part) ---
    H_elec, D_elec, ndim_elec = _build_fci_matrices(
        h1eff_act, eri_eff_act, dip_act,
        ncas, nelec_act, e_core,
    )
    H_elec = 0.5 * (H_elec + H_elec.T)
    D_elec = 0.5 * (D_elec + D_elec.T)

    # Total active-space dipole operator (electronic, no nuclear)
    # acting on the CAS Hilbert space includes the constant core
    # contribution as a uniform diagonal shift.
    D_total = D_elec + d_core_const * np.eye(ndim_elec)

    # --- Coherent-state (displaced) photon basis ---
    # Displace b = a + z with z = d0/√(2Ω), where d0 = ⟨D̂_total⟩ in the
    # QED-HF reference determinant (doubly-occupied core + active-occupied
    # orbitals). Exact unitary transform of the Pauli-Fierz Hamiltonian:
    #   diagonal block:  H_elec → H_elec − d0·D_total + ½·d0²·I
    #   photon coupling: D_total → D_total − d0·I   (fluctuation dipole)
    # At full active space d0 equals the run_qed_fci displacement, so the
    # two methods stay bit-identical. Matches CS-QED-CASCI of Vu et al.
    cs_displacement = 0.0
    if coherent_state and lam > 0:
        d0 = d_core_const + (
            sum(dip_act[i, i] for i in range(nelecas_a))
            + sum(dip_act[i, i] for i in range(nelecas_b))
        )
        eye = np.eye(ndim_elec)
        H_elec = H_elec - d0 * D_total + 0.5 * d0 ** 2 * eye
        D_total = D_total - d0 * eye
        cs_displacement = float(d0 / np.sqrt(2.0 * omega))

    # --- Standard CASCI (no cavity) in the same reference orbitals ---
    # Used to report the bare correlation energy alongside the cavity
    # one. We feed the *reference* MO coefficients explicitly so the
    # CASCI is consistent with the chosen orbital basis (QED-HF or HF).
    mc = mcscf.CASCI(mf, ncas, (nelecas_a, nelecas_b))
    mc.verbose = 0
    e_casci_tot = mc.kernel(mo_coeff)[0]
    e_casci = float(e_casci_tot)

    # --- Product-space Pauli-Fierz Hamiltonian ---
    nph = nph_max + 1
    ndim_total = ndim_elec * nph
    H_total = np.zeros((ndim_total, ndim_total))

    for n in range(nph):
        # Diagonal block: H_elec + Ω·n·I
        r0, r1 = n * ndim_elec, (n + 1) * ndim_elec
        H_total[r0:r1, r0:r1] = (
            H_elec + omega * n * np.eye(ndim_elec)
        )

        # Off-diagonal block (n → n+1): √(Ω/2) √(n+1) D_total
        if n + 1 < nph:
            m = n + 1
            m0, m1 = m * ndim_elec, (m + 1) * ndim_elec
            coupling_matrix = (
                np.sqrt(omega / 2.0) * np.sqrt(n + 1) * D_total
            )
            H_total[m0:m1, r0:r1] += coupling_matrix
            H_total[r0:r1, m0:m1] += coupling_matrix.T

    H_total = 0.5 * (H_total + H_total.T)

    eigenvalues, eigenvectors = eigh(H_total)
    e_gs = float(eigenvalues[0])
    psi_gs = eigenvectors[:, 0]

    # Photon-number expectation value in the ground state
    n_photon = 0.0
    for n in range(nph):
        r0, r1 = n * ndim_elec, (n + 1) * ndim_elec
        n_photon += n * float(np.sum(psi_gs[r0:r1] ** 2))

    # --- Correlation energies ---
    e_hf = float(mf.e_tot) if getattr(mf, 'e_tot', None) is not None else None
    e_corr = (e_casci - e_hf) if e_hf is not None else None
    e_corr_qed = (e_gs - e_qed_hf) if e_qed_hf is not None else None

    return {
        'e_qed_casci': e_gs,
        'e_casci': e_casci,
        'e_qed_hf': e_qed_hf,
        'e_hf': e_hf,
        'e_corr_qed': e_corr_qed,
        'e_corr': e_corr,
        'reference': reference,
        'ncas': int(ncas),
        'ncore': int(ncore),
        'nelecas': nelec_act,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'nph_max': int(nph_max),
        'ndim_elec': int(ndim_elec),
        'ndim_total': int(ndim_total),
        'n_photon': float(n_photon),
        'd_core_const': float(d_core_const),
        'coherent_state': bool(coherent_state and lam > 0),
        'cs_displacement': cs_displacement,
        'proper_dse': bool(proper_dse and lam > 0),
        'dse_correction_norm': dse_correction_norm,
    }
