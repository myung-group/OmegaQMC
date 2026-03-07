"""
QED-FCI: Exact diagonalization of the Pauli-Fierz Hamiltonian.

Constructs the full Hamiltonian in the product basis:
    |FCI_electronic⟩ ⊗ |n_photon⟩

and diagonalizes using scipy.linalg.eigh.

Supports the dipole gauge Pauli-Fierz Hamiltonian:
    H = H_elec + DSE + √(Ω/2)(a+a†) D + Ω a†a

where:
    DSE = ½(λ ε·Σ_i r_i)²  (dipole self-energy, purely electronic)
    D   = λ Σ_ij <i|r·ε|j> c†_iσ c_jσ  (dipole operator)

Reference: arXiv:2410.18838
"""

import numpy as np
from scipy.linalg import eigh
from pyscf import ao2mo, fci
from pyscf.fci import direct_spin1, cistring


def _build_fci_matrices(h1e, eri, dip_mo, norb, nelec, enuc):
    """Build the electronic Hamiltonian and dipole operator matrices in FCI basis.

    Args:
        h1e: one-body integrals (norb, norb).
        eri: two-electron integrals with DSE, chemist notation (norb,norb,norb,norb).
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


def qed_fci(mf, omega, coupling_vec, nph_max=10):
    """QED-FCI: exact diagonalization of the Pauli-Fierz Hamiltonian.

    Builds the full Hamiltonian in the product basis |FCI⟩ ⊗ |n_ph⟩
    and finds the ground state energy.

    Dipole gauge:
        H = H_elec + DSE + √(Ω/2)(a+a†) D + Ω a†a

    The photon Fock space is truncated at nph_max photons.

    Args:
        mf: PySCF mean-field object (must have run kernel()).
        omega: Photon frequency in Hartree.
        coupling_vec: Light-matter coupling vector (3,). Direction gives
            polarization ε, magnitude gives coupling strength λ.
        nph_max: Maximum number of photons in Fock space truncation.

    Returns:
        dict with:
            'e_qed_fci': QED-FCI ground state energy.
            'e_fci': standard FCI energy (no cavity).
            'eigenvalues': all eigenvalues of the product Hamiltonian.
            'eigenvectors': corresponding eigenvectors.
            'nph_max': photon truncation used.
            'ndim_elec': electronic FCI dimension.
            'ndim_total': total product space dimension.
            'n_photon': expectation value of photon number in ground state.
    """
    mol = mf.mol
    norb = mol.nao_nr()
    nelec = mol.nelec
    mo_coeff = np.asarray(mf.mo_coeff)
    enuc = mol.energy_nuc()

    # --- Electronic integrals in MO basis ---
    h1e = mo_coeff.T @ np.asarray(mf.get_hcore()) @ mo_coeff
    eri_mo = ao2mo.full(mol, mo_coeff)
    eri_mo_full = ao2mo.restore(1, eri_mo, norb)

    # --- Dipole matrix ---
    coupling_vec = np.asarray(coupling_vec, dtype=np.float64)
    lam = np.linalg.norm(coupling_vec)

    if lam > 1e-15:
        epsilon = coupling_vec / lam
    else:
        epsilon = np.array([0.0, 0.0, 1.0])
        lam = 0.0

    dip_ao = mol.intor('int1e_r', comp=3)
    dip_ao_proj = lam * np.einsum('k,kpq->pq', epsilon, dip_ao)
    dip_mo = mo_coeff.T @ dip_ao_proj @ mo_coeff

    # --- Standard FCI (no cavity) ---
    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()

    # --- Add DSE to two-electron integrals ---
    # DSE: (pq|rs) += d_pq * d_rs  (chemist notation)
    eri_mo_dse = eri_mo_full + np.einsum('pq,rs->pqrs', dip_mo, dip_mo)

    # --- Build FCI matrices ---
    H_elec, D_elec, ndim_elec = _build_fci_matrices(
        h1e, eri_mo_dse, dip_mo, norb, nelec, enuc)

    # Symmetrize (should already be symmetric, but ensure numerical stability)
    H_elec = 0.5 * (H_elec + H_elec.T)
    D_elec = 0.5 * (D_elec + D_elec.T)

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

    return {
        'e_qed_fci': e_gs,
        'e_fci': e_fci,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'nph_max': nph_max,
        'ndim_elec': ndim_elec,
        'ndim_total': ndim_total,
        'n_photon': n_photon,
    }
