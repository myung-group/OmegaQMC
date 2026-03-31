"""
QED-specific integral preparation for QED-AFQMC.

Augments standard Cholesky-decomposed ERIs with the
dipole self-energy (DSE) term from the Pauli-Fierz
Hamiltonian in the dipole gauge.
"""

import numpy as np
import jax.numpy as jnp

from OmegaQMC.integrals.cholesky import chunked_cholesky


def prepare_qed_integrals(
    mf, omega, coupling_vec, chol_cut=1e-5,
):
    """Prepare AFQMC integrals augmented with QED DSE.

    In the dipole gauge Pauli-Fierz Hamiltonian:
        H = sum h_ij(q) c+_is c_js
            + 1/2 sum v_ijkl c+c+cc
            + Omega/2 (Pi^2 + q^2 - 1)
    where:
        h_ij(q) = h_ij^0 + sqrt(Omega) * q * d_ij
        v_ijkl  = v_ijkl^Coulomb + d_ik * d_jl  (DSE)
        d_ij    = lambda * <i|r.eps|j>

    The DSE adds one extra Cholesky vector (d_ij).

    Args:
        mf: PySCF mean-field object
            (must have run kernel()).
        omega: Photon frequency in Hartree.
        coupling_vec: Light-matter coupling vector (3,).
            Direction = polarization, magnitude = lambda.
        chol_cut: Cholesky decomposition threshold.

    Returns:
        dict with keys:
            'h1e': shape (nbasis, nbasis)
            'h1e_mod_0': shape (nbasis, nbasis)
            'chol_qed': shape (naux+1, nbasis, nbasis)
            'dip_mo': shape (nbasis, nbasis)
            'enuc': float
            'nbasis', 'nup', 'ndown': int
            'mo_coeff': jnp.array
            'omega': float
    """
    mol = mf.mol
    nbasis = mol.nao_nr()
    mo_coeff = np.asarray(mf.mo_coeff)
    nup, ndown = mol.nelec

    # --- Standard electronic integrals ---
    hcore_ao = np.asarray(mf.get_hcore())
    h1e = mo_coeff.T @ hcore_ao @ mo_coeff

    chol_ao = chunked_cholesky(mol, chol_cut=chol_cut)
    chol_mo = np.einsum(
        'ab,gbc,cd->gad',
        mo_coeff.T, chol_ao, mo_coeff,
    )

    # --- QED: dipole matrix elements ---
    coupling_vec = np.asarray(
        coupling_vec, dtype=np.float64,
    )
    lam = np.linalg.norm(coupling_vec)

    if lam > 1e-15:
        epsilon = coupling_vec / lam
    else:
        epsilon = np.array([0.0, 0.0, 1.0])
        lam = 0.0

    # Dipole integrals in AO basis: (3, nao, nao)
    dip_ao = mol.intor('int1e_r', comp=3)
    # Project onto polarization and scale by lambda
    dip_ao_proj = lam * np.einsum(
        'k,kpq->pq', epsilon, dip_ao,
    )
    # Transform to MO basis
    dip_mo = mo_coeff.T @ dip_ao_proj @ mo_coeff

    # --- Augment Cholesky vectors with DSE ---
    chol_qed = np.concatenate(
        [chol_mo, dip_mo[None, :, :]], axis=0,
    )

    # --- Modified one-body Hamiltonian ---
    v0 = np.einsum(
        'gij,gkj->ik', chol_qed, chol_qed,
    ) * (-0.5)
    h1e_mod_0 = h1e + v0

    enuc = mol.energy_nuc()

    return {
        'h1e': jnp.array(h1e),
        'h1e_mod_0': jnp.array(h1e_mod_0),
        'chol_qed': jnp.array(chol_qed),
        'dip_mo': jnp.array(dip_mo),
        'enuc': float(enuc),
        'nbasis': nbasis,
        'nup': nup,
        'ndown': ndown,
        'mo_coeff': jnp.array(mo_coeff),
        'omega': float(omega),
    }
