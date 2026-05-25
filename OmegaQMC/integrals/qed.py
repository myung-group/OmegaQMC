"""
QED-specific integral preparation for QED-AFQMC.

Augments standard Cholesky-decomposed ERIs with the
dipole self-energy (DSE) term from the Pauli-Fierz
Hamiltonian in the dipole gauge.
"""

import numpy as np
import jax.numpy as jnp

from OmegaQMC.integrals.cholesky import (
    chunked_cholesky,
    DiskChol,
    h5py,
)


def prepare_qed_integrals(
    mf, omega, coupling_vec, chol_cut=1e-5,
    chol_h5_path=None, chol_chunk_g=128,
    mo_coeff=None, proper_dse=True,
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

    The DSE adds one extra Cholesky vector (d_ij), so the augmented
    tensor has shape ``(naux + 1, nbasis, nbasis)``.

    Note on the DSE in a finite basis (``proper_dse``):
        The extra Cholesky vector implements the DSE in its
        *operator-squared* form ``(1/2) D̂²`` with
        ``D̂ = sum_pq d_pq Ê_pq``.  Its one-body part is the matrix
        product ``d̃²_pq = sum_r d_pr d_rq``.  The true Pauli-Fierz DSE,
        ``(1/2) λ² (ε·∑_i r̂_i)²``, instead carries the one-body
        *quadrupole* matrix ``Q_pq = λ² <p|(ε·r̂)²|q>``.  The two agree
        only in the complete-basis limit (resolution of identity).  With
        ``proper_dse=True`` (default) we add the residual
        ``(1/2)(λ² Q − d̃²)`` to ``h1e`` so the implemented Hamiltonian
        matches the exact Pauli-Fierz form in any basis — and therefore
        QED-AFQMC reproduces QED-FCI run with the same ``proper_dse=True``
        convention.  Set ``proper_dse=False`` for the legacy
        operator-squared behaviour.

    Args:
        mf: PySCF mean-field object
            (must have run kernel()).
        omega: Photon frequency in Hartree.
        coupling_vec: Light-matter coupling vector (3,).
            Direction = polarization, magnitude = lambda.
        chol_cut: Cholesky decomposition threshold.
        chol_h5_path: If set, write the augmented Cholesky tensor to
            this HDF5 file (dataset ``chol_mo``) and return a
            :class:`DiskChol` as ``chol_qed``. AO→MO transform and v0
            accumulation are done in ``chol_chunk_g`` slabs so peak
            RAM is independent of naux.
        chol_chunk_g: Slab size along the auxiliary axis. Default 128.
        mo_coeff: Optional (nao, nmo) orbital coefficients to build the
            MO basis from.  Defaults to ``mf.mo_coeff``.  Pass the
            QED-Hartree-Fock orbitals here to use the cavity-relaxed
            reference instead of plain (RHF) orbitals.
        proper_dse: If True (default), add the one-body quadrupole
            correction ``(1/2)(λ² Q − d̃²)`` to ``h1e`` so the DSE is
            exact in any basis.  If False, keep the legacy
            operator-squared DSE.

    Returns:
        dict with keys:
            'h1e': shape (nbasis, nbasis)
            'h1e_mod_0': shape (nbasis, nbasis)
            'chol_qed': shape (naux+1, nbasis, nbasis); jnp array
                        (default) or :class:`DiskChol` if path set
            'dip_mo': shape (nbasis, nbasis)
            'enuc': float
            'nbasis', 'nup', 'ndown': int
            'mo_coeff': jnp.array
            'omega': float
    """
    mol = mf.mol
    nbasis = mol.nao_nr()
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    mo_coeff = np.asarray(mo_coeff)
    nup, ndown = mol.nelec
    nmo = mo_coeff.shape[1]

    # --- One-body and dipole integrals ---
    hcore_ao = np.asarray(mf.get_hcore())
    h1e = mo_coeff.T @ hcore_ao @ mo_coeff

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

    # --- Proper-DSE one-body correction ---
    # The augmented Cholesky vector below contributes the DSE through the
    # *genuine* two-body operator (1/2)∑_{i≠j} d̂_i d̂_j (the one-body
    # byproduct of D̂² is the normal-ordering shift v0, used only by the
    # propagator, and is absent from the local-energy estimator).  The exact
    # Pauli-Fierz DSE additionally carries the one-body self term
    # (1/2)∑_i d̂_i² = (1/2) λ² <p|(ε·r̂)²|q>.  Add this quadrupole matrix to
    # h1e so the implemented Hamiltonian is the exact Pauli-Fierz form in any
    # basis (matching QED-FCI with proper_dse=True).  It is the same
    # correction applied in OmegaQMC.addons.qed_fci.
    if proper_dse and lam > 0.0:
        quad_ao = mol.intor('int1e_rr', comp=9).reshape(3, 3, nbasis, nbasis)
        quad_ao_proj = (lam ** 2) * np.einsum(
            'a,b,abpq->pq', epsilon, epsilon, quad_ao)
        quad_mo = mo_coeff.T @ quad_ao_proj @ mo_coeff
        h1e = h1e + 0.5 * quad_mo

    # --- Cholesky in AO basis (chunked_cholesky picks dense/direct) ---
    chol_ao = chunked_cholesky(mol, chol_cut=chol_cut)
    naux = chol_ao.shape[0]

    if chol_h5_path is None:
        # In‑memory path: two GEMMs for AO→MO, single GEMM for v0
        tmp = chol_ao.reshape(naux * nbasis, nbasis) @ mo_coeff
        tmp = tmp.reshape(naux, nbasis, nmo)
        chol_mo = (
            mo_coeff.T
            @ tmp.transpose(1, 0, 2).reshape(nbasis, naux * nmo)
        ).reshape(nmo, naux, nmo).transpose(1, 0, 2)

        # Augment with dipole self‑energy as the (naux+1)-th vector
        chol_qed = np.concatenate(
            [chol_mo, dip_mo[None, :, :]], axis=0,
        )

        # v0[i,k] = -0.5 sum_{g,j} L[g,i,j] L[g,k,j]
        A = chol_qed.transpose(0, 2, 1).reshape((naux + 1) * nmo, nmo)
        v0 = -0.5 * (A.T @ A)
        chol_out = jnp.array(chol_qed)

    else:
        if h5py is None:
            raise ImportError(
                "h5py is required for chol_h5_path — install with "
                "`pip install h5py`"
            )
        v0 = np.zeros((nmo, nmo))
        with h5py.File(chol_h5_path, 'w') as f:
            ds = f.create_dataset(
                'chol_mo',
                shape=(naux + 1, nmo, nmo),
                dtype=np.float64,
                chunks=(min(chol_chunk_g, naux + 1), nmo, nmo),
            )
            # Stream AO→MO transform for the naux Coulomb vectors
            for g0 in range(0, naux, chol_chunk_g):
                g1 = min(g0 + chol_chunk_g, naux)
                chunk_ao = chol_ao[g0:g1]
                ng = g1 - g0
                tmp = chunk_ao.reshape(ng * nbasis, nbasis) @ mo_coeff
                tmp = tmp.reshape(ng, nbasis, nmo)
                chunk_mo = (
                    mo_coeff.T
                    @ tmp.transpose(1, 0, 2).reshape(nbasis, ng * nmo)
                ).reshape(nmo, ng, nmo).transpose(1, 0, 2)
                ds[g0:g1] = chunk_mo
                A = chunk_mo.transpose(0, 2, 1).reshape(ng * nmo, nmo)
                v0 += A.T @ A
                del tmp, chunk_mo, A
            # Append the DSE row as the (naux)-th index
            ds[naux] = dip_mo
            # DSE contribution: sum_j dip_mo[i,j] * dip_mo[k,j]
            v0 += dip_mo @ dip_mo.T
        v0 *= -0.5
        chol_out = DiskChol(chol_h5_path, dataset='chol_mo',
                            chunk_g=chol_chunk_g)

    h1e_mod_0 = h1e + v0
    enuc = mol.energy_nuc()

    return {
        'h1e': jnp.array(h1e),
        'h1e_mod_0': jnp.array(h1e_mod_0),
        'chol_qed': chol_out,
        'dip_mo': jnp.array(dip_mo),
        'enuc': float(enuc),
        'nbasis': nbasis,
        'nup': nup,
        'ndown': ndown,
        'mo_coeff': jnp.array(mo_coeff),
        'omega': float(omega),
    }
