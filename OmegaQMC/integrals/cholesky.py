"""
Cholesky decomposition of ERIs and integral preparation
for AFQMC.

Functions here run at setup time (not inside JIT hot loops).
"""

import numpy as np
import jax
import jax.numpy as jnp


def extract_casscf_trial(mc, coeff_threshold=1e-4):
    """Extract multi-det trial from PySCF CASSCF/CASCI.

    Args:
        mc: PySCF CASSCF or CASCI object
            (must have run kernel()).
        coeff_threshold: Threshold on |c_I| for
            truncating the CI expansion.

    Returns:
        dict with keys:
            'ci_coeffs': jnp.array, shape (ndet,).
            'occ_up': jnp.array int, shape (ndet, nup).
            'occ_dn': jnp.array int, shape (ndet, ndown).
            'ndet': Number of determinants retained.
            'mo_coeff': MO coefficients, shape (nao, nmo).
    """
    from pyscf.fci import cistring

    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nelecas = mc.nelecas  # (nalpha_cas, nbeta_cas)
    nup = ncore + nelecas[0]
    ndown = ncore + nelecas[1]

    # Generate occupation string lists
    occslst_a = cistring.gen_occslst(
        range(ncas), nelecas[0],
    )
    occslst_b = cistring.gen_occslst(
        range(ncas), nelecas[1],
    )

    ci = np.asarray(mc.ci)
    core_indices = list(range(ncore))

    coeffs = []
    occ_up_list = []
    occ_dn_list = []

    for ia in range(len(occslst_a)):
        for ib in range(len(occslst_b)):
            c = ci[ia, ib]
            if abs(c) > coeff_threshold:
                coeffs.append(c)
                occ_a = core_indices + [
                    ncore + j for j in occslst_a[ia]
                ]
                occ_b = core_indices + [
                    ncore + j for j in occslst_b[ib]
                ]
                occ_up_list.append(occ_a)
                occ_dn_list.append(occ_b)

    ndet = len(coeffs)

    return {
        'ci_coeffs': jnp.array(np.array(coeffs)),
        'occ_up': jnp.array(
            np.array(occ_up_list, dtype=np.int32),
        ),
        'occ_dn': jnp.array(
            np.array(occ_dn_list, dtype=np.int32),
        ),
        'ndet': ndet,
        'mo_coeff': np.asarray(mc.mo_coeff),
    }


def chunked_cholesky(mol, chol_cut=1e-5, max_vecs=None):
    """Modified Cholesky decomposition of the ERI tensor.

    Computes the full ERI tensor and performs pivoted
    Cholesky decomposition on the reshaped
    (nbasis^2, nbasis^2) matrix:
        (pq|rs) ~ sum_g L^g_{pq} L^g_{rs}

    Args:
        mol: PySCF Mole object.
        chol_cut: Convergence threshold
            (max diagonal residual).
        max_vecs: Maximum Cholesky vectors.
            Defaults to 10*nbasis.

    Returns:
        chol_vecs: np.ndarray, shape (naux, nbasis, nbasis).
    """
    nbasis = mol.nao_nr()
    nao2 = nbasis * nbasis
    if max_vecs is None:
        max_vecs = 10 * nbasis

    # Full ERI tensor: (pq|rs)
    eri = mol.intor('int2e', aosym='s1') \
        .reshape(nbasis, nbasis, nbasis, nbasis)
    eri_2d = eri.reshape(nao2, nao2)

    # Pivoted Cholesky decomposition
    diag = np.diag(eri_2d).copy()

    chol_list = []
    for ivec in range(max_vecs):
        delta_max = np.max(diag)
        if delta_max < chol_cut:
            break

        # Select pivot
        nu = np.argmax(diag)

        # Compute Cholesky vector
        col = eri_2d[:, nu].copy()

        # Subtract previous contributions
        for prev_vec in chol_list:
            col -= prev_vec * prev_vec[nu]

        # Normalize
        col /= np.sqrt(delta_max)
        chol_list.append(col)

        # Update diagonal
        diag -= col * col
        diag = np.maximum(diag, 0.0)

    naux = len(chol_list)
    chol_vecs = np.array(chol_list).reshape(
        naux, nbasis, nbasis,
    )
    return chol_vecs


def prepare_afqmc_integrals(
    mf, chol_cut=1e-5, mo_coeff=None,
):
    """Prepare all integrals for AFQMC.

    Computes:
    1. Cholesky-decomposed ERIs in MO basis
    2. Modified one-body Hamiltonian (h1e_mod)
    3. Nuclear repulsion energy

    Args:
        mf: PySCF mean-field object (RHF or UHF,
            must have run kernel()).
        chol_cut: Cholesky decomposition threshold.
        mo_coeff: MO coefficient matrix to use instead
            of mf.mo_coeff.  Useful for CASSCF trials.

    Returns:
        dict with keys:
            'h1e': shape (nbasis, nbasis)
            'h1e_mod': shape (nbasis, nbasis)
            'chol': shape (naux, nbasis, nbasis)
            'enuc': float
            'nbasis', 'nup', 'ndown': int
            'mo_coeff': jnp.array
    """
    mol = mf.mol
    nbasis = mol.nao_nr()

    # Get MO coefficients
    if mo_coeff is None:
        mo_coeff = np.asarray(mf.mo_coeff)
    else:
        mo_coeff = np.asarray(mo_coeff)

    # Electron counts
    nup = mol.nelec[0]
    ndown = mol.nelec[1]

    # 1. One-body integrals in MO basis
    hcore_ao = np.asarray(mf.get_hcore())
    h1e = mo_coeff.T @ hcore_ao @ mo_coeff

    # 2. Cholesky decomposition in AO basis
    chol_ao = chunked_cholesky(mol, chol_cut=chol_cut)

    # Transform to MO basis
    chol_mo = np.einsum(
        'ab,gbc,cd->gad', mo_coeff.T, chol_ao, mo_coeff,
    )

    # 3. Modified one-body Hamiltonian
    v0 = np.einsum('gij,gkj->ik', chol_mo, chol_mo)
    v0 *= -0.5
    h1e_mod = h1e + v0

    # 4. Nuclear repulsion energy
    enuc = mol.energy_nuc()

    return {
        'h1e': jnp.array(h1e),
        'h1e_mod': jnp.array(h1e_mod),
        'chol': jnp.array(chol_mo),
        'enuc': float(enuc),
        'nbasis': nbasis,
        'nup': nup,
        'ndown': ndown,
        'mo_coeff': jnp.array(mo_coeff),
    }


def half_rotate_cholesky(chol, trial_up, trial_dn):
    """Half-rotate Cholesky vectors with trial wavefunction.

    Precomputes rchol[g,i,q] = sum_p trial[p,i]* L^g_{pq}
    to reduce cost from O(M^2 * naux) to O(nocc * M * naux).

    Args:
        chol: Cholesky vectors,
            shape (naux, nbasis, nbasis).
        trial_up: Trial alpha orbitals,
            shape (nbasis, nup).
        trial_dn: Trial beta orbitals,
            shape (nbasis, ndown).

    Returns:
        rchol_a: shape (naux, nup, nbasis).
        rchol_b: shape (naux, ndown, nbasis).
    """
    rchol_a = jnp.einsum(
        'pi,gpq->giq', trial_up.conj(), chol,
    )
    rchol_b = jnp.einsum(
        'pi,gpq->giq', trial_dn.conj(), chol,
    )
    return rchol_a, rchol_b


def half_rotate_cholesky_multidet(
    chol, trials_up, trials_dn,
):
    """Half-rotate Cholesky vectors for multi-det trial.

    Args:
        chol: Cholesky vectors,
            shape (naux, nbasis, nbasis).
        trials_up: Trial alpha orbitals,
            shape (ndet, nbasis, nup).
        trials_dn: Trial beta orbitals,
            shape (ndet, nbasis, ndown).

    Returns:
        rchols_a: shape (ndet, naux, nup, nbasis).
        rchols_b: shape (ndet, naux, ndown, nbasis).
    """
    _hr = lambda trial: jnp.einsum(
        'pi,gpq->giq', trial.conj(), chol,
    )
    rchols_a = jax.vmap(_hr)(trials_up)
    rchols_b = jax.vmap(_hr)(trials_dn)
    return rchols_a, rchols_b
