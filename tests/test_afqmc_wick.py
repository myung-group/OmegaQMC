"""Generalized Wick's theorem vs the explicit multi-det kernels.

Li2/STO-3G with CAS trials: (4e,6o) above one core orbital, so that
core, active and external orbitals all occur, and (6e,7o), which has
triple excitations in each spin.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from pyscf import gto, scf, mcscf

from OmegaQMC import get_afqmc_func
from OmegaQMC.psi.gto import extract_casscf_trial
from OmegaQMC.observables.greens_th4 import greens_function_multidet
from OmegaQMC.observables.energy_th4 import local_energy_multidet
from OmegaQMC.observables.wick import (
    cofactors, wick_force_bias, wick_local_energy, wick_overlap,
)


@pytest.mark.parametrize('k', [1, 2, 3, 4])
def test_cofactors(k):
    rng = np.random.default_rng(k)
    A = jnp.asarray(rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k)))
    d, C1, C2 = cofactors(A[None])
    det = lambda M: jnp.linalg.det(M)
    np.testing.assert_allclose(d[0], det(A), atol=1e-12)
    np.testing.assert_allclose(
        C1[0], jax.jacfwd(det, holomorphic=True)(A), atol=1e-12)
    np.testing.assert_allclose(
        C2[0], jax.hessian(det, holomorphic=True)(A), atol=1e-11)
    # Exactly singular: the zero matrix
    d, C1, C2 = cofactors(jnp.zeros((1, k, k), dtype=complex))
    assert np.all(np.isfinite(C1)) and np.all(np.isfinite(C2))
    np.testing.assert_allclose(C1[0], np.eye(k) if k == 1 else 0.0)


def _setup(ncas=6, nelecas=4, chol_cut=1e-8):
    mol = gto.M(atom='Li 0 0 0; Li 0 0 2.67', basis='sto-3g',
                verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.kernel()
    rng = np.random.default_rng(0)
    exact = extract_casscf_trial(mc, coeff_threshold=0.0)

    # Random complex coefficients, shuffled determinants and permuted
    # columns (which define each determinant's sign) beyond the first
    ndet = exact['ndet']
    keep = np.sort(rng.choice(np.arange(1, ndet), ndet // 2,
                              replace=False))
    order = np.concatenate([[0], rng.permutation(keep)])
    occ_up = np.asarray(exact['occ_up'])[order]
    occ_dn = np.asarray(exact['occ_dn'])[order]
    for occ in (occ_up, occ_dn):
        for i in range(1, len(occ)):
            occ[i] = rng.permutation(occ[i])
    ci = rng.normal(size=len(order)) + 1j * rng.normal(size=len(order))
    ci[0] = 3.0
    trial = {'ci_coeffs': jnp.asarray(ci), 'occ_up': jnp.asarray(occ_up),
             'occ_dn': jnp.asarray(occ_dn), 'ndet': len(order),
             'mo_coeff': exact['mo_coeff']}
    drv = get_afqmc_func(mf, chol_cut=chol_cut, trial=trial,
                         verbose=False, multidet_method='wick')
    return mf, mc, exact, drv


def _random_walkers(drv, nw, rng):
    shape = (nw, drv.nbasis)
    pa = rng.normal(size=shape + (drv.nup,)) \
        + 1j * rng.normal(size=shape + (drv.nup,))
    pb = rng.normal(size=shape + (drv.ndown,)) \
        + 1j * rng.normal(size=shape + (drv.ndown,))
    return jnp.asarray(pa), jnp.asarray(pb)


@pytest.mark.parametrize('ncas, nelecas', [(6, 4), (7, 6)])
def test_wick_vs_explicit(ncas, nelecas):
    _, _, _, drv = _setup(ncas, nelecas)
    levels = [h.shape[1] for h in drv.wick_trial.up.holes]
    assert levels == list(range(1, nelecas // 2 + 1))
    pa, pb = _random_walkers(drv, 7, np.random.default_rng(1))
    args = (drv.trials_up, drv.trials_dn, drv.ci_coeffs)

    Ga, Gb, ovlp = greens_function_multidet(pa, pb, *args)
    vbias_ref = (jnp.einsum('gpq,wqp->gw', drv.chol, Ga)
                 + jnp.einsum('gpq,wqp->gw', drv.chol, Gb))
    e_ref = local_energy_multidet(drv.h1e, drv.chol, pa, pb, *args,
                                  drv.enuc)

    for batch in (None, 3):
        wt = drv.wick_trial
        np.testing.assert_allclose(
            wick_overlap(pa, pb, wt, batch_size=batch), ovlp,
            rtol=1e-10)
        vbias, ovlp_w = wick_force_bias(pa, pb, wt, batch_size=batch)
        np.testing.assert_allclose(ovlp_w, ovlp, rtol=1e-10)
        np.testing.assert_allclose(vbias, vbias_ref, atol=1e-9)
        e_w = wick_local_energy(pa, pb, wt, drv.enuc,
                                 batch_size=batch)
        for x, y in zip(e_w, e_ref):
            np.testing.assert_allclose(x, y, atol=1e-9)


def test_wick_walker_on_reference():
    """Zero-variance check where the explicit kernels drop dets."""
    mf, mc, exact, _ = _setup()
    drv = get_afqmc_func(mf, chol_cut=1e-10, trial=exact, verbose=False,
                         multidet_method='wick')
    pa, pb = drv.trials_up[:1] + 0j, drv.trials_dn[:1] + 0j
    e_tot, _, _ = wick_local_energy(pa, pb, drv.wick_trial, drv.enuc)
    np.testing.assert_allclose(e_tot.real, mc.e_tot, atol=1e-8)
    np.testing.assert_allclose(e_tot.imag, 0.0, atol=1e-10)

    # ... and the force bias agrees with the explicit kernels at a
    # slightly displaced walker
    eps = 1e-7 * np.random.default_rng(2).normal(size=pa.shape)
    Ga, Gb, _ = greens_function_multidet(
        pa + eps, pb + eps, drv.trials_up, drv.trials_dn,
        drv.ci_coeffs)
    vbias_ref = (jnp.einsum('gpq,wqp->gw', drv.chol, Ga)
                 + jnp.einsum('gpq,wqp->gw', drv.chol, Gb))
    vbias, _ = wick_force_bias(pa, pb, drv.wick_trial)
    np.testing.assert_allclose(vbias, vbias_ref, atol=1e-5)


def test_wick_afqmc_exact_trial():
    """AFQMC with the exact CAS trial has zero variance in the CAS."""
    mol = gto.M(atom='Li 0 0 0; Li 0 0 2.67', basis='sto-3g',
                verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, mol.nao, mol.nelectron)
    mc.kernel()
    drv = get_afqmc_func(mf, chol_cut=1e-10,
                         trial=extract_casscf_trial(mc, 1e-12),
                         verbose=False, multidet_method='wick')
    res = drv(rng_key=jax.random.key(3), num_walkers=20, num_blocks=3,
              num_steps_per_block=10, num_blocks_equil=0, fname_log='')
    np.testing.assert_allclose(res['energy_blocks'], mc.e_tot,
                               atol=1e-7)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
