"""Multi-det AFQMC estimators against exact CI contractions.

LiH/STO-3G with a CASCI(2e,5o) trial (11 determinants). For random
complex walkers |phi> = |phi_a phi_b>, the walker's full-space CI vector
c[Ia, Ib] = det(phi_a[occ_Ia]) det(phi_b[occ_Ib]) gives the exact
<psi_T|phi> and <psi_T|H|phi> via the PySCF FCI engine. The chunked
multi-det overlap and local energy must reproduce them for chunk sizes
that do and do not divide the number of determinants.
"""
import itertools

import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import ao2mo, fci, gto, mcscf, scf

from OmegaQMC import extract_casscf_trial, get_afqmc_func
from OmegaQMC.observables.energy_th4 import local_energy_multidet
from OmegaQMC.observables.greens_th4 import greens_function_multidet_overlap_only


@pytest.fixture(scope="module")
def lih_casci():
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.595", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASCI(mf, 5, (1, 1), ncore=1)
    mc.verbose = 0
    mc.kernel()
    trial = extract_casscf_trial(mc, coeff_threshold=1e-12)
    drv = get_afqmc_func(mf, chol_cut=1e-10, verbose=False, trial=trial)
    return mf, trial, drv


def _exact(mf, trial, pa, pb):
    """Exact <psi_T|phi> and <psi_T|H|phi>/<psi_T|phi> for each walker."""
    norb = mf.mo_coeff.shape[1]
    na, nb = mf.mol.nelec
    c = mf.mo_coeff
    h1 = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, c), norb)
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, norb, (na, nb), 0.5)
    occ_a = fci.cistring.gen_occslst(range(norb), na)
    occ_b = fci.cistring.gen_occslst(range(norb), nb)

    ct = np.zeros((len(occ_a), len(occ_b)))
    for ci, oa, ob in zip(np.asarray(trial["ci_coeffs"]), np.asarray(trial["occ_up"]),
                          np.asarray(trial["occ_dn"])):
        ia = fci.cistring.str2addr(norb, na, int(sum(1 << int(i) for i in oa)))
        ib = fci.cistring.str2addr(norb, nb, int(sum(1 << int(i) for i in ob)))
        ct[ia, ib] = ci

    ovlp, eloc = [], []
    for w in range(pa.shape[0]):
        da = np.array([np.linalg.det(pa[w][list(o)]) for o in occ_a])
        db = np.array([np.linalg.det(pb[w][list(o)]) for o in occ_b])
        cw = np.outer(da, db)
        hc = (fci.direct_spin1.contract_2e(h2, cw.real, norb, (na, nb))
              + 1j * fci.direct_spin1.contract_2e(h2, cw.imag, norb, (na, nb)))
        o = np.sum(ct * cw)
        ovlp.append(o)
        eloc.append(np.sum(ct * hc) / o + mf.mol.energy_nuc())
    return np.array(ovlp), np.array(eloc)


@pytest.mark.parametrize("det_chunk_size", [3, 5, 11])
def test_multidet_overlap_and_local_energy_exact(lih_casci, det_chunk_size):
    mf, trial, drv = lih_casci
    assert trial["ndet"] == 11
    nbasis, nup, ndn = drv.nbasis, drv.nup, drv.ndown
    rng = np.random.default_rng(1)
    nw = 5
    pa = rng.normal(size=(nw, nbasis, nup)) + 1j * rng.normal(size=(nw, nbasis, nup))
    pb = rng.normal(size=(nw, nbasis, ndn)) + 1j * rng.normal(size=(nw, nbasis, ndn))
    for k in range(min(nup, ndn)):  # keep the overlap with the trial away from zero
        pa[:, k, k] += 3.0
        pb[:, k, k] += 3.0

    ovlp_ref, eloc_ref = _exact(mf, trial, pa, pb)

    ovlp = greens_function_multidet_overlap_only(
        jnp.array(pa), jnp.array(pb), drv.trials_up, drv.trials_dn, drv.ci_coeffs,
        det_chunk_size=det_chunk_size)
    np.testing.assert_allclose(np.asarray(ovlp), ovlp_ref, rtol=1e-10, atol=0)

    e_tot, _, _ = local_energy_multidet(
        drv.h1e, drv.chol, jnp.array(pa), jnp.array(pb), drv.trials_up, drv.trials_dn,
        drv.ci_coeffs, drv.enuc, det_chunk_size=det_chunk_size)
    # chol_cut=1e-10 → the Cholesky ERIs are exact to ~1e-10 Ha
    assert np.max(np.abs(np.asarray(e_tot) - eloc_ref)) < 1e-10
