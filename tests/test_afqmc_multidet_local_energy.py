"""Multi-det AFQMC estimators against exact CI contractions.

For walkers |phi> = |phi_a phi_b>, the walker's CI vector
c[Ia, Ib] = det(phi_a[occ_Ia]) det(phi_b[occ_Ib]) gives the exact
<psi_T|phi> and <psi_T|H|phi> via the PySCF FCI engine.

Cases:
* LiH/STO-3G full space, CASCI(2e,5o) trial (11 dets, 2 electrons/spin).
* LiH active space (2e,5o) from CASCI integrals, same trial (1 electron/spin).
* H2/STO-3G full space, CASCI(2e,2o) trial (1 electron/spin).

Walker kinds:
* random walkers: overlap and local energy exact to 1e-10.
* copies of the leading determinant D0 (the initial walkers of every run):
  exactly orthogonal to the other determinants, which are masked, so the
  local energy is <D0|H|D0> (finite, to 1e-10) rather than the exact
  <psi_T|H|D0>/<psi_T|D0>. This only affects the first step of a run.
* D0 + eps*noise, re-orthonormalised (small but non-zero overlaps, as
  after the first step): exact to 1e-8 at eps = 1e-6.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import ao2mo, fci, gto, mcscf, scf

from OmegaQMC import extract_casscf_trial, get_afqmc_func, get_afqmc_func_from_integrals
from OmegaQMC.observables.energy_th4 import local_energy_multidet
from OmegaQMC.observables.greens_th4 import greens_function_multidet_overlap_only


def _casci(atom, ncas, nelecas, ncore):
    mf = scf.RHF(gto.M(atom=atom, basis="sto-3g", verbose=0)).run(conv_tol=1e-12)
    mc = mcscf.CASCI(mf, ncas, nelecas, ncore=ncore)
    mc.verbose = 0
    mc.kernel()
    return mf, mc


def _full_ham(mf):
    c = mf.mo_coeff
    norb = c.shape[1]
    return (c.T @ mf.get_hcore() @ c, ao2mo.restore(1, ao2mo.kernel(mf.mol, c), norb),
            mf.mol.energy_nuc(), norb, mf.mol.nelec)


def _exact_chol(eri, norb):
    """Exact Cholesky-like factor of the (pq|rs) supermatrix via eigendecomposition."""
    w, v = np.linalg.eigh(eri.reshape(norb * norb, norb * norb))
    keep = w > 1e-12
    return (v[:, keep] * np.sqrt(w[keep])).T.reshape(-1, norb, norb)


def _addr(norb, nel, occ):
    return fci.cistring.str2addr(norb, nel, int(sum(1 << int(i) for i in occ)))


def _exact(ham, trial, pa, pb):
    """Exact <psi_T|phi> and <psi_T|H|phi>/<psi_T|phi> for each walker."""
    h1, eri, ecore, norb, (na, nb) = ham
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, norb, (na, nb), 0.5)
    occ_a = fci.cistring.gen_occslst(range(norb), na)
    occ_b = fci.cistring.gen_occslst(range(norb), nb)
    ct = np.zeros((len(occ_a), len(occ_b)))
    for c, oa, ob in zip(np.asarray(trial["ci_coeffs"]), np.asarray(trial["occ_up"]),
                         np.asarray(trial["occ_dn"])):
        ct[_addr(norb, na, oa), _addr(norb, nb, ob)] = c
    ovlp, eloc = [], []
    for w in range(pa.shape[0]):
        da = np.array([np.linalg.det(pa[w][list(o)]) for o in occ_a])
        db = np.array([np.linalg.det(pb[w][list(o)]) for o in occ_b])
        cw = np.outer(da, db)
        hc = (fci.direct_spin1.contract_2e(h2, cw.real, norb, (na, nb))
              + 1j * fci.direct_spin1.contract_2e(h2, cw.imag, norb, (na, nb)))
        o = np.sum(ct * cw)
        ovlp.append(o)
        eloc.append(np.sum(ct * hc) / o + ecore)
    return np.array(ovlp), np.array(eloc)


def _d0_energy(ham, trial):
    """<D0|H|D0> of the leading determinant."""
    h1, eri, ecore, norb, (na, nb) = ham
    ci = np.zeros((fci.cistring.num_strings(norb, na), fci.cistring.num_strings(norb, nb)))
    ci[_addr(norb, na, np.asarray(trial["occ_up"])[0]),
       _addr(norb, nb, np.asarray(trial["occ_dn"])[0])] = 1.0
    return float(fci.direct_spin1.energy(h1, eri, ci, norb, (na, nb)) + ecore)


def _lih_full():
    mf, mc = _casci("Li 0 0 0; H 0 0 1.595", 5, (1, 1), 1)
    trial = extract_casscf_trial(mc, coeff_threshold=1e-12)
    drv = get_afqmc_func(mf, chol_cut=1e-10, verbose=False, trial=trial)
    return drv, _full_ham(mf), trial, 11


def _lih_active():
    mf, mc = _casci("Li 0 0 0; H 0 0 1.595", 5, (1, 1), 1)
    h1, ecore = mc.get_h1eff()
    eri = ao2mo.restore(1, mc.get_h2eff(), mc.ncas)
    t = extract_casscf_trial(mc, coeff_threshold=1e-12)
    trial = {"ci_coeffs": t["ci_coeffs"], "occ_up": t["occ_up"][:, 1:] - 1,
             "occ_dn": t["occ_dn"][:, 1:] - 1, "ndet": t["ndet"], "mo_coeff": None}
    drv = get_afqmc_func_from_integrals(h1, _exact_chol(eri, mc.ncas), ecore, mc.nelecas,
                                        trial=trial, verbose=False)
    return drv, (h1, eri, ecore, mc.ncas, mc.nelecas), trial, 11


def _h2_full():
    mf, mc = _casci("H 0 0 0; H 0 0 0.74", 2, (1, 1), 0)
    trial = extract_casscf_trial(mc, coeff_threshold=1e-12)
    drv = get_afqmc_func(mf, chol_cut=1e-10, verbose=False, trial=trial)
    return drv, _full_ham(mf), trial, 2


CASES = {"LiH_full": _lih_full, "LiH_active": _lih_active, "H2_full": _h2_full}
_BUILT = {}


def _case(name):
    if name not in _BUILT:
        _BUILT[name] = CASES[name]()
    return _BUILT[name]


def _omega(drv, pa, pb, det_chunk_size):
    ovlp = greens_function_multidet_overlap_only(
        jnp.array(pa), jnp.array(pb), drv.trials_up, drv.trials_dn, drv.ci_coeffs,
        det_chunk_size=det_chunk_size)
    e_tot, _, _ = local_energy_multidet(
        drv.h1e, drv.chol, jnp.array(pa), jnp.array(pb), drv.trials_up, drv.trials_dn,
        drv.ci_coeffs, drv.enuc, det_chunk_size=det_chunk_size)
    return np.asarray(ovlp), np.asarray(e_tot)


def _d0_walkers(drv, n, eps, rng):
    """D0 + eps*noise, re-orthonormalised per spin (eps = 0: exact copies)."""
    out = []
    for t in (np.asarray(drv.trial_up), np.asarray(drv.trial_dn)):
        noise = rng.normal(size=(n, *t.shape)) + 1j * rng.normal(size=(n, *t.shape))
        phi = t[None] + eps * noise
        out.append(np.array([np.linalg.qr(p)[0] for p in phi]))
    return out


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("det_chunk_size", [3, 5, 11])
def test_random_walkers_exact(case, det_chunk_size):
    drv, ham, trial, ndet = _case(case)
    assert trial["ndet"] == ndet
    nbasis, nup, ndn = drv.nbasis, drv.nup, drv.ndown
    rng = np.random.default_rng(1)
    nw = 5
    pa = rng.normal(size=(nw, nbasis, nup)) + 1j * rng.normal(size=(nw, nbasis, nup))
    pb = rng.normal(size=(nw, nbasis, ndn)) + 1j * rng.normal(size=(nw, nbasis, ndn))
    for k in range(min(nup, ndn)):  # keep the overlap with the trial away from zero
        pa[:, k, k] += 3.0
        pb[:, k, k] += 3.0
    ovlp_ref, eloc_ref = _exact(ham, trial, pa, pb)
    ovlp, e_tot = _omega(drv, pa, pb, det_chunk_size)
    np.testing.assert_allclose(ovlp, ovlp_ref, rtol=1e-10, atol=0)
    assert np.max(np.abs(e_tot - eloc_ref)) < 1e-10


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("det_chunk_size", [3, 5, 11])
def test_d0_copies_masked(case, det_chunk_size):
    drv, ham, trial, _ = _case(case)
    pa, pb = _d0_walkers(drv, 2, 0.0, np.random.default_rng(2))
    ovlp_ref, _ = _exact(ham, trial, pa, pb)
    ovlp, e_tot = _omega(drv, pa, pb, det_chunk_size)
    np.testing.assert_allclose(ovlp, ovlp_ref, rtol=1e-10, atol=0)
    assert np.all(np.isfinite(e_tot))
    assert np.max(np.abs(e_tot - _d0_energy(ham, trial))) < 1e-10


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-8])
def test_perturbed_d0_walkers(case, eps, record_property):
    drv, ham, trial, _ = _case(case)
    pa, pb = _d0_walkers(drv, 4, eps, np.random.default_rng(3))
    _, eloc_ref = _exact(ham, trial, pa, pb)
    _, e_tot = _omega(drv, pa, pb, 5)
    err = float(np.max(np.abs(e_tot - eloc_ref)))
    record_property("max_abs_err", err)
    print(f"{case} eps={eps:.0e}: max|E_L - exact| = {err:.2e}")
    assert np.all(np.isfinite(e_tot))
    if eps == 1e-6:
        assert err < 1e-8
