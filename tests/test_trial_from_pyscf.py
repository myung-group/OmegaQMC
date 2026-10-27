"""trial_from_pyscf: PySCF solvers -> multi-determinant trial dict.

Each trial is mapped back to a full-space FCI vector (sorting every
occupation list and tracking the permutation sign), whose energy must
equal the solver's own energy; this checks all determinant signs.
CISD vectors are also compared with PySCF's ``to_fcivec`` element by
element.  The drivers are checked end to end on UHF-based trials:
AFQMC's Wick's and explicit kernels (beta orbitals rotated into the
alpha MO basis) and the GTO VMC wavefunction (alpha and beta MOs side
by side).  Li2/STO-3G (10 MOs) keeps every FCI space small; the
doublet Li2+ gives UHF orbitals that differ between spins.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from pyscf import ao2mo, cc, ci, fci, gto, mcscf, scf
from pyscf.fci import cistring, direct_uhf

from OmegaQMC import get_afqmc_func
from OmegaQMC.psi.gto import _PsiGTO, extract_casscf_trial
from OmegaQMC.psi.trial import trial_from_pyscf
from OmegaQMC.observables.greens_th4 import greens_function_multidet
from OmegaQMC.observables.energy_th4 import local_energy_multidet
from OmegaQMC.observables.wick import (
    wick_force_bias, wick_local_energy, wick_overlap,
)

ATOM = 'Li 0 0 0; Li 0 0 2.67'


def _mol(charge=0, spin=0):
    return gto.M(atom=ATOM, basis='sto-3g', charge=charge, spin=spin,
                 verbose=0)


@pytest.fixture(scope='module')
def rhf():
    return scf.RHF(_mol()).run()


@pytest.fixture(scope='module')
def uhf():
    mf = scf.UHF(_mol(charge=1, spin=1)).run()
    assert abs(mf.mo_coeff[0] - mf.mo_coeff[1]).max() > 0.1
    return mf


def _addr_sign(occ, norb, nel):
    occ = np.asarray(occ)
    inv = sum(int(np.sum(occ[k] > occ[k + 1:])) for k in range(nel))
    s = sum(1 << int(o) for o in occ)
    return cistring.str2addr(norb, nel, s), (-1) ** inv


def _full_fcivec(trial, norb, nelec):
    na = cistring.num_strings(norb, nelec[0])
    nb = cistring.num_strings(norb, nelec[1])
    vec = np.zeros((na, nb))
    seen = np.zeros((na, nb), dtype=bool)
    for c, up, dn in zip(np.asarray(trial['ci_coeffs']),
                         np.asarray(trial['occ_up']),
                         np.asarray(trial['occ_dn'])):
        a, sa = _addr_sign(up, norb, nelec[0])
        b, sb = _addr_sign(dn, norb, nelec[1])
        assert not seen[a, b], "duplicate determinant"
        seen[a, b] = True
        vec[a, b] = c * sa * sb
    return vec


def _energy(mf, trial):
    """Variational energy of the trial, restricted or UHF-based."""
    mo = np.asarray(trial['mo_coeff'])
    norb = mo.shape[-1]
    nelec = (len(trial['occ_up'][0]), len(trial['occ_dn'][0]))
    v = _full_fcivec(trial, norb, nelec)
    hcore = mf.get_hcore()
    mol = mf.mol
    if mo.ndim == 2:
        h1 = mo.T @ hcore @ mo
        e = fci.direct_spin1.energy(h1, ao2mo.full(mol, mo), v, norb,
                                    nelec)
    else:
        ca, cb = mo
        h1 = (ca.T @ hcore @ ca, cb.T @ hcore @ cb)
        eri = (ao2mo.full(mol, ca), ao2mo.general(mol, (ca, ca, cb, cb)),
               ao2mo.full(mol, cb))
        e = direct_uhf.energy(h1, eri, v, norb, nelec)
    return e / np.vdot(v, v) + mf.energy_nuc()


def _check_sorted(trial):
    c = np.abs(np.asarray(trial['ci_coeffs']))
    assert np.all(np.diff(c) <= 0)
    assert trial['ndet'] == len(c)


# -------------------------------------------------------------------
# Restricted orbitals
# -------------------------------------------------------------------

@pytest.mark.parametrize('kind', ['casci', 'casscf'])
def test_cas(rhf, kind):
    cls = mcscf.CASCI if kind == 'casci' else mcscf.CASSCF
    mc = cls(rhf, 6, 4).run()
    trial = trial_from_pyscf(mc, coeff_threshold=0.0)
    _check_sorted(trial)
    assert trial['mo_coeff'].ndim == 2
    assert trial['ci_weight'] == pytest.approx(1.0, abs=1e-12)
    assert _energy(rhf, trial) == pytest.approx(mc.e_tot, abs=1e-9)

    ref = extract_casscf_trial(mc, coeff_threshold=0.0)
    for key in ('ci_coeffs', 'occ_up', 'occ_dn'):
        np.testing.assert_array_equal(ref[key], trial[key])


def test_threshold_and_cap(rhf):
    mc = mcscf.CASCI(rhf, 6, 4).run()
    full = trial_from_pyscf(mc, coeff_threshold=0.0)
    cut = trial_from_pyscf(mc, coeff_threshold=1e-2)
    assert 1 < cut['ndet'] < full['ndet']
    assert np.all(np.abs(np.asarray(cut['ci_coeffs'])) > 1e-2)
    assert 0.9 < cut['ci_weight'] < 1.0
    top = trial_from_pyscf(mc, coeff_threshold=0.0, max_ndet=5)
    assert top['ndet'] == 5
    for key in ('ci_coeffs', 'occ_up', 'occ_dn'):
        np.testing.assert_array_equal(top[key], full[key][:5])
    with pytest.raises(ValueError):
        trial_from_pyscf(mc, coeff_threshold=10.0)


def test_state_average(rhf):
    mc = mcscf.CASSCF(rhf, 6, 4).state_average_([0.5, 0.5]).run()
    for k in range(2):
        trial = trial_from_pyscf(mc, coeff_threshold=0.0, state=k)
        assert _energy(rhf, trial) == pytest.approx(mc.e_states[k],
                                                    abs=1e-9)
    with pytest.raises(ValueError, match='state'):
        trial_from_pyscf(mc)


def test_selected_ci(rhf):
    mc = mcscf.CASCI(rhf, 8, 6)
    mc.fcisolver = fci.SCI(rhf.mol)
    mc.kernel()
    assert hasattr(mc.ci, '_strs')
    trial = trial_from_pyscf(mc, coeff_threshold=0.0)
    _check_sorted(trial)
    assert _energy(rhf, trial) == pytest.approx(mc.e_tot, abs=1e-8)


@pytest.mark.parametrize('open_shell', [False, True])
def test_fci_solver(open_shell):
    if open_shell:
        mf = scf.ROHF(_mol(charge=1, spin=1)).run()
    else:
        mf = scf.RHF(_mol()).run()
    cis = fci.FCI(mf)
    cis.nroots = 2
    e, c = cis.kernel()
    for k in range(2):
        trial = trial_from_pyscf(cis, coeff_threshold=0.0, state=k,
                                 mo_coeff=mf.mo_coeff)
        assert _energy(mf, trial) == pytest.approx(e[k], abs=1e-9)
    # Bare CI vector
    bare = trial_from_pyscf(c[0], coeff_threshold=0.0,
                            mo_coeff=mf.mo_coeff, nelec=cis.nelec)
    assert _energy(mf, bare) == pytest.approx(e[0], abs=1e-9)
    with pytest.raises(ValueError, match='mo_coeff'):
        trial_from_pyscf(cis, state=0)


def test_bare_vector_with_core(rhf):
    """A CASCI vector passed bare, with ncore and norb."""
    mc = mcscf.CASCI(rhf, 6, 4).run()
    trial = trial_from_pyscf(mc.ci, coeff_threshold=0.0,
                             mo_coeff=mc.mo_coeff, nelec=4,
                             ncore=mc.ncore, norb=6)
    assert _energy(rhf, trial) == pytest.approx(mc.e_tot, abs=1e-9)


@pytest.mark.parametrize('frozen', [None, 1, [1], [0, 9]])
def test_cisd(rhf, frozen):
    myci = ci.CISD(rhf, frozen=frozen).run()
    trial = trial_from_pyscf(myci, coeff_threshold=0.0)
    _check_sorted(trial)
    assert trial['ci_weight'] == pytest.approx(1.0, abs=1e-12)
    norb = rhf.mo_coeff.shape[1]
    nelec = rhf.mol.nelec
    ref = ci.cisd.to_fcivec(myci.ci, norb, nelec, frozen=frozen)
    np.testing.assert_allclose(_full_fcivec(trial, norb, nelec), ref,
                               atol=1e-14)
    assert _energy(rhf, trial) == pytest.approx(myci.e_tot, abs=1e-9)

    cut = trial_from_pyscf(myci, coeff_threshold=1e-3)
    assert np.all(np.abs(np.asarray(cut['ci_coeffs'])) > 1e-3)
    assert cut['ci_weight'] < 1.0


# -------------------------------------------------------------------
# UHF-based expansions
# -------------------------------------------------------------------

@pytest.mark.parametrize('kind', ['ucasci', 'ucasscf'])
def test_ucas(uhf, kind):
    cls = mcscf.UCASCI if kind == 'ucasci' else mcscf.UCASSCF
    mc = cls(uhf, 6, (2, 1)).run()
    trial = trial_from_pyscf(mc, coeff_threshold=0.0)
    _check_sorted(trial)
    assert trial['mo_coeff'].shape == (2, 10, 10)
    assert _energy(uhf, trial) == pytest.approx(mc.e_tot, abs=1e-9)


def test_uhf_fci_solver(uhf):
    cis = fci.FCI(uhf)
    e, c = cis.kernel()
    trial = trial_from_pyscf(cis, coeff_threshold=0.0,
                             mo_coeff=uhf.mo_coeff)
    assert trial['mo_coeff'].ndim == 3
    assert _energy(uhf, trial) == pytest.approx(e, abs=1e-9)
    bare = trial_from_pyscf(c, coeff_threshold=0.0,
                            mo_coeff=uhf.mo_coeff, nelec=cis.nelec)
    assert _energy(uhf, bare) == pytest.approx(e, abs=1e-9)


@pytest.mark.parametrize('ref, frozen', [
    ('uhf', None), ('uhf', 1), ('uhf', [[0], [0, 9]]),
    ('rohf', None), ('rohf', 1),
])
def test_ucisd(ref, frozen):
    mol = _mol(charge=1, spin=1)
    mf = (scf.UHF if ref == 'uhf' else scf.ROHF)(mol).run()
    myci = ci.CISD(mf, frozen=frozen).run()
    assert isinstance(myci, ci.ucisd.UCISD)
    trial = trial_from_pyscf(myci, coeff_threshold=0.0)
    _check_sorted(trial)
    assert trial['mo_coeff'].ndim == 3
    assert trial['ci_weight'] == pytest.approx(1.0, abs=1e-12)
    norb, nelec = 10, mol.nelec
    vec = ci.ucisd.to_fcivec(myci.ci, norb, nelec, frozen=frozen)
    np.testing.assert_allclose(_full_fcivec(trial, norb, nelec), vec,
                               atol=1e-14)
    assert _energy(mf, trial) == pytest.approx(myci.e_tot, abs=1e-9)


def test_rejects(rhf):
    with pytest.raises(NotImplementedError, match='GCISD'):
        trial_from_pyscf(ci.GCISD(scf.GHF(rhf.mol).run()))
    with pytest.raises(TypeError, match='unsupported'):
        trial_from_pyscf(cc.CCSD(rhf))
    mc = mcscf.CASCI(rhf, 6, 4).run()
    with pytest.raises(TypeError):
        trial_from_pyscf(mc, mo_coeff=rhf.mo_coeff)
    with pytest.raises(ValueError):
        trial_from_pyscf(mc, max_ndet=0)
    with pytest.raises(ValueError, match='kernel'):
        trial_from_pyscf(ci.CISD(rhf))
    with pytest.raises(ValueError, match='nelec'):
        trial_from_pyscf(mc.ci, mo_coeff=rhf.mo_coeff)
    with pytest.raises(ValueError, match='shape'):
        trial_from_pyscf(mc.ci, mo_coeff=np.zeros((3, 10, 10)),
                         nelec=4)


# -------------------------------------------------------------------
# Drivers
# -------------------------------------------------------------------

def _solver(rhf, uhf, source):
    """(mf, trial, energy) of a solver whose state is an eigenvector
    of H projected onto a space containing its leading determinant."""
    if source == 'cisd':
        obj = ci.CISD(rhf, frozen=1).run()
        return rhf, trial_from_pyscf(obj, coeff_threshold=0.0), obj.e_tot
    if source == 'sa-casscf':
        obj = mcscf.CASSCF(rhf, 6, 4).state_average_([0.5, 0.5]).run()
        trial = trial_from_pyscf(obj, coeff_threshold=0.0, state=1)
        return rhf, trial, obj.e_states[1]
    if source == 'ucisd':
        obj = ci.CISD(uhf, frozen=1).run()
        return uhf, trial_from_pyscf(obj, coeff_threshold=0.0), obj.e_tot
    obj = mcscf.UCASSCF(uhf, 6, (2, 1)).run()
    return uhf, trial_from_pyscf(obj, coeff_threshold=0.0), obj.e_tot


@pytest.mark.parametrize('source',
                         ['cisd', 'sa-casscf', 'ucisd', 'ucasscf'])
def test_afqmc_wick_reference(rhf, uhf, source):
    """Mixed energy at the leading determinant equals the solver's:
    <Psi|H|D0> / <Psi|D0> = E for an eigenvector of PHP with D0 in P.
    """
    mf, trial, e_ref = _solver(rhf, uhf, source)
    drv = get_afqmc_func(mf, chol_cut=1e-10, trial=trial,
                         verbose=False, multidet_method='wick')
    pa = drv.trials_up[:1] + 0j
    pb = drv.trials_dn[:1] + 0j
    e_tot, _, _ = wick_local_energy(pa, pb, drv.wick_trial, drv.enuc)
    np.testing.assert_allclose(jnp.real(e_tot), e_ref, atol=1e-7)
    np.testing.assert_allclose(jnp.imag(e_tot), 0.0, atol=1e-9)


@pytest.mark.parametrize('source', ['ucisd', 'ucasscf'])
def test_afqmc_uhf_wick_vs_explicit(rhf, uhf, source):
    """Rotated beta orbitals: Wick's and explicit kernels agree."""
    mf, trial, _ = _solver(rhf, uhf, source)
    trial = dict(trial, ci_coeffs=trial['ci_coeffs'][:40],
                 occ_up=trial['occ_up'][:40],
                 occ_dn=trial['occ_dn'][:40], ndet=40)
    drv = get_afqmc_func(mf, chol_cut=1e-8, trial=trial, verbose=False,
                         multidet_method='wick')
    assert drv.wick_trial.dn.orbs is not None
    rng = np.random.default_rng(5)
    shape = (6, drv.nbasis)
    pa = jnp.asarray(rng.normal(size=shape + (drv.nup,))
                     + 1j * rng.normal(size=shape + (drv.nup,)))
    pb = jnp.asarray(rng.normal(size=shape + (drv.ndown,))
                     + 1j * rng.normal(size=shape + (drv.ndown,)))
    args = (drv.trials_up, drv.trials_dn, drv.ci_coeffs)
    Ga, Gb, ovlp = greens_function_multidet(pa, pb, *args)
    vbias_ref = (jnp.einsum('gpq,wqp->gw', drv.chol, Ga)
                 + jnp.einsum('gpq,wqp->gw', drv.chol, Gb))
    e_ref = local_energy_multidet(drv.h1e, drv.chol, pa, pb, *args,
                                  drv.enuc)
    wt = drv.wick_trial
    np.testing.assert_allclose(wick_overlap(pa, pb, wt), ovlp,
                               rtol=1e-9)
    vbias, _ = wick_force_bias(pa, pb, wt)
    np.testing.assert_allclose(vbias, vbias_ref, atol=1e-8)
    for x, y in zip(wick_local_energy(pa, pb, wt, drv.enuc), e_ref):
        np.testing.assert_allclose(x, y, atol=1e-8)


def test_afqmc_uhf_run(uhf):
    """Short AFQMC run on a UHF-based trial, both kernels agreeing."""
    obj = mcscf.UCASSCF(uhf, 4, (2, 1)).run()
    trial = trial_from_pyscf(obj, coeff_threshold=1e-6)
    res = []
    for method in ('explicit', 'wick'):
        drv = get_afqmc_func(uhf, chol_cut=1e-8, trial=trial,
                             verbose=False, multidet_method=method)
        res.append(drv(rng_key=jax.random.key(1), num_walkers=8,
                       num_blocks=2, num_steps_per_block=5,
                       num_blocks_equil=0, fname_log=''))
    np.testing.assert_allclose(res[0]['energy_blocks'],
                               res[1]['energy_blocks'], atol=1e-8)
    assert abs(res[1]['energy_blocks'][0] - obj.e_tot) < 0.05


def test_vmc_uhf_wavefunction(uhf):
    """GTO VMC log|psi| and KE of a UHF-based trial, against a direct
    evaluation with separate alpha and beta MOs."""
    mc = mcscf.UCASSCF(uhf, 6, (2, 1)).run()
    trial = trial_from_pyscf(mc, coeff_threshold=1e-3)
    psi = _PsiGTO(uhf, params_cusp=None, trial=trial)
    mol = uhf.mol
    ca, cb = trial['mo_coeff']
    nuc = jnp.asarray(mol.atom_coords())
    rng = np.random.default_rng(3)
    for _ in range(3):
        elec = rng.normal(size=(mol.nelectron, 3)) + [0, 0, 2.5]
        # Electrons are interleaved: alpha at even, beta at odd rows
        ao_a = mol.eval_gto('GTOval_sph', elec[::2])
        ao_b = mol.eval_gto('GTOval_sph', elec[1::2])
        val = sum(
            c * np.linalg.det(ao_a @ ca[:, ua])
            * np.linalg.det(ao_b @ cb[:, ub])
            for c, ua, ub in zip(np.asarray(trial['ci_coeffs']),
                                 np.asarray(trial['occ_up']),
                                 np.asarray(trial['occ_dn'])))
        log_psi = psi.log_trial_wavefunction(jnp.asarray(elec), nuc, {})
        assert float(log_psi) == pytest.approx(np.log(abs(val)),
                                               abs=1e-9)
        ke = psi.local_energy_ke(jnp.asarray(elec), nuc, {})
        ke_h = psi._local_energy_ke_hessian(jnp.asarray(elec), nuc, {})
        assert float(ke) == pytest.approx(float(ke_h), abs=1e-7)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
