"""trial_from_pyscf: PySCF solvers -> multi-determinant trial dict.

Each trial is mapped back to a full-space FCI vector (sorting every
occupation list and tracking the permutation sign), whose energy must
equal the solver's own energy; this checks all determinant signs.
CISD vectors are also compared with ``pyscf.ci.cisd.to_fcivec``
element by element, and an AFQMC run on the Wick's trial checks the
dict end to end.  Li2/STO-3G (10 MOs) keeps every FCI space small.
"""

import numpy as np
import jax.numpy as jnp
import pytest
from pyscf import ao2mo, cc, ci, fci, gto, mcscf, scf
from pyscf.fci import cistring

from OmegaQMC import get_afqmc_func
from OmegaQMC.psi.gto import extract_casscf_trial
from OmegaQMC.psi.trial import trial_from_pyscf
from OmegaQMC.observables.wick import wick_local_energy

ATOM = 'Li 0 0 0; Li 0 0 2.67'


def _mol(charge=0, spin=0):
    return gto.M(atom=ATOM, basis='sto-3g', charge=charge, spin=spin,
                 verbose=0)


@pytest.fixture(scope='module')
def rhf():
    return scf.RHF(_mol()).run()


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
    C = trial['mo_coeff']
    norb = C.shape[1]
    nelec = (len(trial['occ_up'][0]), len(trial['occ_dn'][0]))
    h1 = C.T @ mf.get_hcore() @ C
    eri = ao2mo.full(mf.mol, C)
    v = _full_fcivec(trial, norb, nelec)
    e = fci.direct_spin1.energy(h1, eri, v, norb, nelec)
    return e / np.vdot(v, v) + mf.energy_nuc()


def _check_sorted(trial):
    c = np.abs(np.asarray(trial['ci_coeffs']))
    assert np.all(np.diff(c) <= 0)
    assert trial['ndet'] == len(c)


@pytest.mark.parametrize('kind', ['casci', 'casscf'])
def test_cas(rhf, kind):
    cls = mcscf.CASCI if kind == 'casci' else mcscf.CASSCF
    mc = cls(rhf, 6, 4).run()
    trial = trial_from_pyscf(mc, coeff_threshold=0.0)
    _check_sorted(trial)
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


def test_rejects(rhf):
    mol = rhf.mol
    uhf = scf.UHF(mol).run()
    for obj in (mcscf.UCASSCF(uhf, 6, 4), mcscf.UCASCI(uhf, 6, 4),
                ci.CISD(uhf), fci.FCI(uhf)):
        with pytest.raises(NotImplementedError, match='UHF'):
            trial_from_pyscf(obj)
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


@pytest.mark.parametrize('source', ['cisd', 'sa-casscf'])
def test_afqmc_wick_reference(rhf, source):
    """Mixed energy at the leading determinant equals the solver's.

    Both solvers' states are eigenvectors of H projected onto a space
    that contains the leading determinant D0, so
    <Psi|H|D0> / <Psi|D0> is their energy.
    """
    if source == 'cisd':
        obj = ci.CISD(rhf, frozen=1).run()
        trial = trial_from_pyscf(obj, coeff_threshold=0.0)
        e_ref = obj.e_tot
    else:
        obj = mcscf.CASSCF(rhf, 6, 4).state_average_([0.5, 0.5]).run()
        trial = trial_from_pyscf(obj, coeff_threshold=0.0, state=1)
        e_ref = obj.e_states[1]
    drv = get_afqmc_func(rhf, chol_cut=1e-10, trial=trial,
                         verbose=False, multidet_method='wick')
    pa = drv.trials_up[:1] + 0j
    pb = drv.trials_dn[:1] + 0j
    e_tot, _, _ = wick_local_energy(pa, pb, drv.wick_trial, drv.enuc)
    np.testing.assert_allclose(jnp.real(e_tot), e_ref, atol=1e-7)
    np.testing.assert_allclose(jnp.imag(e_tot), 0.0, atol=1e-9)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
