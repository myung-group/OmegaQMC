"""Open-shell QED-CCSD / QED-RPA / QED-FCI / QED-CASCI.

Lightweight parity + smoke tests for the spin-unrestricted (QED-UHF)
paths of the four cavity-QED correlated methods. The emphasis (per the
chosen validation depth) is closed-shell *parity* — the generalized
code must reproduce the established QED-RHF-based results — plus one
open-shell smoke test per method.

Parity checks
-------------
* QED-CCSD / QED-RPA: feeding a closed-shell QED-UHF reference dict
  (Cα = Cβ) must reproduce the QED-RHF-dict result. The residual / A-B
  equations are spin-orbital and reference-agnostic, so this validates
  the new spin-orbital build.
* QED-FCI / QED-CASCI: λ = 0 reduces to the bare method; the open-shell
  full-active-space QED-(U)CASCI equals open-shell QED-FCI; open-shell
  QED-CCSD at λ = 0 equals pyscf UCCSD.

Smoke tests
-----------
A doublet (OH / STO-3G) in a cavity runs, converges and returns a
sensible (negative) QED correlation energy for every method.
"""

import numpy as np
import pytest
from pyscf import gto, scf, cc

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_uhf import run_qed_uhf
from OmegaQMC.addons.qed_ccsd import run_qed_ccsd
from OmegaQMC.addons.qed_rpa import run_qed_rpa
from OmegaQMC.addons.qed_fci import run_qed_fci
from OmegaQMC.addons.qed_casci import run_qed_casci


OMEGA = 3.0 / 27.211386245988
LAM = (0.0, 0.0, 0.1)


def _h2():
    return gto.M(atom='H 0 0 0; H 0 0 0.74', basis='STO-3G',
                 unit='Angstrom', verbose=0)


def _h2o():
    return gto.M(atom='O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24',
                 basis='STO-3G', unit='Angstrom', verbose=0)


def _oh_doublet():
    return gto.M(atom='O 0 0 0; H 0 0 0.97', basis='STO-3G',
                 spin=1, unit='Angstrom', verbose=0)


# --- QED-CCSD --------------------------------------------------------------

def test_ccsd_uhf_reference_matches_rhf_closed_shell():
    """A closed-shell QED-UHF reference must give the same QED-CCSD as the
    QED-RHF reference (the spin-orbital build reduces to the restricted one)."""
    mol = _h2()
    rhf = run_qed_hf(mol, OMEGA, LAM)
    uhf = run_qed_uhf(mol, OMEGA, LAM)
    # Default tolerances: the amplitude-step-norm criterion (tol_amp)
    # guarantees both reference paths (interleaved vs occ-first spin
    # ordering) leave the DIIS plateau where |dE| alone would stop
    # ~1e-7 short of the fixed point — this doubles as a regression
    # test for the dual energy/amplitude convergence check.
    flags = dict(do_t1_01=True, do_t2_11=True, do_t2_21=True, verbose=False)
    rr = run_qed_ccsd(rhf, **flags)
    ru = run_qed_ccsd(uhf, **flags)
    assert rr['converged'] and ru['converged']
    assert ru['E_qed_ccsd_corr'] == pytest.approx(rr['E_qed_ccsd_corr'],
                                                  abs=1e-8)


def test_ccsd_open_shell_zero_coupling_matches_uccsd():
    """Open-shell QED-CCSD at λ=0 must equal pyscf UCCSD."""
    mol = _oh_doublet()
    mf = scf.UHF(mol)
    mf.kernel()
    e_uccsd = cc.UCCSD(mf).kernel()[0]

    uhf0 = run_qed_uhf(mol, OMEGA, (0.0, 0.0, 0.0))
    e0 = run_qed_ccsd(uhf0, verbose=False)['E_qed_ccsd_corr']
    assert e0 == pytest.approx(e_uccsd, abs=1e-7)


def test_ccsd_open_shell_cavity_runs():
    """Open-shell QED-CCSD-21 with coupling converges to a negative corr."""
    mol = _oh_doublet()
    uhf = run_qed_uhf(mol, OMEGA, LAM)
    res = run_qed_ccsd(uhf, do_t1_01=True, do_t2_11=True, do_t2_21=True,
                       verbose=False)
    assert res['E_qed_ccsd_corr'] < 0.0
    assert res['E_qed_hf'] == pytest.approx(uhf['E_qed_uhf'])


# --- QED-RPA ---------------------------------------------------------------

@pytest.mark.parametrize('direct', [True, False])
def test_rpa_uhf_reference_matches_rhf_closed_shell(direct):
    """A closed-shell QED-UHF reference must give the same QED-(d)RPA
    correlation energy as the QED-RHF reference."""
    mol = _h2o()
    omega = 0.4
    rhf = run_qed_hf(mol, omega, (0.0, 0.0, 0.05))
    uhf = run_qed_uhf(mol, omega, (0.0, 0.0, 0.05))
    er = run_qed_rpa(rhf, direct=direct, verbose=False)['E_qed_rpa_corr']
    eu = run_qed_rpa(uhf, direct=direct, verbose=False)['E_qed_rpa_corr']
    assert eu == pytest.approx(er, abs=1e-7)


def test_rpa_open_shell_runs():
    """Open-shell QED-dRPA runs and returns a negative correlation energy."""
    mol = _oh_doublet()
    uhf = run_qed_uhf(mol, 0.4, (0.0, 0.0, 0.05))
    res = run_qed_rpa(uhf, direct=True, verbose=False)
    assert res['E_qed_rpa_corr'] < 0.0


# --- QED-FCI ---------------------------------------------------------------

def test_fci_open_shell_zero_coupling_reduces_to_bare():
    """Open-shell QED-FCI at λ=0 reduces to the bare FCI and bare UHF."""
    mol = _oh_doublet()
    mf = scf.UHF(mol)
    e_uhf = mf.kernel()
    r = run_qed_fci(mf, omega=OMEGA, coupling_vec=(0.0, 0.0, 0.0), nph_max=3)
    assert r['reference'] == 'QED-UHF'
    assert r['e_qed_fci'] == pytest.approx(r['e_fci'], abs=1e-9)
    assert r['e_qed_hf'] == pytest.approx(e_uhf, abs=1e-9)


def test_fci_open_shell_cavity_runs():
    """Open-shell QED-FCI with coupling: negative QED correlation energy."""
    mol = _oh_doublet()
    mf = scf.UHF(mol)
    mf.kernel()
    r = run_qed_fci(mf, omega=OMEGA, coupling_vec=LAM, nph_max=4)
    assert r['e_corr_qed'] < 0.0


# --- QED-CASCI -------------------------------------------------------------

def test_ucasci_full_active_space_equals_fci():
    """Open-shell QED-UCASCI at full active space equals open-shell QED-FCI."""
    mol = _oh_doublet()
    mf = scf.UHF(mol)
    mf.kernel()
    nmo, ne = mol.nao_nr(), mol.nelec
    for lam in (0.0, 0.05):
        cv = [0.0, 0.0, lam]
        rf = run_qed_fci(mf, omega=OMEGA, coupling_vec=cv, nph_max=4)
        rc = run_qed_casci(mf, ncas=nmo, nelecas=ne,
                           omega=OMEGA, coupling_vec=cv, nph_max=4)
        assert rc['e_qed_casci'] == pytest.approx(rf['e_qed_fci'], abs=1e-9)
        assert rc['e_qed_hf'] == pytest.approx(rf['e_qed_hf'], abs=1e-9)


def test_ucasci_spin_polarized_core_and_variational():
    """A truncated open-shell active space with a spin-polarized core
    (ncore_α ≠ ncore_β) reduces to bare UCASCI at λ=0 and stays variational
    above QED-FCI with coupling."""
    mol = _oh_doublet()
    mf = scf.UHF(mol)
    mf.kernel()
    # OH/STO-3G: nmo=6, nelec=(5,4). CAS(4,(3,3)) -> ncore=(2,1).
    r0 = run_qed_casci(mf, ncas=4, nelecas=(3, 3),
                       omega=OMEGA, coupling_vec=[0, 0, 0.0], nph_max=3)
    assert r0['ncore'] == (2, 1)
    assert r0['e_qed_casci'] == pytest.approx(r0['e_casci'], abs=1e-9)

    rc = run_qed_casci(mf, ncas=4, nelecas=(3, 3),
                       omega=OMEGA, coupling_vec=list(LAM), nph_max=4)
    rf = run_qed_fci(mf, omega=OMEGA, coupling_vec=list(LAM), nph_max=4)
    assert rc['e_qed_casci'] >= rf['e_qed_fci'] - 1e-9
