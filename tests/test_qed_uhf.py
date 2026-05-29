"""Tests for :func:`OmegaQMC.addons.qed_uhf.run_qed_uhf`.

Covers:
1. λ = 0 reproduces plain pyscf UHF exactly (open-shell doublet).
2. For a closed-shell molecule QED-UHF collapses onto the QED-RHF
   reference of :mod:`OmegaQMC.addons.qed_hf` (<S^2> ≈ 0).
3. With non-zero coupling the iteration converges and returns the
   expected output structure and a sensible spin multiplicity.
4. Polarisation directions give different energies (sanity check that
   the cavity coupling actually enters).
"""

import numpy as np
import pytest
from pyscf import gto, scf

from OmegaQMC.addons.qed_uhf import run_qed_uhf
from OmegaQMC.addons.qed_hf import run_qed_hf


def _ch3():
    """Methyl radical (doublet) / STO-3G."""
    return gto.M(
        atom='C 0 0 0; H 0 0 1.08; H 1.02 0 -0.36; H -0.51 0.88 -0.36',
        basis='STO-3G', spin=1, unit='Angstrom', verbose=0)


def _o2_triplet():
    return gto.M(atom='O 0 0 0; O 0 0 1.208', basis='STO-3G',
                 spin=2, unit='Angstrom', verbose=0)


def _h2o():
    return gto.M(atom='O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24',
                 basis='STO-3G', unit='Angstrom', verbose=0)


def test_zero_coupling_matches_uhf():
    """With λ = 0 the QED-UHF energy must equal the plain UHF energy."""
    mol = _ch3()
    mf = scf.UHF(mol)
    e_uhf = mf.kernel()

    res = run_qed_uhf(mol, omega=0.5, lambda_cav=(0.0, 0.0, 0.0))

    assert res['E_qed_uhf'] == pytest.approx(e_uhf, abs=1e-9)
    assert res['E_uhf'] == pytest.approx(e_uhf, abs=1e-9)
    # Dressed Fock at λ=0 should yield the canonical UHF eigenvalues.
    fa = res['Ca'].T @ res['Fa'] @ res['Ca']
    fb = res['Cb'].T @ res['Fb'] @ res['Cb']
    np.testing.assert_allclose(np.sort(np.diag(fa)),
                               np.sort(mf.mo_energy[0]), atol=1e-5)
    np.testing.assert_allclose(np.sort(np.diag(fb)),
                               np.sort(mf.mo_energy[1]), atol=1e-5)


def test_closed_shell_matches_qed_rhf():
    """For a closed-shell molecule QED-UHF must reproduce QED-RHF."""
    mol = _h2o()
    omega = 3.0 / 27.211386245988
    lam = (0.0, 0.0, 0.1)

    rhf = run_qed_hf(mol, omega, lambda_cav=lam)
    uhf = run_qed_uhf(mol, omega, lambda_cav=lam)

    assert uhf['E_qed_uhf'] == pytest.approx(rhf['E_qed_hf'], abs=1e-9)
    assert uhf['s_squared'] == pytest.approx(0.0, abs=1e-8)
    assert uhf['multiplicity'] == pytest.approx(1.0, abs=1e-6)


def test_output_structure_and_multiplicity():
    """Non-zero coupling converges; output structure and <S^2> are sane."""
    mol = _o2_triplet()
    omega = 3.0 / 27.211386245988
    res = run_qed_uhf(mol, omega, lambda_cav=(0.0, 0.0, 0.1))

    required = {
        'E_qed_uhf', 'E_uhf', 'E_nuc',
        'Ca', 'Cb', 'Fa', 'Fb', 'mo_energy_a', 'mo_energy_b',
        'H_core', 'oei', 'eri_ao',
        'mu_x_ao', 'mu_y_ao', 'mu_z_ao', 'dipole_x_lambda_tot',
        'nocc_a', 'nocc_b', 'nmo_spatial',
        's_squared', 'multiplicity',
        'lambda_cav', 'omega', 'mol',
    }
    assert required <= set(res.keys())
    nao = mol.nao_nr()
    assert res['Ca'].shape == (nao, nao)
    assert res['Fb'].shape == (nao, nao)
    assert res['eri_ao'].shape == (nao, nao, nao, nao)
    assert (res['nocc_a'], res['nocc_b']) == mol.nelec
    assert res['nmo_spatial'] == nao
    # Triplet O2: multiplicity close to 3 (modest spin contamination).
    assert res['multiplicity'] == pytest.approx(3.0, abs=0.05)
    # The cavity raises the energy (positive dipole self-energy).
    assert res['E_qed_uhf'] > res['E_uhf']


def test_polarization_direction_matters():
    """Different cavity polarizations give different QED-UHF energies."""
    mol = _o2_triplet()
    omega = 3.0 / 27.211386245988
    e_z = run_qed_uhf(mol, omega, lambda_cav=(0.0, 0.0, 0.1))['E_qed_uhf']
    e_x = run_qed_uhf(mol, omega, lambda_cav=(0.1, 0.0, 0.0))['E_qed_uhf']
    # O2 lies along z, so the parallel and perpendicular couplings differ.
    assert abs(e_z - e_x) > 1e-6
