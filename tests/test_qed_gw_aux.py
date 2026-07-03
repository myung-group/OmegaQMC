"""Tests for the auxiliary-basis dielectric backend of the singlet QED-GW.

The aux path (``screening='aux-cd' | 'aux-pade'``) replaces the dense
(nov+1)-dimensional QED-dRPA eigensolve of :func:`run_qed_gw_singlet` by
Woodbury dielectric inversions in the (naux+1)-dimensional auxiliary space,
with the cavity photon folded in exactly (frequency-dependent weight on the
dipole channel). These tests pin the two contracts:

1. The screened-interaction identity — the aux-dielectric Wt_pm(i w')
   equals the dense sum-over-poles built from `_screening_at_eps` to
   machine precision, for every (DSE, photon) flag combination.
2. Quasiparticle energies — contour deformation reproduces the dense
   Newton solves for ALL orbitals; Pade continuation reproduces the
   frontier states (its documented domain).

Small basis (sto-3g) on purpose: the dense reference diagonalisation is
cheap and the whole file runs in seconds (see
tools/qed_ccsd_df_derivation/validate_gw_aux_screening.py for the larger
6-31g / density-fitted validation).
"""

import numpy as np
import pytest
from pyscf import gto

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_polariton_singlet import (
    _build_spatial_quantities, _screening_at_eps, _aux_setup,
    _aux_wtilde_grid, _aux_w_static_blocks, run_qed_gw_singlet,
    run_qed_bse_polaritonic_singlet,
)

OMEGA = 0.4
LAMBDA = (0.0, 0.0, 0.05)


@pytest.fixture(scope='module')
def qedhf():
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='sto-3g', verbose=0)
    return run_qed_hf(mol, OMEGA, LAMBDA, verbose=False)


@pytest.mark.parametrize('dse', [True, False])
@pytest.mark.parametrize('photon', [True, False])
def test_wtilde_identity(qedhf, dse, photon):
    """Aux dielectric == dense sum-over-poles on the imaginary axis."""
    sq = _build_spatial_quantities(qedhf)
    eps = np.diag(sq['F'])
    freqs = np.array([0.0, 0.05, 0.3, 1.0, 5.0])

    Omega, M = _screening_at_eps(sq, eps, include_dse=dse,
                                 include_photon=photon)
    pole = 2.0 * Omega / (-freqs[:, None] ** 2 - Omega[None, :] ** 2)
    Wt_dense = np.einsum('pqs,ks->kpq', M * M, pole)

    setup = _aux_setup(sq, eps, include_dse=dse, include_photon=photon)
    Wt_aux = _aux_wtilde_grid(setup, freqs)

    assert np.max(np.abs(Wt_aux - Wt_dense)) < 1e-10


@pytest.mark.parametrize('mode', ['G0W0', 'evGW'])
def test_qp_aux_cd_matches_dense(qedhf, mode):
    """Contour deformation reproduces the dense QP energies, all orbitals."""
    ref = run_qed_gw_singlet(qedhf, mode=mode, verbose=False,
                             screening='dense')['eps_QP']
    qp = run_qed_gw_singlet(qedhf, mode=mode, verbose=False,
                            screening='aux-cd', n_freq=200)['eps_QP']
    assert np.max(np.abs(qp - ref)) < 1e-5


def test_qp_aux_pade_frontier(qedhf):
    """Pade continuation reproduces the frontier QP energies."""
    no = qedhf['nocc_spatial']
    ref = run_qed_gw_singlet(qedhf, mode='evGW', verbose=False,
                             screening='dense')['eps_QP']
    qp = run_qed_gw_singlet(qedhf, mode='evGW', verbose=False,
                            screening='aux-pade', n_freq=200)['eps_QP']
    assert abs(qp[no - 1] - ref[no - 1]) < 5e-4     # HOMO
    assert abs(qp[no] - ref[no]) < 5e-4             # LUMO


def test_w_static_blocks_match_dense(qedhf):
    """Static BSE W blocks: nu=0 Woodbury == dense sum over dRPA poles.

    The nu=0 dielectric is a single exact inversion (no quadrature), so
    this identity holds to machine precision."""
    sq = _build_spatial_quantities(qedhf)
    eps = np.diag(sq['F'])
    no = sq['nocc']
    Omega, M = _screening_at_eps(sq, eps, include_dse=True,
                                 include_photon=False)
    inv_Om = 1.0 / Omega
    ref_ijab = -2.0 * np.einsum('ijs,abs,s->ijab', M[:no, :no, :],
                                M[no:, no:, :], inv_Om)
    ref_ibaj = -2.0 * np.einsum('ibs,ajs,s->ibaj', M[:no, no:, :],
                                M[no:, :no, :], inv_Om)
    Wc_ijab, Wc_ibaj = _aux_w_static_blocks(sq, eps)
    assert np.max(np.abs(Wc_ijab - ref_ijab)) < 1e-12
    assert np.max(np.abs(Wc_ibaj - ref_ibaj)) < 1e-12


@pytest.mark.parametrize('tda', [False, True])
def test_bse_aux_w_matches_dense(qedhf, tda):
    """pol-BSE spectrum with w_screening='aux' == 'dense' (same QP input)."""
    eps_QP = run_qed_gw_singlet(qedhf, mode='evGW', verbose=False)['eps_QP']
    dense = run_qed_bse_polaritonic_singlet(qedhf, eps_QP=eps_QP, tda=tda,
                                            verbose=False)
    aux = run_qed_bse_polaritonic_singlet(qedhf, eps_QP=eps_QP, tda=tda,
                                          verbose=False, w_screening='aux')
    assert np.max(np.abs(dense['Omega'] - aux['Omega'])) < 1e-10
    assert np.max(np.abs(dense['f_osc'] - aux['f_osc'])) < 1e-10


@pytest.mark.parametrize('tda', [False, True])
def test_bse_davidson_matches_dense(qedhf, tda):
    """Matrix-free paired Davidson == dense diagonalisation, lowest roots."""
    eps_QP = run_qed_gw_singlet(qedhf, mode='evGW', verbose=False)['eps_QP']
    dense = run_qed_bse_polaritonic_singlet(qedhf, eps_QP=eps_QP, tda=tda,
                                            verbose=False)
    dav = run_qed_bse_polaritonic_singlet(qedhf, eps_QP=eps_QP, tda=tda,
                                          verbose=False, solver='davidson',
                                          nroots=5, davidson_tol=1e-9)
    n = len(dav['Omega'])
    assert np.max(np.abs(dense['Omega'][:n] - dav['Omega'])) < 1e-8
    assert np.max(np.abs(dense['f_osc'][:n] - dav['f_osc'])) < 1e-8
    assert np.max(np.abs(dense['photon_weight'][:n]
                         - dav['photon_weight'])) < 1e-8


def test_unknown_screening_raises(qedhf):
    with pytest.raises(ValueError, match='screening'):
        run_qed_gw_singlet(qedhf, verbose=False, screening='bogus')
    with pytest.raises(ValueError, match='w_screening'):
        run_qed_bse_polaritonic_singlet(qedhf, eps_QP=np.zeros(7),
                                        verbose=False, w_screening='bogus')
    with pytest.raises(ValueError, match='solver'):
        run_qed_bse_polaritonic_singlet(qedhf, eps_QP=np.zeros(7),
                                        verbose=False, solver='bogus')
