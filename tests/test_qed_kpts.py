"""Periodic (k-point) QED-BSE and its indefinite auxiliary metric.
"""
import os
import sys
import numpy as np
import pytest
from pyscf.pbc import scf as pbcscf

from OmegaQMC.addons.qed_polariton_kpts import (  # noqa: E402
    build_kpts_quantities,
    run_qed_bse_kpts,
    _lambda_eff_q,
)


# ---------------------------------------------------------------------------
# From test_qed_bse_kpts.py
#
# Regression test: k-point polaritonic QED-BSE vs Gamma-only supercell.
#
# A Gamma-centred 1x2 k-mesh on the primitive hBN cell must reproduce the
# Gamma-only 1x2 supercell exactly (same BvK boundary conditions): HF
# energy/eigenvalues, and the polaritonic BSE@HF spectra (TDA and full)
# with the q = 0 photon channel. BSE@HF (eps_QP = HF) keeps the runtime
# small while exercising the full DF/conjugation/W(q)/photon machinery;
# the GW step is validated by examples/qed_kpts/validate_qed_bse_kpts_hbn.py.
#
# Both sides use the BARE velocity route: bare momentum matrix elements
# conserve the primitive crystal momentum in either representation, so
# the supercell identity holds to arithmetic precision. The exact
# velocity route is representation-covariant only up to
# basis-incompleteness residuals (see qed_polariton_kpts docstring) and
# is validated separately in examples/qed_kpts/run_qed_velocity_exact_check.py.
# ---------------------------------------------------------------------------


pyscf = pytest.importorskip('pyscf')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'examples', 'qed_kpts'))
from run_qed_bse_hbn_gamma import (build_hbn_cell, run_gamma_rhf,  # noqa: E402
                                   gamma_df_factor, velocity_gauge_dipole,
                                   build_sq, run_bse_polaritonic)
LAM = 0.05
W_CAV = 0.5  # Ha; detuned is fine — the equivalence must hold anyway


@pytest.fixture(scope='module')
def supercell_ref():
    cell = build_hbn_cell(basis='gth-szv', nsc=(1, 2))
    mf = run_gamma_rhf(cell)
    B_ao = gamma_df_factor(mf)
    r_eff = velocity_gauge_dipole(mf, exact=False)   # machinery check
    sq, eps_HF = build_sq(mf, B_ao, r_eff, (LAM, 0.0, 0.0), W_CAV)
    full = run_bse_polaritonic(sq, eps_HF, r_eff, lambda_on=True)
    tda = run_bse_polaritonic(sq, eps_HF, r_eff, lambda_on=True, tda=True)
    return {'E': mf.e_tot, 'eps': np.sort(eps_HF), 'full': full, 'tda': tda}


@pytest.fixture(scope='module')
def kmesh_run():
    cell = build_hbn_cell(basis='gth-szv', nsc=(1, 1))
    kpts = cell.make_kpts([1, 2, 1])
    kmf = pbcscf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
    kmf.conv_tol = 1e-10
    kmf.kernel()
    assert kmf.converged
    kq = build_kpts_quantities(kmf, verbose=False,
                               velocity='bare')      # machinery check
    full = run_qed_bse_kpts(kq, W_CAV, (LAM, 0.0, 0.0), verbose=False)
    tda = run_qed_bse_kpts(kq, W_CAV, (LAM, 0.0, 0.0), tda=True,
                           verbose=False)
    return {'E': kmf.e_tot, 'nk': len(kpts),
            'eps': np.sort(np.concatenate(kq['eps'])),
            'full': full, 'tda': tda}


def test_hf_equivalence(supercell_ref, kmesh_run):
    assert abs(supercell_ref['E']
               - kmesh_run['nk'] * kmesh_run['E']) < 1e-7
    assert np.max(np.abs(supercell_ref['eps'] - kmesh_run['eps'])) < 1e-7


@pytest.mark.parametrize('kind', ['tda', 'full'])
def test_bse_subset(supercell_ref, kmesh_run, kind):
    """Every k-space root (q = 0 block + photon) appears in the
    supercell spectrum; bright/photon roots carry matching weights."""
    om_k = kmesh_run[kind]['Omega']
    om_sc = supercell_ref[kind]['Omega']
    dev = max(np.min(np.abs(om_sc - x)) for x in om_k)
    assert dev < 1e-6

    # LP/UP (largest photon weight) match in energy and weight
    idx_k = np.argsort(kmesh_run[kind]['photon_weight'])[::-1][:2]
    idx_s = np.argsort(supercell_ref[kind]['photon_weight'])[::-1][:2]
    lpup_k = np.sort(om_k[idx_k])
    lpup_s = np.sort(om_sc[idx_s])
    assert np.max(np.abs(lpup_k - lpup_s)) < 1e-6
    pw_k = np.sort(kmesh_run[kind]['photon_weight'][idx_k])
    pw_s = np.sort(supercell_ref[kind]['photon_weight'][idx_s])
    assert np.max(np.abs(pw_k - pw_s)) < 1e-5


# ---------------------------------------------------------------------------
# From test_qed_kpts_metric.py
#
# Unit tests for the indefinite (sign-split GDF) auxiliary metric in
# the k-point polaritonic QED-BSE module.
#
# Pure linear algebra — no pyscf runs. The screened-interaction kernel
# with an indefinite decomposition v = B^T S B (S = diag(+1...,-1...),
# pyscf dimension=2 low-dimensional GDF) is
#
#     Lam_eff = S 4Pi (1 - S 4Pi)^{-1} S,
#
# and must reproduce the channel-space RPA series
# W = v chi0 (1 - v chi0)^{-1} v for any metric. The heavy end-to-end
# check (dimension=2 vs vacuum-slab L_z -> inf on hBN) is the manual
# script examples/qed_gw/validate_qed_bse_kpts_2d.py.
# ---------------------------------------------------------------------------


def _random_channels(seed, naux=7, nch=12):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(naux, nch)) + 1j * rng.normal(size=(naux, nch))
    dE = rng.uniform(0.5, 2.0, size=nch)
    return b, dE


@pytest.mark.parametrize('sgn', [
    None,
    np.ones(7),
    np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0]),
    np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
])
def test_lambda_eff_matches_channel_space_series(sgn):
    """b^T Lam_eff b* must equal the dense channel-space RPA screened
    interaction for definite and indefinite metrics."""
    b, dE = _random_channels(0)
    nch = b.shape[1]
    nu2 = -0.3    # imaginary axis, nu2 = -w^2
    s = np.ones(b.shape[0]) if sgn is None else sgn
    x = dE / (nu2 - dE * dE)
    v = np.einsum('X,Xn,Xm->nm', s, b, b.conj())
    chi0 = np.diag(4.0 * x)
    W_ref = v @ chi0 @ np.linalg.solve(np.eye(nch) - v @ chi0, v)
    lam = _lambda_eff_q(b, dE, nu2, sgn)
    W_aux = np.einsum('Xn,XY,Ym->nm', b, lam, b.conj())
    assert np.abs(W_aux - W_ref).max() < 1e-11


def test_lambda_eff_hermitian_for_real_frequency():
    """S 4Pi (1 - S 4Pi)^{-1} S is Hermitian whenever Pi is (real nu2),
    which the contour-deformation GW relies on."""
    b, dE = _random_channels(1)
    sgn = np.array([1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
    for nu2 in (0.0, -0.7):
        lam = _lambda_eff_q(b, dE, nu2, sgn)
        assert np.abs(lam - lam.conj().T).max() < 1e-12


def test_positive_metric_reduces_to_legacy_kernel():
    """sgn all +1 must agree with the sgn=None (legacy 3D) branch."""
    b, dE = _random_channels(2)
    for nu2 in (0.0, -0.5, (0.2 + 1e-3j) ** 2):
        lam0 = _lambda_eff_q(b, dE, nu2)
        lam1 = _lambda_eff_q(b, dE, nu2, np.ones(b.shape[0]))
        assert np.abs(lam0 - lam1).max() < 1e-12
