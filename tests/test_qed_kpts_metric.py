"""Unit tests for the indefinite (sign-split GDF) auxiliary metric in
the k-point polaritonic QED-BSE module.

Pure linear algebra — no pyscf runs. The screened-interaction kernel
with an indefinite decomposition v = B^T S B (S = diag(+1...,-1...),
pyscf dimension=2 low-dimensional GDF) is

    Lam_eff = S 4Pi (1 - S 4Pi)^{-1} S,

and must reproduce the channel-space RPA series
W = v chi0 (1 - v chi0)^{-1} v for any metric. The heavy end-to-end
check (dimension=2 vs vacuum-slab L_z -> inf on hBN) is the manual
script examples/qed_gw/validate_qed_bse_kpts_2d.py.
"""

import numpy as np
import pytest

from OmegaQMC.addons.qed_polariton_kpts import _lambda_eff_q


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
