"""Fast pytest checks for the QED-BSE kernel structure and the
QED-GW spectral function (H2O/STO-3G — suitable for the suite).

Covers the index-convention fix of the BSE kernel:
* W → v̄ limit of the TDA-BSE matrix equals the QED-CIS matrix
  (TDA on the antisymmetric QED-RPA kernel) to machine precision;
* triplets lie below singlets and carry no oscillator strength;
* the kernel channel toggles (include_dse / include_photon) are inert
  at λ = 0;
* the spectral-function peak coincides with the Newton G0W0 root.
"""
import math

import numpy as np
import pytest
from pyscf import gto

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_bse import (run_qed_bse, _W_static,
                                     _eh_exchange_kernel)
from OmegaQMC.addons.qed_gw import run_qed_gw, _build_static_quantities
from OmegaQMC.addons.qed_spectral import qed_gw_spectral_function

OMEGA = 0.415668
LAMBDA = 0.05


@pytest.fixture(scope='module')
def mol():
    half = math.radians(104.5 / 2.0)
    hx, hz = math.sin(half), -math.cos(half)
    return gto.M(atom=[['O', (0, 0, 0)], ['H', (hx, 0, hz)],
                       ['H', (-hx, 0, hz)]],
                 basis='sto-3g', unit='Angstrom', symmetry=False, verbose=0)


@pytest.fixture(scope='module')
def qedhf(mol):
    return run_qed_hf(mol, OMEGA, (0.0, 0.0, LAMBDA), verbose=False)


@pytest.fixture(scope='module')
def qedhf0(mol):
    return run_qed_hf(mol, OMEGA, (0.0, 0.0, 0.0), verbose=False)


def test_bse_bare_limit_equals_cis(qedhf):
    """A^BSE with W → v̄ must equal the QED-CIS (antisymmetric RPA TDA)
    electronic block: the e-h exchange enters once, the direct
    attraction once (inside W)."""
    static = _build_static_quantities(qedhf, direct=True)
    so = static['so']
    nocc, nso = so['nocc'], so['nso']
    nvir = nso - nocc
    nov = nvir * nocc
    eps = static['eps_HF']
    diag = np.diag((eps[nocc:, None] - eps[None, :nocc]).reshape(-1))

    M0 = np.zeros((nso, nso, 1))
    K_x = _eh_exchange_kernel(static, 'aj,ib')
    W_b = _W_static(static, np.ones(1), M0, 'ij,ab').transpose(2, 0, 3, 1)
    A_bse = (K_x - W_b).reshape(nov, nov) + diag

    g_a = so['g_phys_a']
    d_so = so['d_so']
    d_vo, d_ov = d_so[nocc:, :nocc], d_so[:nocc, nocc:]
    d_oo, d_vv = d_so[:nocc, :nocc], d_so[nocc:, nocc:]
    A_cis = diag + (g_a[:nocc, nocc:, nocc:, :nocc]
                    .transpose(2, 0, 1, 3).reshape(nov, nov))
    A_cis = A_cis + (np.einsum('ai,jb->aibj', d_vo, d_ov)
                     - np.einsum('ab,ij->aibj', d_vv, d_oo)
                     ).reshape(nov, nov)
    assert np.max(np.abs(A_bse - A_cis)) < 1e-12


def test_spin_structure_and_brightness(qedhf0):
    bse = run_qed_bse(qedhf0, gw_mode='evGW', tda=True, verbose=False)
    spin = bse['spin']
    w_s = bse['w_singlet']
    f = bse['f_osc']
    # clean spin classification
    assert np.all((w_s < 1e-6) | (w_s > 1.0 - 1e-6))
    # triplets dark
    assert np.max(f[spin == 'T']) < 1e-10
    # lowest root is a triplet, below the lowest singlet
    assert spin[0] == 'T'
    assert bse['Omega_T1'] < bse['Omega_S1']
    # exciton binding positive (basis-confined LUMO)
    assert bse['E_b'] > 0.0


def test_channel_toggles_inert_at_lambda0(qedhf0):
    full = run_qed_bse(qedhf0, gw_mode='evGW', tda=True, verbose=False)
    off = run_qed_bse(qedhf0, gw_mode='evGW', tda=True,
                      include_dse=False, include_photon=False,
                      verbose=False)
    assert np.max(np.abs(full['Omega_BSE'] - off['Omega_BSE'])) < 1e-12


def test_eps_qp_override_matches_internal(qedhf0):
    full = run_qed_bse(qedhf0, gw_mode='evGW', tda=True, verbose=False)
    over = run_qed_bse(qedhf0, tda=True, eps_QP=full['eps_QP'],
                       verbose=False)
    assert np.max(np.abs(full['Omega_BSE'] - over['Omega_BSE'])) < 1e-12


def test_spectral_peak_at_qp_energy(qedhf):
    gw = run_qed_gw(qedhf, mode='G0W0', verbose=False)
    homo = qedhf['nocc_spatial'] - 1
    grid = np.linspace(-0.6, -0.1, 20001)
    sp = qed_gw_spectral_function(qedhf, orbs=[homo], omega_grid=grid,
                                  eta=2e-3, verbose=False)
    w_peak = grid[np.argmax(sp['A'][0])]
    assert abs(w_peak - gw['eps_QP'][homo]) < 1e-4
