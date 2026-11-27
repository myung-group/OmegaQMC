"""Tests for :mod:`OmegaQMC.addons.qed_noci` and :mod:`OmegaQMC.addons.qed_noci_scf`.

1. Photon factors D, X, B equal brute-force displaced-Fock-space matrices.
2. One determinant, N = 0, own z: QED-NOCI energy = coherent-state QED-RHF energy,
   = bare-frame energy, = OmegaQMC.addons.qed_hf.run_qed_hf.
3. QED-NOCI over the complete determinant space with one common photon frame
   reproduces every eigenvalue of OmegaQMC.addons.qed_fci.run_qed_fci.
4. With per-determinant z_I the complete-space spectrum converges to the same
   QED-FCI values in N; the spectrum is origin invariant (HeH+ is charged).
5. Spin adaptation of a BS-ROHF pair: singlet + triplet roots = unadapted roots.
6. QED-BS-ROHF converges to a stationary point (orbital gradient ~ 0) and the
   spin pair has equal energies and z.
"""

import numpy as np
import pytest
from pyscf import gto, scf

from OmegaQMC.addons.qed_noci import (QEDNOCIIntegrals, QEDNOCI, photon_factors,
                                      run_qed_noci, spin_flip)
from OmegaQMC.addons.qed_noci_scf import QEDRHF, run_qed_bs_rohf, bs_grad
from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_fci import run_qed_fci

OMEGA = 0.5
LAM = np.array([0., 0., 0.08])


def _hehp(basis='6-31g'):
    return gto.M(atom='He 0 0 0; H 0 0 0.79', basis=basis, unit='Angstrom', charge=1, verbose=0)


def _qedrhf(mol, origin=None):
    mf = QEDRHF(mol, OMEGA, LAM, origin=origin)
    mf.verbose, mf.conv_tol = 0, 1e-12
    mf.kernel()
    assert mf.converged
    return mf


def _all_dets(c, nelec):
    """Every M_s = 0 determinant of a 1 alpha + 1 beta electron system in orbitals c."""
    assert nelec == (1, 1)
    n = c.shape[1]
    return [(c[:, [i]], c[:, [j]]) for i in range(n) for j in range(n)]


def test_photon_factors():
    from scipy.linalg import expm
    nbig, zi, zj, ni, nj = 60, 0.3, -0.45, 4, 6
    b = np.diag(np.sqrt(np.arange(1, nbig)), 1)          # annihilation
    u = lambda z: expm(z * (b.T - b))                     # noqa: E731
    vi, vj = u(zi)[:, :ni + 1], u(zj)[:, :nj + 1]
    dm, x, bb = photon_factors(zi, zj, ni, nj)
    assert np.allclose(dm, vi.T @ vj, atol=1e-12)
    assert np.allclose(x, vi.T @ (b + b.T) @ vj, atol=1e-12)
    assert np.allclose(bb, vi.T @ (b.T @ b) @ vj, atol=1e-12)


def test_single_det_equals_qed_rhf():
    mol = gto.M(atom='O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24', basis='sto-3g',
                unit='Angstrom', verbose=0)
    mf = _qedrhf(mol)
    nocc = mol.nelec[0]
    det = (mf.mo_coeff[:, :nocc], mf.mo_coeff[:, :nocc])
    res = run_qed_noci(mol, OMEGA, LAM, [det], nph_max=0)
    assert abs(res['e_qed_noci'] - mf.e_tot) < 1e-9
    assert abs(mf.energy_bare_frame() - mf.e_tot) < 1e-9
    ref = run_qed_hf(mol, OMEGA, LAM, tol=1e-12)
    assert abs(ref['E_qed_hf'] - mf.e_tot) < 1e-8


def test_full_space_equals_qed_fci():
    mol = _hehp()
    nph = 3
    mf = _qedrhf(mol)
    dets = _all_dets(mf.mo_coeff, mol.nelec)
    ints = QEDNOCIIntegrals(mol, OMEGA, LAM)
    q = QEDNOCI(ints, dets, zs=ints.z_of(dets[0]), nph=nph)
    e, _ = q.kernel()
    ref = run_qed_fci(scf.RHF(mol).run(), OMEGA, LAM, nph_max=nph)
    assert len(e) == len(ref['eigenvalues'])
    assert np.allclose(e, ref['eigenvalues'], atol=1e-8)


def test_state_specific_z_and_origin():
    mol = _hehp()
    mf = _qedrhf(mol)
    dets = _all_dets(mf.mo_coeff, mol.nelec)
    ref = run_qed_fci(scf.RHF(mol).run(), OMEGA, LAM, nph_max=12)['eigenvalues'][:4]
    for origin in (None, np.array([0., 0., 3.0])):
        ints = QEDNOCIIntegrals(mol, OMEGA, LAM, origin=origin)
        q = QEDNOCI(ints, dets, nph=10)                    # own z_I per determinant
        e, _ = q.kernel()
        assert np.allclose(e[:4], ref, atol=1e-8)


def _ct_pair(mol):
    nao = mol.nao
    he, h = mol.aoslice_by_atom()[:, 2:4]
    a = np.zeros((nao, 1)); b = np.zeros((nao, 1))
    a[he[0]] = 1.; b[h[0]] = 1.                          # He 1s-like, H 1s-like guess
    mf = run_qed_bs_rohf(mol, OMEGA, LAM, np.zeros((nao, 0)), a, b)
    assert mf.converged
    return mf


def test_bs_rohf_stationary_and_spin_pair():
    mol = _hehp()
    mf = _ct_pair(mol)
    c = mf.mo_coeff
    fock = mf.get_fock(dm=mf.make_rdm1())
    g = bs_grad(c.T @ fock.focka @ c, c.T @ fock.fockb @ c, mf.mo_occ)
    assert np.abs(g).max() < 1e-5
    det = mf.det()
    ints = QEDNOCIIntegrals(mol, OMEGA, LAM)
    assert abs(ints.z_of(det) - ints.z_of(spin_flip(det))) < 1e-12
    e1 = run_qed_noci(mol, OMEGA, LAM, [det], nph_max=0)['e_qed_noci']
    e2 = run_qed_noci(mol, OMEGA, LAM, [spin_flip(det)], nph_max=0)['e_qed_noci']
    assert abs(e1 - e2) < 1e-10
    assert abs(e1 - mf.e_tot) < 1e-8


def test_spin_adaptation():
    mol = _hehp()
    hf = _qedrhf(mol)
    det = _ct_pair(mol).det()
    c0 = hf.mo_coeff[:, :1]
    dets = [(c0, c0), det, spin_flip(det)]
    ints = QEDNOCIIntegrals(mol, OMEGA, LAM)
    q = QEDNOCI(ints, dets, nph=3)
    e_all = np.sort(q.kernel()[0])
    e_s = q.kernel(spin_pairs=[(1, 2)], sign=+1)[0]
    # M_s = 0 triplet: the antisymmetric pair combination alone (decoupled from the singlets)
    e_t = QEDNOCI(ints, dets[1:], nph=3).kernel(spin_pairs=[(0, 1)], sign=-1)[0]
    assert len(e_s) + len(e_t) == len(e_all)
    assert np.allclose(np.sort(np.r_[e_s, e_t]), e_all, atol=1e-9)
