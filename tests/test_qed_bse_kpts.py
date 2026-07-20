"""Regression test: k-point polaritonic QED-BSE vs Gamma-only supercell.

A Gamma-centred 1x2 k-mesh on the primitive hBN cell must reproduce the
Gamma-only 1x2 supercell exactly (same BvK boundary conditions): HF
energy/eigenvalues, and the polaritonic BSE@HF spectra (TDA and full)
with the q = 0 photon channel. BSE@HF (eps_QP = HF) keeps the runtime
small while exercising the full DF/conjugation/W(q)/photon machinery;
the GW step is validated by examples/qed_gw/validate_qed_bse_kpts_hbn.py.

Both sides use the BARE velocity route: bare momentum matrix elements
conserve the primitive crystal momentum in either representation, so
the supercell identity holds to arithmetic precision. The exact
velocity route is representation-covariant only up to
basis-incompleteness residuals (see qed_polariton_kpts docstring) and
is validated separately in examples/qed_gw/run_qed_velocity_exact_check.py.
"""

import math
import os
import sys

import numpy as np
import pytest

pyscf = pytest.importorskip('pyscf')
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import tools as pbctools

from OmegaQMC.addons.qed_polariton_kpts import (build_kpts_quantities,
                                                run_qed_bse_kpts)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'examples', 'qed_gw'))
from run_qed_bse_hbn_gamma import (build_hbn_cell, run_gamma_rhf,
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
