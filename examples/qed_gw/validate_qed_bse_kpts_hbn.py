"""Validate the k-point polaritonic QED-BSE against the Gamma-only
supercell path, then demonstrate a genuine k-point run.

Part 1 (validation, hBN gth-szv): a Gamma-centred 2x2 k-mesh on the
primitive cell is numerically identical to a Gamma-only 2x2 supercell
(same BvK boundary conditions, exxdiv=None on both). The k-space
q = 0 excitation block is a SUBSET of the supercell excitation space
(the supercell also carries the finite-momentum, cavity-dark exciton
blocks), so the checks are:

  * E_HF(supercell) = N_k * E_HF(KRHF per cell)
  * HF and evGW quasiparticle energies agree as multisets
    (dense sum-over-poles supercell evGW vs aux-CD k-point evGW)
  * every k-space polaritonic BSE root (TDA and full) appears in the
    supercell spectrum; LP/UP energies and photon weights agree

Part 2 (demo, 3x3 mesh): the 3x3 Gamma-centred mesh folds the K point
into the sampling — physics a Gamma-only primitive cell cannot see —
and the cavity is tuned to the lowest bright exciton of that mesh.

Run:  python validate_qed_bse_kpts_hbn.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_qed_bse_hbn_gamma import (EV, build_hbn_cell, run_gamma_rhf,
                                   gamma_df_factor, velocity_gauge_dipole,
                                   build_sq, run_gw_electronic,
                                   run_bse_polaritonic)

from pyscf.pbc import scf as pbcscf

from OmegaQMC.addons.qed_polariton_kpts import (build_kpts_quantities,
                                                run_kgw_electronic,
                                                run_qed_bse_kpts)

BASIS = 'gth-szv'
LAM = 0.05


def lp_up(bse):
    idx = np.argsort(bse['photon_weight'])[::-1][:2]
    lo, hi = sorted(float(bse['Omega'][i]) for i in idx)
    return lo, hi, [float(bse['photon_weight'][i]) for i in idx]


def subset_dev(sub, full):
    """max over `sub` of the distance to the nearest element of `full`."""
    sub = np.asarray(sub)
    full = np.asarray(full)
    return float(max(np.min(np.abs(full - x)) for x in sub))


def run_supercell():
    print('=== supercell reference: hBN 2x2, Gamma-only ===')
    cell = build_hbn_cell(basis=BASIS, nsc=(2, 2))
    mf = run_gamma_rhf(cell)
    B_ao = gamma_df_factor(mf)
    r_eff = velocity_gauge_dipole(mf)
    sq0, eps_HF = build_sq(mf, B_ao, r_eff, (0.0, 0.0, 0.0), omega_cav=1.0)
    eps_QP = run_gw_electronic(sq0, eps_HF, mode='evGW')
    bse0 = run_bse_polaritonic(sq0, eps_QP, r_eff, lambda_on=False)
    ib = int(np.where(bse0['f_osc'] > 1e-2)[0][0])
    w_cav = float(bse0['Omega'][ib])
    sq, _ = build_sq(mf, B_ao, r_eff, (LAM, 0.0, 0.0), w_cav)
    bse_full = run_bse_polaritonic(sq, eps_QP, r_eff, lambda_on=True)
    bse_tda = run_bse_polaritonic(sq, eps_QP, r_eff, lambda_on=True,
                                  tda=True)
    return {
        'E': float(mf.e_tot),
        'eps_HF': np.sort(eps_HF),
        'eps_QP': np.sort(eps_QP),
        'w_cav': w_cav,
        'full': bse_full, 'tda': bse_tda,
    }


def run_kmesh(w_cav):
    print('\n=== k-point run: hBN primitive cell, 2x2 Gamma-centred ===')
    cell = build_hbn_cell(basis=BASIS, nsc=(1, 1))
    kpts = cell.make_kpts([2, 2, 1])
    kmf = pbcscf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
    kmf.conv_tol = 1e-10
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError('KRHF did not converge')
    kq = build_kpts_quantities(kmf)
    eps_QP = run_kgw_electronic(kq, mode='evGW')
    bse_full = run_qed_bse_kpts(kq, w_cav, (LAM, 0.0, 0.0), eps_QP=eps_QP)
    bse_tda = run_qed_bse_kpts(kq, w_cav, (LAM, 0.0, 0.0), eps_QP=eps_QP,
                               tda=True)
    return {
        'E': float(kmf.e_tot),
        'nk': len(kpts),
        'eps_HF': np.sort(np.concatenate(kq['eps'])),
        'eps_QP': np.sort(np.concatenate(eps_QP)),
        'full': bse_full, 'tda': bse_tda,
    }


def run_demo_33():
    print('\n=== demo: hBN primitive cell, 3x3 mesh (K folded in) ===')
    cell = build_hbn_cell(basis=BASIS, nsc=(1, 1))
    kpts = cell.make_kpts([3, 3, 1])
    kmf = pbcscf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
    kmf.conv_tol = 1e-10
    kmf.kernel()
    kq = build_kpts_quantities(kmf)
    eps_QP = run_kgw_electronic(kq, mode='evGW')
    bse0 = run_qed_bse_kpts(kq, 1.0, (0.0, 0.0, 0.0), eps_QP=eps_QP,
                            verbose=False)
    f0 = bse0['f_osc']
    ib = int(np.where(f0 > 1e-2)[0][0])
    w_cav = float(bse0['Omega'][ib])
    print('  lambda=0 excitons (eV / f_osc):')
    for i in range(6):
        mark = ' <- cavity tuned here' if i == ib else ''
        print(f'    {bse0["Omega"][i] * EV:8.3f}  {f0[i]:8.4f}{mark}')
    bse = run_qed_bse_kpts(kq, w_cav, (LAM, 0.0, 0.0), eps_QP=eps_QP)
    lo, hi, pw = lp_up(bse)
    print(f'  LP = {lo * EV:.3f} eV, UP = {hi * EV:.3f} eV, '
          f'Rabi = {(hi - lo) * EV:.3f} eV, photon weights = '
          f'{pw[0]:.3f}/{pw[1]:.3f}')
    return {
        'omega_cav_eV': w_cav * EV,
        'exciton_eV': [float(x) for x in bse0['Omega'][:8] * EV],
        'exciton_f': [float(x) for x in f0[:8]],
        'LP_eV': lo * EV, 'UP_eV': hi * EV,
        'rabi_eV': (hi - lo) * EV, 'photon_weight': pw,
    }


def main():
    sc = run_supercell()
    km = run_kmesh(sc['w_cav'])

    print('\n=== validation: 2x2 k-mesh vs 2x2 Gamma-only supercell ===')
    dev_E = abs(sc['E'] - km['nk'] * km['E'])
    dev_hf = float(np.max(np.abs(sc['eps_HF'] - km['eps_HF'])))
    dev_qp = float(np.max(np.abs(sc['eps_QP'] - km['eps_QP'])))
    print(f'  |E_HF(sc) - Nk*E_HF(k)|          = {dev_E:.3e} Ha')
    print(f'  max |eps_HF(sc) - eps_HF(k)|     = {dev_hf:.3e} Ha')
    print(f'  max |eps_QP(sc) - eps_QP(k)|     = {dev_qp:.3e} Ha '
          f'(dense poles vs aux-CD)')

    devs = {}
    for kind in ('tda', 'full'):
        dev_sub = subset_dev(km[kind]['Omega'], sc[kind]['Omega'])
        lo_s, hi_s, pw_s = lp_up(sc[kind])
        lo_k, hi_k, pw_k = lp_up(km[kind])
        dev_lp = abs(lo_s - lo_k) * EV
        dev_up = abs(hi_s - hi_k) * EV
        print(f'  [{kind}] max_k min_sc |dOmega|       = {dev_sub:.3e} Ha')
        print(f'  [{kind}] LP: sc {lo_s * EV:8.4f} / k {lo_k * EV:8.4f} eV '
              f'(|d| = {dev_lp:.2e} eV)')
        print(f'  [{kind}] UP: sc {hi_s * EV:8.4f} / k {hi_k * EV:8.4f} eV '
              f'(|d| = {dev_up:.2e} eV)')
        print(f'  [{kind}] photon weights: sc {pw_s[0]:.4f}/{pw_s[1]:.4f}, '
              f'k {pw_k[0]:.4f}/{pw_k[1]:.4f}')
        devs[kind] = {'subset_Ha': dev_sub, 'LP_eV': dev_lp, 'UP_eV': dev_up}

    demo = run_demo_33()

    out = {
        'basis': BASIS, 'lambda': LAM,
        'dev_E_Ha': dev_E, 'dev_eps_HF_Ha': dev_hf, 'dev_eps_QP_Ha': dev_qp,
        'bse_devs': devs,
        'omega_cav_eV': sc['w_cav'] * EV,
        'demo_3x3': demo,
    }
    with open('qed_bse_kpts_validation.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('\nresults saved to qed_bse_kpts_validation.json')


if __name__ == '__main__':
    main()
