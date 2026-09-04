"""Quantify the exact-within-basis velocity operator against the bare
momentum route for the hBN polaritonic QED-BSE@evGW pipeline.

The bare route  r_eff = i p / dE  (int1e_ipovlp) neglects the
[r, V_nl(pseudo)] and [r, Sigma_x] commutators and carries the
basis-incompleteness inequivalence of p- and r-matrix elements; the
exact route (OmegaQMC.addons.qed_polariton_kpts.velocity_dipole_exact)
is the nonorthogonal-basis k-derivative formula plus the intra-cell
dipole term and is exact within the basis. This script measures, for
hBN monolayers (gth-pade):

  * per-system ov-block dipole statistics (exact vs bare),
  * bright-exciton oscillator strengths f (lambda = 0),
  * Rabi splitting Omega_R at lambda = 0.05 with the cavity tuned to
    the bright exciton (same tuning for both routes),
  * the 2x2 supercell <-> 2x2 k-mesh consistency of the exact route.

Systems: gth-szv Gamma 1x1 / Gamma 2x2 supercell / 2x2 mesh / 3x3 mesh,
and gth-dzvp Gamma 1x1 (basis-convergence indicator).

Run:  python run_qed_velocity_exact_check.py
"""

import json
import os
import sys
import time

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
kpts_examples = os.path.abspath(os.path.join(here, '..', 'qed_kpts'))
sys.path.insert(0, kpts_examples)
from run_qed_bse_hbn_gamma import (EV, build_hbn_cell, run_gamma_rhf,
                                   gamma_df_factor, velocity_gauge_dipole,
                                   build_sq, run_gw_electronic,
                                   run_bse_polaritonic)

from pyscf.pbc import scf as pbcscf

from OmegaQMC.addons.qed_polariton_kpts import (build_kpts_quantities,
                                                velocity_dipole_bare,
                                                run_kgw_electronic,
                                                run_qed_bse_kpts)

LAM = 0.05


def ov_stats(r_ex, r_bp, no, inplane=True):
    """Statistics of exact-vs-bare ov-block dipoles over all k."""
    comps = (0, 1) if inplane else (0, 1, 2)
    mx_e = mx_b = dev = 0.0
    rel = []
    for ke, kb in zip(r_ex, r_bp):
        ex = np.array([ke[x][:no, no:] for x in comps])
        bp = np.array([kb[x][:no, no:] for x in comps])
        mx_e = max(mx_e, float(np.max(np.abs(ex))))
        mx_b = max(mx_b, float(np.max(np.abs(bp))))
        dev = max(dev, float(np.max(np.abs(ex - bp))))
        big = np.abs(ex) > 0.05 * np.max(np.abs(ex))
        rel.append(np.abs(ex - bp)[big] / np.abs(ex)[big])
    rel = np.concatenate(rel)
    return {'max_r_exact': mx_e, 'max_r_bare': mx_b, 'max_abs_dev': dev,
            'mean_rel_dev_pct': float(np.mean(rel) * 100),
            'max_rel_dev_pct': float(np.max(rel) * 100)}


def lp_up(bse):
    coord = bse.get('photon_coordinate_strength', bse['photon_weight'])
    idx = np.argsort(coord)[::-1][:2]
    lo, hi = sorted(float(bse['Omega'][i]) for i in idx)
    return lo, hi, [float(coord[i]) for i in idx]


def summarize(tag, w_cav, ib, bse0, rabi):
    print(f'  [{tag}] bright exciton #{ib}: {w_cav * EV:.3f} eV')
    print(f'  [{tag}]           f(exact) = {bse0["exact"]:8.4f}   '
          f'f(bare) = {bse0["bare"]:8.4f}   '
          f'ratio = {bse0["exact"] / bse0["bare"]:6.3f}')
    print(f'  [{tag}]     Rabi(exact) = {rabi["exact"] * EV:7.3f} eV   '
          f'Rabi(bare) = {rabi["bare"] * EV:7.3f} eV   '
          f'ratio = {rabi["exact"] / rabi["bare"]:6.3f}')


# ---------------------------------------------------------------------------
# Gamma-only systems (1x1, 2x2 supercell; szv + dzvp 1x1)
# ---------------------------------------------------------------------------
def run_gamma_system(nsc, basis, gw_mode='evGW'):
    tag = f'{nsc[0]}x{nsc[1]}-G-{basis}'
    t0 = time.time()
    print(f'\n=== {tag} ===')
    cell = build_hbn_cell(basis=basis, nsc=nsc)
    mf = run_gamma_rhf(cell)
    B_ao = gamma_df_factor(mf)
    no = int(np.count_nonzero(mf.mo_occ > 0))

    # supercell: phase-transported exchange derivative (primitive sub-cell
    # offsets) so the exact route is representation covariant
    cell_prim = build_hbn_cell(basis=basis, nsc=(1, 1)) if nsc != (1, 1) else None
    r_exact = velocity_gauge_dipole(mf, exact=True, cell_prim=cell_prim)
    r_bare = velocity_gauge_dipole(mf, exact=False)
    stats = ov_stats([r_exact], [r_bare], no)
    print(f'  ov dipoles: max|r| exact/bare = {stats["max_r_exact"]:.4f}/'
          f'{stats["max_r_bare"]:.4f}, mean rel dev = '
          f'{stats["mean_rel_dev_pct"]:.1f}%')

    sq0, eps_HF = build_sq(mf, B_ao, r_exact, (0.0, 0.0, 0.0), omega_cav=1.0)
    eps_QP = run_gw_electronic(sq0, eps_HF, mode=gw_mode, verbose=False)

    f0 = {}
    for label, r_eff in (('exact', r_exact), ('bare', r_bare)):
        bse0 = run_bse_polaritonic(sq0, eps_QP, r_eff, lambda_on=False)
        f0[label] = bse0
    Omega0 = f0['exact']['Omega']
    ib = int(np.where(f0['exact']['f_osc'] > 1e-2)[0][0])
    w_cav = float(Omega0[ib])

    fvals = {lab: float(f0[lab]['f_osc'][ib]) for lab in ('exact', 'bare')}
    rabi = {}
    pw = {}
    for label, r_eff in (('exact', r_exact), ('bare', r_bare)):
        sq, _ = build_sq(mf, B_ao, r_eff, (LAM, 0.0, 0.0), w_cav)
        bse = run_bse_polaritonic(sq, eps_QP, r_eff, lambda_on=True)
        lo, hi, w = lp_up(bse)
        rabi[label] = hi - lo
        pw[label] = w
    summarize(tag, w_cav, ib, fvals, rabi)
    return {
        'tag': tag, 'basis': basis, 'ncell': nsc[0] * nsc[1],
        'E_HF': float(mf.e_tot), 'dipole_stats': stats,
        'omega_cav_eV': w_cav * EV, 'bright_index': ib,
        'exciton_eV': [float(x) for x in Omega0[:8] * EV],
        'f_exact': [float(x) for x in f0['exact']['f_osc'][:8]],
        'f_bare': [float(x) for x in f0['bare']['f_osc'][:8]],
        'f_bright': fvals,
        'rabi_eV': {k: v * EV for k, v in rabi.items()},
        'photon_weight': pw,
        't_wall_s': time.time() - t0,
    }


# ---------------------------------------------------------------------------
# k-mesh systems (2x2, 3x3; szv)
# ---------------------------------------------------------------------------
def run_kmesh_system(mesh, basis='gth-szv', gw_mode='evGW'):
    tag = f'{mesh[0]}x{mesh[1]}-k-{basis}'
    t0 = time.time()
    print(f'\n=== {tag} ===')
    cell = build_hbn_cell(basis=basis, nsc=(1, 1))
    kpts = cell.make_kpts([mesh[0], mesh[1], 1])
    kmf = pbcscf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
    kmf.conv_tol = 1e-10
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError('KRHF did not converge')

    kq = build_kpts_quantities(kmf, velocity='exact')
    kq_bare = dict(kq)
    kq_bare['r_eff'] = velocity_dipole_bare(kmf)
    kq_bare['velocity'] = 'bare'
    stats = ov_stats(kq['r_eff'], kq_bare['r_eff'], kq['nocc'])
    print(f'  ov dipoles: max|r| exact/bare = {stats["max_r_exact"]:.4f}/'
          f'{stats["max_r_bare"]:.4f}, mean rel dev = '
          f'{stats["mean_rel_dev_pct"]:.1f}%')

    eps_QP = run_kgw_electronic(kq, mode=gw_mode, verbose=False)

    f0 = {}
    for label, kqi in (('exact', kq), ('bare', kq_bare)):
        f0[label] = run_qed_bse_kpts(kqi, 1.0, (0.0, 0.0, 0.0),
                                     eps_QP=eps_QP, verbose=False)
    Omega0 = f0['exact']['Omega']
    ib = int(np.where(f0['exact']['f_osc'] > 1e-2)[0][0])
    w_cav = float(Omega0[ib])

    fvals = {lab: float(f0[lab]['f_osc'][ib]) for lab in ('exact', 'bare')}
    rabi = {}
    pw = {}
    bse_pol = {}
    for label, kqi in (('exact', kq), ('bare', kq_bare)):
        bse = run_qed_bse_kpts(kqi, w_cav, (LAM, 0.0, 0.0), eps_QP=eps_QP,
                               verbose=False)
        lo, hi, w = lp_up(bse)
        rabi[label] = hi - lo
        pw[label] = w
        bse_pol[label] = bse
    summarize(tag, w_cav, ib, fvals, rabi)
    return {
        'tag': tag, 'basis': basis, 'nk': len(kpts),
        'E_HF': float(kmf.e_tot), 'dipole_stats': stats,
        'omega_cav_eV': w_cav * EV, 'bright_index': ib,
        'exciton_eV': [float(x) for x in Omega0[:8] * EV],
        'f_exact': [float(x) for x in f0['exact']['f_osc'][:8]],
        'f_bare': [float(x) for x in f0['bare']['f_osc'][:8]],
        'f_bright': fvals,
        'rabi_eV': {k: v * EV for k, v in rabi.items()},
        'photon_weight': pw,
        't_wall_s': time.time() - t0,
    }, bse_pol


def main():
    results = {'lambda': LAM, 'systems': []}

    g11 = run_gamma_system((1, 1), 'gth-szv')
    g22 = run_gamma_system((2, 2), 'gth-szv')
    k22, _ = run_kmesh_system((2, 2))
    k33, _ = run_kmesh_system((3, 3))
    g11_dzvp = run_gamma_system((1, 1), 'gth-dzvp')
    results['systems'] = [g11, g22, k22, k33, g11_dzvp]

    # consistency: 2x2 supercell (Gamma) vs 2x2 mesh, exact route
    print('\n=== consistency: 2x2 supercell vs 2x2 mesh (exact route) ===')
    dev_w = abs(g22['omega_cav_eV'] - k22['omega_cav_eV'])
    dev_f = abs(g22['f_bright']['exact'] - k22['f_bright']['exact'])
    dev_r = abs(g22['rabi_eV']['exact'] - k22['rabi_eV']['exact'])
    print(f'  |d omega_cav| = {dev_w:.2e} eV, |d f| = {dev_f:.2e}, '
          f'|d Rabi| = {dev_r:.2e} eV')
    results['consistency_2x2'] = {
        'd_omega_cav_eV': dev_w, 'd_f_bright': dev_f, 'd_rabi_eV': dev_r}

    print('\n=== summary (bright exciton) ===')
    print('  system              f_ex     f_bare   f ratio  '
          'Rabi_ex  Rabi_bare  ratio')
    for s in results['systems']:
        fe, fb = s['f_bright']['exact'], s['f_bright']['bare']
        re_, rb = s['rabi_eV']['exact'], s['rabi_eV']['bare']
        print(f'  {s["tag"]:<18s} {fe:8.4f} {fb:8.4f} {fe / fb:8.3f}  '
              f'{re_:7.3f}  {rb:8.3f} {re_ / rb:7.3f}')

    out = os.path.join(kpts_examples, 'qed_velocity_exact_results.json')
    with open(out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'\nresults saved to {out}')


if __name__ == '__main__':
    main()
