"""Validate the dimension=2 low-dimensional GDF path of the k-point
polaritonic QED-BSE against the 3D vacuum-slab path in the L_z -> inf
limit (hBN monolayer, gth-szv, 2x2 Gamma-centred mesh).

pyscf's dimension=2 GDF kernel factorises the TRUNCATED 2D Coulomb
operator, which is indefinite: with_df.sr_loop emits sign-split factor
blocks (sign = -1), i.e. (pq|rs) = B.B - Bbar.Bbar. The module carries
the resulting auxiliary metric S = diag(+1...,-1...) through the
polarizability Woodbury kernel, the static W and every BSE
contraction. The 3D vacuum-slab treatment of the same monolayer keeps
the untruncated Coulomb interaction, so periodic images screen
spuriously and observables converge only ~ 1/L_z in the interlayer
distance [Hueser, Olsen, Thygesen, PRB 88, 245309 (2013)]; the
dimension=2 result is the L_z -> inf limit of the slab series.

At a FIXED k-mesh the slab limit is only well defined per observable
class, so the validation runs two tracks:

Track A (charged observables, exxdiv='ewald'): HF and evGW
  quasiparticle gaps. The probe-charge correction is essential here:
  with exxdiv=None the discarded G=0 exchange grows ~ L_z at fixed
  mesh and the slab gap diverges linearly. With it, the slab HF gap
  approaches the dimension=2 value with an exact 1/L_z tail, and the
  evGW gap extrapolates onto the dimension=2 value to a few meV.
  No BSE in this track: the response kernels use the bare DF
  interaction, whose uncorrected q=0 elements grow ~ L_z in the slab,
  so an ewald-referenced slab BSE loses internal consistency as L_z
  grows (by L_z = 40 A it is RPA-unstable) — while the dimension=2
  BSE is stable and height-independent.

Track B (neutral excitations, exxdiv=None throughout): lambda=0
  exciton energies and the polaritonic LP/UP. With one bare DF
  interaction in the reference AND every kernel the setup is
  internally consistent at every L_z, and the dimension=2 results are
  height-independent through the whole stack. The slab series is kept
  as documentation of a negative result: even here the fixed-mesh
  slab excitons do NOT admit a clean L_z -> inf limit — the
  ~L_z-growing q=0 electrostatics cancels only to ~98% between the
  quasiparticle gap and the electron-hole attraction, leaving a
  slowly diverging exciton drift, and the velocity-gauge dipoles
  r_eff = p/dE carry the exxdiv=None dE that grows ~ L_z, so the
  light-matter coupling (and Rabi splitting) is suppressed. At a
  fixed k-mesh, well-defined 2D polaritons exist ONLY on the
  dimension=2 path.

Track C (exact equivalence, exxdiv=None): the sharp correctness test
  of the sign-split BSE contractions. A Gamma-only 2x2 supercell of
  the dimension=2 cell and the 2x2 k-mesh on the dimension=2
  primitive cell describe the same Born-von-Karman system, so E_HF,
  the HF and G0W0 quasiparticle multisets must agree to numerical
  precision, and every k-space polaritonic BSE@HF root must appear in
  the supercell spectrum (the supercell also carries the
  finite-momentum, cavity-dark blocks). Both sides run through the
  new sign-split module, but exercise completely different metric
  code paths (one q = 0 channel with a single indefinite metric vs
  four q-channels with per-q metrics and complex factors). This is
  the dimension=2 analogue of the 3D supercell validation of
  validate_qed_bse_kpts_hbn.py. G0W0 (not evGW) keeps the comparison
  free of eigenvalue-iteration ambiguity; BSE@HF keeps the BSE test
  free of GW noise.

Run:  python validate_qed_bse_kpts_2d.py
"""

import json
import math

import numpy as np

from pyscf.pbc import gto as pbcgto, scf as pbcscf
from pyscf.pbc import tools as pbctools

from OmegaQMC.addons.qed_polariton_kpts import (EV, build_kpts_quantities,
                                                run_kgw_electronic,
                                                run_qed_bse_kpts)

BASIS = 'gth-szv'
MESH = [2, 2, 1]
LAM = 0.05
LZ_GAPS = [15.0, 20.0, 30.0, 40.0, 60.0]     # track A (ewald, gaps)
LZ_BSE = [15.0, 20.0, 30.0, 40.0]            # track B (exxdiv=None)
C_2D = [20.0, 30.0]


def build_hbn_cell(dimension, vacuum, basis=BASIS, verbose=0):
    """hBN monolayer: a = 2.504 A, B at the origin, N at
    (1/3)a1 + (2/3)a2. dimension=3 -> vacuum slab (untruncated
    Coulomb); dimension=2 -> truncated 2D Coulomb (vacuum sets the
    truncation length c)."""
    a = 2.504
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([-a / 2.0, a * math.sqrt(3.0) / 2.0, 0.0])
    a3 = np.array([0.0, 0.0, vacuum])
    r_N = a1 / 3.0 + 2.0 * a2 / 3.0
    cell = pbcgto.Cell()
    cell.a = np.array([a1, a2, a3])
    cell.atom = [('B', (0.0, 0.0, 0.0)), ('N', tuple(r_N))]
    cell.basis = basis
    cell.pseudo = 'gth-pade'
    cell.unit = 'angstrom'
    cell.dimension = dimension
    cell.verbose = verbose
    cell.build()
    return cell


def lp_up(bse):
    idx = np.argsort(bse['photon_weight'])[::-1][:2]
    lo, hi = sorted(float(bse['Omega'][i]) for i in idx)
    return lo, hi, [float(bse['photon_weight'][i]) for i in idx]


def run_electronic(dimension, vacuum, exxdiv, tag='', nsc=None, mesh=None,
                   gw_mode='evGW', gw_max_iter=30):
    """KRHF/GDF -> GW. Returns (record, kq, eps_QP). nsc builds a
    (nsc x nsc) supercell of the primitive cell; mesh overrides the
    k-mesh (default MESH)."""
    print(f'\n--- {tag}: dimension={dimension}, c = {vacuum:.0f} A, '
          f'exxdiv={exxdiv} ---')
    cell = build_hbn_cell(dimension, vacuum)
    if nsc is not None:
        cell = pbctools.super_cell(cell, [nsc, nsc, 1])
        cell.verbose = 0
    kmf = pbcscf.KRHF(cell, kpts=cell.make_kpts(mesh or MESH),
                      exxdiv=exxdiv).density_fit()
    kmf.conv_tol = 1e-10
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError('KRHF did not converge')
    # machinery validation is pinned to the bare velocity route: both
    # sides of every comparison use identical r_eff = i p / dE dipoles
    kq = build_kpts_quantities(kmf, velocity='bare')
    no = kq['nocc']
    homo = max(float(e[no - 1]) for e in kq['eps'])
    lumo = min(float(e[no]) for e in kq['eps'])
    eps_QP = run_kgw_electronic(kq, mode=gw_mode, max_iter=gw_max_iter)
    homo_qp = max(float(e[no - 1]) for e in eps_QP)
    lumo_qp = min(float(e[no]) for e in eps_QP)
    rec = {
        'dimension': dimension, 'vacuum_A': vacuum, 'exxdiv': exxdiv,
        'E_HF': float(kmf.e_tot),
        'gap_HF_eV': (lumo - homo) * EV,
        'gap_GW_eV': (lumo_qp - homo_qp) * EV,
    }
    print(f'  HF gap = {rec["gap_HF_eV"]:.4f} eV, '
          f'evGW gap = {rec["gap_GW_eV"]:.4f} eV')
    return rec, kq, eps_QP


def add_excitations(rec, kq, eps_QP, w_cav=None):
    """lambda=0 BSE (bright exciton) and, given w_cav, the polaritonic
    BSE at coupling LAM."""
    bse0 = run_qed_bse_kpts(kq, 1.0, (0.0, 0.0, 0.0), eps_QP=eps_QP,
                            verbose=False)
    f0 = bse0['f_osc']
    ib = int(np.where(f0 > 1e-2)[0][0])    # lowest bright state
    rec.update({
        'exciton_eV': [float(x) * EV for x in bse0['Omega'][:4]],
        'exciton_f': [float(x) for x in f0[:4]],
        'bright_eV': float(bse0['Omega'][ib]) * EV,
        'bright_f': float(f0[ib]),
    })
    print(f'  bright exciton = {rec["bright_eV"]:.4f} eV '
          f'(f = {rec["bright_f"]:.4f})')
    if w_cav is not None:
        bse = run_qed_bse_kpts(kq, w_cav, (LAM, 0.0, 0.0), eps_QP=eps_QP,
                               verbose=False)
        lo, hi, pw = lp_up(bse)
        rec.update({'LP_eV': lo * EV, 'UP_eV': hi * EV,
                    'rabi_eV': (hi - lo) * EV, 'photon_weight': pw})
        print(f'  LP = {rec["LP_eV"]:.4f} eV, UP = {rec["UP_eV"]:.4f} eV, '
              f'Rabi = {rec["rabi_eV"]:.4f} eV')
    return rec


def extrap_1_over_L(Lz, y, npts=3):
    """Linear fit y = y_inf + c / L_z over the last `npts` points."""
    x = 1.0 / np.asarray(Lz[-npts:])
    coef = np.polyfit(x, np.asarray(y[-npts:]), 1)
    return float(coef[1]), float(coef[0])   # y_inf, slope (eV * A)


def report_series(name, Lz, series, val2d, npts=3):
    y_inf, slope = extrap_1_over_L(Lz, series, npts)
    dev = abs(y_inf - val2d)
    print(f'  {name:12s} ' + ''.join(f'{v:9.4f}' for v in series)
          + f' | extrap {y_inf:9.4f}  2D {val2d:9.4f}  '
          f'|dev| {dev:.2e} eV  slope {slope:7.3f} eV.A')
    return {'series_eV': series, 'extrap_eV': y_inf, 'slope_eVA': slope,
            'val2d_eV': val2d, 'dev_eV': dev}


def subset_dev(sub, full):
    """max over `sub` of the distance to the nearest element of `full`."""
    sub = np.asarray(sub)
    full = np.asarray(full)
    return float(max(np.min(np.abs(full - x)) for x in sub))


def run_track_c():
    """Exact equivalence: Gamma-only 2x2 supercell of the dimension=2
    cell vs 2x2 k-mesh on the dimension=2 primitive cell (same BvK
    system). G0W0 + polaritonic BSE@HF, both through the sign-split
    module."""
    print('\n===== track C: dimension=2 supercell equivalence =====')
    rsc, kqsc, qpsc = run_electronic(
        2, C_2D[0], None, tag='2D supercell 2x2, Gamma-only',
        nsc=2, mesh=[1, 1, 1], gw_mode='G0W0')
    rk, kqk, qpk = run_electronic(
        2, C_2D[0], None, tag='2D primitive, 2x2 k-mesh',
        gw_mode='G0W0')

    nk = kqk['nk']
    dev_E = abs(rsc['E_HF'] - nk * rk['E_HF'])
    eps_sc = np.sort(np.concatenate(kqsc['eps']))
    eps_k = np.sort(np.concatenate(kqk['eps']))
    dev_hf = float(np.max(np.abs(eps_sc - eps_k)))
    qp_sc = np.sort(np.concatenate(qpsc))
    qp_k = np.sort(np.concatenate(qpk))
    dqp = np.abs(qp_sc - qp_k)
    # far-from-gap states can have several solutions of the QP
    # equation; the secant may select different (equally valid) roots
    # on the two sides, so count matching bands rather than max only
    n_match = int(np.sum(dqp < 1e-6))
    no_sc, no_k = kqsc['nocc'], kqk['nocc']
    homo_s = max(float(e[no_sc - 1]) for e in qpsc)
    lumo_s = min(float(e[no_sc]) for e in qpsc)
    homo_k = max(float(e[no_k - 1]) for e in qpk)
    lumo_k = min(float(e[no_k]) for e in qpk)
    dev_qp_fr = max(abs(homo_s - homo_k), abs(lumo_s - lumo_k))
    print(f'  |E_HF(sc) - Nk*E_HF(k)|      = {dev_E:.3e} Ha')
    print(f'  max |eps_HF(sc) - eps_HF(k)| = {dev_hf:.3e} Ha')
    print(f'  G0W0: HOMO/LUMO dev = {dev_qp_fr:.3e} Ha; '
          f'{n_match}/{len(dqp)} bands < 1e-6 Ha '
          f'(rest: secant root selection), max = {np.max(dqp):.3e} Ha')

    # cavity tuned to the lowest bright BSE@HF exciton of the k-mesh
    bse0 = run_qed_bse_kpts(kqk, 1.0, (0.0, 0.0, 0.0), verbose=False)
    ib = int(np.where(bse0['f_osc'] > 1e-2)[0][0])
    w_cav = float(bse0['Omega'][ib])
    devs = {'E_Ha': dev_E, 'eps_HF_Ha': dev_hf,
            'eps_QP_frontier_Ha': dev_qp_fr,
            'eps_QP_bands_matched': [n_match, len(dqp)],
            'eps_QP_max_Ha': float(np.max(dqp)),
            'omega_cav_eV': w_cav * EV}
    for kind, tda in (('tda', True), ('full', False)):
        bk = run_qed_bse_kpts(kqk, w_cav, (LAM, 0.0, 0.0), tda=tda,
                              verbose=False)
        bs = run_qed_bse_kpts(kqsc, w_cav, (LAM, 0.0, 0.0), tda=tda,
                              verbose=False)
        dev_sub = subset_dev(bk['Omega'], bs['Omega'])
        lo_s, hi_s, pw_s = lp_up(bs)
        lo_k, hi_k, pw_k = lp_up(bk)
        print(f'  [{kind}] max_k min_sc |dOmega|   = {dev_sub:.3e} Ha')
        print(f'  [{kind}] LP: sc {lo_s * EV:8.4f} / k {lo_k * EV:8.4f} eV, '
              f'UP: sc {hi_s * EV:8.4f} / k {hi_k * EV:8.4f} eV')
        print(f'  [{kind}] photon weights: sc {pw_s[0]:.4f}/{pw_s[1]:.4f}, '
              f'k {pw_k[0]:.4f}/{pw_k[1]:.4f}')
        devs[kind] = {'subset_Ha': dev_sub,
                      'LP_eV': abs(lo_s - lo_k) * EV,
                      'UP_eV': abs(hi_s - hi_k) * EV}
    return devs


def main():
    out = {'basis': BASIS, 'mesh': MESH, 'lambda': LAM}

    out['trackC'] = run_track_c()

    # ---- track A: quasiparticle gaps, probe-charge corrected --------
    print('\n===== track A: HF/evGW gaps (exxdiv=ewald) =====')
    a2d = [run_electronic(2, c, 'ewald', tag=f'2D truncated, c = {c:.0f}')[0]
           for c in C_2D]
    aslab = [run_electronic(3, lz, 'ewald', tag=f'3D slab, L_z = {lz:.0f}')[0]
             for lz in LZ_GAPS]

    print('\n=== track A summary: L_z(A) = '
          + ', '.join(f'{lz:.0f}' for lz in LZ_GAPS) + ' ===')
    devs_c_A = {k: abs(a2d[0][k] - a2d[1][k])
                for k in ('gap_HF_eV', 'gap_GW_eV')}
    print(f'  2D height independence: d(HF gap) = '
          f'{devs_c_A["gap_HF_eV"]:.2e} eV, d(GW gap) = '
          f'{devs_c_A["gap_GW_eV"]:.2e} eV  (c = 20 vs 30 A)')
    extr_A = {}
    for k in ('gap_HF_eV', 'gap_GW_eV'):
        extr_A[k] = report_series(k, LZ_GAPS, [s[k] for s in aslab],
                                  a2d[0][k])
    out['trackA'] = {'dim2': a2d, 'slab_Lz_A': LZ_GAPS, 'slab': aslab,
                     'dim2_height_devs_eV': devs_c_A,
                     'extrapolation': extr_A}

    # ---- track B: neutral excitations, one bare DF interaction ------
    print('\n===== track B: excitons & polaritons (exxdiv=None) =====')
    rec, kq, eqp = run_electronic(2, C_2D[0], None,
                                  tag=f'2D truncated, c = {C_2D[0]:.0f}')
    rec = add_excitations(rec, kq, eqp)
    w_cav = rec['bright_eV'] / EV
    b2d = [add_excitations(rec, kq, eqp, w_cav=w_cav)]
    rec, kq, eqp = run_electronic(2, C_2D[1], None,
                                  tag=f'2D truncated, c = {C_2D[1]:.0f}')
    b2d.append(add_excitations(rec, kq, eqp, w_cav=w_cav))
    bslab = []
    for lz in LZ_BSE:
        rec, kq, eqp = run_electronic(3, lz, None,
                                      tag=f'3D slab, L_z = {lz:.0f}')
        bslab.append(add_excitations(rec, kq, eqp, w_cav=w_cav))

    print('\n=== track B summary: L_z(A) = '
          + ', '.join(f'{lz:.0f}' for lz in LZ_BSE) + ' ===')
    keys_c = ['gap_HF_eV', 'gap_GW_eV', 'bright_eV', 'LP_eV', 'UP_eV',
              'rabi_eV']
    devs_c_B = {k: abs(b2d[0][k] - b2d[1][k]) for k in keys_c}
    for k in keys_c:
        print(f'  2D height independence: d({k}) = {devs_c_B[k]:.2e} eV')
    # documented NEGATIVE result: fixed-mesh slab excitons/polaritons
    # have no clean L_z -> inf limit (residual electrostatic drift;
    # coupling suppressed through the exxdiv=None velocity-gauge dE)
    print('  fixed-mesh slab series (no L_z limit — see docstring):')
    for k in ('bright_eV', 'bright_f', 'rabi_eV', 'LP_eV', 'UP_eV'):
        print(f'  slab {k:9s}: '
              + ''.join(f'{s[k]:9.4f}' for s in bslab)
              + f'   [2D: {b2d[0][k]:9.4f}]')
    out['trackB'] = {'dim2': b2d, 'slab_Lz_A': LZ_BSE, 'slab': bslab,
                     'dim2_height_devs_eV': devs_c_B,
                     'omega_cav_eV': w_cav * EV}

    with open('qed_bse_kpts_2d_validation.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('\nresults saved to qed_bse_kpts_2d_validation.json')


if __name__ == '__main__':
    main()
