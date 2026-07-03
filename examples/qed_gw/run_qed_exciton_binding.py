"""Cavity-modified exciton binding energy from QED-BSE@QED-evGW.

For a molecule in a z-polarized cavity (omega_cav = 0.415668 Ha):

    E_gap(lambda) = eps^QP_LUMO - eps^QP_HOMO     (QED-evGW fundamental gap)
    Omega_S1(lambda) = lowest singlet TDA-BSE root (optical gap)
    E_b(lambda)  = E_gap - Omega_S1               (exciton binding energy)

Three computations:

1. Four-molecule table (H2O, HF, NH3, CH4 / cc-pVDZ) at lambda = 0 and
   0.05, including the lowest triplet Omega_T1 and the singlet-triplet
   gap.
2. Water lambda-scan (fig_exciton_h2o.pdf): cavity-induced shifts of
   E_gap, Omega_S1, Omega_T1 and E_b versus lambda.
3. Per-lambda channel decomposition of the water shifts: the cavity
   enters the BSE through the QP energies (reference channel), the DSE
   d(x)d kernel terms, and the photon-pole channel of W; the three
   increments are resolved at every scan point by toggling the kernel
   channels at fixed cavity QP energies (cumulative ladder
   QP -> +DSE -> +photon), so delta X = QP + DSE + photon for X in
   {E_gap, Omega_S1, E_b}. E_gap depends only on the QP energies, so its
   DSE and photon channels vanish identically (a built-in check).

Results are printed and dumped to qed_exciton_results.json. Two figures
go into OmegaQMC/paper/: fig_exciton_h2o.pdf (the cavity-induced shifts
vs lambda) and fig_exciton_binding_h2o.pdf (the delta E_b QP/DSE/photon
channel breakdown vs lambda).
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyscf import gto

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_bse import run_qed_bse

EV = 27.211386245988
OMEGA = 0.415668
BASIS = 'cc-pVDZ'

_h2o_half = math.radians(104.5 / 2.0)
_nh3_rho, _nh3_z = 0.93786, -0.38129          # r=1.0124 A, HNH=106.67 deg
_ch4_t = 1.087 / math.sqrt(3.0)

GEOMETRIES = {
    'H2O': [['O', (0.0, 0.0, 0.0)],
            ['H', (math.sin(_h2o_half), 0.0, -math.cos(_h2o_half))],
            ['H', (-math.sin(_h2o_half), 0.0, -math.cos(_h2o_half))]],
    'HF':  [['F', (0.0, 0.0, 0.0)],
            ['H', (0.0, 0.0, 0.917)]],
    'NH3': [['N', (0.0, 0.0, 0.0)],
            ['H', (_nh3_rho, 0.0, _nh3_z)],
            ['H', (-0.5 * _nh3_rho, 0.8660254 * _nh3_rho, _nh3_z)],
            ['H', (-0.5 * _nh3_rho, -0.8660254 * _nh3_rho, _nh3_z)]],
    'CH4': [['C', (0.0, 0.0, 0.0)],
            ['H', (_ch4_t, _ch4_t, _ch4_t)],
            ['H', (_ch4_t, -_ch4_t, -_ch4_t)],
            ['H', (-_ch4_t, _ch4_t, -_ch4_t)],
            ['H', (-_ch4_t, -_ch4_t, _ch4_t)]],
}


def build(name):
    return gto.M(atom=GEOMETRIES[name], basis=BASIS, unit='Angstrom',
                 symmetry=False, verbose=0)


def bse_full(name, lam):
    """Full QED-BSE@QED-evGW; returns the result dict (energies in Ha)."""
    qedhf = run_qed_hf(build(name), OMEGA, (0.0, 0.0, lam),
                       verbose=False, tol=1e-12)
    bse = run_qed_bse(qedhf, gw_mode='evGW', tda=False, verbose=False)
    bse['qedhf'] = qedhf
    return bse


def summarize(b):
    return {
        'E_gap': b['E_gap'] * EV,
        'Omega_S1': b['Omega_S1'] * EV,
        'Omega_T1': b['Omega_T1'] * EV,
        'E_b': b['E_b'] * EV,
        'Delta_ST': (b['Omega_S1'] - b['Omega_T1']) * EV,
    }


results = {'omega_cav': OMEGA, 'basis': BASIS, 'gw_mode': 'evGW',
           'tda': False}

# ----------------------------------------------------------------------
# 1. Four-molecule table at lambda = 0 and 0.05
# ----------------------------------------------------------------------
print("=" * 78)
print(f"QED-BSE@QED-evGW exciton binding, {BASIS}, "
      f"omega_cav = {OMEGA} Ha, z-polarized")
print("=" * 78)
header = (f"{'system':8s} {'lam':>5s} {'E_gap':>9s} {'Om_S1':>9s} "
          f"{'Om_T1':>9s} {'E_b':>9s} {'D_ST':>9s}   (eV)")
table = {}
print(header)
for name in ('H2O', 'HF', 'NH3', 'CH4'):
    table[name] = {}
    for lam in (0.0, 0.05):
        b = bse_full(name, lam)
        s = summarize(b)
        table[name][lam] = s
        if lam == 0.05:
            table[name]['bse_at_0.05'] = b      # reused below
        print(f"{name:8s} {lam:5.2f} {s['E_gap']:9.4f} {s['Omega_S1']:9.4f} "
              f"{s['Omega_T1']:9.4f} {s['E_b']:9.4f} {s['Delta_ST']:9.4f}")
    d = {k: table[name][0.05][k] - table[name][0.0][k]
         for k in table[name][0.0]}
    print(f"{'':8s} {'shift':>5s} {d['E_gap']:9.4f} {d['Omega_S1']:9.4f} "
          f"{d['Omega_T1']:9.4f} {d['E_b']:9.4f} {d['Delta_ST']:9.4f}")
results['table'] = {n: {str(l): v for l, v in table[n].items()
                        if not str(l).startswith('bse')}
                    for n in table}

# ----------------------------------------------------------------------
# 2. Water lambda-scan + per-lambda QP/DSE/photon decomposition.
#    Each lambda is solved once; the kernel ladder QP -> +DSE -> +photon
#    reuses the same cavity QP energies (no extra GW).
# ----------------------------------------------------------------------
lams = [0.0, 0.0125, 0.025, 0.0375, 0.05, 0.075, 0.10]
KEYS = ('E_gap', 'Omega_S1', 'E_b')

scan = {}
channels = {key: {} for key in KEYS}     # channels[key][lam] = QP/DSE/photon
ref0 = None
print("\nH2O lambda-scan (eV):")
print(header.replace('system', 'H2O'))
for lam in lams:
    b_full = bse_full('H2O', lam)
    s = summarize(b_full)
    scan[lam] = s
    print(f"{'':8s} {lam:5.3f} {s['E_gap']:9.4f} {s['Omega_S1']:9.4f} "
          f"{s['Omega_T1']:9.4f} {s['E_b']:9.4f} {s['Delta_ST']:9.4f}")
    if lam == 0.0:
        ref0 = s
        for key in KEYS:
            channels[key][0.0] = {'total': 0.0, 'QP': 0.0,
                                  'DSE': 0.0, 'photon': 0.0}
        continue
    qedhf = b_full['qedhf']
    eps_qp = b_full['eps_QP']
    b_qp = run_qed_bse(qedhf, tda=False, eps_QP=eps_qp,
                       include_dse=False, include_photon=False, verbose=False)
    b_dse = run_qed_bse(qedhf, tda=False, eps_QP=eps_qp,
                        include_dse=True, include_photon=False, verbose=False)
    qp_eV = {k: b_qp[k] * EV for k in KEYS}
    dse_eV = {k: b_dse[k] * EV for k in KEYS}
    for key in KEYS:
        channels[key][lam] = {
            'total': s[key] - ref0[key],
            'QP': qp_eV[key] - ref0[key],
            'DSE': dse_eV[key] - qp_eV[key],
            'photon': s[key] - dse_eV[key],
        }
results['scan_H2O'] = {str(l): v for l, v in scan.items()}
results['decomposition_H2O'] = {key: {str(l): channels[key][l] for l in lams}
                                for key in KEYS}

# ----------------------------------------------------------------------
# 3. Channel decomposition table at lambda = 0.05 and 0.10
# ----------------------------------------------------------------------
print("\nChannel decomposition of the water cavity shifts (eV):")
for lam in (0.05, 0.10):
    for key in KEYS:
        r = channels[key][lam]
        print(f"  lam={lam:5.3f}  {key:9s}: total {r['total']:+8.4f} "
              f"= QP {r['QP']:+8.4f} + DSE {r['DSE']:+8.4f} "
              f"+ photon {r['photon']:+8.4f}")

# ----------------------------------------------------------------------
# Figure 1 (unchanged): cavity-induced shifts vs lambda (water)
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.4), dpi=300,
                               sharex=True)
lam_arr = np.array(lams)
for key, lab, col, mk in (('E_gap', r'$\delta E_\mathrm{gap}$',
                           '#1b9e77', 'o'),
                          ('Omega_S1', r'$\delta\Omega_{S_1}$',
                           '#d95f02', 's'),
                          ('Omega_T1', r'$\delta\Omega_{T_1}$',
                           '#7570b3', '^')):
    shift = np.array([scan[l][key] - scan[0.0][key] for l in lams])
    ax1.plot(lam_arr, shift, marker=mk, ms=3.5, lw=1.0, color=col,
             label=lab)
ax1.axhline(0.0, color='k', lw=0.5)
ax1.set_ylabel('shift (eV)')
ax1.legend(fontsize=7, frameon=False)

shift_eb = np.array([scan[l]['E_b'] - scan[0.0]['E_b'] for l in lams])
ax2.plot(lam_arr, shift_eb, marker='D', ms=3.5, lw=1.0, color='#e7298a',
         label=r'$\delta E_b$')
ax2.axhline(0.0, color='k', lw=0.5)
ax2.set_xlabel(r'$\lambda$ (a.u.)')
ax2.set_ylabel(r'$\delta E_b$ (eV)')
ax2.legend(fontsize=7, frameon=False)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'OmegaQMC', 'paper',
                   'fig_exciton_h2o.pdf')
fig.savefig(os.path.abspath(out))
print(f"\nwrote {os.path.abspath(out)}")

# ----------------------------------------------------------------------
# Figure 2 (new): delta E_b QP/DSE/photon channel breakdown (water)
# ----------------------------------------------------------------------
fig2, (bx1, bx2) = plt.subplots(2, 1, figsize=(3.6, 4.6), dpi=300,
                                sharex=True)
for key, lab, col, mk in (('E_gap', r'$\delta E_\mathrm{gap}$',
                           '#1b9e77', 'o'),
                          ('Omega_S1', r'$\delta\Omega_{S_1}$',
                           '#d95f02', 's'),
                          ('E_b', r'$\delta E_b$', '#e7298a', 'D')):
    shift = np.array([scan[l][key] - ref0[key] for l in lams])
    bx1.plot(lam_arr, shift, marker=mk, ms=3.5, lw=1.0, color=col, label=lab)
bx1.axhline(0.0, color='k', lw=0.5)
bx1.set_ylabel('shift (eV)')
bx1.legend(fontsize=7, frameon=False)
bx1.set_title(rf'H$_2$O, $\omega_\mathrm{{cav}}={OMEGA * EV:.2f}$ eV '
              r'(default)', fontsize=8)

for ch, lab, col, mk in (('total', 'total', '#e7298a', 'D'),
                         ('QP', 'QP', '#1b9e77', 'o'),
                         ('DSE', 'DSE', '#d95f02', 's'),
                         ('photon', 'photon', '#7570b3', '^')):
    y = np.array([channels['E_b'][l][ch] for l in lams])
    lw = 1.4 if ch == 'total' else 1.0
    bx2.plot(lam_arr, y, marker=mk, ms=3.5, lw=lw, color=col, label=lab)
bx2.axhline(0.0, color='k', lw=0.5)
bx2.set_xlabel(r'$\lambda$ (a.u.)')
bx2.set_ylabel(r'$\delta E_b$ channels (eV)')
bx2.legend(fontsize=7, frameon=False, ncol=2)
fig2.tight_layout()
out2 = os.path.join(os.path.dirname(__file__), '..', 'OmegaQMC', 'paper',
                    'fig_exciton_binding_h2o.pdf')
fig2.savefig(os.path.abspath(out2))
fig2.savefig(os.path.abspath(out2)[:-4] + '.png')
print(f"wrote {os.path.abspath(out2)}")

with open(os.path.join(os.path.dirname(__file__),
                       'qed_exciton_results.json'), 'w') as f:
    json.dump(results, f, indent=1)
print("wrote qed_exciton_results.json")
