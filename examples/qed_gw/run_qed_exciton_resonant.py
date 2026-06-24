"""Exciton-resonant cavity: water lambda-scan with omega_cav retuned to
the BSE S1 excitation.

The default tuning of the benchmark set (omega_cav = 0.415668 Ha =
11.31 eV) is resonant with the lowest dipole-allowed *RPA* excitation
of water; the correlated TDA-BSE@QED-evGW optical gap lies much lower
(Omega_S1 = 7.98 eV at lambda = 0). Here the cavity is retuned to that
value -- a true exciton-resonant cavity -- and the lambda-scan and the
kernel channel decomposition are repeated. At lambda = 0 the photon
mode is decoupled, so Omega_S1(lambda=0) is tuning-independent and the
retuning requires no self-consistency cycle.

Physics checks of interest:
* does the static folding cancellation (DSE vs photon kernel) survive
  at resonance with the exciton?
* the static BSE has no photon in its excitation space, so even an
  exciton-resonant cavity cannot Rabi-split the BSE root -- the shifts
  must remain smooth lambda^2 curves inherited from the QP channel.

Reads the default-tuning scan from qed_exciton_results.json, writes
qed_exciton_resonant_results.json, and regenerates fig_exciton_h2o.pdf
with both tunings overlaid in the E_b panel.
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
HERE = os.path.dirname(os.path.abspath(__file__))

_h2o_half = math.radians(104.5 / 2.0)
mol = gto.M(atom=[['O', (0.0, 0.0, 0.0)],
                  ['H', (math.sin(_h2o_half), 0.0, -math.cos(_h2o_half))],
                  ['H', (-math.sin(_h2o_half), 0.0, -math.cos(_h2o_half))]],
            basis='cc-pVDZ', unit='Angstrom', symmetry=False, verbose=0)


def bse_full(omega_cav, lam):
    qedhf = run_qed_hf(mol, omega_cav, (0.0, 0.0, lam),
                       verbose=False, tol=1e-12)
    bse = run_qed_bse(qedhf, gw_mode='evGW', tda=True, verbose=False)
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


# ----------------------------------------------------------------------
# Retune: Omega_S1 at lambda = 0 (photon decoupled -> tuning-free).
# ----------------------------------------------------------------------
b0 = bse_full(0.415668, 0.0)
OMEGA_RES = float(b0['Omega_S1'])
print(f"BSE S1 at lambda=0: {OMEGA_RES:.6f} Ha = {OMEGA_RES * EV:.4f} eV")
print(f"-> exciton-resonant cavity: omega_cav = {OMEGA_RES:.6f} Ha")

results = {'omega_cav': OMEGA_RES, 'basis': 'cc-pVDZ',
           'gw_mode': 'evGW', 'tda': True}

# ----------------------------------------------------------------------
# Lambda-scan at the resonant tuning
# ----------------------------------------------------------------------
lams = [0.0, 0.0125, 0.025, 0.0375, 0.05, 0.075, 0.10]
scan = {0.0: summarize(b0)}
print(f"\nH2O lambda-scan, omega_cav = {OMEGA_RES * EV:.3f} eV (eV):")
print(f"{'lam':>6s} {'E_gap':>9s} {'Om_S1':>9s} {'Om_T1':>9s} "
      f"{'E_b':>9s} {'D_ST':>9s}")
bse_cache = {0.0: b0}
for lam in lams:
    if lam not in scan:
        b = bse_full(OMEGA_RES, lam)
        bse_cache[lam] = b
        scan[lam] = summarize(b)
    s = scan[lam]
    print(f"{lam:6.3f} {s['E_gap']:9.4f} {s['Omega_S1']:9.4f} "
          f"{s['Omega_T1']:9.4f} {s['E_b']:9.4f} {s['Delta_ST']:9.4f}")
results['scan_H2O'] = {str(l): v for l, v in scan.items()}

# ----------------------------------------------------------------------
# Channel decomposition at lambda = 0.05, 0.10
# ----------------------------------------------------------------------
print("\nChannel decomposition (kernel ladder at fixed QP, eV):")
results['decomposition_H2O'] = {}
ref0 = scan[0.0]
for lam in (0.05, 0.10):
    b_full = bse_cache[lam]
    qedhf = b_full['qedhf']
    eps_qp = b_full['eps_QP']
    b_qp = run_qed_bse(qedhf, tda=True, eps_QP=eps_qp,
                       include_dse=False, include_photon=False,
                       verbose=False)
    b_dse = run_qed_bse(qedhf, tda=True, eps_QP=eps_qp,
                        include_dse=True, include_photon=False,
                        verbose=False)
    rows = {}
    for key in ('E_gap', 'Omega_S1', 'E_b'):
        full = summarize(b_full)[key]
        qp = {'E_gap': b_qp['E_gap'], 'Omega_S1': b_qp['Omega_S1'],
              'E_b': b_qp['E_b']}[key] * EV
        dse = {'E_gap': b_dse['E_gap'], 'Omega_S1': b_dse['Omega_S1'],
               'E_b': b_dse['E_b']}[key] * EV
        rows[key] = {
            'total_shift': full - ref0[key],
            'QP': float(qp - ref0[key]),
            'DSE': float(dse - qp),
            'photon': float(full - dse),
        }
        r = rows[key]
        print(f"  lam={lam:5.3f}  {key:9s}: total {r['total_shift']:+9.5f} "
              f"= QP {r['QP']:+9.5f} + DSE {r['DSE']:+9.5f} "
              f"+ photon {r['photon']:+9.5f}")
    results['decomposition_H2O'][str(lam)] = rows

with open(os.path.join(HERE, 'qed_exciton_resonant_results.json'),
          'w') as f:
    json.dump(results, f, indent=1)
print("\nwrote qed_exciton_resonant_results.json")

# ----------------------------------------------------------------------
# Combined figure: default tuning (solid) + resonant tuning (dashed)
# ----------------------------------------------------------------------
with open(os.path.join(HERE, 'qed_exciton_results.json')) as f:
    default = json.load(f)
scan_def = {float(l): v for l, v in default['scan_H2O'].items()}
lam_arr = np.array(lams)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.4), dpi=300,
                               sharex=True)
styles = ((scan_def, '-', 1.0), (scan, '--', 0.8))
for key, lab, col, mk in (('E_gap', r'$\delta E_\mathrm{gap}$',
                           '#1b9e77', 'o'),
                          ('Omega_S1', r'$\delta\Omega_{S_1}$',
                           '#d95f02', 's'),
                          ('Omega_T1', r'$\delta\Omega_{T_1}$',
                           '#7570b3', '^')):
    for data, ls, alpha in styles:
        shift = np.array([data[l][key] - data[0.0][key] for l in lams])
        ax1.plot(lam_arr, shift, marker=mk, ms=3.0, lw=1.0, ls=ls,
                 color=col, alpha=alpha,
                 label=lab if ls == '-' else None)
ax1.axhline(0.0, color='k', lw=0.5)
ax1.set_ylabel('shift (eV)')
ax1.legend(fontsize=7, frameon=False)

for data, ls, alpha, lab in (
        (scan_def, '-', 1.0,
         r'$\omega_\mathrm{cav}=11.31$ eV (RPA-resonant)'),
        (scan, '--', 0.8,
         rf'$\omega_\mathrm{{cav}}={OMEGA_RES * EV:.2f}$ eV '
         r'(exciton-resonant)')):
    shift_eb = np.array([data[l]['E_b'] - data[0.0]['E_b'] for l in lams])
    ax2.plot(lam_arr, shift_eb, marker='D', ms=3.0, lw=1.0, ls=ls,
             color='#e7298a', alpha=alpha, label=lab)
ax2.axhline(0.0, color='k', lw=0.5)
ax2.set_xlabel(r'$\lambda$ (a.u.)')
ax2.set_ylabel(r'$\delta E_b$ (eV)')
ax2.legend(fontsize=6.5, frameon=False)
fig.tight_layout()
out = os.path.join(HERE, '..', 'OmegaQMC', 'paper', 'fig_exciton_h2o.pdf')
fig.savefig(os.path.abspath(out))
print(f"wrote {os.path.abspath(out)}")
