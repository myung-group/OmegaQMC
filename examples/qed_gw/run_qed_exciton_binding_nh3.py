"""Cavity-induced exciton-binding shift of NH3, decomposed into the
QP / DSE / photon channels of QED-BSE@QED-evGW.

For ammonia (cc-pVDZ) in a z-polarized cavity,

    E_gap(lambda)    = eps^QP_LUMO - eps^QP_HOMO     (QED-evGW fundamental gap)
    Omega_S1(lambda) = lowest singlet TDA-BSE root   (optical gap)
    E_b(lambda)      = E_gap - Omega_S1              (exciton binding energy)

the cavity-induced shift delta E_b(lambda) = E_b(lambda) - E_b(0) is split
into the three channels through which the cavity enters the BSE:

    QP     -- the cavity-dressed evGW quasiparticle energies (BSE diagonal
              and screening evaluated at eps^QP(lambda), bare kernel),
    DSE    -- the dipole self-energy d(x)d terms of the BSE kernel,
    photon -- the photon-pole channel of the screened interaction W.

They are resolved by toggling the kernel channels at FIXED cavity QP
energies (the cumulative ladder QP -> +DSE -> +photon), so that

    delta X(lambda) = QP + DSE + photon ,  X in {E_gap, Omega_S1, E_b}.

E_gap depends only on the QP energies, so its DSE and photon channels
vanish identically (a built-in check).

The cavity is tuned to the exciton resonance -- omega_cav = Omega_S1 of
the bare (lambda = 0) BSE, where the photon is decoupled and the tuning is
self-consistency-free -- so the photon channel is probed at resonance,
where it is largest. The static (TDA) BSE has no photon in its excitation
space, so even at resonance the S1 root cannot Rabi-split: the shifts stay
smooth lambda^2 curves and delta E_b is well defined.

Prints the lambda-scan and the channel decomposition, writes
qed_exciton_binding_nh3_results.json, and a two-panel figure
fig_exciton_binding_nh3.pdf into OmegaQMC/paper/.
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
BASIS = 'cc-pVDZ'
HERE = os.path.dirname(os.path.abspath(__file__))

_rho, _z = 0.93786, -0.38129          # r = 1.0124 A, HNH = 106.67 deg
mol = gto.M(atom=[['N', (0.0, 0.0, 0.0)],
                  ['H', (_rho, 0.0, _z)],
                  ['H', (-0.5 * _rho, 0.8660254 * _rho, _z)],
                  ['H', (-0.5 * _rho, -0.8660254 * _rho, _z)]],
            basis=BASIS, unit='Angstrom', symmetry=False, verbose=0)


def bse_full(omega_cav, lam):
    """Full QED-BSE@QED-evGW; returns the result dict (energies in Ha)."""
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
# Tune the cavity to the exciton resonance (lambda = 0 -> tuning-free).
# ----------------------------------------------------------------------
b0 = bse_full(0.415668, 0.0)
OMEGA_RES = float(b0['Omega_S1'])
print(f"NH3 BSE S1 at lambda=0: {OMEGA_RES:.6f} Ha = {OMEGA_RES * EV:.4f} eV")
print(f"-> exciton-resonant cavity: omega_cav = {OMEGA_RES:.6f} Ha")

results = {'system': 'NH3', 'omega_cav': OMEGA_RES, 'basis': BASIS,
           'gw_mode': 'evGW', 'tda': True}

# ----------------------------------------------------------------------
# Lambda-scan + per-lambda QP/DSE/photon decomposition.
# ----------------------------------------------------------------------
lams = [0.0, 0.0125, 0.025, 0.0375, 0.05, 0.075, 0.10]
KEYS = ('E_gap', 'Omega_S1', 'E_b')

scan = {0.0: summarize(b0)}
# channels[key][lam] = dict(total, QP, DSE, photon); lambda=0 -> all zero.
channels = {key: {0.0: {'total': 0.0, 'QP': 0.0, 'DSE': 0.0, 'photon': 0.0}}
            for key in KEYS}
ref0 = scan[0.0]

print(f"\nNH3 lambda-scan, omega_cav = {OMEGA_RES * EV:.3f} eV (eV):")
print(f"{'lam':>6s} {'E_gap':>9s} {'Om_S1':>9s} {'Om_T1':>9s} "
      f"{'E_b':>9s} {'D_ST':>9s}")
s = scan[0.0]
print(f"{0.0:6.3f} {s['E_gap']:9.4f} {s['Omega_S1']:9.4f} "
      f"{s['Omega_T1']:9.4f} {s['E_b']:9.4f} {s['Delta_ST']:9.4f}")

for lam in lams[1:]:
    b_full = bse_full(OMEGA_RES, lam)
    qedhf = b_full['qedhf']
    eps_qp = b_full['eps_QP']
    # Cumulative kernel ladder at the cavity QP energies.
    b_qp = run_qed_bse(qedhf, tda=True, eps_QP=eps_qp,
                       include_dse=False, include_photon=False, verbose=False)
    b_dse = run_qed_bse(qedhf, tda=True, eps_QP=eps_qp,
                        include_dse=True, include_photon=False, verbose=False)
    s = summarize(b_full)
    scan[lam] = s
    print(f"{lam:6.3f} {s['E_gap']:9.4f} {s['Omega_S1']:9.4f} "
          f"{s['Omega_T1']:9.4f} {s['E_b']:9.4f} {s['Delta_ST']:9.4f}")

    val = {'full': s,
           'qp': {k: b_qp[k] * EV for k in KEYS},
           'dse': {k: b_dse[k] * EV for k in KEYS}}
    for key in KEYS:
        full = val['full'][key]
        qp = val['qp'][key]
        dse = val['dse'][key]
        channels[key][lam] = {
            'total': full - ref0[key],
            'QP': qp - ref0[key],
            'DSE': dse - qp,
            'photon': full - dse,
        }

results['scan'] = {str(l): v for l, v in scan.items()}
results['decomposition'] = {key: {str(l): channels[key][l] for l in lams}
                            for key in KEYS}

# ----------------------------------------------------------------------
# Decomposition table (mirrors run_qed_exciton_binding / _resonant).
# ----------------------------------------------------------------------
print("\nChannel decomposition of the NH3 cavity shifts (eV):")
for lam in (0.05, 0.10):
    for key in KEYS:
        r = channels[key][lam]
        print(f"  lam={lam:5.3f}  {key:9s}: total {r['total']:+8.4f} "
              f"= QP {r['QP']:+8.4f} + DSE {r['DSE']:+8.4f} "
              f"+ photon {r['photon']:+8.4f}")

with open(os.path.join(HERE, 'qed_exciton_binding_nh3_results.json'),
          'w') as f:
    json.dump(results, f, indent=1)
print("\nwrote qed_exciton_binding_nh3_results.json")

# ----------------------------------------------------------------------
# Figure: shifts vs lambda (top) and delta E_b channel breakdown (bottom).
# ----------------------------------------------------------------------
lam_arr = np.array(lams)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.6, 4.6), dpi=300,
                               sharex=True)

for key, lab, col, mk in (('E_gap', r'$\delta E_\mathrm{gap}$',
                           '#1b9e77', 'o'),
                          ('Omega_S1', r'$\delta\Omega_{S_1}$',
                           '#d95f02', 's'),
                          ('E_b', r'$\delta E_b$', '#e7298a', 'D')):
    shift = np.array([scan[l][key] - ref0[key] for l in lams])
    ax1.plot(lam_arr, shift, marker=mk, ms=3.5, lw=1.0, color=col, label=lab)
ax1.axhline(0.0, color='k', lw=0.5)
ax1.set_ylabel('shift (eV)')
ax1.legend(fontsize=7, frameon=False)
ax1.set_title(r'NH$_3$, $\omega_\mathrm{cav}='
              rf'{OMEGA_RES * EV:.2f}$ eV (exciton-resonant)', fontsize=8)

for ch, lab, col, mk in (('total', r'total', '#e7298a', 'D'),
                         ('QP', r'QP', '#1b9e77', 'o'),
                         ('DSE', r'DSE', '#d95f02', 's'),
                         ('photon', r'photon', '#7570b3', '^')):
    y = np.array([channels['E_b'][l][ch] for l in lams])
    lw = 1.4 if ch == 'total' else 1.0
    ax2.plot(lam_arr, y, marker=mk, ms=3.5, lw=lw, color=col, label=lab)
ax2.axhline(0.0, color='k', lw=0.5)
ax2.set_xlabel(r'$\lambda$ (a.u.)')
ax2.set_ylabel(r'$\delta E_b$ channels (eV)')
ax2.legend(fontsize=7, frameon=False, ncol=2)
fig.tight_layout()
out = os.path.join(HERE, '..', 'OmegaQMC', 'paper',
                   'fig_exciton_binding_nh3.pdf')
fig.savefig(os.path.abspath(out))
fig.savefig(os.path.abspath(out)[:-4] + '.png')
print(f"wrote {os.path.abspath(out)}")
