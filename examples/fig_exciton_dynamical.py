"""Figure: static vs dynamical cavity-induced exciton shifts (water).

Water/cc-pVDZ lambda-scan (omega_cav = 0.415668 Ha, z-polarized) of the
cavity-induced shifts delta_lambda X = X(lambda) - X(0) of the optical
gap Omega_S1 and the exciton binding energy E_b, computed with the
static TDA-BSE kernel and with the renormalized dynamical kernel
(qed_bse_dynamical, perturbative). The dynamical kernel amplifies the
binding shift, breaking the static cavity-transparency.

Writes fig_exciton_dynamical_h2o.{pdf,png} into OmegaQMC/paper/.
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyscf import gto

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_bse_dynamical import run_qed_bse_dynamical

EV = 27.211386245988
OMEGA = 0.415668
_h = math.radians(104.5 / 2.0)
H2O = [['O', (0.0, 0.0, 0.0)],
       ['H', (math.sin(_h), 0.0, -math.cos(_h))],
       ['H', (-math.sin(_h), 0.0, -math.cos(_h))]]

lams = [0.0, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.10]
_cache = os.path.join(os.path.dirname(__file__), 'fig_exciton_dyn_scan.npz')

if os.path.exists(_cache):
    d = np.load(_cache)
    lam = d['lam']
    S1_stat, S1_dyn, Eb_stat, Eb_dyn = (d['S1_stat'], d['S1_dyn'],
                                        d['Eb_stat'], d['Eb_dyn'])
    print(f"loaded cached scan from {_cache}")
else:
    S1_stat, S1_dyn, Eb_stat, Eb_dyn = [], [], [], []
    for lam in lams:
        mol = gto.M(atom=H2O, basis='cc-pVDZ', unit='Angstrom',
                    symmetry=False, verbose=0)
        qedhf = run_qed_hf(mol, OMEGA, (0.0, 0.0, lam), verbose=False,
                           tol=1e-12)
        r = run_qed_bse_dynamical(qedhf, gw_mode='evGW', mode='perturbative',
                                  n_states=30, verbose=False)
        spin = np.asarray(r['spin'])
        s1_stat = float(r['Omega_stat'][spin == 'S'][0]) * EV
        egap = r['E_gap'] * EV
        S1_stat.append(s1_stat)
        S1_dyn.append(r['Omega_S1'] * EV)
        Eb_stat.append(egap - s1_stat)
        Eb_dyn.append(r['E_b'] * EV)
        print(f"lam={lam:6.4f}  S1_stat={s1_stat:8.4f} "
              f"S1_dyn={r['Omega_S1']*EV:8.4f}  Eb_stat={egap-s1_stat:8.4f} "
              f"Eb_dyn={r['E_b']*EV:8.4f} eV")
    lam = np.array(lams)
    S1_stat, S1_dyn = np.array(S1_stat), np.array(S1_dyn)
    Eb_stat, Eb_dyn = np.array(Eb_stat), np.array(Eb_dyn)
    np.savez(_cache, lam=lam, S1_stat=S1_stat, S1_dyn=S1_dyn,
             Eb_stat=Eb_stat, Eb_dyn=Eb_dyn)
dS1_stat = S1_stat - S1_stat[0]
dS1_dyn = S1_dyn - S1_dyn[0]
dEb_stat = Eb_stat - Eb_stat[0]
dEb_dyn = Eb_dyn - Eb_dyn[0]

C_STAT, C_DYN = '#7570b3', '#d95f02'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.8), dpi=300, sharex=True)

ax1.plot(lam, dS1_stat, 'o-', ms=3.5, lw=1.1, color=C_STAT, label='static')
ax1.plot(lam, dS1_dyn, 's--', ms=3.5, lw=1.1, color=C_DYN, label='dynamical')
ax1.axhline(0.0, color='k', lw=0.5)
ax1.set_ylabel(r'$\delta_\lambda\Omega_{S_1}$ (eV)')
ax1.set_title('(a) optical-gap shift', loc='left', fontsize=8.5)
ax1.legend(fontsize=7.5, frameon=False, loc='upper left')

ax2.plot(lam, dEb_stat, 'o-', ms=3.5, lw=1.1, color=C_STAT, label='static')
ax2.plot(lam, dEb_dyn, 's--', ms=3.5, lw=1.1, color=C_DYN, label='dynamical')
ax2.axhline(0.0, color='k', lw=0.5)
ax2.set_xlabel(r'$\lambda$ (a.u.)')
ax2.set_ylabel(r'$\delta_\lambda E_b$ (eV)')
ax2.set_title('(b) exciton-binding shift', loc='left', fontsize=8.5)
ax2.legend(fontsize=7.5, frameon=False, loc='upper left')

fig.tight_layout()
base = os.path.join(os.path.dirname(__file__), '..', 'OmegaQMC', 'paper',
                    'fig_exciton_dynamical_h2o')
fig.savefig(os.path.abspath(base + '.pdf'))
fig.savefig(os.path.abspath(base + '.png'))
print(f"\nwrote {os.path.abspath(base)}.pdf / .png")
print(f"at lam=0.05: dEb static={dEb_stat[lams.index(0.05)]:+.4f} "
      f"dynamical={dEb_dyn[lams.index(0.05)]:+.4f} eV")
print(f"at lam=0.10: dEb static={dEb_stat[-1]:+.4f} "
      f"dynamical={dEb_dyn[-1]:+.4f} eV")
