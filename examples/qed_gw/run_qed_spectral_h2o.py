"""HOMO spectral function of H2O/cc-pVDZ in a resonant cavity:
polariton satellites in photoemission.

A_HOMO(omega) = (1/pi) |Im [omega - eps^HF - Sigma_c(omega)]^{-1}| at
the one-shot QED-G0W0 level, for lambda = 0, 0.05, 0.10. The poles of
Sigma_c on the occupied branch sit at eps_i - Omega_m; at lambda = 0
the satellite below the HOMO quasiparticle peak is the usual
"plasmon"-like replica at the bare RPA excitation energy, while at
finite coupling each replica splits in two, tracking the lower/upper
polariton: photoelectron sidebands at the polariton energies, the
cavity analogue of phonon sidebands in ARPES.

Prints the dominant Sigma_c poles (with the photon weight of the RPA
mode that generates them) and the integrated satellite weight; writes
fig_spectral_h2o.pdf into OmegaQMC/paper/ and the raw data to
qed_spectral_results.json.
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
from OmegaQMC.addons.qed_gw import run_qed_gw
from OmegaQMC.addons.qed_spectral import qed_gw_spectral_function

EV = 27.211386245988
OMEGA = 0.415668
ETA = 0.005                      # Ha, ~0.14 eV Lorentzian broadening

half = math.radians(104.5 / 2.0)
hx, hz = math.sin(half), -math.cos(half)
mol = gto.M(atom=[['O', (0, 0, 0)], ['H', (hx, 0, hz)],
                  ['H', (-hx, 0, hz)]],
            basis='cc-pVDZ', unit='Angstrom', symmetry=False, verbose=0)

grid_ev = np.linspace(-32.0, -6.0, 5201)
grid = grid_ev / EV

results = {'omega_cav': OMEGA, 'eta': ETA, 'grid_ev': grid_ev.tolist()}
curves = {}
for lam in (0.0, 0.05, 0.10):
    qedhf = run_qed_hf(mol, OMEGA, (0.0, 0.0, lam), verbose=False,
                       tol=1e-12)
    homo = qedhf['nocc_spatial'] - 1
    gw = run_qed_gw(qedhf, mode='G0W0', orbs=[homo], verbose=False)
    print(f"\nlambda = {lam:.2f}: eps_QP(HOMO) = "
          f"{gw['eps_QP'][homo] * EV:+.3f} eV")
    sp = qed_gw_spectral_function(qedhf, orbs=[homo], omega_grid=grid,
                                  eta=ETA, verbose=False)
    A_ev = sp['A'][0] / EV       # 1/Ha -> 1/eV
    curves[lam] = A_ev

    # Strongest Sigma_c poles inside the satellite window: these are
    # the polariton replicas eps_i - Omega_m.
    sat_poles = [(pos * EV, wgt, m, pw)
                 for pos, wgt, m, pw in sp['poles'][homo]
                 if -30.0 < pos * EV < -18.0][:8]
    print("  satellite-window poles (eV, |M|^2, mode, photon weight):")
    for pos, wgt, m, pw in sat_poles:
        print(f"    {pos:+9.3f}   {wgt:10.3e}   {m:4d}   {pw:8.3f}")

    # Integrated weight: QP region vs satellite region (eV windows).
    mask_qp = (grid_ev > -16) & (grid_ev < -8)
    mask_sat = (grid_ev > -30) & (grid_ev < -18)
    w_qp = float(np.trapezoid(A_ev[mask_qp], grid_ev[mask_qp]))
    w_sat = float(np.trapezoid(A_ev[mask_sat], grid_ev[mask_sat]))
    print(f"  integrated weight: QP window [-16,-8] eV = {w_qp:.4f},  "
          f"satellite window [-30,-18] eV = {w_sat:.4f}")
    results[f'lam_{lam}'] = {
        'eps_QP_ev': gw['eps_QP'][homo] * EV,
        'A_per_ev': A_ev.tolist(),
        'weight_qp_window': w_qp,
        'weight_sat_window': w_sat,
        'poles_satellite_window': sat_poles,
    }

# ----------------------------------------------------------------------
# Figure: QP peak + zoomed satellite region
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(3.4, 2.6), dpi=300, sharey=False,
    gridspec_kw={'width_ratios': [1.25, 1.0], 'wspace': 0.06})
colors = {0.0: '#555555', 0.05: '#d95f02', 0.10: '#7570b3'}
offset_qp, offset_sat = 6.0, 0.0035
for k, lam in enumerate((0.0, 0.05, 0.10)):
    A = curves[lam]
    ax2.plot(grid_ev, A + k * offset_qp, lw=0.9, color=colors[lam])
    ax1.plot(grid_ev, A + k * offset_sat, lw=0.9, color=colors[lam])
    ax1.text(-31.6, k * offset_sat + 0.0012, rf'$\lambda={lam:.2f}$',
             fontsize=7, color=colors[lam])
    # mark the photon-replica pole position eps_i - Omega_m
    for pos, wgt, m, pw in results[f'lam_{lam}']['poles_satellite_window']:
        ax1.plot([pos], [k * offset_sat + 0.0024], marker='v', ms=3,
                 color=colors[lam], clip_on=False)
ax2.set_xlim(-14.5, -10.0)
ax1.set_xlim(-32, -18)
ax1.set_ylim(0.0, 3 * offset_sat + 0.0035)
ax2.set_ylim(-0.5, 2 * offset_qp + 12.0)
ax2.set_yticks([])
ax1.set_yticks([])
ax1.set_title(r'satellite region ($\times\,10^3$)', fontsize=8)
ax2.set_title('QP peak', fontsize=8)
ax1.set_xlabel(r'$\omega$ (eV)', fontsize=9)
ax2.set_xlabel(r'$\omega$ (eV)', fontsize=9)
ax1.set_ylabel(r'$A_\mathrm{HOMO}(\omega)$ (offset)', fontsize=9)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'OmegaQMC', 'paper',
                   'fig_spectral_h2o.pdf')
fig.savefig(os.path.abspath(out))
print(f"\nwrote {os.path.abspath(out)}")

with open(os.path.join(os.path.dirname(__file__),
                       'qed_spectral_results.json'), 'w') as f:
    json.dump(results, f, indent=1)
print("wrote qed_spectral_results.json")
