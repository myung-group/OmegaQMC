"""Apples-to-apples comparison of the cavity-induced exciton-binding
shift in H2O and NH3, each at ITS OWN exciton resonance.

For each molecule (cc-pVDZ, z-polarized cavity) the cavity is tuned to the
bare BSE S1 (omega_cav = Omega_S1 at lambda = 0, where the photon is
decoupled and the tuning is self-consistency-free), and delta E_b(lambda)
is decomposed into the QP / DSE / photon channels of QED-BSE@QED-evGW by
the cumulative kernel ladder QP -> +DSE -> +photon at fixed cavity QP
energies (see run_qed_exciton_binding[_nh3].py):

    delta X(lambda) = QP + DSE + photon ,  X in {E_gap, Omega_S1, E_b}.

Putting both molecules ON their S1 resonance removes the cavity-tuning
dependence, so any difference in the binding-shift channels is intrinsic
to the electronic structure. (It also tests whether H2O's near-exact
DSE/photon cancellation, seen at the off-resonant default cavity in
run_qed_exciton_binding.py, survives at resonance.)

Prints a side-by-side table, writes qed_exciton_binding_compare_results.json
and a 2x2 figure fig_exciton_binding_compare.pdf into OmegaQMC/paper/
(rows: shifts [delta E_gap, Omega_S1, Omega_T1, E_b] / delta E_b channels;
columns: H2O / NH3). The water shift panel overlays the default-cavity
(11.31 eV) scan as dashed curves on the S1-resonant solid curves, showing
the <=2% detuning insensitivity of the static BSE.
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyscf import gto

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_bse import run_qed_bse

EV = 27.211386245988
BASIS = 'cc-pVDZ'
HERE = os.path.dirname(os.path.abspath(__file__))

_h2o_half = math.radians(104.5 / 2.0)
_nh3_rho, _nh3_z = 0.93786, -0.38129
GEOMETRIES = {
    'H2O': [['O', (0.0, 0.0, 0.0)],
            ['H', (math.sin(_h2o_half), 0.0, -math.cos(_h2o_half))],
            ['H', (-math.sin(_h2o_half), 0.0, -math.cos(_h2o_half))]],
    'NH3': [['N', (0.0, 0.0, 0.0)],
            ['H', (_nh3_rho, 0.0, _nh3_z)],
            ['H', (-0.5 * _nh3_rho, 0.8660254 * _nh3_rho, _nh3_z)],
            ['H', (-0.5 * _nh3_rho, -0.8660254 * _nh3_rho, _nh3_z)]],
}
LAMS = [0.0, 0.0125, 0.025, 0.0375, 0.05, 0.075, 0.10]
KEYS = ('E_gap', 'Omega_S1', 'E_b')
DEFAULT_OMEGA = 0.415668          # Ha = 11.31 eV, the RPA-resonant cavity

# Shift curves shown in the top panels: (key, label, colour, marker).
SHIFT_SPECS = (('E_gap', r'$\delta E_\mathrm{gap}$', '#1b9e77', 'o'),
               ('Omega_S1', r'$\delta\Omega_{S_1}$', '#d95f02', 's'),
               ('Omega_T1', r'$\delta\Omega_{T_1}$', '#7570b3', '^'),
               ('E_b', r'$\delta E_b$', '#e7298a', 'D'))


def build(name):
    mol = gto.M(atom=GEOMETRIES[name], basis=BASIS, unit='Angstrom',
                symmetry=False, verbose=0)
    # Recenter at the nuclear centre of mass (paper convention): the
    # QP energies entering the BSE are origin-dependent at lambda > 0.
    masses = mol.atom_mass_list()
    coords = mol.atom_coords()                      # Bohr
    com = masses @ coords / masses.sum()
    mol.set_geom_(coords - com, unit='Bohr', symmetry=False)
    return mol


def bse_at(mol, omega_cav, lam):
    qedhf = run_qed_hf(mol, omega_cav, (0.0, 0.0, lam),
                       verbose=False, tol=1e-12)
    return run_qed_bse(qedhf, gw_mode='evGW', tda=False, verbose=False)


def scan_shifts(mol, omega_cav):
    """{lambda: summarize} at a fixed cavity (shift curves only)."""
    return {lam: summarize(bse_at(mol, omega_cav, lam)) for lam in LAMS}


def summarize(b):
    return {
        'E_gap': b['E_gap'] * EV,
        'Omega_S1': b['Omega_S1'] * EV,
        'Omega_T1': b['Omega_T1'] * EV,
        'E_b': b['E_b'] * EV,
    }


def resonant_decomposition(name):
    """Scan + per-lambda QP/DSE/photon channels at the S1-resonant cavity."""
    mol = build(name)
    # Tune to the exciton resonance (lambda = 0 -> tuning-free).
    b0 = run_qed_bse(run_qed_hf(mol, 0.415668, (0.0, 0.0, 0.0),
                                verbose=False, tol=1e-12),
                     gw_mode='evGW', tda=False, verbose=False)
    omega_res = float(b0['Omega_S1'])

    scan = {0.0: summarize(b0)}
    channels = {k: {0.0: {'total': 0.0, 'QP': 0.0, 'DSE': 0.0, 'photon': 0.0}}
                for k in KEYS}
    ref0 = scan[0.0]
    for lam in LAMS[1:]:
        qedhf = run_qed_hf(mol, omega_res, (0.0, 0.0, lam),
                           verbose=False, tol=1e-12)
        b_full = run_qed_bse(qedhf, gw_mode='evGW', tda=False, verbose=False)
        eps_qp = b_full['eps_QP']
        b_qp = run_qed_bse(qedhf, tda=False, eps_QP=eps_qp,
                           include_dse=False, include_photon=False,
                           verbose=False)
        b_dse = run_qed_bse(qedhf, tda=False, eps_QP=eps_qp,
                            include_dse=True, include_photon=False,
                            verbose=False)
        s = summarize(b_full)
        scan[lam] = s
        qp_eV = {k: b_qp[k] * EV for k in KEYS}
        dse_eV = {k: b_dse[k] * EV for k in KEYS}
        for k in KEYS:
            channels[k][lam] = {
                'total': s[k] - ref0[k],
                'QP': qp_eV[k] - ref0[k],
                'DSE': dse_eV[k] - qp_eV[k],
                'photon': s[k] - dse_eV[k],
            }
    return {'omega_cav': omega_res, 'scan': scan, 'channels': channels}


data = {}
for name in ('H2O', 'NH3'):
    print(f"... {name}: tuning to S1 + resonant lambda-scan with decomposition")
    data[name] = resonant_decomposition(name)
    print(f"    {name} S1 (lambda=0): omega_cav = "
          f"{data[name]['omega_cav'] * EV:.4f} eV")

# Water also at the default (RPA-resonant) cavity, for the detuning overlay.
print("... H2O: default-cavity scan (11.31 eV) for the detuning overlay")
data['H2O']['default_scan'] = scan_shifts(build('H2O'), DEFAULT_OMEGA)

# ----------------------------------------------------------------------
# Side-by-side table
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("delta E_b channel decomposition at the S1-resonant cavity (eV)")
print("=" * 72)
print(f"{'mol':5s} {'w_cav':>8s} {'lam':>6s} {'dE_b':>9s} "
      f"{'QP':>9s} {'DSE':>9s} {'photon':>9s}")
for name in ('H2O', 'NH3'):
    oc = data[name]['omega_cav'] * EV
    for lam in (0.05, 0.10):
        r = data[name]['channels']['E_b'][lam]
        print(f"{name:5s} {oc:8.3f} {lam:6.3f} {r['total']:+9.4f} "
              f"{r['QP']:+9.4f} {r['DSE']:+9.4f} {r['photon']:+9.4f}")

out_json = {
    name: {
        'omega_cav': data[name]['omega_cav'],
        'scan': {str(l): v for l, v in data[name]['scan'].items()},
        'decomposition': {k: {str(l): data[name]['channels'][k][l]
                              for l in LAMS} for k in KEYS},
    } for name in ('H2O', 'NH3')
}
out_json['H2O']['default_scan'] = {
    str(l): v for l, v in data['H2O']['default_scan'].items()}
with open(os.path.join(HERE, 'qed_exciton_binding_compare_results.json'),
          'w') as f:
    json.dump(out_json, f, indent=1)
print("\nwrote qed_exciton_binding_compare_results.json")

# ----------------------------------------------------------------------
# Figure: rows = shifts / delta E_b channels; columns = H2O / NH3.
# Shared y per row -> directly comparable magnitudes.
# ----------------------------------------------------------------------
lam_arr = np.array(LAMS)
fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), dpi=300,
                         sharex=True, sharey='row')
for j, name in enumerate(('H2O', 'NH3')):
    scan = data[name]['scan']
    channels = data[name]['channels']
    ref0 = scan[0.0]
    oc = data[name]['omega_cav'] * EV
    ax_top, ax_bot = axes[0, j], axes[1, j]

    for key, lab, col, mk in SHIFT_SPECS:
        y = np.array([scan[l][key] - ref0[key] for l in LAMS])
        ax_top.plot(lam_arr, y, marker=mk, ms=3.3, lw=1.0, color=col,
                    label=lab)
    if name == 'H2O':
        # Overlay the default-cavity (11.31 eV) scan as dashed curves: the
        # two tunings coincide to <=2% (static-BSE detuning insensitivity).
        dscan = data['H2O']['default_scan']
        dref = dscan[0.0]
        for key, _lab, col, _mk in SHIFT_SPECS:
            yd = np.array([dscan[l][key] - dref[key] for l in LAMS])
            ax_top.plot(lam_arr, yd, ls='--', lw=1.0, color=col, alpha=0.8)
    ax_top.axhline(0.0, color='k', lw=0.5)
    sym = r'H$_2$O' if name == 'H2O' else r'NH$_3$'
    ax_top.set_title(rf'{sym}, $\omega_{{S_1}}={oc:.2f}$ eV', fontsize=8.5)

    for ch, lab, col, mk in (('total', 'total', '#e7298a', 'D'),
                             ('QP', 'QP', '#1b9e77', 'o'),
                             ('DSE', 'DSE', '#d95f02', 's'),
                             ('photon', 'photon', '#7570b3', '^')):
        y = np.array([channels['E_b'][l][ch] for l in LAMS])
        lw = 1.4 if ch == 'total' else 1.0
        ax_bot.plot(lam_arr, y, marker=mk, ms=3.3, lw=lw, color=col,
                    label=lab)
    ax_bot.axhline(0.0, color='k', lw=0.5)
    ax_bot.set_xlabel(r'$\lambda$ (a.u.)')
    if j == 0:
        ax_top.set_ylabel('shift (eV)')
        ax_bot.set_ylabel(r'$\delta E_b$ channels (eV)')
        q_leg = ax_top.legend(fontsize=7, frameon=False, loc='upper left')
        ax_top.add_artist(q_leg)
        tuning_handles = [
            Line2D([0], [0], color='0.35', ls='-', lw=1.2, label='resonant'),
            Line2D([0], [0], color='0.35', ls='--', lw=1.2,
                   label=r'default ($11.31$ eV)')]
        ax_top.legend(handles=tuning_handles, fontsize=6.5, frameon=False,
                      loc='upper right')
        ax_bot.legend(fontsize=7, frameon=False, ncol=2)
fig.tight_layout()
out = os.path.join(HERE, '..', '..', 'OmegaQMC', 'paper',
                   'fig_exciton_binding_compare.pdf')
fig.savefig(os.path.abspath(out))
fig.savefig(os.path.abspath(out)[:-4] + '.png')
print(f"wrote {os.path.abspath(out)}")
