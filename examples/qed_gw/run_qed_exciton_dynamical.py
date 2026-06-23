"""Dynamical-BSE cavity-modified exciton observables (Tasks 1 & 2).

Companion to run_qed_exciton_binding.py: recomputes the cavity-modified
exciton binding with the *frequency-dependent* (dynamical) BSE kernel of
qed_bse_dynamical (renormalised first-order correction, dTDA; with a
self-consistent cross-check), to quantify how much the static
"cavity-transparent" picture changes once the screened e-h interaction
is resolved at its true (polariton) frequency.

Outputs (printed + qed_exciton_dynamical_results.json):
  1. Four-molecule table: static vs dynamical Omega_S1, Omega_T1, E_b at
     lambda = 0 and 0.05, and the cavity-induced shifts of each.
  2. Water: perturbative vs self-consistent dynamical correction.
  3. Water channel decomposition (QP fixed at lambda=0.05): the
     dynamical correction of S1 with (DSE, photon) kernel channels
     toggled -- isolates whether the photon channel, which CANCELS in
     the static kernel, contributes once the kernel is dynamical.
  4. Retuning test: omega_cav = 0.293113 Ha (exciton-resonant) vs
     0.415668 Ha (RPA-resonant). The static BSE gave a <=2% null; here
     we test whether the dynamical kernel develops a resonance response.
"""
import json
import math
import os

import numpy as np
from pyscf import gto

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_bse import run_qed_bse, _resolve_eps_QP
from OmegaQMC.addons.qed_bse_dynamical import run_qed_bse_dynamical

EV = 27.211386245988
OMEGA = 0.415668
OMEGA_EXC = 0.293113          # exciton-resonant retuning (= BSE optical gap of water)
BASIS = 'cc-pVDZ'

_h2o_half = math.radians(104.5 / 2.0)
_nh3_rho, _nh3_z = 0.93786, -0.38129
_ch4_t = 1.087 / math.sqrt(3.0)
GEOMETRIES = {
    'H2O': [['O', (0.0, 0.0, 0.0)],
            ['H', (math.sin(_h2o_half), 0.0, -math.cos(_h2o_half))],
            ['H', (-math.sin(_h2o_half), 0.0, -math.cos(_h2o_half))]],
    'HF':  [['F', (0.0, 0.0, 0.0)], ['H', (0.0, 0.0, 0.917)]],
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


def lowest_S_T(res):
    """(Omega_S1, Omega_T1, E_b) in eV from a dynamical-BSE result dict."""
    return (res['Omega_S1'] * EV,
            (res['Omega_T1'] * EV) if res['Omega_T1'] is not None else None,
            res['E_b'] * EV)


results = {'omega_cav': OMEGA, 'basis': BASIS, 'gw_mode': 'evGW'}

# ----------------------------------------------------------------------
# 1. Four-molecule table: static vs dynamical
# ----------------------------------------------------------------------
print("=" * 88)
print(f"Static vs DYNAMICAL QED-BSE@QED-evGW exciton binding, {BASIS}, "
      f"omega_cav={OMEGA} Ha")
print("=" * 88)
print(f"{'sys':5s} {'lam':>5s} | {'S1_stat':>8s} {'S1_dyn':>8s} | "
      f"{'Eb_stat':>8s} {'Eb_dyn':>8s}   (eV)")
table = {}
for name in ('H2O', 'HF', 'NH3', 'CH4'):
    table[name] = {}
    for lam in (0.0, 0.05):
        qedhf = run_qed_hf(build(name), OMEGA, (0.0, 0.0, lam),
                           verbose=False, tol=1e-12)
        st = run_qed_bse(qedhf, gw_mode='evGW', tda=True, verbose=False)
        dy = run_qed_bse_dynamical(qedhf, gw_mode='evGW',
                                   mode='perturbative', n_states=30,
                                   verbose=False)
        table[name][lam] = {
            'S1_stat': st['Omega_S1'] * EV, 'S1_dyn': dy['Omega_S1'] * EV,
            'T1_stat': st['Omega_T1'] * EV, 'T1_dyn': dy['Omega_T1'] * EV,
            'Eb_stat': st['E_b'] * EV, 'Eb_dyn': dy['E_b'] * EV,
        }
        r = table[name][lam]
        print(f"{name:5s} {lam:5.2f} | {r['S1_stat']:8.4f} {r['S1_dyn']:8.4f} "
              f"| {r['Eb_stat']:8.4f} {r['Eb_dyn']:8.4f}")
    d0, d5 = table[name][0.0], table[name][0.05]
    dS1_s = d5['S1_stat'] - d0['S1_stat']
    dS1_d = d5['S1_dyn'] - d0['S1_dyn']
    dEb_s = d5['Eb_stat'] - d0['Eb_stat']
    dEb_d = d5['Eb_dyn'] - d0['Eb_dyn']
    table[name]['shifts'] = {'dS1_stat': dS1_s, 'dS1_dyn': dS1_d,
                             'dEb_stat': dEb_s, 'dEb_dyn': dEb_d}
    print(f"{'':5s} {'d_lam':>5s} | dS1: stat {dS1_s:+.4f} dyn {dS1_d:+.4f} "
          f"| dEb: stat {dEb_s:+.4f} dyn {dEb_d:+.4f}")
results['table'] = table

# ----------------------------------------------------------------------
# 2. Water: perturbative vs self-consistent
# ----------------------------------------------------------------------
print("\nWater perturbative vs self-consistent (lam=0.05):")
qedhf5 = run_qed_hf(build('H2O'), OMEGA, (0.0, 0.0, 0.05),
                    verbose=False, tol=1e-12)
pert = run_qed_bse_dynamical(qedhf5, mode='perturbative', n_states=30,
                             verbose=False)
sc = run_qed_bse_dynamical(qedhf5, mode='selfconsistent', n_states=30,
                           verbose=False)
print(f"  S1_dyn  perturbative={pert['Omega_S1']*EV:.4f}  "
      f"self-consistent={sc['Omega_S1']*EV:.4f} eV")
print(f"  E_b_dyn perturbative={pert['E_b']*EV:.4f}  "
      f"self-consistent={sc['E_b']*EV:.4f} eV")
results['pert_vs_sc'] = {
    'S1_pert': pert['Omega_S1'] * EV, 'S1_sc': sc['Omega_S1'] * EV,
    'Eb_pert': pert['E_b'] * EV, 'Eb_sc': sc['E_b'] * EV}

# ----------------------------------------------------------------------
# 3. Channel decomposition of the DYNAMICAL correction (QP fixed)
# ----------------------------------------------------------------------
print("\nDynamical correction of S1 with QP fixed at lam=0.05 "
      "(isolates kernel channel):")
eps_fix = _resolve_eps_QP(qedhf5, 'evGW', 1e-3, None, False)
decomp = {}
for dse, ph in ((True, True), (True, False), (False, False)):
    r = run_qed_bse_dynamical(qedhf5, mode='perturbative', n_states=30,
                              include_dse=dse, include_photon=ph,
                              eps_QP=eps_fix, verbose=False)
    s_idx = [i for i in r['idx'] if r['spin'][i] == 'S']
    s0 = int(s_idx[0])
    k = list(r['idx']).index(s0)
    decomp[f'dse{int(dse)}_ph{int(ph)}'] = {
        'S1_stat': float(r['Omega_stat'][s0] * EV),
        'dOmega_dyn': float(r['dOmega'][k] * EV),
        'Z': float(r['Z'][k])}
    print(f"  dse={dse!s:5} photon={ph!s:5}: S1_stat={r['Omega_stat'][s0]*EV:.4f} "
          f"dOmega_dyn={r['dOmega'][k]*EV:+.5f} eV  Z={r['Z'][k]:.4f}")
results['decomp_H2O'] = decomp

# ----------------------------------------------------------------------
# 4. Retuning test: exciton-resonant vs RPA-resonant cavity
# ----------------------------------------------------------------------
print("\nRetuning test (water, lam=0.05): dynamical correction of S1")
retune = {}
for tag, om in (('RPA-res(11.31eV)', OMEGA), ('exc-res(7.98eV)', OMEGA_EXC)):
    qh = run_qed_hf(build('H2O'), om, (0.0, 0.0, 0.05),
                    verbose=False, tol=1e-12)
    r = run_qed_bse_dynamical(qh, mode='perturbative', n_states=30,
                              verbose=False)
    s_idx = [i for i in r['idx'] if r['spin'][i] == 'S']
    s0 = int(s_idx[0])
    k = list(r['idx']).index(s0)
    retune[tag] = {'omega_cav': om,
                   'S1_stat': float(r['Omega_stat'][s0] * EV),
                   'S1_dyn': float(r['Omega_dyn'][k] * EV),
                   'dOmega_dyn': float(r['dOmega'][k] * EV),
                   'Z': float(r['Z'][k])}
    print(f"  {tag:18s}: S1_stat={retune[tag]['S1_stat']:.4f} "
          f"dOmega_dyn={retune[tag]['dOmega_dyn']:+.5f} eV  Z={retune[tag]['Z']:.4f}")
results['retune_H2O'] = retune

with open(os.path.join(os.path.dirname(__file__),
                       'qed_exciton_dynamical_results.json'), 'w') as f:
    json.dump(results, f, indent=1)
print("\nwrote qed_exciton_dynamical_results.json")
