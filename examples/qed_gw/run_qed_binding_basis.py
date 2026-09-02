"""Basis-set robustness of the cavity-modified electron-hole binding.

Referee check (Sec.~exciton / Sec.~exciton-selrule): every neutral-excitation
number of the manuscript is computed in cc-pVDZ, where the LUMO is a
basis-confined virtual.  Because the cavity-induced fundamental-gap shift
delta_lambda E_gap = delta_lambda IP - delta_lambda EA is dominated by the
attachment channel, and delta_lambda EA grows by a factor ~3 when diffuse
functions are added (run_qed_ea_basis_check.py), delta_lambda E_gap grows from
+0.044 eV (cc-pVDZ) to ~+0.21 eV (aug-cc-pVDZ) for water.  The question this
script answers is whether the *optical* gap shift delta_lambda Omega_S1 --- and
hence the quasiparticle channel delta_lambda E_b^elec of the binding
decomposition --- tracks that growth, and whether the DSE/photon kernel
residual delta_lambda E_b^ker that reverses the sign for NH3 survives the basis
change.

For each molecule and basis it recomputes the full QED-BSE@QED-evGW
observables of Tables IV and V at lambda = 0 and 0.05
(omega_cav = 0.415668 Ha, z-polarized):

    E_gap, Omega_S1, Omega_T1, Delta E_ST, E_b = E_gap - Omega_S1

and the channel decomposition

    delta_lambda E_b^elec  (cavity-dressed QP energies, both kernel channels off)
    delta_lambda E_b^ker   = delta_lambda E_b - delta_lambda E_b^elec

together with the selection-rule diagnostics mu_z^2(S1) and
Delta d_z = d^z_LL - d^z_HH.

Usage:  python run_qed_binding_basis.py [MOL ...] [--basis B ...]
Default: H2O and NH3 in cc-pVDZ and aug-cc-pVDZ.
Results accumulate in qed_binding_basis_results.json (re-runs update in place).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import scipy.linalg as la
from pyscf import gto

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_bse import (run_qed_bse, _assemble_A_static,
                                     _classify_and_brightness)
from OmegaQMC.addons.qed_gw import _build_static_quantities, _rpa_at_eps

EV = 27.211386245988
OMEGA = 0.415668
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'qed_binding_basis_results.json')


def c3v(heavy, r, angle_deg):
    """C3v AH3 geometry: heavy atom at origin, H3 below, C3 axis = z."""
    ang = math.radians(angle_deg)
    cth = math.sqrt((0.5 + math.cos(ang)) / 1.5)   # 1.5cos^2 - 0.5 = cos(HXH)
    sth = math.sqrt(1.0 - cth ** 2)
    rho, z = r * sth, -r * cth
    return [[heavy, (0.0, 0.0, 0.0)],
            ['H', (rho, 0.0, z)],
            ['H', (-0.5 * rho, 0.8660254 * rho, z)],
            ['H', (-0.5 * rho, -0.8660254 * rho, z)]]


_h2o_half = math.radians(104.5 / 2.0)
_ch4_t = 1.087 / math.sqrt(3.0)
GEOMETRIES = {
    'H2O': [['O', (0.0, 0.0, 0.0)],
            ['H', (math.sin(_h2o_half), 0.0, -math.cos(_h2o_half))],
            ['H', (-math.sin(_h2o_half), 0.0, -math.cos(_h2o_half))]],
    'HF':  [['F', (0.0, 0.0, 0.0)],
            ['H', (0.0, 0.0, 0.917)]],
    'NH3': c3v('N', 1.0124, 106.67),
    'CH4': [['C', (0.0, 0.0, 0.0)],
            ['H', (_ch4_t, _ch4_t, _ch4_t)],
            ['H', (_ch4_t, -_ch4_t, -_ch4_t)],
            ['H', (-_ch4_t, _ch4_t, -_ch4_t)],
            ['H', (-_ch4_t, -_ch4_t, _ch4_t)]],
    'PH3': c3v('P', 1.4200, 93.345),
}


def build(name, basis):
    mol = gto.M(atom=GEOMETRIES[name], basis=basis, unit='Angstrom',
                symmetry=False, verbose=0)
    # Recenter at the nuclear centre of mass (paper convention): the QP
    # energies entering the BSE are origin-dependent at lambda > 0.
    masses = mol.atom_mass_list()
    coords = mol.atom_coords()                      # Bohr
    com = masses @ coords / masses.sum()
    mol.set_geom_(coords - com, unit='Bohr', symmetry=False)
    return mol


def s1_props(qedhf, eps_QP):
    """(mu_z^2, Delta d_z) for the lowest singlet S1 (length gauge, a.u.)."""
    static = _build_static_quantities(qedhf, direct=True,
                                      include_dse=True, include_photon=True)
    Omega_RPA, M_full = _rpa_at_eps(static, eps_QP)
    A_BSE = _assemble_A_static(static, Omega_RPA, M_full, eps_QP)
    Omega, X = la.eigh(A_BSE)
    so = static['so']
    nocc, nso = so['nocc'], so['nso']
    nvir = nso - nocc
    X_4d = X.reshape(nvir, nocc, -1)
    spin, _w, _f = _classify_and_brightness(qedhf, X_4d, Omega, nocc)
    s1 = int(np.where(spin == 'S')[0][0])
    Xs1 = X_4d[:, :, s1]

    C = np.asarray(qedhf['C'])
    idx = np.arange(nso)
    same = (idx[:, None] % 2) == (idx[None, :] % 2)
    muz_sf = C.T @ np.asarray(qedhf['mu_z_ao']) @ C
    muz_so = same * muz_sf[idx[:, None] // 2, idx[None, :] // 2]
    muz_S1 = float(np.einsum('ai,ai->', muz_so[nocc:, :nocc], Xs1))
    nsp = nocc // 2                                  # HOMO/LUMO spatial idx
    delta_dz = float(muz_sf[nsp, nsp] - muz_sf[nsp - 1, nsp - 1])
    return muz_S1 ** 2, delta_dz


def run(name, basis, lam=0.05):
    t0 = time.time()
    mol = build(name, basis)
    qh0 = run_qed_hf(mol, OMEGA, (0, 0, 0.0), verbose=False, tol=1e-12)
    b0 = run_qed_bse(qh0, gw_mode='evGW', tda=False, verbose=False)
    qh1 = run_qed_hf(mol, OMEGA, (0, 0, lam), verbose=False, tol=1e-12)
    b1 = run_qed_bse(qh1, gw_mode='evGW', tda=False, verbose=False)
    # quasiparticle channel: cavity-dressed QP energies, kernel channels off
    b1e = run_qed_bse(qh1, tda=False, eps_QP=b1['eps_QP'],
                      include_dse=False, include_photon=False, verbose=False)
    muz2, ddz = s1_props(qh0, b0['eps_QP'])

    def pack(b):
        return {k: b[k] * EV for k in ('E_gap', 'Omega_S1', 'Omega_T1', 'E_b')}

    r0, r1 = pack(b0), pack(b1)
    r0['dE_ST'] = r0['Omega_S1'] - r0['Omega_T1']
    r1['dE_ST'] = r1['Omega_S1'] - r1['Omega_T1']
    dEb = r1['E_b'] - r0['E_b']
    dEb_elec = b1e['E_b'] * EV - r0['E_b']
    rec = {
        'nao': int(mol.nao), 'lambda': lam, 'omega_cav': OMEGA,
        'lam0': r0, 'lam1': r1,
        'delta': {k: r1[k] - r0[k] for k in r0},
        'mu_z2_S1': muz2, 'Delta_d_z': ddz,
        'dEb_meV': dEb * 1e3,
        'dEb_elec_meV': dEb_elec * 1e3,
        'dEb_ker_meV': (dEb - dEb_elec) * 1e3,
        'walltime_s': time.time() - t0,
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mols', nargs='*', default=['H2O', 'NH3'])
    ap.add_argument('--basis', nargs='*', default=['cc-pVDZ', 'aug-cc-pVDZ'])
    ap.add_argument('--lam', type=float, default=0.05)
    args = ap.parse_args()

    out = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            out = json.load(f)

    for name in args.mols:
        for basis in args.basis:
            key = f'{name}|{basis}'
            print(f'--- {key} ...', flush=True)
            rec = run(name, basis, args.lam)
            out[key] = rec
            with open(OUT, 'w') as f:
                json.dump(out, f, indent=1)
            d = rec['delta']
            print(f"    nao={rec['nao']}  ({rec['walltime_s']:.0f} s)\n"
                  f"    E_gap  {rec['lam0']['E_gap']:9.3f} -> "
                  f"{rec['lam1']['E_gap']:9.3f}   d = {d['E_gap']:+.4f} eV\n"
                  f"    Om_S1  {rec['lam0']['Omega_S1']:9.3f} -> "
                  f"{rec['lam1']['Omega_S1']:9.3f}   d = {d['Omega_S1']:+.4f} eV\n"
                  f"    Om_T1  {rec['lam0']['Omega_T1']:9.3f} -> "
                  f"{rec['lam1']['Omega_T1']:9.3f}   d = {d['Omega_T1']:+.4f} eV\n"
                  f"    E_b    {rec['lam0']['E_b']:9.3f} -> "
                  f"{rec['lam1']['E_b']:9.3f}   d = {d['E_b']:+.4f} eV\n"
                  f"    mu_z^2(S1) = {rec['mu_z2_S1']:.3f}   "
                  f"Delta d_z = {rec['Delta_d_z']:+.2f} a.u.\n"
                  f"    dE_b = {rec['dEb_meV']:+.1f} meV "
                  f"= elec {rec['dEb_elec_meV']:+.1f} + ker "
                  f"{rec['dEb_ker_meV']:+.1f}", flush=True)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
