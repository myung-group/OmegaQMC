"""PH3 vs NH3: a discriminating test of the exciton-binding selection rule.

Referee check (Sec.~exciton, Major comment on generality): NH3 is the only
transparency-breaking (binding-weakening) molecule in the four-molecule
set. The sign of the cavity-induced binding shift is governed by a
polarization selection rule --- the DSE kernel residual reaches the lowest
exciton S1 only when S1 is both z-bright (mu_z^2 != 0) and z-dipolar
(Delta d_z large). This script tests that rule on PH3, a heavier polar
C3v hydride beyond the ten-electron set, with NH3 recomputed as a
validation that reproduces Table~\ref{tab:excitonchannel}.

Finding: PH3's lowest exciton is z-DARK (mu_z^2(S1)=0.00, an E-symmetry
x,y state), so the kernel residual never reaches it (ker = 0) and the
cavity BINDS it (delta E_b = +25 meV, pure quasiparticle channel) ---
the transparent response the rule predicts when the z-bright/z-dipolar
conditions are not both met. PH3 is therefore a discriminating
confirmation of the rule, not a second transparency-breaking case.

Default cavity omega_cav = 0.415668 Ha, z-polarized, cc-pVDZ,
full QED-BSE@QED-evGW, lambda = 0 and 0.05. Covers all five molecules
of Table excitonchannel (H2O, HF, NH3, CH4, PH3).
"""
import json
import math
import os

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


results = {}
print(f"{'mol':5s} {'Om_S1':>7s} {'E_b':>7s} {'muz2':>6s} {'Dd_z':>6s} "
      f"{'dE_b':>8s} {'elec':>8s} {'ker':>8s}  (eV unless noted)")
for name in ('H2O', 'HF', 'NH3', 'CH4', 'PH3'):
    mol = gto.M(atom=GEOMETRIES[name], basis='cc-pVDZ', unit='Angstrom',
                symmetry=False, verbose=0)
    # Recenter at the nuclear centre of mass (paper convention): the QP
    # energies entering the BSE are origin-dependent at lambda > 0.
    masses = mol.atom_mass_list()
    coords = mol.atom_coords()                      # Bohr
    com = masses @ coords / masses.sum()
    mol.set_geom_(coords - com, unit='Bohr', symmetry=False)
    qh0 = run_qed_hf(mol, OMEGA, (0, 0, 0.0), verbose=False, tol=1e-12)
    b0 = run_qed_bse(qh0, gw_mode='evGW', tda=False, verbose=False)
    qh1 = run_qed_hf(mol, OMEGA, (0, 0, 0.05), verbose=False, tol=1e-12)
    b1 = run_qed_bse(qh1, gw_mode='evGW', tda=False, verbose=False)
    b1_elec = run_qed_bse(qh1, tda=False, eps_QP=b1['eps_QP'],
                          include_dse=False, include_photon=False,
                          verbose=False)
    Eb0, Eb1, Eb1e = b0['E_b'] * EV, b1['E_b'] * EV, b1_elec['E_b'] * EV
    dEb, dEb_elec = Eb1 - Eb0, Eb1e - Eb0
    muz2, ddz = s1_props(qh0, b0['eps_QP'])
    results[name] = {
        'Omega_S1': b0['Omega_S1'] * EV, 'E_b': Eb0,
        'mu_z2_S1': muz2, 'Delta_d_z': ddz,
        'dEb_total_meV': dEb * 1000, 'dEb_elec_meV': dEb_elec * 1000,
        'dEb_ker_meV': (dEb - dEb_elec) * 1000,
        # full observable set per lambda (the Table~exciton row)
        'obs': {str(lam): {k: b[k] * EV for k in
                           ('E_gap', 'Omega_S1', 'Omega_T1', 'E_b')}
                for lam, b in ((0.0, b0), (0.05, b1))},
    }
    print(f"{name:5s} {b0['Omega_S1']*EV:7.3f} {Eb0:7.3f} {muz2:6.3f} "
          f"{ddz:+6.2f} {dEb*1000:+7.1f}m {dEb_elec*1000:+7.1f}m "
          f"{(dEb-dEb_elec)*1000:+7.1f}m")

with open(os.path.join(HERE, 'qed_binding_ph3_results.json'), 'w') as f:
    json.dump(results, f, indent=1)
print("\n(full QED-BSE@QED-evGW, shifts in meV; Table excitonchannel: "
      "NH3 muz2~0.27, Dd_z~-1.2, dE_b<0 = elec>0 + ker<0; PH3 z-dark, ker~0)")
print("wrote qed_binding_ph3_results.json")
