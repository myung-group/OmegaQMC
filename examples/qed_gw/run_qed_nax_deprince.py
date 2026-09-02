"""QED-GW vs published QED-CC cavity-modulated IPs and EAs (sodium halides).

Referee check (placement against the existing method): cavity-modulated IPs
and EAs were first computed at the QED-CC / EOM-QED-CC level by DePrince
[J. Chem. Phys. 154, 094112 (2021)] and Liebenthal, Vu and DePrince
[J. Chem. Phys. 156, 054105 (2022)].  Those studies use sodium halides, not
the hydrides of the present benchmark, so this script reproduces their setup
exactly and adds the QED-GW columns, giving a like-for-like numerical
comparison on a shared system/basis/cavity:

    NaF and NaCl, def2-TZVPPD, omega_cav = 2.0 eV, coupling polarized along
    the molecular (z) axis, lambda = 0.00 ... 0.05,
    IP = E(N-1) - E(N),  EA = E(N) - E(N+1)   (vertical, neutral geometry).

Reference values (DePrince 2021, Tables I and II, eV):

    IP  lambda   NaF          NaCl            EA  lambda   NaF     NaCl
        0.00     9.96 / 8.10  9.00 / 7.99         0.00     0.43/0.37  0.64/0.56
        0.05     9.92 / 8.05  8.94 / 7.90         0.05     0.26/0.13  0.49/0.34
                 (Delta-QED-CC / Delta-QED-HF)

The Delta-QED-HF column is the validation handle: it is a mean-field number
that any correct implementation of the same Pauli-Fierz Hamiltonian,
coherent-state reference and gauge origin must reproduce, so agreement there
certifies that the QED-GW numbers below are computed in the same convention.

Geometries are B3LYP/def2-TZVPPD equilibrium bond lengths, as in the
reference; they are located here by a parabolic fit to a short bond scan
(printed, so the value used is on the record).

The closed-shell QED-GW step uses the singlet-adapted (spatial-orbital)
implementation, which reproduces the spin-orbital quasiparticle energies to
~1e-7 Ha and avoids the nso^4 tensor that puts def2-TZVPPD out of reach of
the spin-orbital code on a small-memory machine.

Results -> qed_nax_deprince_results.json.
"""
import json
import os
import time

import numpy as np
from pyscf import gto, dft

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_uhf import run_qed_uhf
from OmegaQMC.addons.qed_polariton_singlet import run_qed_gw_singlet

EV = 27.211386245988
OMEGA = 2.0 / EV                      # 2.0 eV, as in DePrince (2021)
BASIS = 'def2-TZVPPD'
LAMBDAS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'qed_nax_deprince_results.json')

# DePrince, J. Chem. Phys. 154, 094112 (2021), Tables I and II (eV).
REF = {
    'NaF': {'IP_CC': {0.0: 9.96, 0.01: 9.96, 0.02: 9.95, 0.03: 9.95,
                      0.04: 9.93, 0.05: 9.92},
            'IP_HF': {0.0: 8.10, 0.01: 8.10, 0.02: 8.09, 0.03: 8.08,
                      0.04: 8.07, 0.05: 8.05},
            'EA_CC': {0.0: 0.43, 0.01: 0.42, 0.02: 0.40, 0.03: 0.36,
                      0.04: 0.32, 0.05: 0.26},
            'EA_HF': {0.0: 0.37, 0.01: 0.36, 0.02: 0.33, 0.03: 0.28,
                      0.04: 0.21, 0.05: 0.13}},
    'NaCl': {'IP_CC': {0.0: 9.00, 0.01: 9.00, 0.02: 8.99, 0.03: 8.98,
                       0.04: 8.96, 0.05: 8.94},
             'IP_HF': {0.0: 7.99, 0.01: 7.98, 0.02: 7.97, 0.03: 7.96,
                       0.04: 7.93, 0.05: 7.90},
             'EA_CC': {0.0: 0.64, 0.01: 0.64, 0.02: 0.62, 0.03: 0.59,
                       0.04: 0.54, 0.05: 0.49},
             'EA_HF': {0.0: 0.56, 0.01: 0.55, 0.02: 0.52, 0.03: 0.47,
                       0.04: 0.41, 0.05: 0.34}},
}

GUESS = {'NaF': 1.93, 'NaCl': 2.36}          # A, starting point of the scan


def mol_at(name, r, charge=0, spin=0):
    """Diatomic along z, nuclear centre of mass at the origin."""
    x = name.replace('Na', '')
    m = gto.M(atom=[['Na', (0.0, 0.0, 0.0)], [x, (0.0, 0.0, r)]],
              basis=BASIS, unit='Angstrom', charge=charge, spin=spin,
              symmetry=False, verbose=0)
    masses = m.atom_mass_list()
    coords = m.atom_coords()
    com = masses @ coords / masses.sum()
    m.set_geom_(coords - com, unit='Bohr', symmetry=False)
    return m


def optimize_b3lyp(name):
    """Equilibrium bond length from a parabolic fit to a B3LYP scan."""
    r0 = GUESS[name]
    for step in (0.04, 0.01):
        rs = np.array([r0 - step, r0, r0 + step])
        es = []
        for r in rs:
            mf = dft.RKS(mol_at(name, r))
            mf.xc = 'b3lyp'
            mf.verbose = 0
            es.append(mf.kernel())
        c = np.polyfit(rs, es, 2)
        r0 = float(-c[1] / (2 * c[0]))
    return r0


def ipea(name, r, lam):
    cv = (0.0, 0.0, lam)
    t0 = time.time()
    qn = run_qed_hf(mol_at(name, r), OMEGA, cv, verbose=False, tol=1e-11)
    nocc = qn['nocc_spatial']
    out = {'E_N_hf': qn['E_qed_hf']}
    for label, charge in (('cation', 1), ('anion', -1)):
        try:
            q = run_qed_uhf(mol_at(name, r, charge, 1), OMEGA, cv,
                            verbose=False, tol=1e-10)
        except RuntimeError:
            q = run_qed_uhf(mol_at(name, r, charge, 1), OMEGA, cv,
                            verbose=False, tol=1e-9, damping=0.4,
                            max_iter=4000)
        out[f'E_{label}_hf'] = q['E_qed_uhf']
    out['dHF_IP'] = (out['E_cation_hf'] - out['E_N_hf']) * EV
    out['dHF_EA'] = (out['E_N_hf'] - out['E_anion_hf']) * EV
    for mode, tag in (('G0W0', 'g0w0'), ('evGW', 'evgw')):
        gw = run_qed_gw_singlet(qn, mode=mode, verbose=False)
        eps = np.asarray(gw['eps_QP'])
        out[f'{tag}_IP'] = -float(eps[nocc - 1]) * EV
        out[f'{tag}_EA'] = -float(eps[nocc]) * EV
    C = np.asarray(qn['C'])
    eps_hf = np.diag(C.T @ np.asarray(qn['F']) @ C)
    out['koop_IP'] = -float(eps_hf[nocc - 1]) * EV
    out['koop_EA'] = -float(eps_hf[nocc]) * EV
    out['walltime_s'] = time.time() - t0
    return out


def main():
    res = {}
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    for name in ('NaF', 'NaCl'):
        r = res.get(name, {}).get('r_B3LYP')
        if r is None:
            r = optimize_b3lyp(name)
            print(f'{name}: B3LYP/{BASIS} r_e = {r:.4f} A', flush=True)
        rec = res.setdefault(name, {})
        rec['r_B3LYP'] = r
        rec.setdefault('scan', {})
        print(f'\n{name}  (r = {r:.4f} A, omega_cav = 2.0 eV, '
              f'{BASIS})')
        print(f"{'lam':>5s} {'dHF_IP':>8s} {'ref':>6s} {'evGW_IP':>8s} "
              f"{'G0W0_IP':>8s} | {'dHF_EA':>8s} {'ref':>6s} "
              f"{'evGW_EA':>8s} {'G0W0_EA':>8s}")
        for lam in LAMBDAS:
            key = f'{lam:g}'
            if key not in rec['scan']:
                rec['scan'][key] = ipea(name, r, lam)
                json.dump(res, open(OUT, 'w'), indent=1)
            o = rec['scan'][key]
            print(f'{lam:5.2f} {o["dHF_IP"]:8.3f} '
                  f'{REF[name]["IP_HF"][lam]:6.2f} {o["evgw_IP"]:8.3f} '
                  f'{o["g0w0_IP"]:8.3f} | {o["dHF_EA"]:8.3f} '
                  f'{REF[name]["EA_HF"][lam]:6.2f} {o["evgw_EA"]:8.3f} '
                  f'{o["g0w0_EA"]:8.3f}', flush=True)
        # cavity-induced shifts vs the published Delta-QED-CC reference
        s0, s5 = rec['scan']['0'], rec['scan']['0.05']
        d = {}
        for q in ('IP', 'EA'):
            for k in ('koop', 'dHF', 'g0w0', 'evgw'):
                d[f'{k}_{q}'] = s5[f'{k}_{q}'] - s0[f'{k}_{q}']
            d[f'refCC_{q}'] = (REF[name][f'{q}_CC'][0.05]
                               - REF[name][f'{q}_CC'][0.0])
            d[f'refHF_{q}'] = (REF[name][f'{q}_HF'][0.05]
                               - REF[name][f'{q}_HF'][0.0])
        rec['shifts_lam0.05'] = d
        json.dump(res, open(OUT, 'w'), indent=1)
        print(f'  cavity-induced shifts at lambda = 0.05 (eV):')
        for q in ('IP', 'EA'):
            print(f'    d{q}: dHF {d[f"dHF_{q}"]:+.3f} '
                  f'(ref QED-HF {d[f"refHF_{q}"]:+.2f}) | '
                  f'G0W0 {d[f"g0w0_{q}"]:+.3f}  evGW {d[f"evgw_{q}"]:+.3f} '
                  f'(ref QED-CC {d[f"refCC_{q}"]:+.2f})')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
