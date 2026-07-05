"""Vertical IP and EA benchmark: QED self-energy flavours vs the Delta-method.

For each molecule (H2O, HF, NH3, CH4) coupled to a single z-polarized
cavity mode (omega_cav = 0.415668 Ha) at lambda = 0 and 0.05:

  IP = E0(N-1) - E0(N),   EA = E0(N) - E0(N+1)

with both energies ground states of the same Pauli-Fierz Hamiltonian
(vertical: same geometry; each geometry recentered so its nuclear
centre of mass sits at the coordinate origin — the dipole operator is
origin-dependent for the charged species, so the gauge origin matters;
cation and anion are doublets treated with a QED-UHF reference).
Methods:

  Koopmans (-eps_HOMO / -eps_LUMO of QED-HF), Delta-QED-HF,
  linG0W0 / G0W0 / evGW (QED-dRPA-screened self-energy),
  G0W0+SOSEX(static) vertex correction, Delta-QED-CCSD(-21),
  and Delta-QED-FCI in STO-3G for H2O and HF (dense polaritonic FCI;
  NH3/CH4 minimal-basis FCI dimensions exceed the dense solver).

Results are printed as a table and dumped to qed_ipea_results.json.
"""
import json
import math
import numpy as np
from pyscf import gto, scf

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_uhf import run_qed_uhf
from OmegaQMC.addons.qed_ccsd import run_qed_ccsd
from OmegaQMC.addons.qed_gw import run_qed_gw, run_qed_gw_sosex
from OmegaQMC.addons.qed_fci import run_qed_fci

EV = 27.211386245988
OMEGA = 0.415668

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

FCI_MOLECULES = ('H2O', 'HF')      # minimal-basis dense-FCI anchors


def build(name, basis, charge=0, spin=0):
    mol = gto.M(atom=GEOMETRIES[name], basis=basis, unit='Angstrom',
                charge=charge, spin=spin, symmetry=False, verbose=0)
    # Recenter at the nuclear centre of mass: the length-gauge dipole
    # operator is origin-dependent for charged species, so the cation
    # and anion energies (hence IP/EA) depend on this choice.
    masses = mol.atom_mass_list()
    coords = mol.atom_coords()                      # Bohr
    com = masses @ coords / masses.sum()
    mol.set_geom_(coords - com, unit='Bohr', symmetry=False)
    return mol


def run_molecule(name, basis, lam, do_fci=False):
    cv = (0.0, 0.0, lam)
    out = {}

    # --- neutral reference + correlated ground state ---
    qedhf = run_qed_hf(build(name, basis), OMEGA, cv, verbose=False,
                       tol=1e-12)
    homo = qedhf['nocc_spatial'] - 1
    lumo = homo + 1
    rn = run_qed_ccsd(qedhf, max_iter=300, verbose=False)
    assert rn['converged'], f"{name}/{basis} neutral CCSD not converged"
    E_N_hf = qedhf['E_qed_hf']
    E_N_cc = rn['E_qed_ccsd_total']

    # --- cation / anion (doublets, QED-UHF) ---
    species = {}
    for label, charge in (('cation', 1), ('anion', -1)):
        mol_ion = build(name, basis, charge=charge, spin=1)
        try:
            quhf = run_qed_uhf(mol_ion, OMEGA, cv, verbose=False, tol=1e-12)
        except RuntimeError:
            # Orbitally degenerate holes (e.g. the t2 hole of CH4+) make
            # the plain fixed-point iteration oscillate; damp it.
            quhf = run_qed_uhf(mol_ion, OMEGA, cv, verbose=False,
                               tol=1e-11, damping=0.4, max_iter=4000)
        rc = run_qed_ccsd(quhf, max_iter=300, verbose=False)
        assert rc['converged'], (
            f"{name}/{basis} {label} CCSD not converged")
        species[label] = (quhf['E_qed_uhf'], rc['E_qed_ccsd_total'])

    out['dHF_IP'] = (species['cation'][0] - E_N_hf) * EV
    out['dHF_EA'] = (E_N_hf - species['anion'][0]) * EV
    out['dCC_IP'] = (species['cation'][1] - E_N_cc) * EV
    out['dCC_EA'] = (E_N_cc - species['anion'][1]) * EV

    # --- self-energy flavours (HOMO -> IP, LUMO -> EA) ---
    lin = run_qed_gw(qedhf, mode='linG0W0', verbose=False)
    evg = run_qed_gw(qedhf, mode='evGW', verbose=False)
    sx = run_qed_gw_sosex(qedhf, verbose=False)
    out['koop_IP'] = -lin['eps_HF'][homo] * EV
    out['koop_EA'] = -lin['eps_HF'][lumo] * EV
    out['lin_IP'] = -lin['eps_QP'][homo] * EV
    out['lin_EA'] = -lin['eps_QP'][lumo] * EV
    out['g0w0_IP'] = -sx['eps_QP_GW'][homo] * EV
    out['g0w0_EA'] = -sx['eps_QP_GW'][lumo] * EV
    out['evgw_IP'] = -evg['eps_QP'][homo] * EV
    out['evgw_EA'] = -evg['eps_QP'][lumo] * EV
    out['sosex_IP'] = -sx['eps_QP'][homo] * EV
    out['sosex_EA'] = -sx['eps_QP'][lumo] * EV

    # --- Delta-QED-FCI anchor (minimal basis only) ---
    if do_fci:
        def fci_energy(charge, spin):
            mf = (scf.RHF if spin == 0 else scf.UHF)(
                build(name, basis, charge=charge, spin=spin))
            mf.conv_tol = 1e-12
            mf.kernel()
            return run_qed_fci(mf, omega=OMEGA, coupling_vec=cv,
                               nph_max=4)['e_qed_fci']
        e_n = fci_energy(0, 0)
        out['dFCI_IP'] = (fci_energy(1, 1) - e_n) * EV
        out['dFCI_EA'] = (e_n - fci_energy(-1, 1)) * EV

    return out


METHODS = ['koop', 'dHF', 'lin', 'g0w0', 'evgw', 'sosex', 'dCC', 'dFCI']
LABELS = {'koop': 'Koopmans', 'dHF': 'dQED-HF', 'lin': 'linG0W0',
          'g0w0': 'G0W0', 'evgw': 'evGW', 'sosex': 'G0W0+SOSEX',
          'dCC': 'dQED-CCSD', 'dFCI': 'dQED-FCI'}

if __name__ == '__main__':
    results = {}
    for basis, mols in (('sto-3g', FCI_MOLECULES),
                        ('cc-pVDZ', tuple(GEOMETRIES))):
        for name in mols:
            for lam in (0.0, 0.05):
                key = f"{name}|{basis}|{lam:.2f}"
                print(f"... running {key}", flush=True)
                results[key] = run_molecule(
                    name, basis, lam,
                    do_fci=(basis == 'sto-3g' and name in FCI_MOLECULES))
                # incremental save so partial runs are not lost
                with open('qed_ipea_results.json', 'w') as fh:
                    json.dump(results, fh, indent=1)

    for quantity in ('IP', 'EA'):
        print("\n" + "=" * 100)
        print(f"Vertical {quantity} (eV), omega_cav = {OMEGA} Ha")
        print("=" * 100)
        hdr = f"{'system':>14} {'lam':>5}" + ''.join(
            f"{LABELS[m]:>11}" for m in METHODS)
        print(hdr)
        for key, row in results.items():
            name, basis, lam = key.split('|')
            cells = ''.join(
                f"{row[f'{m}_{quantity}']:11.3f}"
                if f'{m}_{quantity}' in row else f"{'---':>11}"
                for m in METHODS)
            print(f"{name + '/' + basis:>14} {lam:>5}" + cells)
