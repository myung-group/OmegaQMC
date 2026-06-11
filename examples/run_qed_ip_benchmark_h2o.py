"""Delta-method IP benchmark vs QED-GW for water.

IP = E0(N-1) - E0(N), both ground states of the same Pauli-Fierz
Hamiltonian (omega = 0.415668 Ha, z-polarised lambda), vertical
(same geometry, O at origin, matching manuscript tab:gw).

 - Delta-QED-CCSD-21: neutral on QED-RHF, cation (doublet) on QED-UHF.
 - Delta-QED-FCI: STO-3G only (dense polaritonic FCI).
 - QED-GW (linG0W0 / G0W0 / evGW): IP = -eps_QP(HOMO).
"""
import math
import numpy as np
from pyscf import gto, scf

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_uhf import run_qed_uhf
from OmegaQMC.addons.qed_ccsd import run_qed_ccsd
from OmegaQMC.addons.qed_gw import run_qed_gw
from OmegaQMC.addons.qed_fci import run_qed_fci

EV = 27.211386245988
OMEGA = 0.415668

half = math.radians(104.5 / 2.0)
hx, hz = math.sin(half), -math.cos(half)
ATOMS = [['O', (0.0, 0.0, 0.0)], ['H', (hx, 0.0, hz)], ['H', (-hx, 0.0, hz)]]


def build(basis, charge=0, spin=0):
    return gto.M(atom=ATOMS, basis=basis, unit='Angstrom', charge=charge,
                 spin=spin, symmetry=False, verbose=0)


def delta_ccsd(basis, lam):
    cv = (0.0, 0.0, lam)
    qedhf = run_qed_hf(build(basis), OMEGA, cv, verbose=False, tol=1e-12)
    rn = run_qed_ccsd(qedhf, max_iter=200, verbose=False)
    qeduhf = run_qed_uhf(build(basis, charge=1, spin=1), OMEGA, cv,
                         verbose=False, tol=1e-12)
    rc = run_qed_ccsd(qeduhf, max_iter=200, verbose=False)
    assert rn['converged'] and rc['converged']
    ip = (rc['E_qed_ccsd_total'] - rn['E_qed_ccsd_total']) * EV
    ip_hf = (qeduhf['E_qed_uhf'] - qedhf['E_qed_hf']) * EV  # Delta-QED-HF
    return ip, ip_hf, qedhf


def delta_fci(basis, lam, nph_max):
    cv = (0.0, 0.0, lam)
    mfn = scf.RHF(build(basis)); mfn.conv_tol = 1e-12; mfn.kernel()
    rn = run_qed_fci(mfn, omega=OMEGA, coupling_vec=cv, nph_max=nph_max)
    mfc = scf.UHF(build(basis, charge=1, spin=1))
    mfc.conv_tol = 1e-12; mfc.kernel()
    rc = run_qed_fci(mfc, omega=OMEGA, coupling_vec=cv, nph_max=nph_max)
    return (rc['e_qed_fci'] - rn['e_qed_fci']) * EV


def gw_ips(qedhf):
    homo = qedhf['nocc_spatial'] - 1
    out = {}
    for mode in ('linG0W0', 'G0W0', 'evGW'):
        g = run_qed_gw(qedhf, mode=mode, verbose=False)
        out[mode] = -float(g['eps_QP'][homo]) * EV
    return out


for basis in ('sto-3g', 'cc-pVDZ'):
    print("=" * 76)
    print(f"H2O / {basis}   (vertical IP, eV;  omega = {OMEGA} Ha)")
    print("=" * 76)
    hdr = (f"{'lam':>6} {'dQED-HF':>9} {'linG0W0':>9} {'G0W0':>9} "
           f"{'evGW':>9} {'dQED-CCSD':>10}")
    if basis == 'sto-3g':
        hdr += f" {'dQED-FCI':>9}"
    print(hdr)
    for lam in (0.0, 0.05):
        ip_ccsd, ip_hf, qedhf = delta_ccsd(basis, lam)
        gw = gw_ips(qedhf)
        row = (f"{lam:6.2f} {ip_hf:9.3f} {gw['linG0W0']:9.3f} "
               f"{gw['G0W0']:9.3f} {gw['evGW']:9.3f} {ip_ccsd:10.3f}")
        if basis == 'sto-3g':
            ip_fci4 = delta_fci(basis, lam, nph_max=4)
            ip_fci6 = delta_fci(basis, lam, nph_max=6)
            row += f" {ip_fci6:9.3f}"
            if abs(ip_fci6 - ip_fci4) > 1e-4:
                row += f"  (nph 4->6 shift {ip_fci6 - ip_fci4:+.1e} eV)"
        print(row, flush=True)
    print()
