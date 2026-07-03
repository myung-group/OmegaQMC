"""Validate qed_polariton_singlet against the spin-orbital modules
(qed_rpa, qed_gw, qed_bse_polaritonic) on closed-shell H2O/6-31g."""
import numpy as np
from pyscf import gto
from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons import qed_rpa as so_rpa
from OmegaQMC.addons import qed_gw as so_gw
from OmegaQMC.addons import qed_bse_polaritonic as so_bse
from OmegaQMC.addons import qed_polariton_singlet as sg

mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
            basis="6-31g", verbose=0)
omega = 0.40
qedhf = run_qed_hf(mol, omega, (0.0, 0.0, 0.05), verbose=False, tol=1e-12)

ok = True
def check(label, d, thr=1e-9):
    global ok
    flag = 'OK ' if d < thr else 'FAIL'
    ok = ok and d < thr
    print(f"{flag} {label:34s} {d:.3e}")

# ---- QED-RPA -----------------------------------------------------------
for direct in (True, False):
    r_so = so_rpa.run_qed_rpa(qedhf, direct=direct, verbose=False)
    r_sg = sg.run_qed_rpa_singlet(qedhf, direct=direct, verbose=False)
    lab = 'dRPA' if direct else 'RPA '
    check(f"{lab} E_corr", abs(r_so['E_qed_rpa_corr']
                               - r_sg['E_qed_rpa_corr']))
    # spectrum: union of singlet(+photon) and 3x triplet = spin-orbital
    union = np.sort(np.concatenate(
        [r_sg['Omega'], np.repeat(r_sg['Omega_triplet'], 3)]))
    check(f"{lab} full spectrum union", abs(np.sort(r_so['Omega_full'])
                                            - union).max())
    union_tda = np.sort(np.concatenate(
        [r_sg['Omega_tda_rwa'], np.repeat(r_sg['Omega_tda_rwa_triplet'], 3)]))
    check(f"{lab} TDA-RWA spectrum union",
          abs(np.sort(r_so['Omega_tda_rwa']) - union_tda).max())

# ---- QED-evGW ----------------------------------------------------------
gw_so = so_gw.run_qed_gw(qedhf, mode='evGW', eta=1e-3, verbose=False)
gw_sg = sg.run_qed_gw_singlet(qedhf, mode='evGW', eta=1e-3, verbose=False)
eps_so_spatial = gw_so['eps_GW_all'][0::2]        # interleaved -> alpha
check("evGW eps_QP (all orbitals)",
      abs(eps_so_spatial - gw_sg['eps_QP']).max(), 1e-7)

# ---- polaritonic QED-BSE@evGW ------------------------------------------
for tda in (True, False):
    b_so = so_bse.run_qed_bse_polaritonic(
        qedhf, gw_mode='evGW', tda=tda, verbose=False,
        eps_QP=gw_so['eps_GW_all'])
    b_sg = sg.run_qed_bse_polaritonic_singlet(
        qedhf, gw_mode='evGW', tda=tda, verbose=False,
        eps_QP=gw_sg['eps_QP'])
    lab = 'pol-BSE TDA ' if tda else 'pol-BSE full'
    # every singlet root must appear in the spin-orbital spectrum
    d = max(abs(b_so['Omega'] - w).min() for w in b_sg['Omega'])
    check(f"{lab} singlet roots subset", d, 1e-7)
    # bright roots: energies, oscillator strengths, photon weights
    bs = np.where(b_so['f_osc'] > 1e-4)[0]
    bg = np.where(b_sg['f_osc'] > 1e-4)[0]
    check(f"{lab} #bright roots", abs(len(bs) - len(bg)), 0.5)
    n = min(len(bs), len(bg))
    check(f"{lab} bright Omega", abs(b_so['Omega'][bs[:n]]
                                     - b_sg['Omega'][bg[:n]]).max(), 1e-7)
    check(f"{lab} bright f_osc", abs(b_so['f_osc'][bs[:n]]
                                     - b_sg['f_osc'][bg[:n]]).max(), 1e-6)
    check(f"{lab} bright photon wt",
          abs(b_so['photon_weight'][bs[:n]]
              - b_sg['photon_weight'][bg[:n]]).max(), 1e-6)

print("ALL OK" if ok else "FAILURES PRESENT")
