"""Paper-grade FCI validation of the polaritonic QED-BSE@GW.

Benchmarks the resonant Rabi splitting of the photon-augmented (non-TDA)
polaritonic QED-BSE [Eq. bsepol] against exact polaritonic QED-FCI for
H2O/STO-3G in a z-polarized cavity. The cavity is tuned (at lambda=0, the
photon decoupled) to a strongly z-bright state near 15.5 eV; at finite
coupling that bright state Rabi-splits into a lower and an upper polariton
(the two states of largest photon weight), and the splitting

    Omega_R = E_UP - E_LP

is compared between the two methods and against the two-level law
Omega_R = sqrt(3) * lambda * sqrt(f), with f the lambda=0 oscillator
strength of the bright root. Each method is tuned to its OWN bright state
(BSE and FCI place it at slightly different energies), so the residual is
oscillator-strength (exciton) quality, not the polaritonic construction.

Prints a lambda-scan table and writes qed_bse_polaritonic_fci_results.json.
"""
import json
import math
import os

import numpy as np
from pyscf import gto, scf, ao2mo
from pyscf.fci import direct_spin1, cistring

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_bse_polaritonic import run_qed_bse_polaritonic
from OmegaQMC.addons.qed_fci import _build_fci_matrices

EV = 27.211386245988
HERE = os.path.dirname(os.path.abspath(__file__))
LAMS = [0.025, 0.05, 0.075]
WIN_LO, WIN_HI = 13.0, 18.0      # eV window bracketing the bright state


def build_mol():
    half = math.radians(104.5 / 2.0)
    hx, hz = math.sin(half), -math.cos(half)
    return gto.M(atom=[['O', (0, 0, 0)], ['H', (hx, 0, hz)],
                       ['H', (-hx, 0, hz)]],
                 basis='sto-3g', unit='Angstrom', symmetry=False, verbose=0)


def bright_bse(mol, omega, lam):
    """(energy_eV, f) of the brightest polaritonic-BSE root in the window."""
    qh = run_qed_hf(mol, omega, (0, 0, lam), verbose=False, tol=1e-12)
    pol = run_qed_bse_polaritonic(qh, gw_mode='evGW', tda=False, verbose=False)
    om = np.asarray(pol['Omega']) * EV
    f = np.asarray(pol['f_osc'])
    m = (om >= WIN_LO) & (om <= WIN_HI)
    i = np.where(m)[0][np.argmax(f[m])]
    return float(om[i]), float(f[i])


def bse_split(mol, omega_res, lam):
    """Polaritonic-BSE Rabi splitting (eV) at cavity omega_res."""
    qh = run_qed_hf(mol, omega_res / EV, (0, 0, lam), verbose=False, tol=1e-12)
    pol = run_qed_bse_polaritonic(qh, gw_mode='evGW', tda=False, verbose=False)
    return _doublet_split(np.asarray(pol['Omega']) * EV,
                          np.asarray(pol['photon_weight']), omega_res)


def _doublet_split(energies, photon_wt, e_center, half_window=1.5):
    """|dE| between the two highest-photon-weight states within
    +/- half_window of e_center (the lower/upper polaritons)."""
    e = np.asarray(energies)
    w = np.asarray(photon_wt)
    m = np.abs(e - e_center) <= half_window
    e_in, w_in = e[m], w[m]
    if np.sum(w_in > 0.02) < 2:
        return 0.0
    top2 = np.argsort(w_in)[-2:]
    pair = np.sort(e_in[top2])
    return float(pair[1] - pair[0])


def fci_roots(mol, omega, lam, nph_max=4):
    """Exact polaritonic-FCI roots as (energy_eV, f, photon_number)."""
    norb = mol.nao_nr()
    nelec = mol.nelec
    enuc = mol.energy_nuc()
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    hcore = mf.get_hcore()
    mu_ao = mol.intor('int1e_r', comp=3)
    quad_ao = mol.intor('int1e_rr', comp=9).reshape(3, 3, norb, norb)

    qh = run_qed_hf(mol, omega, (0, 0, lam), verbose=False, tol=1e-12)
    C = np.asarray(qh['C'])
    h1e = C.T @ hcore @ C
    eri = ao2mo.restore(1, ao2mo.full(mol, C), norb)
    dip_mo = lam * (C.T @ mu_ao[2] @ C)
    eri_dse = eri + np.einsum('pq,rs->pqrs', dip_mo, dip_mo)
    h1e_used = h1e + 0.5 * lam ** 2 * (C.T @ quad_ao[2, 2] @ C)

    H_e, D_e, nd = _build_fci_matrices(h1e_used, eri_dse, dip_mo, norb,
                                       nelec, enuc)
    H_e = 0.5 * (H_e + H_e.T); D_e = 0.5 * (D_e + D_e.T)
    if lam > 0:
        na, nb = nelec
        d0 = sum(dip_mo[i, i] for i in range(na)) + \
            sum(dip_mo[i, i] for i in range(nb))
        eye = np.eye(nd)
        H_e = H_e - d0 * D_e + 0.5 * d0 ** 2 * eye
        D_e = D_e - d0 * eye

    muz_mo = C.T @ mu_ao[2] @ C
    na = cistring.num_strings(norb, nelec[0])
    nbk = cistring.num_strings(norb, nelec[1])
    muz = np.zeros((nd, nd))
    for i in range(nd):
        ci = np.zeros((na, nbk)); ia, ib = divmod(i, nbk); ci[ia, ib] = 1.0
        muz[:, i] = direct_spin1.contract_1e(muz_mo, ci, norb, nelec).ravel()
    muz = 0.5 * (muz + muz.T)

    nph = nph_max + 1
    N = nd * nph
    H = np.zeros((N, N))
    for n in range(nph):
        r0, r1 = n * nd, (n + 1) * nd
        H[r0:r1, r0:r1] = H_e + omega * n * np.eye(nd)
        if n + 1 < nph:
            cpl = math.sqrt(omega / 2.0) * math.sqrt(n + 1) * D_e
            H[(n + 1) * nd:(n + 2) * nd, r0:r1] += cpl
            H[r0:r1, (n + 1) * nd:(n + 2) * nd] += cpl.T
    H = 0.5 * (H + H.T)
    E, V = np.linalg.eigh(H)
    psi0, E0 = V[:, 0], E[0]
    out = []
    for n in range(1, min(60, N)):
        psin = V[:, n]; dE = (E[n] - E0) * EV
        t = sum(psi0[b * nd:(b + 1) * nd] @ (muz @ psin[b * nd:(b + 1) * nd])
                for b in range(nph))
        f = (2.0 / 3.0) * (E[n] - E0) * t * t
        npho = sum(b * np.sum(psin[b * nd:(b + 1) * nd] ** 2)
                   for b in range(nph))
        out.append((dE, f, npho))
    return out


def fci_split(mol, omega_res, lam):
    fci = fci_roots(mol, omega_res / EV, lam)
    return _doublet_split([e for e, f, n in fci],
                          [n for e, f, n in fci], omega_res)


# ----------------------------------------------------------------------
mol = build_mol()

# lambda=0 bright state (resonance) and oscillator strength, each method.
E_bse0, f_bse0 = bright_bse(mol, 15.0 / EV, 0.0)
fci0 = fci_roots(mol, 15.0 / EV, 0.0)
E_fci0, f_fci0 = max(((e, f) for e, f, n in fci0 if WIN_LO <= e <= WIN_HI),
                     key=lambda t: t[1])

print(f"bright resonance (lambda=0):  BSE {E_bse0:.3f} eV (f={f_bse0:.3f}) "
      f" |  FCI {E_fci0:.3f} eV (f={f_fci0:.3f})")
print("\nResonant Rabi splitting Omega_R (eV):")
print(f"{'lambda':>7s} {'BSE':>8s} {'FCI':>8s} {'2-level':>8s} "
      f"{'BSE/FCI':>8s} {'BSE/2lvl':>9s}")

rows = []
for lam in LAMS:
    s_bse = bse_split(mol, E_bse0, lam)
    s_fci = fci_split(mol, E_fci0, lam)
    s_2lvl = math.sqrt(3.0) * lam * math.sqrt(f_bse0) * EV
    dev_fci = (s_bse - s_fci) / s_fci if s_fci else float('nan')
    dev_2lvl = (s_bse - s_2lvl) / s_2lvl if s_2lvl else float('nan')
    print(f"{lam:7.3f} {s_bse:8.4f} {s_fci:8.4f} {s_2lvl:8.4f} "
          f"{dev_fci:+7.1%} {dev_2lvl:+8.1%}")
    rows.append({'lambda': lam, 'Omega_R_BSE': s_bse, 'Omega_R_FCI': s_fci,
                 'Omega_R_2level': s_2lvl, 'dev_vs_FCI': dev_fci,
                 'dev_vs_2level': dev_2lvl})

results = {
    'system': 'H2O/STO-3G', 'gw_mode': 'evGW', 'tda': False,
    'resonance_eV': {'BSE': E_bse0, 'FCI': E_fci0},
    'f_lambda0': {'BSE': f_bse0, 'FCI': f_fci0},
    'scan': rows,
}
with open(os.path.join(HERE, 'qed_bse_polaritonic_fci_results.json'),
          'w') as fp:
    json.dump(results, fp, indent=1)
print("\nwrote qed_bse_polaritonic_fci_results.json")
