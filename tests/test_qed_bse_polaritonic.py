"""Regression tests for the polaritonic QED-BSE@GW (qed_bse_polaritonic).

Covers:
* lambda=0 structure: the polaritonic spectrum is the electronic BSE
  exciton spectrum plus one DARK photon sitting at omega_cav.
* two-level Rabi law: tuned to a bright BSE state, the resonant splitting
  follows Omega_R = sqrt(3) * lambda * sqrt(f) (light-matter coupling
  implemented correctly), and scales linearly in lambda.
* FCI anchor (slow): the resonant Rabi splitting matches exact polaritonic
  QED-FCI (STO-3G water) to within ~15% at a well-defined bright state.
  The residual is exciton/oscillator-strength quality, not construction.

Tolerances are loose enough for numerical drift but tight enough to catch
a wrong coupling prefactor, a dropped photon row, or a sign error.
"""
import math

import numpy as np
import pytest
from pyscf import gto, scf, ao2mo
from pyscf.fci import direct_spin1, cistring

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_bse import run_qed_bse
from OmegaQMC.addons.qed_bse_polaritonic import run_qed_bse_polaritonic
from OmegaQMC.addons.qed_fci import _build_fci_matrices

EV = 27.211386245988


@pytest.fixture(scope='module')
def mol():
    half = math.radians(104.5 / 2.0)
    hx, hz = math.sin(half), -math.cos(half)
    return gto.M(atom=[['O', (0, 0, 0)], ['H', (hx, 0, hz)],
                       ['H', (-hx, 0, hz)]],
                 basis='sto-3g', unit='Angstrom', symmetry=False, verbose=0)


def _bright_bse(mol, omega, lam, lo=13.0, hi=18.0):
    """(energy_eV, f) of the brightest polaritonic-BSE root in a window."""
    qh = run_qed_hf(mol, omega, (0, 0, lam), verbose=False, tol=1e-12)
    pol = run_qed_bse_polaritonic(qh, gw_mode='evGW', tda=False, verbose=False)
    om = np.asarray(pol['Omega']) * EV
    f = np.asarray(pol['f_osc'])
    m = (om >= lo) & (om <= hi)
    i = np.where(m)[0][np.argmax(f[m])]
    return float(om[i]), float(f[i])


def test_lambda0_structure(mol):
    """At lambda=0: electronic excitons unchanged + one dark photon at w_cav."""
    omega = 15.0 / EV
    qh = run_qed_hf(mol, omega, (0, 0, 0.0), verbose=False, tol=1e-12)
    pol = run_qed_bse_polaritonic(qh, gw_mode='evGW', tda=False, verbose=False)
    std = run_qed_bse(qh, gw_mode='evGW', tda=False, verbose=False)

    om_pol = np.asarray(pol['Omega'])
    pw = np.asarray(pol['photon_weight'])

    # exactly one almost-pure photon root, at omega_cav, optically dark
    iph = int(np.argmax(pw))
    assert pw[iph] > 0.99
    assert abs(om_pol[iph] * EV - 15.0) < 1e-3
    assert pol['f_osc'][iph] < 1e-6

    # the remaining (electronic) roots reproduce the standard BSE excitons
    elec = np.sort(np.delete(om_pol, iph))
    std_om = np.sort(np.asarray(std['Omega_BSE']))
    n = min(len(elec), len(std_om), 8)
    assert np.max(np.abs(elec[:n] - std_om[:n])) < 1e-8


def test_two_level_rabi_law(mol):
    """Resonant splitting follows Omega_R = sqrt(3) lambda sqrt(f), linear in lambda."""
    # find a bright state at lambda=0 and tune the cavity onto it
    E0, f0 = _bright_bse(mol, 15.0 / EV, 0.0)
    omega = E0 / EV

    splits = {}
    for lam in (0.02, 0.04):
        qh = run_qed_hf(mol, omega, (0, 0, lam), verbose=False, tol=1e-12)
        pol = run_qed_bse_polaritonic(qh, gw_mode='evGW', tda=False, verbose=False)
        om = np.asarray(pol['Omega']) * EV
        f = np.asarray(pol['f_osc'])
        m = (om >= E0 - 2.0) & (om <= E0 + 2.0) & (f > 1e-4)
        es = np.sort(om[m])
        splits[lam] = float(es[-1] - es[0])
        pred = math.sqrt(3.0) * lam * math.sqrt(f0) * EV
        # within 6% of the analytic two-level resonant splitting
        assert abs(splits[lam] - pred) / pred < 0.06, (lam, splits[lam], pred)

    # splitting scales linearly with lambda (doubling lambda ~ doubles split)
    ratio = splits[0.04] / splits[0.02]
    assert abs(ratio - 2.0) < 0.15


# ---------------------------------------------------------------------------
# Exact FCI anchor (slow): polaritonic-BSE splitting vs polaritonic QED-FCI.
# ---------------------------------------------------------------------------

def _doublet_split(energies, photon_wt, e_center, half_window=1.5):
    """Rabi splitting = |dE| between the two highest-photon-weight states
    within +/- half_window of e_center (these are the LP/UP polaritons)."""
    e = np.asarray(energies)
    w = np.asarray(photon_wt)
    m = np.abs(e - e_center) <= half_window
    e_in, w_in = e[m], w[m]
    if np.sum(w_in > 0.02) < 2:
        return 0.0
    top2 = np.argsort(w_in)[-2:]
    pair = np.sort(e_in[top2])
    return float(pair[1] - pair[0])


def _fci_bright_split(mol, omega, lam, nph_max=4):
    """Polaritonic-FCI roots as (energy_eV, f, photon_number)."""
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

    H_e, D_e, nd = _build_fci_matrices(h1e_used, eri_dse, dip_mo, norb, nelec, enuc)
    H_e = 0.5 * (H_e + H_e.T); D_e = 0.5 * (D_e + D_e.T)
    if lam > 0:
        na, nb = nelec
        d0 = sum(dip_mo[i, i] for i in range(na)) + sum(dip_mo[i, i] for i in range(nb))
        eye = np.eye(nd)
        H_e = H_e - d0 * D_e + 0.5 * d0 ** 2 * eye
        D_e = D_e - d0 * eye

    # bare z-dipole in the FCI basis (z-polarised cavity -> z-bright)
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
            H[(n+1)*nd:(n+2)*nd, r0:r1] += cpl
            H[r0:r1, (n+1)*nd:(n+2)*nd] += cpl.T
    H = 0.5 * (H + H.T)
    E, V = np.linalg.eigh(H)
    psi0, E0 = V[:, 0], E[0]
    out = []
    for n in range(1, min(60, N)):
        psin = V[:, n]; dE = (E[n] - E0) * EV
        t = sum(psi0[b*nd:(b+1)*nd] @ (muz @ psin[b*nd:(b+1)*nd]) for b in range(nph))
        f = (2.0 / 3.0) * (E[n] - E0) * t * t
        npho = sum(b * np.sum(psin[b*nd:(b+1)*nd] ** 2) for b in range(nph))
        out.append((dE, f, npho))
    return out


@pytest.mark.slow
def test_fci_rabi_splitting(mol):
    """Polaritonic-BSE resonant splitting matches exact FCI to ~15%."""
    # FCI brightest state at lambda=0 -> resonance
    f0 = _fci_bright_split(mol, 15.0 / EV, 0.0)
    bright = sorted([(e, f) for e, f, _ in f0 if 13.0 <= e <= 18.0 and f > 1e-3],
                    key=lambda t: -t[1])
    E_res = bright[0][0]
    omega = E_res / EV

    # FCI splitting at lambda=0.05 (LP/UP = two highest photon-number states)
    fci = _fci_bright_split(mol, omega, 0.05)
    split_fci = _doublet_split([e for e, f, n in fci],
                               [n for e, f, n in fci], E_res)

    # polaritonic-BSE splitting at its own resonance, lambda=0.05
    Eb, fb = _bright_bse(mol, 15.0 / EV, 0.0)
    qh = run_qed_hf(mol, Eb / EV, (0, 0, 0.05), verbose=False, tol=1e-12)
    pol = run_qed_bse_polaritonic(qh, gw_mode='evGW', tda=False, verbose=False)
    split_bse = _doublet_split(np.asarray(pol['Omega']) * EV,
                               np.asarray(pol['photon_weight']), Eb)

    assert split_fci > 0.3            # sanity: there is a real splitting
    assert abs(split_bse - split_fci) / split_fci < 0.15, (split_bse, split_fci)
