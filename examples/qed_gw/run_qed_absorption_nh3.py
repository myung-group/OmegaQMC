"""Cavity absorption of NH3/cc-pVDZ at two levels of QED theory.

NH3 analogue of :mod:`run_qed_absorption_h2o`. Two side-by-side spectra
of ammonia in an optical cavity, lambda-scanned (lambda = 0.00, 0.05,
0.10) and Gaussian-broadened from excitation energies Omega_m and
electronic oscillator strengths

    f_m = (2/3) * Omega_m * sum_kappa |sum_ai mu^kappa_ai (X+Y)_ai,m|^2 :

(a) QED-RPA                   -- photon explicit in the RPA excitation
                                 space.
(b) polaritonic QED-BSE@evGW  -- photon placed explicitly in the BSE
                                 excitation manifold (kept out of W to
                                 avoid double counting) on the
                                 eigenvalue-self-consistent evGW
                                 reference; the bright exciton
                                 Rabi-splits into lower/upper polaritons.

C3v ammonia: the cavity field is polarised along the C3 axis (z), so it
couples only to z-polarised A1 transitions. The cavity is therefore tuned
to NH3's lowest dipole-allowed A1 root -- 8.45 eV at the RPA level, and
its BSE analogue near 7.86 eV (the BSE cavity is retuned, at lambda = 0
where the photon is decoupled and tuning is self-consistency-free, to the
bright BSE singlet nearest the RPA resonance). The view is focused on this
resonant band: the A1 root Rabi-splits into lower/upper polaritons, while
the neighbouring ~10.1 eV degenerate E (x,y) peak -- dark to the z-cavity
-- stays put as an unsplit reference (NH3's dominant ~13.7 eV E feature
lies just above the window). Each curve is normalised to its tallest peak
in view and offset vertically. Writes fig_absorption_nh3.pdf into
OmegaQMC/paper/.
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyscf import gto

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_rpa import (_build_spin_orbital_quantities,
                                     _assemble_AB)
from OmegaQMC.addons.qed_gw import _solve_qed_rpa_eigensystem, run_qed_gw
from OmegaQMC.addons.qed_bse import run_qed_bse
from OmegaQMC.addons.qed_bse_polaritonic import run_qed_bse_polaritonic

EV = 27.211386245988
SIGMA_EV = 0.25          # Gaussian broadening
F_BRIGHT = 0.04          # oscillator-strength cut for a "bright" root

_rho, _z = 0.93786, -0.38129
mol = gto.M(atom=[['N', (0.0, 0.0, 0.0)],
            ['H', (_rho, 0.0, _z)],
            ['H', (-0.5 * _rho, 0.8660254 * _rho, _z)],
            ['H', (-0.5 * _rho, -0.8660254 * _rho, _z)]],
            basis='cc-pVDZ', unit='Angstrom', symmetry=False, verbose=0)
# Recenter at the nuclear centre of mass (paper convention): the QP and
# RPA/BSE excitation energies are origin-dependent at lambda > 0.
_masses = mol.atom_mass_list()
_coords = mol.atom_coords()                          # Bohr
mol.set_geom_(_coords - _masses @ _coords / _masses.sum(), unit='Bohr',
              symmetry=False)

LAMBDAS = (0.0, 0.05, 0.10)
COLORS = {0.0: '#444444', 0.05: '#d95f02', 0.10: '#7570b3'}


def _broaden(om_ev, f_osc, grid_ev):
    spec = np.zeros_like(grid_ev)
    for w0, f in zip(om_ev, f_osc):
        if f > 0.0:
            spec += f * np.exp(-0.5 * ((grid_ev - w0) / SIGMA_EV) ** 2)
    return spec


# ----------------------------------------------------------------------
# (a) photon-augmented QED-RPA
# ----------------------------------------------------------------------
def _rpa_solve(qedhf, omega_cav):
    """Photon-augmented QED-RPA roots (Ha) and oscillator strengths.

    The bilinear vertex g_ai = -sqrt(w_cav/2) d_ai is built from the
    coupling-scaled dipole d = lambda * mu, so at lambda = 0 the photon
    decouples (g = 0) and only the bare electronic RPA roots are bright.
    """
    so = _build_spin_orbital_quantities(qedhf)
    nocc, nso = so['nocc'], so['nso']
    nvir = nso - nocc
    nov = nvir * nocc

    A_t, B_t, d_vo_flat = _assemble_AB(so['F_so'], so['d_so'],
                                       so['g_phys_a'], nocc, nso,
                                       direct=False)
    g_vec = -math.sqrt(omega_cav / 2.0) * d_vo_flat
    dim = nov + 1
    A_big = np.zeros((dim, dim))
    B_big = np.zeros((dim, dim))
    A_big[:nov, :nov] = A_t
    A_big[:nov, -1] = A_big[-1, :nov] = g_vec
    A_big[-1, -1] = omega_cav
    B_big[:nov, :nov] = B_t
    B_big[:nov, -1] = B_big[-1, :nov] = g_vec
    Om, U, V = _solve_qed_rpa_eigensystem(A_big, B_big)
    XpY = (U + V)[:nov, :].reshape(nvir, nocc, -1)

    # Spin-orbital dipole matrices (interleaved even=alpha/odd=beta).
    C = np.asarray(qedhf['C'])
    idx = np.arange(nso)
    same = (idx[:, None] % 2) == (idx[None, :] % 2)
    f_osc = np.zeros_like(Om)
    for key in ('mu_x_ao', 'mu_y_ao', 'mu_z_ao'):
        mu_sf = C.T @ np.asarray(qedhf[key]) @ C
        mu_so = same * mu_sf[idx[:, None] // 2, idx[None, :] // 2]
        mu_m = np.einsum('ai,ais->s', mu_so[nocc:, :nocc], XpY)
        f_osc += (2.0 / 3.0) * Om * mu_m ** 2
    return Om, f_osc


def _tune_rpa_cavity():
    """Lowest dipole-allowed RPA root (Ha); tuning-free at lambda = 0."""
    qedhf0 = run_qed_hf(mol, 0.4, (0.0, 0.0, 0.0), verbose=False, tol=1e-12)
    Om, f = _rpa_solve(qedhf0, 0.4)
    return float(Om[f > F_BRIGHT].min())


OMEGA_RPA = _tune_rpa_cavity()
print(f"RPA cavity tuned to lowest bright RPA root: "
      f"omega_cav = {OMEGA_RPA:.6f} Ha = {OMEGA_RPA * EV:.2f} eV")


def rpa_spectrum(lam, grid_ev):
    qedhf = run_qed_hf(mol, OMEGA_RPA, (0.0, 0.0, lam), verbose=False,
                       tol=1e-12)
    Om, f_osc = _rpa_solve(qedhf, OMEGA_RPA)
    om_ev = Om * EV
    return _broaden(om_ev, f_osc, grid_ev), om_ev, f_osc


# ----------------------------------------------------------------------
# BSE cavity frequency: the bright BSE singlet nearest the RPA resonance.
# The evGW QP energies are cached per lambda and shared by panels (b),(c).
# ----------------------------------------------------------------------
def _tune_bse_cavity():
    qedhf0 = run_qed_hf(mol, OMEGA_RPA, (0.0, 0.0, 0.0), verbose=False,
                        tol=1e-12)
    bse0 = run_qed_bse(qedhf0, gw_mode='evGW', tda=True,
                       include_photon=True, verbose=False)
    bright = (bse0['f_osc'] > F_BRIGHT) & (bse0['spin'] == 'S')
    cand = bse0['Omega_BSE'][bright]
    return float(cand[np.argmin(np.abs(cand - OMEGA_RPA))])


OMEGA_BSE = _tune_bse_cavity()
print(f"BSE cavity tuned to bright BSE singlet:       "
      f"omega_cav = {OMEGA_BSE:.6f} Ha = {OMEGA_BSE * EV:.2f} eV")

_bse_ctx = {}


def _bse_context(lam):
    """(qedhf, eps_QP) at omega_cav = OMEGA_BSE; one evGW per lambda."""
    if lam not in _bse_ctx:
        qedhf = run_qed_hf(mol, OMEGA_BSE, (0.0, 0.0, lam), verbose=False,
                           tol=1e-12)
        gw = run_qed_gw(qedhf, mode='evGW', verbose=False)
        _bse_ctx[lam] = (qedhf, np.asarray(gw['eps_GW_all']))
    return _bse_ctx[lam]


# ----------------------------------------------------------------------
# (b) polaritonic QED-BSE@G0W0  (photon explicit -> Rabi splitting;
#     one-shot G0W0 quasiparticle reference)
# ----------------------------------------------------------------------
def bse_pol_g0w0_spectrum(lam, grid_ev):
    qedhf, _eps_QP = _bse_context(lam)
    pol = run_qed_bse_polaritonic(qedhf, gw_mode='G0W0', tda=False,
                                  verbose=False)
    om_ev = pol['Omega'] * EV
    f_osc = pol['f_osc']
    return _broaden(om_ev, f_osc, grid_ev), om_ev, f_osc


# ----------------------------------------------------------------------
# (c) polaritonic QED-BSE@evGW  (photon explicit -> Rabi splitting;
#     eigenvalue-self-consistent evGW quasiparticle reference)
# ----------------------------------------------------------------------
def bse_pol_evgw_spectrum(lam, grid_ev):
    qedhf, eps_QP = _bse_context(lam)
    pol = run_qed_bse_polaritonic(qedhf, gw_mode='evGW', tda=False,
                                  eps_QP=eps_QP, verbose=False)
    om_ev = pol['Omega'] * EV
    f_osc = pol['f_osc']
    return _broaden(om_ev, f_osc, grid_ev), om_ev, f_osc


# ----------------------------------------------------------------------
# Assemble the two-panel figure.
# ----------------------------------------------------------------------
PANELS = (
    ('(a) QED-RPA', 'photon explicit', OMEGA_RPA, rpa_spectrum),
    (r'(b) polaritonic QED-BSE@ev$GW$', 'photon explicit', OMEGA_BSE,
     bse_pol_evgw_spectrum),
)

grid = np.linspace(6.0, 12.0, 1501)
offset = 0.85
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.4), dpi=300, sharey=True)

for ax, (title, sub, omega_cav, specfun) in zip(axes, PANELS):
    print(f"\n=== {title} (omega_cav = {omega_cav * EV:.2f} eV) ===")
    for k, lam in enumerate(LAMBDAS):
        spec, om_ev, f_osc = specfun(lam, grid)
        peak = spec.max()
        norm = spec / peak if peak > 0 else spec
        ax.plot(grid, norm + k * offset, lw=1.3, color=COLORS[lam])
        ax.text(6.7, k * offset + 0.07, rf'$\lambda = {lam:.2f}$',
                fontsize=8, color=COLORS[lam])
        bright = f_osc > 1e-3
        print(f"  lambda={lam:.2f}: bright roots (eV, f):")
        for w0, f in zip(om_ev[bright][:8], f_osc[bright][:8]):
            print(f"     {w0:8.3f}  {f:8.4f}")

    ocav_ev = omega_cav * EV
    ax.axvline(ocav_ev, color='k', ls=':', lw=0.8)
    ax.text(ocav_ev + 0.3, offset + 0.62,
            rf'$\omega_\mathrm{{cav}} = {ocav_ev:.2f}$ eV',
            fontsize=8, ha='left')
    ax.text(0.5, 0.965, sub, transform=ax.transAxes, fontsize=8.5,
            style='italic', color='#888888', ha='center', va='top')
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(r'$\omega$ (eV)')
    ax.set_xlim(6.6, 11.6)
    ax.set_xticks([7, 8, 9, 10, 11])

axes[0].set_ylabel('absorption (norm., offset)')
axes[0].set_ylim(-0.08, 2 * offset + 1.25)
axes[0].set_yticks([])
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', '..', 'OmegaQMC',
                   'paper', 'fig_absorption_nh3.pdf')
fig.savefig(os.path.abspath(out))
fig.savefig(os.path.abspath(out)[:-4] + '.png')
print(f"\nwrote {os.path.abspath(out)}")
