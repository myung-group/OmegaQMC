"""Cavity absorption of naphthalene/cc-pVDZ at two levels of QED theory
-- a replacement candidate for Fig. 2 of paper/main.tex (in place of NH3).

Naphthalene analogue of :mod:`run_qed_absorption_nh3`. Two side-by-side
spectra, lambda-scanned (lambda = 0.00, 0.05, 0.10) and Gaussian-broadened
from excitation energies Omega_m and oscillator strengths

    f_m = (2/3) * Omega_m * sum_kappa |sum_ai mu^kappa_ai (X+Y)_ai,m|^2 :

(a) QED-RPA                   -- photon explicit in the RPA excitation space.
(b) polaritonic QED-BSE@evGW  -- photon placed explicitly in the BSE
                                 excitation manifold (kept out of W to avoid
                                 double counting) on the evGW reference;
                                 the bright exciton Rabi-splits into
                                 lower/upper polaritons.

Orientation: the idealized planar acene lies in the x-z plane with the
long (spine) axis along x and the SHORT in-plane axis along z, so the
z-polarized cavity couples to the short-axis-polarized bright transition
(the strong 1La-type band of the acene). The cavity is auto-tuned, at
lambda = 0 (photon decoupled, tuning-free), to the lowest bright root:
OMEGA_RPA for panel (a) and the nearest bright BSE singlet for panel (b).

=====================================================================
MEMORY WARNING -- run on a large-RAM machine.
The QED-RPA (panel a) and the full polaritonic QED-BSE (panel b) both
dense-diagonalize a matrix of dimension ~2*Nov. For naphthalene/cc-pVDZ
(180 AO, Nov ~ 2e4) that matrix is ~13 GB (needs ~40 GB working set with
la.eig). Not feasible in 8 GB. Reduce to a smaller basis (e.g. 6-31G) or
use a bigger machine. Writes fig_absorption_naph.pdf into OmegaQMC/paper/.
=====================================================================
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
SIGMA_EV = 0.15          # Gaussian broadening (eV)
F_BRIGHT = 0.03          # oscillator-strength cut for a "bright" root
N_RINGS = 2              # 2 = naphthalene (1 benzene, 3 anthracene)
BASIS = 'cc-pVDZ'


def acene(n_rings, aCC=1.40, aCH=1.09):
    """Idealized planar linear acene in the x-z plane (short axis = z)."""
    ang = np.deg2rad([30, 90, 150, 210, 270, 330])
    cx0 = (n_rings - 1) / 2.0 * math.sqrt(3.0) * aCC
    verts = []
    for k in range(n_rings):
        cx = k * math.sqrt(3.0) * aCC - cx0
        for t in ang:
            verts.append((cx + aCC * math.cos(t), aCC * math.sin(t)))
    uniq = []
    for v in verts:
        if not any(abs(v[0] - u[0]) < 1e-3 and abs(v[1] - u[1]) < 1e-3
                   for u in uniq):
            uniq.append(v)
    C = np.array(uniq)
    nbr = [[j for j in range(len(C)) if j != i
            and np.linalg.norm(C[i] - C[j]) < 1.2 * aCC]
           for i in range(len(C))]
    atoms = [['C', (float(c[0]), 0.0, float(c[1]))] for c in C]
    for i, c in enumerate(C):
        if len(nbr[i]) == 2:
            d = c - C[nbr[i]].mean(axis=0)
            d = d / np.linalg.norm(d)
            h = c + aCH * d
            atoms.append(['H', (float(h[0]), 0.0, float(h[1]))])
    return atoms


mol = gto.M(atom=acene(N_RINGS), basis=BASIS, unit='Angstrom',
            symmetry=False, verbose=0)

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
    qedhf0 = run_qed_hf(mol, 0.2, (0.0, 0.0, 0.0), verbose=False, tol=1e-12)
    Om, f = _rpa_solve(qedhf0, 0.2)
    return float(Om[f > F_BRIGHT].min())


OMEGA_RPA = _tune_rpa_cavity()
print(f"RPA cavity tuned to lowest bright RPA root: "
      f"omega_cav = {OMEGA_RPA:.6f} Ha = {OMEGA_RPA * EV:.2f} eV")


def rpa_spectrum(lam, grid_ev):
    qedhf = run_qed_hf(mol, OMEGA_RPA, (0.0, 0.0, lam), verbose=False,
                       tol=1e-12)
    Om, f_osc = _rpa_solve(qedhf, OMEGA_RPA)
    return _broaden(Om * EV, f_osc, grid_ev), Om * EV, f_osc


# ----------------------------------------------------------------------
# BSE cavity frequency: bright BSE singlet nearest the RPA resonance.
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
    if lam not in _bse_ctx:
        qedhf = run_qed_hf(mol, OMEGA_BSE, (0.0, 0.0, lam), verbose=False,
                           tol=1e-12)
        gw = run_qed_gw(qedhf, mode='evGW', verbose=False)
        _bse_ctx[lam] = (qedhf, np.asarray(gw['eps_GW_all']))
    return _bse_ctx[lam]


# ----------------------------------------------------------------------
# (b) polaritonic QED-BSE@evGW  (photon explicit -> Rabi splitting)
# ----------------------------------------------------------------------
def bse_pol_evgw_spectrum(lam, grid_ev):
    qedhf, eps_QP = _bse_context(lam)
    pol = run_qed_bse_polaritonic(qedhf, gw_mode='evGW', tda=False,
                                  eps_QP=eps_QP, verbose=False)
    return _broaden(pol['Omega'] * EV, pol['f_osc'], grid_ev), \
        pol['Omega'] * EV, pol['f_osc']


# ----------------------------------------------------------------------
# Assemble the two-panel figure. The frequency window is centred on the
# tuned cavity so it adapts to whatever acene/basis is chosen.
# ----------------------------------------------------------------------
PANELS = (
    ('(a) QED-RPA', 'photon explicit', OMEGA_RPA, rpa_spectrum),
    (r'(b) polaritonic QED-BSE@ev$GW$', 'photon explicit', OMEGA_BSE,
     bse_pol_evgw_spectrum),
)

c_ev = 0.5 * (OMEGA_RPA + OMEGA_BSE) * EV
lo, hi = c_ev - 2.5, c_ev + 3.5
grid = np.linspace(lo - 0.5, hi + 0.5, 1801)
offset = 0.85
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.4), dpi=300, sharey=True)

for ax, (title, sub, omega_cav, specfun) in zip(axes, PANELS):
    print(f"\n=== {title} (omega_cav = {omega_cav * EV:.2f} eV) ===")
    for k, lam in enumerate(LAMBDAS):
        spec, om_ev, f_osc = specfun(lam, grid)
        peak = spec.max()
        norm = spec / peak if peak > 0 else spec
        ax.plot(grid, norm + k * offset, lw=1.3, color=COLORS[lam])
        ax.text(lo + 0.1, k * offset + 0.07, rf'$\lambda = {lam:.2f}$',
                fontsize=8, color=COLORS[lam])
        bright = f_osc > 1e-3
        print(f"  lambda={lam:.2f}: bright roots (eV, f):")
        for w0, f in zip(om_ev[bright][:8], f_osc[bright][:8]):
            print(f"     {w0:8.3f}  {f:8.4f}")

    ocav_ev = omega_cav * EV
    ax.axvline(ocav_ev, color='k', ls=':', lw=0.8)
    ax.text(ocav_ev + 0.2, offset + 0.62,
            rf'$\omega_\mathrm{{cav}} = {ocav_ev:.2f}$ eV',
            fontsize=8, ha='left')
    ax.text(0.5, 0.965, sub, transform=ax.transAxes, fontsize=8.5,
            style='italic', color='#888888', ha='center', va='top')
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(r'$\omega$ (eV)')
    ax.set_xlim(lo, hi)

axes[0].set_ylabel('absorption (norm., offset)')
axes[0].set_ylim(-0.08, 2 * offset + 1.25)
axes[0].set_yticks([])
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'OmegaQMC', 'paper',
                   'fig_absorption_naph.pdf')
fig.savefig(os.path.abspath(out))
fig.savefig(os.path.abspath(out)[:-4] + '.png')
print(f"\nwrote {os.path.abspath(out)}")
