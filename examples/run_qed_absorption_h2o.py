"""QED-RPA absorption spectrum of H2O/cc-pVDZ in an optical cavity.

Cavity analogue of Fig. 2 of Marie, Ammar, and Loos [arXiv:2311.05351]:
the low-frequency absorption spectrum is built from the photon-augmented
QED-RPA excitation energies Omega_m and electronic oscillator strengths

    f_m = (2/3) * Omega_m * sum_kappa |sum_ai mu^kappa_ai (X+Y)_ai,m|^2,

convolved with Gaussians. At lambda = 0 the photon mode is dark and the
spectrum is the bare RPA spectrum; at finite coupling the resonant band
(omega_cav = 0.415668 Ha tuned to the lowest dipole-allowed excitation)
Rabi-splits into lower/upper polaritons whose splitting grows with
lambda. Writes fig_absorption_h2o.pdf into OmegaQMC/paper/.
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
from OmegaQMC.addons.qed_gw import _solve_qed_rpa_eigensystem

EV = 27.211386245988
OMEGA = 0.415668
SIGMA_EV = 0.25          # Gaussian broadening

half = math.radians(104.5 / 2.0)
hx, hz = math.sin(half), -math.cos(half)
mol = gto.M(atom=[['O', (0, 0, 0)], ['H', (hx, 0, hz)], ['H', (-hx, 0, hz)]],
            basis='cc-pVDZ', unit='Angstrom', symmetry=False, verbose=0)


def spectrum(lam, grid_ev):
    qedhf = run_qed_hf(mol, OMEGA, (0.0, 0.0, lam), verbose=False, tol=1e-12)
    so = _build_spin_orbital_quantities(qedhf)
    nocc, nso = so['nocc'], so['nso']
    nvir = nso - nocc
    nov = nvir * nocc

    # Full (antisymmetric) QED-RPA — physical absorption spectrum.
    A_t, B_t, d_vo_flat = _assemble_AB(so['F_so'], so['d_so'],
                                       so['g_phys_a'], nocc, nso,
                                       direct=False)
    g_vec = -math.sqrt(OMEGA / 2.0) * d_vo_flat
    dim = nov + 1
    A_big = np.zeros((dim, dim))
    B_big = np.zeros((dim, dim))
    A_big[:nov, :nov] = A_t
    A_big[:nov, -1] = A_big[-1, :nov] = g_vec
    A_big[-1, -1] = OMEGA
    B_big[:nov, :nov] = B_t
    B_big[:nov, -1] = B_big[-1, :nov] = g_vec
    Om, U, V = _solve_qed_rpa_eigensystem(A_big, B_big)
    XpY = (U + V)[:nov, :].reshape(nvir, nocc, -1)
    ph_weight = (U + V)[-1, :] ** 2

    # Spin-orbital dipole matrices (interleaved even=alpha/odd=beta).
    C = np.asarray(qedhf['C'])
    idx = np.arange(nso)
    same = (idx[:, None] % 2) == (idx[None, :] % 2)
    f_osc = np.zeros_like(Om)
    for key in ('mu_x_ao', 'mu_y_ao', 'mu_z_ao'):
        mu_sf = C.T @ np.asarray(qedhf[key]) @ C
        mu_so = same * mu_sf[idx[:, None] // 2, idx[None, :] // 2]
        mu_vo = mu_so[nocc:, :nocc]
        mu_m = np.einsum('ai,ais->s', mu_vo, XpY)
        f_osc += (2.0 / 3.0) * Om * mu_m ** 2

    om_ev = Om * EV
    spec = np.zeros_like(grid_ev)
    for w0, f in zip(om_ev, f_osc):
        spec += f * np.exp(-0.5 * ((grid_ev - w0) / SIGMA_EV) ** 2)
    return spec, om_ev, f_osc, ph_weight


grid = np.linspace(7.0, 16.0, 1801)
fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=300)
colors = {0.0: '#555555', 0.05: '#d95f02', 0.10: '#7570b3'}
offset = 0.42
for k, lam in enumerate((0.0, 0.05, 0.10)):
    spec, om_ev, f_osc, phw = spectrum(lam, grid)
    ax.plot(grid, spec + k * offset, lw=1.2, color=colors[lam])
    ax.text(7.2, k * offset + 0.06, rf'$\lambda = {lam:.2f}$',
            fontsize=8, color=colors[lam])
    strong = f_osc > 1e-3
    print(f"lambda={lam:.2f}: bright roots (eV, f, photon weight):")
    for w0, f, w in zip(om_ev[strong][:8], f_osc[strong][:8],
                        phw[strong][:8]):
        print(f"   {w0:8.3f}  {f:8.4f}  {w:6.3f}")

ax.axvline(OMEGA * EV, color='k', ls=':', lw=0.8)
ax.text(OMEGA * EV - 0.12, 2 * offset + 0.32, r'$\omega_\mathrm{cav}$',
        fontsize=8, ha='right')
ax.set_xlabel(r'$\omega$ (eV)')
ax.set_ylabel('absorption (arb. units, offset)')
ax.set_xlim(7, 16)
ax.set_yticks([])
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'OmegaQMC', 'paper',
                   'fig_absorption_h2o.pdf')
fig.savefig(os.path.abspath(out))
print(f"\nwrote {os.path.abspath(out)}")
