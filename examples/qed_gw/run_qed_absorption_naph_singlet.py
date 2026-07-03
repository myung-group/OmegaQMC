"""Cavity absorption of naphthalene/cc-pVDZ — singlet-adapted path.

Same physics and layout as run_qed_absorption_naph.py (two side-by-side
lambda-scanned, Gaussian-broadened spectra: (a) QED-RPA with the photon
explicit in the excitation space, (b) polaritonic QED-BSE@evGW), but
computed with :mod:`OmegaQMC.addons.qed_polariton_singlet`: the
excitation space is the spatial singlet block (N_ov ~ 5e3 instead of the
spin-orbital ~2e4), which brings the working set from ~40 GB down to
~3 GB — this figure is computable on the 8 GB laptop.

Validation: the singlet-adapted module reproduces the spin-orbital
spectra/oscillator strengths/photon weights to machine precision on
closed-shell references (see tools/qed_ccsd_df_derivation/).

Writes fig_absorption_naph_singlet.pdf into OmegaQMC/paper/ and
qed_absorption_naph_singlet_results.json next to this script.
"""
import json
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyscf import gto

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_polariton_singlet import (
    run_qed_rpa_singlet, run_qed_gw_singlet,
    run_qed_bse_polaritonic_singlet)

EV = 27.211386245988
SIGMA_EV = 0.15          # Gaussian broadening (eV)
F_BRIGHT = 0.03          # oscillator-strength cut for a "bright" root
N_RINGS = 2              # 2 = naphthalene
BASIS = 'cc-pVDZ'
AUXBASIS = 'cc-pvdz-ri'  # DF factor for the two-electron integrals
LAMBDAS = (0.0, 0.05, 0.10)
COLORS = {0.0: '#444444', 0.05: '#d95f02', 0.10: '#7570b3'}


def acene(n_rings, aCC=1.40, aCH=1.09):
    """Idealized planar linear acene in the x-z plane (short axis = z),
    same construction as run_qed_absorption_naph.py."""
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
print(f"naphthalene / {BASIS}: nao = {mol.nao}, nelec = {mol.nelectron}")


def _broaden(om_ev, f_osc, grid_ev):
    spec = np.zeros_like(grid_ev)
    for w0, f in zip(om_ev, f_osc):
        if f > 0.0:
            spec += f * np.exp(-0.5 * ((grid_ev - w0) / SIGMA_EV) ** 2)
    return spec


def _qedhf(omega_cav, lam):
    return run_qed_hf(mol, omega_cav, (0.0, 0.0, lam), verbose=False,
                      tol=1e-12, auxbasis=AUXBASIS)


# ----------------------------------------------------------------------
# (a) photon-augmented QED-RPA (singlet)
# ----------------------------------------------------------------------
t0 = time.time()
qedhf0 = _qedhf(0.2, 0.0)
r0 = run_qed_rpa_singlet(qedhf0, direct=False, verbose=False)
OMEGA_RPA = float(r0['Omega'][r0['f_osc'] > F_BRIGHT].min())
print(f"RPA cavity tuned to lowest bright RPA singlet: "
      f"omega_cav = {OMEGA_RPA:.6f} Ha = {OMEGA_RPA * EV:.2f} eV"
      f"   [{time.time() - t0:.0f} s]")


def rpa_spectrum(lam, grid_ev):
    qedhf = _qedhf(OMEGA_RPA, lam)
    r = run_qed_rpa_singlet(qedhf, direct=False, verbose=False)
    return (_broaden(r['Omega'] * EV, r['f_osc'], grid_ev),
            r['Omega'] * EV, r['f_osc'])


# ----------------------------------------------------------------------
# (b) polaritonic QED-BSE@evGW (singlet)
# ----------------------------------------------------------------------
_bse_ctx = {}


def _bse_context(omega_cav, lam):
    # At lam = 0 the photon decouples from the screening, so the evGW QP
    # energies are independent of omega_cav — share them between the
    # cavity-tuning step and the lam = 0 panel. QP energies are also
    # cached on disk so an interrupted run resumes without redoing evGW.
    key = 'lam0' if lam == 0.0 else f"w{omega_cav:.8f}_l{lam}"
    if key not in _bse_ctx:
        here = os.path.dirname(os.path.abspath(__file__))
        cache = os.path.join(here, f'qed_naph_evgw_{key}.npy')
        if os.path.exists(cache):
            _bse_ctx[key] = np.load(cache)
            print(f"    evGW(lam={lam}) loaded from {cache}")
        else:
            t1 = time.time()
            qedhf = _qedhf(omega_cav, lam)
            gw = run_qed_gw_singlet(qedhf, mode='evGW', tol=1e-5,
                                    verbose=False)
            print(f"    evGW(lam={lam}) done  [{time.time() - t1:.0f} s]")
            np.save(cache, gw['eps_QP'])
            _bse_ctx[key] = gw['eps_QP']
    # rebuild the (cheap) QED-HF reference at the requested omega_cav so
    # the BSE photon diagonal is correct even when QP energies are shared
    return _qedhf(omega_cav, lam), _bse_ctx[key]


t0 = time.time()
qedhf_t, eps_t = _bse_context(OMEGA_RPA, 0.0)
bse0 = run_qed_bse_polaritonic_singlet(qedhf_t, tda=False, eps_QP=eps_t,
                                       verbose=False)
bright = bse0['f_osc'] > F_BRIGHT
cand = bse0['Omega'][bright]
OMEGA_BSE = float(cand[np.argmin(np.abs(cand - OMEGA_RPA))])
print(f"BSE cavity tuned to bright BSE singlet:        "
      f"omega_cav = {OMEGA_BSE:.6f} Ha = {OMEGA_BSE * EV:.2f} eV"
      f"   [{time.time() - t0:.0f} s]")


def bse_pol_spectrum(lam, grid_ev):
    qedhf, eps_QP = _bse_context(OMEGA_BSE, lam)
    pol = run_qed_bse_polaritonic_singlet(qedhf, tda=False, eps_QP=eps_QP,
                                          verbose=False)
    return (_broaden(pol['Omega'] * EV, pol['f_osc'], grid_ev),
            pol['Omega'] * EV, pol['f_osc'])


# ----------------------------------------------------------------------
# assemble the figure
# ----------------------------------------------------------------------
def main():
    grid = np.linspace(2.0, 10.0, 1200)
    results = {'omega_rpa_Ha': OMEGA_RPA, 'omega_bse_Ha': OMEGA_BSE,
               'basis': BASIS, 'auxbasis': AUXBASIS,
               'sigma_eV': SIGMA_EV, 'lambdas': list(LAMBDAS),
               'panels': {}}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=False)
    panels = (('a', 'QED-RPA (singlet)', rpa_spectrum, OMEGA_RPA),
              ('b', 'polaritonic QED-BSE@evGW (singlet)',
               bse_pol_spectrum, OMEGA_BSE))
    for ax, (tag, title, solver, om_cav) in zip(axes, panels):
        pdat = {}
        for lam in LAMBDAS:
            t1 = time.time()
            spec, om_ev, f_osc = solver(lam, grid)
            print(f"  panel ({tag})  lambda = {lam}:"
                  f" {time.time() - t1:.0f} s")
            ax.plot(grid, spec, color=COLORS[lam], lw=1.6,
                    label=rf"$\lambda_z = {lam}$")
            keep = f_osc > 1e-4
            pdat[str(lam)] = {
                'omega_eV': [float(x) for x in om_ev[keep]],
                'f_osc': [float(x) for x in f_osc[keep]],
            }
        ax.axvline(om_cav * EV, color='k', ls=':', lw=0.8, alpha=0.6)
        ax.set_title(f"({tag}) {title}", fontsize=10)
        ax.set_xlabel(r"$\omega$ (eV)")
        ax.set_xlim(grid[0], grid[-1])
        results['panels'][tag] = pdat
    axes[0].set_ylabel("absorption (arb. units)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()

    here = os.path.dirname(os.path.abspath(__file__))
    paper = os.path.join(here, '..', 'OmegaQMC', 'paper')
    figpath = os.path.join(paper, 'fig_absorption_naph_singlet.pdf')
    fig.savefig(figpath, bbox_inches='tight')
    print(f"figure written to {figpath}")

    jpath = os.path.join(here, 'qed_absorption_naph_singlet_results.json')
    with open(jpath, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"results written to {jpath}")


if __name__ == '__main__':
    main()
