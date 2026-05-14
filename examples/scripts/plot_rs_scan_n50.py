#!/usr/bin/env python
"""Plot the N=50 rs-scan results: E(rs) and S(k_Bragg)(rs) for fluid
and crystal sectors, plus identify the energy crossing rs_c.

Reads runs/rs_scan_n50_rs<rs>_<sector>/summary.json for
sector in {fluid, crystal}, rs in {20, 30, 35, 38, 40, 45, 50, 60}.

Usage:  python scripts/plot_rs_scan_n50.py [--out figure.png]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Smith 2024 reference values (Table I, N=56). Bragg peaks averaged.
SMITH_2024 = {
    20: {'e': -0.04637,  'sk': 1.27},
    30: {'e': -0.03197,  'sk': 1.44},
    35: {'e': -0.02770,  'sk': 1.52},
    38: {'e': -0.02565,  'sk': 4.4},
    40: {'e': -0.02445,  'sk': 5.1},
    45: {'e': -0.02190,  'sk': 6.4},
    50: {'e': -0.01983,  'sk': 7.58},
}


def load_run(rs, sector, base='runs'):
    p = Path(base) / f'rs_scan_n50_rs{rs}_{sector}/summary.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rs', type=int, nargs='+',
                    default=[20, 30, 35, 38, 40, 45, 50, 60])
    ap.add_argument('--out', default='rs_scan_n50.png')
    args = ap.parse_args()

    rs_vals, e_flu, se_flu, e_xtl, se_xtl = [], [], [], [], []
    sk_flu_max, sk_xtl_max = [], []
    e_hf = []

    for rs in sorted(args.rs):
        sf = load_run(rs, 'fluid')
        sc = load_run(rs, 'crystal')
        if sf is None or sc is None:
            print(f'[skip rs={rs}] missing summary.json', file=sys.stderr)
            continue
        rs_vals.append(rs)
        e_flu.append(sf['e_vmc_ha'] * 1000)
        se_flu.append(sf['e_vmc_serr_ha'] * 1000)
        e_xtl.append(sc['e_vmc_ha'] * 1000)
        se_xtl.append(sc['e_vmc_serr_ha'] * 1000)
        e_hf.append(sf['e_hf_ha'] * 1000)

        sk_f = sf.get('observables', {}).get('sk_triangular_bragg', {}).get('mean', [0])
        sk_c = sc.get('observables', {}).get('sk_triangular_bragg', {}).get('mean', [0])
        sk_flu_max.append(max(sk_f) if sk_f else 0)
        sk_xtl_max.append(max(sk_c) if sk_c else 0)

    rs_vals = np.array(rs_vals)
    e_flu = np.array(e_flu); se_flu = np.array(se_flu)
    e_xtl = np.array(e_xtl); se_xtl = np.array(se_xtl)
    e_hf = np.array(e_hf)
    sk_flu_max = np.array(sk_flu_max)
    sk_xtl_max = np.array(sk_xtl_max)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Raw energies vs rs
    ax = axes[0]
    ax.errorbar(rs_vals, e_flu, yerr=se_flu, fmt='o-', label='fluid',
                color='C0', capsize=3, lw=1.5)
    ax.errorbar(rs_vals, e_xtl, yerr=se_xtl, fmt='s-', label='crystal',
                color='C3', capsize=3, lw=1.5)
    ax.plot(rs_vals, e_hf, 'k--', alpha=0.5, label='HF analytic')
    # Bonsall-Maradudin static crystal (lower bound)
    bm = -1.106103e3 / rs_vals  # mHa
    ax.plot(rs_vals, bm, 'g:', alpha=0.5, label='BM static (triangular)')
    # Smith 2024 reference points
    smith_rs = [k for k in SMITH_2024 if k in args.rs]
    if smith_rs:
        ax.plot(smith_rs, [SMITH_2024[k]['e'] * 1000 for k in smith_rs],
                marker='*', markersize=12, color='purple', linestyle='',
                label='Smith 2024 (MP)$^2$NQS')
    ax.set_xlabel('$r_s$ (Bohr)'); ax.set_ylabel('E/N (mHa/elec)')
    ax.set_title('Energy vs $r_s$')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)

    # (2) Energy gap E_xtl - E_flu vs rs (the key phase-boundary signal)
    ax = axes[1]
    delta = e_xtl - e_flu
    sigma_delta = np.sqrt(se_xtl ** 2 + se_flu ** 2)
    ax.errorbar(rs_vals, delta, yerr=sigma_delta, fmt='D-',
                color='C2', capsize=3, lw=1.5,
                label='$\\Delta = E_\\mathrm{xtl} - E_\\mathrm{flu}$')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5,
               label='phase boundary')
    # Identify rs_c by linear interpolation of Δ=0 crossing
    rs_c = None
    for i in range(len(rs_vals) - 1):
        if delta[i] * delta[i + 1] < 0:
            rs_c = rs_vals[i] + (rs_vals[i + 1] - rs_vals[i]) * (
                -delta[i] / (delta[i + 1] - delta[i])
            )
            break
    if rs_c is not None:
        ax.axvline(rs_c, color='red', linestyle=':', alpha=0.7,
                   label=f'$r_s^c \\approx {rs_c:.1f}$')
    ax.axvline(37, color='purple', linestyle=':', alpha=0.5,
               label='Smith 2024: $r_s^c = 37$')
    ax.set_xlabel('$r_s$ (Bohr)')
    ax.set_ylabel('$\\Delta = E_\\mathrm{xtl} - E_\\mathrm{flu}$ (mHa)')
    ax.set_title('Energy gap (negative = WC phase)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (3) S(k) Bragg peak vs rs
    ax = axes[2]
    ax.semilogy(rs_vals, sk_flu_max, 'o-', color='C0', label='fluid',
                lw=1.5, markersize=8)
    ax.semilogy(rs_vals, sk_xtl_max, 's-', color='C3', label='crystal',
                lw=1.5, markersize=8)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5,
               label='S(k)=1 (uncorrelated)')
    smith_rs = [k for k in SMITH_2024 if k in args.rs]
    if smith_rs:
        ax.semilogy(smith_rs, [SMITH_2024[k]['sk'] for k in smith_rs],
                    marker='*', markersize=12, color='purple',
                    linestyle='', label='Smith 2024 (best phase)')
    ax.set_xlabel('$r_s$ (Bohr)'); ax.set_ylabel('max $S(\\mathbf{k}_\\mathrm{Bragg})$')
    ax.set_title('Bragg-peak height vs $r_s$')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    fig.suptitle(
        '2D HEG fluid vs Wigner crystal: rs scan @ N=50 (light ansatz)',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'Saved: {args.out}')

    # Print summary table
    print()
    print(f'{"rs":>4} | {"E_fluid":>20} | {"E_crystal":>20} | {"delta (mHa)":>13} | {"Sk_xtl":>8} | phase')
    print('-' * 96)
    for i, rs in enumerate(rs_vals):
        d = e_xtl[i] - e_flu[i]
        sd = np.sqrt(se_xtl[i] ** 2 + se_flu[i] ** 2)
        phase = 'crystal' if d < -2 * sd else (
            'fluid' if d > 2 * sd else 'tied'
        )
        print(
            f'{rs:>4} | '
            f'{e_flu[i]:>+12.4f} +- {se_flu[i]:.4f} | '
            f'{e_xtl[i]:>+12.4f} +- {se_xtl[i]:.4f} | '
            f'{d:>+9.3f}    | '
            f'{sk_xtl_max[i]:>8.3f} | {phase}'
        )
    if rs_c is not None:
        print(f'\nLinear-interp transition: r_s^c ≈ {rs_c:.2f}  '
              f'(Smith 2024: 37 ± 1)')


if __name__ == '__main__':
    main()
