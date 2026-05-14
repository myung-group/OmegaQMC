#!/usr/bin/env python
"""Plot 2D structure factor S(k) from a run's summary.json.

Produces a 2x2 figure:
  (1) S(k) on the Cartesian k-grid as a 2D heatmap (Bragg-marker overlay)
  (2) Radial S(k) profile (binned by |k|)
  (3) Angular S(k) at the triangular Bragg shell |k| = 4 pi / (a sqrt(3))
  (4) Side-by-side cartoon density map (electron walker positions
      averaged over MCMC, showing positional order or absence thereof)

Usage:
    python scripts/plot_sk_2d.py runs/<project>/summary.json [--out output.png]

Compares two runs side by side:
    python scripts/plot_sk_2d.py \\
        runs/heg2d_rs50_N58_unpol_fluid_smith/summary.json \\
        runs/heg2d_rs50_N58_unpol_crystal_smith/summary.json \\
        --out fluid_vs_crystal_rs50.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_obs(path):
    with open(path) as f:
        return json.load(f)


def plot_one(ax_heat, ax_radial, ax_angular, summary, title):
    obs = summary.get('observables') or {}
    rs = summary['system']['rs']
    L = summary['system']['L']

    # --- 2D heatmap from cartesian grid ---
    cg = obs.get('sk_cartesian_grid')
    if cg is None:
        ax_heat.text(0.5, 0.5, 'no S(k) data', ha='center', va='center')
    else:
        k = np.array(cg['k_vectors'])
        sk = np.array(cg['mean'])

        # Add the (0, 0) point at S=N (Bragg-divergence convention)
        # purely cosmetic so heatmap doesn't have a hole
        all_k = np.concatenate([k, np.zeros((1, 2))], axis=0)
        all_sk = np.concatenate([sk, [float(summary['system']['N'])]])

        # Determine grid spacing -> reshape into 2D
        kx_unique = sorted(set(np.round(k[:, 0], 6)))
        ky_unique = sorted(set(np.round(k[:, 1], 6)))
        nx, ny = len(kx_unique), len(ky_unique)
        # Build a grid view excluding origin (which the run script omits)
        grid = np.full((nx, ny), np.nan)
        kx_idx = {v: i for i, v in enumerate(kx_unique)}
        ky_idx = {v: i for i, v in enumerate(ky_unique)}
        for kv, sv in zip(k, sk):
            i = kx_idx[round(float(kv[0]), 6)]
            j = ky_idx[round(float(kv[1]), 6)]
            grid[i, j] = sv

        extent = [kx_unique[0], kx_unique[-1],
                  ky_unique[0], ky_unique[-1]]
        im = ax_heat.imshow(
            grid.T, origin='lower', extent=extent, cmap='viridis',
            aspect='equal',
        )
        plt.colorbar(im, ax=ax_heat, label='S(k)', fraction=0.046)

        # Overlay triangular Bragg vectors
        from OmegaQMC.observables.structure_factor import (
            reciprocal_lattice_vectors_triangular,
        )
        bragg = reciprocal_lattice_vectors_triangular(rs, n_shell=1)
        ax_heat.scatter(bragg[:, 0], bragg[:, 1],
                        s=80, marker='x', c='red', linewidths=2,
                        label='triangular Bragg')
        ax_heat.legend(loc='upper right', fontsize=8)
        ax_heat.set_xlabel('k_x (Bohr$^{-1}$)')
        ax_heat.set_ylabel('k_y (Bohr$^{-1}$)')
        ax_heat.set_title(f'{title}: S(k) heatmap')

    # --- Radial S(k) profile ---
    if cg is not None:
        knorm = np.linalg.norm(k, axis=-1)
        order = np.argsort(knorm)
        ax_radial.errorbar(
            knorm[order], sk[order],
            yerr=np.array(cg['serr'])[order],
            fmt='o', markersize=3, color='C0', label='Cartesian',
        )

    bb = obs.get('sk_triangular_bragg')
    if bb is not None:
        kbb = np.array(bb['k_vectors'])
        sk_bb = np.array(bb['mean'])
        serr_bb = np.array(bb['serr'])
        knb = np.linalg.norm(kbb, axis=-1)
        ax_radial.errorbar(
            knb, sk_bb, yerr=serr_bb,
            fmt='s', markersize=6, color='red',
            label='triangular Bragg',
        )

    ax_radial.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax_radial.set_xlabel('|k| (Bohr$^{-1}$)')
    ax_radial.set_ylabel('S(k)')
    ax_radial.set_title(f'{title}: S(k) vs |k|')
    ax_radial.legend(fontsize=8)
    if bb is not None:
        ax_radial.axvline(
            float(np.linalg.norm(kbb[0])),
            color='red', linestyle=':', alpha=0.4,
        )

    # --- Angular S(k) at the Bragg shell ---
    if bb is not None:
        kbb = np.array(bb['k_vectors'])
        sk_bb = np.array(bb['mean'])
        serr_bb = np.array(bb['serr'])
        angles = np.arctan2(kbb[:, 1], kbb[:, 0]) * 180 / np.pi
        idx = np.argsort(angles)
        ax_angular.errorbar(
            angles[idx], sk_bb[idx], yerr=serr_bb[idx],
            fmt='o-', color='C1',
        )
        ax_angular.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax_angular.set_xlabel('Bragg-vector angle θ (deg)')
        ax_angular.set_ylabel('S(k=k_Bragg)')
        ax_angular.set_title(
            f'{title}: angular S(k) at |k|='
            f'{float(np.linalg.norm(kbb[0])):.4f}'
        )
        ax_angular.set_xticks([-180, -90, 0, 90, 180])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('summaries', nargs='+', help='one or two summary.json paths')
    p.add_argument('--out', default=None, help='output .png path')
    args = p.parse_args()

    if len(args.summaries) > 2:
        sys.exit('error: max 2 summaries')

    n_runs = len(args.summaries)
    fig, axes = plt.subplots(
        n_runs, 3, figsize=(18, 6 * n_runs),
        squeeze=False,
    )

    for row, summ_path in enumerate(args.summaries):
        s = load_obs(summ_path)
        title = Path(summ_path).parent.name
        plot_one(axes[row, 0], axes[row, 1], axes[row, 2], s, title)

    fig.suptitle(
        '2D HEG fluid vs crystal S(k) comparison'
        if n_runs == 2 else 'S(k) analysis',
        fontsize=12,
    )
    fig.tight_layout()

    if args.out:
        out = args.out
    else:
        if n_runs == 2:
            out = 'sk_compare.png'
        else:
            out = Path(args.summaries[0]).parent.with_suffix('.png').name
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
