#!/usr/bin/env python
"""Plot real-space electron density n(r) for a trained 2D HEG run.

Loads the trained checkpoint, samples walkers from |psi|^2 via plain
Metropolis MCMC, and bins the electron positions onto a 2D grid.
The resulting heatmap reveals the Wigner-crystal site pattern when
the trial wavefunction is in the WC sector, or a featureless density
for the fluid sector.

Usage:
    python scripts/plot_density_2d.py runs/<project>/ \\
        [--n-walkers 256] [--n-equil 500] [--n-sample 200] \\
        [--n-bins 80] [--out density.png]

Compares two runs side-by-side:
    python scripts/plot_density_2d.py \\
        runs/heg2d_rs50_N58_unpol_fluid_smith \\
        runs/heg2d_rs50_N18_unpol_crystal_minimal_test \\
        --out density_compare.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
import yaml


def _build_wf_from_config(cfg, init_seed=0):
    """Reconstruct the wavefunction module + driver from a YAML config."""
    from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig
    from OmegaQMC.heg_2d import build_2deg_system
    from OmegaQMC.vmc_nn_heg import get_vmc_nn_heg_func

    # System
    rs = float(cfg['system']['rs'])
    N = int(cfg['system']['N'])
    pol = cfg['system'].get('polarization', 'unpolarized')
    sys_info = build_2deg_system(rs, N_elec=N, polarization=pol)
    L = float(sys_info['L'])
    n_up, n_down = sys_info['nup'], sys_info['ndown']

    # PsiFormer config
    a = cfg.get('ansatz', {})
    jas_act = a.get('jas_activation', 'tanh')
    config = HEGPsiFormerConfig(
        n_up=n_up, n_down=n_down, L=L,
        n_det=int(a.get('n_det', 1)),
        full_determinant=bool(a.get('full_determinant', False)),
        embedding_dim=int(a.get('embedding_dim', 64)),
        n_interactions=int(a.get('layers', 2)),
        two_particle_stream_dim=int(a.get('two_particle_dim', 16)),
        n_attention_heads=int(a.get('heads', 2)),
        use_cusp=bool(a.get('use_cusp', True)),
        use_deep_jastrow=bool(a.get('deep_jastrow', False)),
        use_pair_jastrow=bool(a.get('pair_jastrow', False)),
        jas_mlp_activation=(None if jas_act in (None, 'none') else jas_act),
        jas_mlp_bias=bool(a.get('jas_bias', True)),
        jas_mlp_zero_init_last=bool(a.get('jas_zero_init_last', True)),
        n_virt_pw=int(a.get('n_virt_pw', 12)),
        det_jitter=float(a.get('det_jitter', 0.02)),
        use_ghost_atom=bool(a.get('use_ghost_atom', True)),
        use_backflow=bool(a.get('use_backflow', True)),
        use_coord_backflow=bool(a.get('use_coord_backflow', False)),
        coord_bf_zero_init=bool(a.get('coord_bf_zero_init', True)),
        envelope_type=str(a.get('envelope_type', 'plane_wave')),
        crystal_sigma_init=float(a.get('crystal_sigma_init', 0.25)),
        crystal_spin_pattern=str(a.get('crystal_spin_pattern', 'neel')),
        crystal_det_jitter=float(a.get('crystal_det_jitter', 0.0)),
        walker_init=str(a.get('walker_init', 'auto')),
        dim=int(cfg['system'].get('dim', 3)),
    )

    ewald_cfg = cfg.get('ewald', {})
    driver = get_vmc_nn_heg_func(
        config, jax.random.key(init_seed),
        ewald_n_real=int(ewald_cfg.get('n_real', 3)),
        ewald_n_recip=int(ewald_cfg.get('n_recip', 6)),
    )
    return driver, sys_info, config


def _sample_walkers(driver, n_walkers, n_equil, n_sample, decorr,
                    mc_timestep, seed=42):
    """Equilibrate walkers from |psi|^2, then collect snapshots."""
    rng = jax.random.key(seed)
    rng, init_key = jax.random.split(rng)
    walkers = driver.initialize_walkers(init_key, n_walkers)
    step_size = (3 * mc_timestep) ** 0.5
    move_fn = driver._metropolis_move_allw

    # Burn-in
    print(f'  burn-in {n_equil} steps ...', end='', flush=True)
    for _ in range(n_equil):
        rng, sk = jax.random.split(rng)
        keys = jax.random.split(sk, n_walkers)
        walkers, _ = move_fn(keys, walkers, step_size, driver.params)
    print(' done.')

    # Sample
    print(f'  sampling {n_sample} blocks (decorr={decorr}) ...', end='', flush=True)
    snapshots = []
    for i in range(n_sample):
        for _ in range(decorr):
            rng, sk = jax.random.split(rng)
            keys = jax.random.split(sk, n_walkers)
            walkers, _ = move_fn(keys, walkers, step_size, driver.params)
        snapshots.append(np.asarray(walkers))
    print(' done.')

    # Stack: (n_sample, n_walkers, n_elec, 2) -> (n_sample*n_walkers*n_elec, 2)
    arr = np.stack(snapshots, axis=0)
    return arr


def _wrap_to_cell(positions, L):
    return np.mod(positions, L)


def plot_density(ax_dens, ax_g, ax_xy, walkers_arr, sys_info, label,
                 cfg, n_bins=80):
    """Plot density heatmap + radial g(r) + scatter overlay."""
    L = float(sys_info['L'])
    n_up, n_down = sys_info['nup'], sys_info['ndown']
    n_elec = sys_info['N_elec']
    rs = sys_info['rs']

    # Flatten all electrons across walkers and snapshots
    pos_all = walkers_arr.reshape(-1, 2)
    pos_all = _wrap_to_cell(pos_all, L)
    H, xedges, yedges = np.histogram2d(
        pos_all[:, 0], pos_all[:, 1],
        bins=n_bins, range=[[0, L], [0, L]],
    )
    # Normalize: density per unit area, total integrates to N_elec
    cell_area = (L / n_bins) ** 2
    n_snapshots_walkers = walkers_arr.shape[0] * walkers_arr.shape[1]
    density = H / (n_snapshots_walkers * cell_area)

    # Heatmap
    im = ax_dens.imshow(
        density.T, origin='lower', extent=[0, L, 0, L],
        cmap='viridis', aspect='equal',
    )
    plt.colorbar(im, ax=ax_dens, label='n(r) (Bohr$^{-2}$)', fraction=0.046)
    ax_dens.set_xlabel('x (Bohr)')
    ax_dens.set_ylabel('y (Bohr)')
    ax_dens.set_title(f'{label}\nElectron density n(r), N={n_elec}, rs={rs}')

    # Overlay triangular Bravais sites if crystal envelope
    a_env = cfg.get('ansatz', {}).get('envelope_type', 'plane_wave')
    if a_env == 'crystal_gaussian':
        from OmegaQMC.psi.nn.env_localized_2d import (
            triangular_lattice_sites,
        )
        sites = triangular_lattice_sites(rs, n_elec)
        sites = np.mod(sites, L)
        ax_dens.scatter(sites[:, 0], sites[:, 1],
                        s=80, marker='+', c='red', linewidths=1.5,
                        label='triangular sites')
        ax_dens.legend(loc='upper right', fontsize=8)

    # Radial pair-correlation g(r): histogram of pairwise distances
    print(f'  computing g(r) for {label}...')
    walkers_per_block = walkers_arr.reshape(-1, n_elec, 2)  # (M, N, 2)
    n_blocks_for_g = min(50, walkers_per_block.shape[0])
    block_idx = np.linspace(
        0, walkers_per_block.shape[0] - 1, n_blocks_for_g, dtype=int,
    )
    r_max = L / 2.0
    n_g_bins = 60
    bin_edges = np.linspace(0, r_max, n_g_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist = np.zeros(n_g_bins)
    for i in block_idx:
        r = walkers_per_block[i]   # (n_elec, 2)
        # Minimum-image pairwise distances
        diff = r[:, None, :] - r[None, :, :]
        diff_mi = diff - L * np.round(diff / L)
        d = np.linalg.norm(diff_mi, axis=-1)
        upper = d[np.triu_indices(n_elec, k=1)]
        h, _ = np.histogram(upper, bins=bin_edges)
        hist = hist + h
    # Normalize: n(r) of an ideal gas pair distribution = 2*pi*r*dr * (n/2)
    # Actually g(r): hist / (2*pi*r*dr * n * (N-1)/2 * n_blocks)
    n_density = n_elec / (L ** 2)
    norm = (
        2 * np.pi * bin_centers * (bin_edges[1] - bin_edges[0])
        * n_density * (n_elec - 1) / 2 * len(block_idx)
    )
    g_of_r = hist / np.maximum(norm, 1e-12)

    # Plot g(r) in units of r_s
    ax_g.plot(bin_centers / rs, g_of_r, color='C0', lw=1.6)
    ax_g.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax_g.set_xlabel('r / r_s')
    ax_g.set_ylabel('g(r)')
    ax_g.set_title(f'{label}\nPair correlation g(r)')
    ax_g.set_xlim(0, r_max / rs)

    # Scatter plot of one snapshot for visual cue (last walker, first snap)
    snap = walkers_arr[0, 0]    # (n_elec, 2)
    snap = _wrap_to_cell(snap, L)
    if a_env == 'crystal_gaussian':
        # Color up vs down spins distinctly
        ax_xy.scatter(snap[:n_up, 0], snap[:n_up, 1],
                      c='red', s=40, marker='^', label='up')
        if n_down > 0:
            ax_xy.scatter(snap[n_up:, 0], snap[n_up:, 1],
                          c='blue', s=40, marker='v', label='down')
        # Overlay sites
        sites = triangular_lattice_sites(rs, n_elec)
        sites = np.mod(sites, L)
        ax_xy.scatter(sites[:, 0], sites[:, 1],
                      s=120, marker='o', c='none',
                      edgecolors='gray', linewidth=1, label='target sites')
        ax_xy.legend(fontsize=8)
    else:
        ax_xy.scatter(snap[:n_up, 0], snap[:n_up, 1],
                      c='red', s=40, marker='^', label='up')
        if n_down > 0:
            ax_xy.scatter(snap[n_up:, 0], snap[n_up:, 1],
                          c='blue', s=40, marker='v', label='down')
        ax_xy.legend(fontsize=8)
    ax_xy.set_xlim(0, L); ax_xy.set_ylim(0, L)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlabel('x (Bohr)'); ax_xy.set_ylabel('y (Bohr)')
    ax_xy.set_title(f'{label}\nOne walker snapshot (electron positions)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('rundirs', nargs='+',
                   help='one or two run directories with .chk.h5 + config.yaml')
    p.add_argument('--n-walkers', type=int, default=256)
    p.add_argument('--n-equil', type=int, default=500)
    p.add_argument('--n-sample', type=int, default=200)
    p.add_argument('--decorr', type=int, default=3)
    p.add_argument('--n-bins', type=int, default=80)
    p.add_argument('--mc-timestep-fluid', type=float, default=0.5,
                   help='mc_timestep for fluid samples')
    p.add_argument('--mc-timestep-crystal', type=float, default=2.0,
                   help='mc_timestep for crystal samples (larger for sigma~10 Bohr peaks)')
    p.add_argument('--out', default='density.png')
    args = p.parse_args()

    if len(args.rundirs) > 2:
        sys.exit('error: max 2 run directories')

    n_runs = len(args.rundirs)
    fig, axes = plt.subplots(
        n_runs, 3, figsize=(18, 6 * n_runs), squeeze=False,
    )

    for row, rundir in enumerate(args.rundirs):
        rundir = Path(rundir)
        cfg_path = rundir / 'config.yaml'
        chk_path = rundir / f'{rundir.name}.chk.h5'

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        a_env = cfg.get('ansatz', {}).get('envelope_type', 'plane_wave')
        mc_timestep = (args.mc_timestep_crystal
                       if a_env == 'crystal_gaussian'
                       else args.mc_timestep_fluid)

        print(f'\n=== {rundir.name} (envelope: {a_env}) ===')
        driver, sys_info, _ = _build_wf_from_config(cfg, init_seed=row)

        # Try to load trained params; fall back to init.
        if chk_path.is_file():
            try:
                from OmegaQMC.nn_checkpoint import load_nn_checkpoint
                params, _ = load_nn_checkpoint(str(chk_path), driver.params)
                driver.params = params
                print(f'  loaded {chk_path}')
            except Exception as e:
                print(
                    f'  WARNING: could not load {chk_path} ({e!r}). '
                    f'Falling back to fresh init params.'
                )
                print(
                    f'           For crystal_gaussian envelope, init params '
                    f'place Gaussians exactly at lattice sites — the density '
                    f'plot will still show the WC pattern.'
                )
        else:
            print(f'  warning: {chk_path} not found, using fresh init params')

        walkers_arr = _sample_walkers(
            driver,
            n_walkers=args.n_walkers,
            n_equil=args.n_equil,
            n_sample=args.n_sample,
            decorr=args.decorr,
            mc_timestep=mc_timestep,
        )

        plot_density(
            axes[row, 0], axes[row, 1], axes[row, 2],
            walkers_arr, sys_info,
            label=rundir.name,
            cfg=cfg, n_bins=args.n_bins,
        )

    fig.suptitle(
        '2D HEG real-space electron distribution',
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
