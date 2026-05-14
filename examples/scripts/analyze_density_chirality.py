"""Density chirality analysis: cylindrical Fourier decomposition of
the ground-state electron density.

Algorithm:
1. Load walker positions from .npz (dumped from scripts/dump_walker_positions.py).
2. Convert each electron to cylindrical (R, theta, z) coords around C3 axis (z).
3. Histogram into a 3D (R, theta, z) grid → density rho.
4. FT in azimuthal angle: a_m(R, z) = (1/N_theta) sum_theta rho * exp(-i m theta).
5. Vacuum D3h symmetry → only m = 0, ±3, ±6 nonzero, and they are real.
   sigma+ chiral cavity → m = 0, ±3, ±6 still nonzero, BUT a_{+3} can be complex.
   Im(a_3) ≠ 0 is the chirality signal.
6. Plot heatmap of Im(a_3)/|a_0| in the (R, z) plane.

Usage:
    python scripts/analyze_density_chirality.py <walkers.npz> [--out <prefix>]
"""
from __future__ import annotations

import argparse
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("walkers_npz", help="path to walkers .npz file")
    ap.add_argument("--out", default=None,
                    help="output prefix (default: alongside input)")
    ap.add_argument("--m", type=int, default=3,
                    help="azimuthal Fourier mode (default 3 for D3h)")
    ap.add_argument("--r-max", type=float, default=4.0)
    ap.add_argument("--z-max", type=float, default=2.0)
    ap.add_argument("--n-r", type=int, default=40)
    ap.add_argument("--n-theta", type=int, default=60)
    ap.add_argument("--n-z", type=int, default=40)
    args = ap.parse_args()

    data = np.load(args.walkers_npz)
    walkers = data["walker_positions"]   # shape: (B, W, N_e, 3)
    nuc = data["nuc_coords"]              # shape: (N_nuc, 3)
    lam = float(data["cavity_lambda"])
    hand = int(data["chiral_handedness"])
    lz = float(data["l_z_mean"])

    print(f"walkers shape: {walkers.shape}")
    print(f"cavity lambda={lam}, handedness={hand:+d}")
    print(f"<L_z> from eval: {lz:+.4f}")

    # Flatten to (N_total, 3) for histogramming
    positions = walkers.reshape(-1, 3)
    print(f"total electron samples: {positions.shape[0]:,}")

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    R = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    # 3D histogram in (R, theta, z)
    R_bins = np.linspace(0, args.r_max, args.n_r + 1)
    theta_bins = np.linspace(-np.pi, np.pi, args.n_theta + 1)
    z_bins = np.linspace(-args.z_max, args.z_max, args.n_z + 1)

    H, _ = np.histogramdd(
        np.stack([R, theta, z], axis=-1),
        bins=[R_bins, theta_bins, z_bins],
    )
    # Normalize by bin volume (dV = R dR dtheta dz) to get true density
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    dR = R_bins[1] - R_bins[0]
    dtheta = theta_bins[1] - theta_bins[0]
    dz = z_bins[1] - z_bins[0]

    # density per bin = counts / (R * dR * dtheta * dz)
    # using bin centers for the R factor
    vol = R_centers[:, None, None] * dR * dtheta * dz + 1e-30
    rho = H / vol / positions.shape[0]
    print(f"rho shape: {rho.shape}, integral: "
          f"{(rho * vol).sum():.4f} (should be ~1)")

    # FT in theta
    m = args.m
    phase = np.exp(-1j * m * theta_centers)     # (N_theta,)
    a_m = np.einsum("rtz,t->rz", rho, phase) * dtheta / (2 * np.pi)

    # m=0 component for normalization
    a_0 = rho.mean(axis=1)   # average over theta

    chirality = np.imag(a_m)
    normalized = chirality / (a_0 + 1e-20)

    # Plot Im(a_m)/a_0 heatmap
    out_prefix = args.out or osp.splitext(args.walkers_npz)[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    im = ax.imshow(
        a_0.T, origin="lower", aspect="auto",
        extent=(0, args.r_max, -args.z_max, args.z_max),
        cmap="viridis",
    )
    ax.set_xlabel("R (Bohr)")
    ax.set_ylabel("z (Bohr)")
    ax.set_title("$a_0(R,z)$  (= $\\rho$ averaged over $\\theta$)")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    vmax = float(np.abs(chirality).max())
    im = ax.imshow(
        chirality.T, origin="lower", aspect="auto",
        extent=(0, args.r_max, -args.z_max, args.z_max),
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    ax.set_xlabel("R (Bohr)")
    ax.set_ylabel("z (Bohr)")
    ax.set_title(f"$\\mathrm{{Im}}\\, a_{m}(R,z)$  (chirality signal)")
    plt.colorbar(im, ax=ax)

    ax = axes[2]
    vmax = float(np.abs(normalized).max())
    im = ax.imshow(
        normalized.T, origin="lower", aspect="auto",
        extent=(0, args.r_max, -args.z_max, args.z_max),
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    ax.set_xlabel("R (Bohr)")
    ax.set_ylabel("z (Bohr)")
    ax.set_title(f"$\\mathrm{{Im}}\\, a_{m} / a_0$  (normalized)")
    plt.colorbar(im, ax=ax)

    fig.suptitle(
        f"CH$_3\\cdot$ density chirality, "
        f"$\\lambda={lam}$, handedness $s={hand:+d}$, "
        f"$\\langle L_z\\rangle = {lz:+.4f}\\,\\hbar$"
    )
    plt.tight_layout()
    plot_path = f"{out_prefix}.chirality.png"
    fig.savefig(plot_path, dpi=140)
    print(f"plot saved to {plot_path}")

    # Save numerical data too
    arr_path = f"{out_prefix}.chirality.npz"
    np.savez_compressed(
        arr_path,
        a_0=a_0, a_m=a_m,
        R_centers=R_centers, z_centers=z_centers, theta_centers=theta_centers,
        m=m, cavity_lambda=lam, chiral_handedness=hand,
        l_z_mean=lz,
    )
    print(f"data saved to {arr_path}")

    # Scalar summary
    total_chirality = float(np.abs(chirality).sum() * dR * dz)
    print(f"\nIntegrated |Im(a_{m})| over (R,z): {total_chirality:.5f}")
    print(f"Max |Im(a_{m})/a_0|: {float(np.abs(normalized).max()):.4f}")


if __name__ == "__main__":
    main()
