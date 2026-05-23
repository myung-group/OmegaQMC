"""More intuitive density chirality visualization in (x, y) plane.

Plots:
1. rho(x, y) for sigma+ at z ~ 0 (top down view of the molecular plane).
2. rho(x, y) for sigma- at z ~ 0.
3. Difference rho_sp - rho_sm: shows the cavity-induced chirality as a
   pinwheel/spiral pattern. Vacuum-baseline-symmetric stuff cancels;
   only the chirality-breaking part survives.
4. Cross-section: rho(theta) at fixed R (bond-midpoint), z=0 — direct
   1D plot of the angular density distribution. Shows how the C3 lobes
   are shifted by the chiral cavity.

Usage:
    python scripts/plot_density_chirality_xy.py <sp.npz> <sm.npz> [--out <prefix>]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def density_xy(npz_path, x_max=3.5, n_xy=80, z_slice=0.3):
    """Return density rho(x, y) summed over |z| < z_slice."""
    data = np.load(npz_path)
    walkers = data["walker_positions"].reshape(-1, 3)
    nuc = data["nuc_coords"]
    x, y, z = walkers[:, 0], walkers[:, 1], walkers[:, 2]

    in_plane = np.abs(z) < z_slice
    xi = x[in_plane]
    yi = y[in_plane]

    x_bins = np.linspace(-x_max, x_max, n_xy + 1)
    y_bins = np.linspace(-x_max, x_max, n_xy + 1)
    H, _, _ = np.histogram2d(xi, yi, bins=[x_bins, y_bins])

    bin_area = (x_bins[1] - x_bins[0]) ** 2
    rho = H / walkers.shape[0] / bin_area / (2 * z_slice)

    return dict(
        rho=rho, x_bins=x_bins, y_bins=y_bins,
        nuc=nuc, lam=float(data["cavity_lambda"]),
        hand=int(data["chiral_handedness"]),
        lz=float(data["l_z_mean"]),
    )


def angular_density(npz_path, R_low=1.0, R_high=2.5, z_slice=0.3,
                    n_theta=72):
    """rho(theta) integrated over R_low <= R < R_high, |z| < z_slice."""
    data = np.load(npz_path)
    walkers = data["walker_positions"].reshape(-1, 3)
    x, y, z = walkers[:, 0], walkers[:, 1], walkers[:, 2]
    R = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    mask = (R >= R_low) & (R < R_high) & (np.abs(z) < z_slice)
    theta_in = theta[mask]

    theta_bins = np.linspace(-np.pi, np.pi, n_theta + 1)
    H, _ = np.histogram(theta_in, bins=theta_bins)

    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    # normalize so integral over theta = 1
    rho_theta = H / max(H.sum(), 1) / (2 * np.pi / n_theta)

    return dict(
        theta=theta_centers, rho=rho_theta,
        n_samples=int(mask.sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sp_npz")
    ap.add_argument("sm_npz")
    ap.add_argument("--out", default="chirality_xy")
    ap.add_argument("--x-max", type=float, default=3.5)
    ap.add_argument("--n-xy", type=int, default=60)
    ap.add_argument("--z-slice", type=float, default=0.5)
    args = ap.parse_args()

    sp = density_xy(args.sp_npz, args.x_max, args.n_xy, args.z_slice)
    sm = density_xy(args.sm_npz, args.x_max, args.n_xy, args.z_slice)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    extent = (-args.x_max, args.x_max, -args.x_max, args.x_max)

    nuc = sp["nuc"]

    # (a) rho(x,y) sigma+
    ax = axes[0, 0]
    vmax = max(sp["rho"].max(), sm["rho"].max())
    im = ax.imshow(
        sp["rho"].T, origin="lower", extent=extent,
        cmap="viridis", vmin=0, vmax=vmax,
    )
    ax.scatter(nuc[1:, 0], nuc[1:, 1],
               color="white", marker="x", s=80, label="H atoms")
    ax.scatter(nuc[0:1, 0], nuc[0:1, 1],
               color="red", marker="o", s=100, edgecolor="white", label="C atom")
    ax.set_title(f"$\\rho(x, y)$ at $z \\approx 0$, $\\sigma+$ "
                 f"($\\langle L_z\\rangle = {sp['lz']:+.3f}$)")
    ax.set_xlabel("x (Bohr)")
    ax.set_ylabel("y (Bohr)")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax)

    # (b) rho(x,y) sigma-
    ax = axes[0, 1]
    im = ax.imshow(
        sm["rho"].T, origin="lower", extent=extent,
        cmap="viridis", vmin=0, vmax=vmax,
    )
    ax.scatter(nuc[1:, 0], nuc[1:, 1],
               color="white", marker="x", s=80)
    ax.scatter(nuc[0:1, 0], nuc[0:1, 1],
               color="red", marker="o", s=100, edgecolor="white")
    ax.set_title(f"$\\rho(x, y)$ at $z \\approx 0$, $\\sigma-$ "
                 f"($\\langle L_z\\rangle = {sm['lz']:+.3f}$)")
    ax.set_xlabel("x (Bohr)")
    plt.colorbar(im, ax=ax)

    # (c) difference: should show pinwheel pattern
    diff = sp["rho"] - sm["rho"]
    dmax = float(np.abs(diff).max())
    ax = axes[1, 0]
    im = ax.imshow(
        diff.T, origin="lower", extent=extent,
        cmap="RdBu_r", vmin=-dmax, vmax=dmax,
    )
    ax.scatter(nuc[1:, 0], nuc[1:, 1],
               color="black", marker="x", s=80)
    ax.scatter(nuc[0:1, 0], nuc[0:1, 1],
               color="black", marker="o", s=100, edgecolor="white")
    ax.set_title("$\\rho(\\sigma+) - \\rho(\\sigma-)$  "
                 "(symmetric parts cancel; only chirality survives)")
    ax.set_xlabel("x (Bohr)")
    ax.set_ylabel("y (Bohr)")
    # mark expected pinwheel sense
    plt.colorbar(im, ax=ax)

    # (d) angular density at bond region, both handedness
    ax = axes[1, 1]
    ang_sp = angular_density(args.sp_npz, R_low=0.5, R_high=2.5,
                             z_slice=args.z_slice)
    ang_sm = angular_density(args.sm_npz, R_low=0.5, R_high=2.5,
                             z_slice=args.z_slice)
    theta_deg = np.degrees(ang_sp["theta"])
    ax.plot(theta_deg, ang_sp["rho"], label="$\\sigma+$", color="C3", lw=2)
    ax.plot(theta_deg, ang_sm["rho"], label="$\\sigma-$", color="C0", lw=2)
    # mark H atoms
    for nh in nuc[1:]:
        ang = np.degrees(np.arctan2(nh[1], nh[0]))
        ax.axvline(ang, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("$\\theta$ (degrees)  [H atoms at dashed lines]")
    ax.set_ylabel("$\\rho(\\theta)$  (integrated over $0.5 < R < 2.5$, $|z|<0.5$ Bohr)")
    ax.set_title("Angular density distribution in the molecular plane\n"
                 "(C3 symmetry: 3 lobes; cavity shifts them by ±χ)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        "CH$_3\\cdot$ in chiral cavity ($\\lambda = 0.5$, $\\omega = 0.5$ Ha): "
        "ground-state density chirality\n"
        "Top: density in $xy$-plane for each handedness.  "
        "Bottom-left: difference (pinwheel = chirality).  "
        "Bottom-right: angular profile."
    )
    plt.tight_layout()
    out_path = f"{args.out}.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")

    # Quant: how big is the asymmetry?
    sym = 0.5 * (sp["rho"] + sm["rho"])
    chir = 0.5 * (sp["rho"] - sm["rho"])
    asym_ratio = float(np.abs(chir).sum() / np.abs(sym).sum())
    print(f"\nIn-plane asymmetry / symmetric density: "
          f"{asym_ratio:.4f}  ({asym_ratio*100:.2f}%)")
    print(f"max |rho(sp) - rho(sm)|: {dmax:.4f}")


if __name__ == "__main__":
    main()
