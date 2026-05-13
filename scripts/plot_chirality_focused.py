"""Final focused chirality plot — 2 panels:
1. xy difference rho(sigma+) - rho(sigma-)  with H atoms marked
2. angular density difference  rho_sp(theta) - rho_sm(theta)  vs theta

Both show only the chirality-breaking part of the GS density.
Vacuum baseline would be exactly zero everywhere.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def load_walkers(npz_path):
    data = np.load(npz_path)
    walkers = data["walker_positions"].reshape(-1, 3)
    return dict(
        walkers=walkers,
        nuc=data["nuc_coords"],
        lam=float(data["cavity_lambda"]),
        hand=int(data["chiral_handedness"]),
        lz=float(data["l_z_mean"]),
    )


def density_xy(walkers, x_max=3.5, n_xy=60, z_slice=0.5):
    in_plane = np.abs(walkers[:, 2]) < z_slice
    xi = walkers[in_plane, 0]
    yi = walkers[in_plane, 1]
    x_bins = np.linspace(-x_max, x_max, n_xy + 1)
    H, _, _ = np.histogram2d(xi, yi, bins=[x_bins, x_bins])
    bin_area = (x_bins[1] - x_bins[0]) ** 2
    return H / walkers.shape[0] / bin_area / (2 * z_slice), x_bins


def angular_profile(walkers, R_low=0.5, R_high=2.5, z_slice=0.5,
                    n_theta=72):
    x, y, z = walkers[:, 0], walkers[:, 1], walkers[:, 2]
    R = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    mask = (R >= R_low) & (R < R_high) & (np.abs(z) < z_slice)
    H, edges = np.histogram(theta[mask],
                            bins=np.linspace(-np.pi, np.pi, n_theta + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    rho = H / max(H.sum(), 1) / (2 * np.pi / n_theta)
    serr = np.sqrt(H) / max(H.sum(), 1) / (2 * np.pi / n_theta)
    return centers, rho, serr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sp_npz")
    ap.add_argument("sm_npz")
    ap.add_argument("--out", default="chirality_focused")
    ap.add_argument("--x-max", type=float, default=3.5)
    ap.add_argument("--n-xy", type=int, default=60)
    args = ap.parse_args()

    sp = load_walkers(args.sp_npz)
    sm = load_walkers(args.sm_npz)

    rho_sp, x_bins = density_xy(
        sp["walkers"], args.x_max, args.n_xy,
    )
    rho_sm, _ = density_xy(
        sm["walkers"], args.x_max, args.n_xy,
    )
    diff = rho_sp - rho_sm

    theta_sp, ang_sp, ang_sp_serr = angular_profile(sp["walkers"])
    theta_sm, ang_sm, ang_sm_serr = angular_profile(sm["walkers"])
    ang_diff = ang_sp - ang_sm
    ang_diff_serr = np.sqrt(ang_sp_serr ** 2 + ang_sm_serr ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: xy difference
    ax = axes[0]
    extent = (-args.x_max, args.x_max, -args.x_max, args.x_max)
    dmax = float(np.abs(diff).max())
    im = ax.imshow(
        diff.T, origin="lower", extent=extent,
        cmap="RdBu_r", vmin=-dmax, vmax=dmax,
    )
    # Mark atoms
    nuc = sp["nuc"]
    ax.scatter(nuc[1:, 0], nuc[1:, 1], color="black",
               marker="x", s=120, lw=2, label="H atoms")
    ax.scatter(nuc[0:1, 0], nuc[0:1, 1], color="black",
               marker="o", s=140, edgecolor="white", lw=2, label="C atom")
    ax.set_title(f"$\\rho(\\sigma+) - \\rho(\\sigma-)$ in molecular plane\n"
                 f"(vacuum would be exactly 0; asymmetry = "
                 f"cavity-induced chirality)")
    ax.set_xlabel("x (Bohr)")
    ax.set_ylabel("y (Bohr)")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="$\\Delta\\rho$ (Bohr$^{-3}$)")

    # Panel 2: angular difference
    ax = axes[1]
    theta_deg = np.degrees(theta_sp)
    ax.errorbar(theta_deg, ang_diff, yerr=ang_diff_serr,
                color="C3", lw=1.5, capsize=2,
                label="$\\rho_{\\sigma+}(\\theta) - \\rho_{\\sigma-}(\\theta)$")
    ax.axhline(0, color="black", lw=0.5)
    for nh in nuc[1:]:
        ang = np.degrees(np.arctan2(nh[1], nh[0]))
        ax.axvline(ang, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("$\\theta$ (degrees)  [H atoms at gray dashes]")
    ax.set_ylabel("$\\Delta\\rho(\\theta)$")
    ax.set_title("Angular density difference in molecular plane\n"
                 "(vacuum would be 0 for all θ; here = chirality breaking)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(
        f"CH$_3\\cdot$: cavity-induced GS chirality at $\\lambda = 0.5$, "
        f"$\\omega = 0.5$ Ha\n"
        f"$\\langle L_z\\rangle_{{\\sigma+}} = {sp['lz']:+.3f}\\,\\hbar$,  "
        f"$\\langle L_z\\rangle_{{\\sigma-}} = {sm['lz']:+.3f}\\,\\hbar$"
    )
    plt.tight_layout()
    out_path = f"{args.out}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")

    sigma_diff = float(
        np.abs(ang_diff).sum() / np.sqrt(np.sum(ang_diff_serr ** 2))
    )
    print(f"angular Δρ integrated significance: {sigma_diff:.1f}σ")


if __name__ == "__main__":
    main()
