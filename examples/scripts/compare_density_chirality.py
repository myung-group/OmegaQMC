"""Side-by-side density chirality comparison: sigma+ vs sigma-.

Loads walker dumps for both handedness, computes Im(a_3)(R,z) for each,
plots them side-by-side. Parity test: chirality should sign-flip.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def chirality_of(npz_path, n_r=40, n_theta=60, n_z=40,
                 r_max=4.0, z_max=2.0, m=3):
    data = np.load(npz_path)
    walkers = data["walker_positions"].reshape(-1, 3)
    x, y, z = walkers[:, 0], walkers[:, 1], walkers[:, 2]
    R = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    R_bins = np.linspace(0, r_max, n_r + 1)
    theta_bins = np.linspace(-np.pi, np.pi, n_theta + 1)
    z_bins = np.linspace(-z_max, z_max, n_z + 1)

    H, _ = np.histogramdd(
        np.stack([R, theta, z], axis=-1),
        bins=[R_bins, theta_bins, z_bins],
    )

    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    dR = R_bins[1] - R_bins[0]
    dtheta = theta_bins[1] - theta_bins[0]
    dz = z_bins[1] - z_bins[0]

    vol = R_centers[:, None, None] * dR * dtheta * dz + 1e-30
    rho = H / vol / walkers.shape[0]

    phase = np.exp(-1j * m * theta_centers)
    a_m = np.einsum("rtz,t->rz", rho, phase) * dtheta / (2 * np.pi)
    a_0 = rho.mean(axis=1)

    return {
        "Re_a_m": np.real(a_m),
        "Im_a_m": np.imag(a_m),
        "a_0": a_0,
        "R_centers": R_centers, "z_centers": z_centers,
        "lambda": float(data["cavity_lambda"]),
        "hand": int(data["chiral_handedness"]),
        "lz_mean": float(data["l_z_mean"]),
        "r_max": r_max, "z_max": z_max,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sp_npz", help="sigma+ walkers.npz")
    ap.add_argument("sm_npz", help="sigma- walkers.npz")
    ap.add_argument("--out", default="L050_chirality_compare",
                    help="output prefix")
    args = ap.parse_args()

    sp = chirality_of(args.sp_npz)
    sm = chirality_of(args.sm_npz)

    print(f"sigma+: <L_z>={sp['lz_mean']:+.4f}")
    print(f"sigma-: <L_z>={sm['lz_mean']:+.4f}")

    # Mask low-density bins for the normalized plot (a_0 < 5% peak)
    a0_thresh = 0.05 * sp["a_0"].max()
    mask = sp["a_0"] < a0_thresh

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    extent = (0, sp["r_max"], -sp["z_max"], sp["z_max"])

    # Row 1: sigma+
    # density a_0 (log scale to see H atoms)
    ax = axes[0, 0]
    log_a0 = np.log10(sp["a_0"] + 1e-6)
    im = ax.imshow(
        log_a0.T, origin="lower", aspect="auto", extent=extent,
        cmap="viridis", vmin=-3, vmax=np.log10(sp["a_0"].max()),
    )
    ax.set_title("$\\log_{10}\\, a_0$ (electron density, $\\sigma+$)")
    ax.set_ylabel("z (Bohr)")
    plt.colorbar(im, ax=ax)

    # Im(a_3) chirality
    vmax = max(np.abs(sp["Im_a_m"]).max(), np.abs(sm["Im_a_m"]).max())
    ax = axes[0, 1]
    im = ax.imshow(
        sp["Im_a_m"].T, origin="lower", aspect="auto", extent=extent,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    ax.set_title(f"$\\mathrm{{Im}}\\, a_3$ ($\\sigma+$, $\\langle L_z\\rangle={sp['lz_mean']:+.3f}$)")
    plt.colorbar(im, ax=ax)

    # Im(a_3)/a_0 normalized (with mask)
    norm_sp = np.where(mask, np.nan, sp["Im_a_m"] / (sp["a_0"] + 1e-20))
    nmax = np.nanmax(np.abs(norm_sp))
    if not np.isfinite(nmax) or nmax == 0:
        nmax = 1.0
    ax = axes[0, 2]
    im = ax.imshow(
        norm_sp.T, origin="lower", aspect="auto", extent=extent,
        cmap="RdBu_r", vmin=-nmax, vmax=nmax,
    )
    ax.set_title(f"$\\mathrm{{Im}}\\, a_3 / a_0$ ($\\sigma+$, masked)")
    plt.colorbar(im, ax=ax)

    # Row 2: sigma-
    ax = axes[1, 0]
    log_a0_sm = np.log10(sm["a_0"] + 1e-6)
    im = ax.imshow(
        log_a0_sm.T, origin="lower", aspect="auto", extent=extent,
        cmap="viridis", vmin=-3, vmax=np.log10(sm["a_0"].max()),
    )
    ax.set_title("$\\log_{10}\\, a_0$ ($\\sigma-$)")
    ax.set_xlabel("R (Bohr)")
    ax.set_ylabel("z (Bohr)")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(
        sm["Im_a_m"].T, origin="lower", aspect="auto", extent=extent,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    ax.set_title(f"$\\mathrm{{Im}}\\, a_3$ ($\\sigma-$, $\\langle L_z\\rangle={sm['lz_mean']:+.3f}$)")
    ax.set_xlabel("R (Bohr)")
    plt.colorbar(im, ax=ax)

    mask_sm = sm["a_0"] < 0.05 * sm["a_0"].max()
    norm_sm = np.where(mask_sm, np.nan, sm["Im_a_m"] / (sm["a_0"] + 1e-20))
    ax = axes[1, 2]
    im = ax.imshow(
        norm_sm.T, origin="lower", aspect="auto", extent=extent,
        cmap="RdBu_r", vmin=-nmax, vmax=nmax,
    )
    ax.set_title(f"$\\mathrm{{Im}}\\, a_3 / a_0$ ($\\sigma-$, masked)")
    ax.set_xlabel("R (Bohr)")
    plt.colorbar(im, ax=ax)

    fig.suptitle(
        "CH$_3\\cdot$ density chirality parity check at $\\lambda=0.5$\n"
        "Top row: $\\sigma+$. Bottom row: $\\sigma-$. "
        "If $\\mathrm{Im}\\,a_3$ sign-flips between rows, "
        "the GS chirality follows the cavity handedness."
    )
    plt.tight_layout()
    out = f"{args.out}.png"
    fig.savefig(out, dpi=140)
    print(f"\nplot saved to {out}")

    # Quantitative parity test:
    # <Im(a_3) sp> + <Im(a_3) sm> should be ~0 if parity holds
    sp_signed = sp["Im_a_m"]
    sm_signed = sm["Im_a_m"]
    parity_residual = np.sum(sp_signed + sm_signed)
    parity_chirality = 0.5 * np.sum(sp_signed - sm_signed)
    print(f"\nParity test:")
    print(f"  Sum of Im(a_3): sigma+ = {np.sum(sp_signed):.5f}, "
          f"sigma- = {np.sum(sm_signed):.5f}")
    print(f"  Average chirality (1/2)(sp - sm) = {parity_chirality:.5f}")
    print(f"  Parity residual (sp + sm)         = {parity_residual:.5f}")
    print(f"  Parity quality: |residual/signal| = "
          f"{abs(parity_residual)/(abs(parity_chirality)+1e-30):.2%}")


if __name__ == "__main__":
    main()
