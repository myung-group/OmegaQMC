"""Master comparison plot for the radical menagerie.

Plots <L_z>(lambda) for all systems on one panel, color-coded by orbital
mechanism (closed shell / open shell non-degenerate / open shell degenerate).

Inputs: hard-coded summary dict; populated as menagerie runs complete.
Re-run as new data lands.

Usage:
    python scripts/plot_menagerie.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# === DATA (update as runs complete) ============================
# format: system → {(lambda, hand): (E_Ha, L_z, L_z_serr)}
# L_z = nan when not yet measured; (placeholder for queued)
# basin? = "✓" (correct sign), "X" (flipped), "noise" (in MC noise)

DATA = {
    "H2": {
        "n_elec": 2,
        "category": "closed-shell",
        "marker": "o",
        "color": "C0",
        "points": {
            # from earlier chiral pilot (H2 R=2A, lambda=0.5)
            (0.5,  +1): (-0.884,  +0.0413, 0.0016),
            (0.5,  -1): (-0.885,  -0.0452, 0.0017),
        },
    },
    "H6": {
        "n_elec": 6,
        "category": "closed-shell aromatic",
        "marker": "s",
        "color": "C1",
        "points": {
            (0.0,  +1): (-3.303,   0.0,    0.0010),
            (0.1,  +1): (-3.411,  -0.0014, 0.0010),
            (0.3,  +1): (-3.258,  +0.0226, 0.0024),
            (0.5,  +1): (-3.084,  +0.0582, 0.0035),
            (0.1,  -1): (-3.388,  -0.0007, 0.0011),
        },
    },
    "CH3": {
        "n_elec": 9,
        "category": "open-shell non-deg",
        "marker": "^",
        "color": "C2",
        "points": {
            (0.0,  +1): (-39.76,   0.0,    0.0001),
            (0.1,  +1): (-39.75,  +0.0107, 0.0012),
            (0.3,  +1): (-39.71,  +0.0206, 0.0018),
            (0.5,  +1): (-39.43,  +0.0530, 0.0018),
            (0.7,  +1): (-39.19,  +0.0533, 0.0023),
            (0.5,  -1): (-39.42,  -0.0369, 0.0016),
        },
    },
    "H3": {
        "n_elec": 3,
        "category": "open-shell deg SOMO",
        "marker": "D",
        "color": "C3",
        "points": {
            (0.0,  +1): (-1.554,   0.0,    0.0010),
            (0.1,  +1): (-1.548,  +0.1368, 0.0037),
            (0.3,  +1): (-1.484,  -0.3436, 0.0087),   # basin-flipped
            (0.5,  +1): (-1.357,  -0.3570, 0.0082),   # basin-flipped
            (0.5,  -1): (-1.366,  -0.2355, 0.0081),
        },
    },
    "NO": {
        "n_elec": 15,
        "category": "open-shell deg π* SOMO",
        "marker": "v",
        "color": "C4",
        "points": {
            # E values undertrained (vacuum gate ~12 Ha above HF); use
            # |<L_z>| as the meaningful signal.
            (0.1,  +1): (-121.28, -0.0451, 0.0064),  # basin-flipped
            (0.3,  +1): (-122.19, +0.0769, 0.0120),
            (0.1,  -1): (-111.83, -0.1378, 0.0131),  # |<L_z>|=0.14, matches H3!
        },
    },
}


def plot(ax):
    for sys_name, d in DATA.items():
        pts = sorted(d["points"].items())
        lambdas_sp = [k[0] for k, _ in pts if k[1] == +1]
        lz_sp = [v[1] for k, v in pts if k[1] == +1]
        lz_err_sp = [v[2] for k, v in pts if k[1] == +1]
        if lambdas_sp:
            ax.errorbar(
                lambdas_sp, lz_sp, yerr=lz_err_sp,
                marker=d["marker"], markersize=9, lw=1.5, capsize=4,
                color=d["color"],
                label=(
                    f"{sys_name} ({d['n_elec']}e⁻, {d['category']})"
                ),
            )
            # σ- if present
            lambdas_sm = [k[0] for k, _ in pts if k[1] == -1]
            lz_sm = [v[1] for k, v in pts if k[1] == -1]
            lz_err_sm = [v[2] for k, v in pts if k[1] == -1]
            if lambdas_sm:
                ax.errorbar(
                    lambdas_sm, lz_sm, yerr=lz_err_sm,
                    marker=d["marker"], markersize=9, lw=1.5, capsize=4,
                    color=d["color"], linestyle="--",
                    fillstyle="none", markerfacecolor="white",
                    alpha=0.7,
                )

    # lambda^2 reference (for CH3 perturbative regime)
    lams = np.linspace(0, 0.5, 100)
    ax.plot(lams, lams**2 * 0.05 / 0.25,
            color="gray", ls=":", lw=1, alpha=0.6,
            label=r"$\lambda^2$ (CH3 perturbative)")

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"Cavity coupling $\lambda$ (a.u.)")
    ax.set_ylabel(r"Ground-state $\langle L_z\rangle$ ($\hbar$)")
    ax.set_title(
        r"Inverse-Faraday response in the radical menagerie ($\omega=0.5$ Ha)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)


def main():
    fig, ax = plt.subplots(figsize=(9, 6))
    plot(ax)
    out = "logs/from_mango/menagerie_master.png"
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
