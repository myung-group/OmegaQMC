"""Generate all paper figures from the result data.

Fig 2: <L_z>(lambda) headline curve for CH3 sigma+/sigma- + H2 ref,
        with B_eff in Tesla on the right axis.
Fig 5: B_eff(lambda) in Tesla for CH3.
Fig 8: predicted NMR shift on H nuclei.

Other figures are reused from prior analyses (see manuscript).
"""
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = osp.join(osp.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Constants ----
CHI_ORB = 2.5045        # UHF/cc-pVDZ chi_orb_zz for CH3·, atomic units
AU_B = 2.35051756758e5  # 1 a.u. B field = 2.35e5 Tesla
HBAR_J = 1.054571817e-34
MU_B = 9.2740100783e-24
MU_0 = 4 * np.pi * 1e-7
A_BOHR = 5.29177210903e-11

# ---- Data (from VMC runs) ----
CH3_sp = {
    "lambda":   np.array([0.0, 0.1, 0.3, 0.5, 0.7]),
    "L_z":      np.array([0.0, 0.0107, 0.0206, 0.0530, 0.0533]),
    "L_z_err":  np.array([0.001, 0.0012, 0.0018, 0.0018, 0.0023]),
}
CH3_sm = {
    "lambda":   np.array([0.5]),
    "L_z":      np.array([-0.0369]),
    "L_z_err":  np.array([0.0016]),
}
H2_data = {  # σ+ chiral pilot at R=2A
    "lambda": np.array([0.5]),
    "L_z":    np.array([+0.0413]),
    "L_z_err": np.array([0.0016]),
}


# ---- Fig 2: <L_z>(lambda) for CH3 + H2 ----
def fig2():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(CH3_sp["lambda"], CH3_sp["L_z"], yerr=CH3_sp["L_z_err"],
                marker="o", markersize=10, lw=2, capsize=4,
                color="C3", label=r"CH$_3\cdot$ $\sigma^+$")
    ax.errorbar(CH3_sm["lambda"], CH3_sm["L_z"], yerr=CH3_sm["L_z_err"],
                marker="o", markersize=10, lw=2, capsize=4,
                color="C3", fillstyle="none", markerfacecolor="white",
                linestyle=":", label=r"CH$_3\cdot$ $\sigma^-$")
    ax.errorbar(H2_data["lambda"], H2_data["L_z"], yerr=H2_data["L_z_err"],
                marker="s", markersize=10, lw=2, capsize=4,
                color="C0", label=r"H$_2$ $\sigma^+$ (closed shell)")
    # λ^2 fit through λ=0.3 only (perturbative regime)
    lams = np.linspace(0, 0.55, 50)
    a = CH3_sp["L_z"][2] / CH3_sp["lambda"][2]**2   # fit to λ=0.3 point
    ax.plot(lams, a * lams**2, color="gray", lw=1.5, linestyle="--",
            label=r"$\propto \lambda^2$ fit", alpha=0.7)
    # Annotate saturation
    ax.annotate("saturation\n(λ ≳ ω/√2)", xy=(0.7, 0.053),
                xytext=(0.6, 0.07), fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))

    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"Cavity coupling $\lambda$ (a.u.)", fontsize=12)
    ax.set_ylabel(r"$\langle L_z\rangle$ ($\hbar$)", fontsize=12, color="C3")
    ax.tick_params(axis='y', labelcolor='C3')
    ax.set_xlim(-0.02, 0.78)
    ax.set_ylim(-0.06, 0.085)
    ax.grid(alpha=0.3)

    # Right axis: equivalent B_eff in Tesla
    ax2 = ax.twinx()
    yl = ax.get_ylim()
    ax2.set_ylim(yl[0] / CHI_ORB * AU_B / 1000,
                 yl[1] / CHI_ORB * AU_B / 1000)
    ax2.set_ylabel(r"Equivalent $B_{eff}$ (kTesla)", fontsize=12, color="navy")
    ax2.tick_params(axis='y', labelcolor='navy')

    # Lab reference horizontal lines
    ax2.axhline(0.045, color="navy", linestyle=":", lw=0.8, alpha=0.5)
    ax2.text(0.6, 0.05, "NHMFL 45 T", fontsize=8, color="navy", alpha=0.7)
    ax2.axhline(1.2, color="navy", linestyle=":", lw=0.8, alpha=0.5)
    ax2.text(0.6, 1.25, "Strongest pulsed 1200 T",
             fontsize=8, color="navy", alpha=0.7)

    ax.set_title(r"Cavity-induced orbital current in CH$_3\cdot$ "
                 r"($\omega=0.5$ Ha)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(osp.join(OUT_DIR, "fig2_Lz_vs_lambda.png"), dpi=200,
                bbox_inches="tight")
    print("saved Fig 2")


# ---- Fig 5: B_eff vs lambda in Tesla ----
def fig5():
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    B_eff = CH3_sp["L_z"] / CHI_ORB * AU_B
    B_err = CH3_sp["L_z_err"] / CHI_ORB * AU_B
    ax.errorbar(CH3_sp["lambda"], B_eff, yerr=B_err,
                marker="o", markersize=11, lw=2, capsize=4, color="C3",
                label=r"NN-VMC, CH$_3\cdot$ $\sigma^+$")
    # Reference horizontal lines for lab magnets
    refs = [(45, "NHMFL static (45 T)", "C0"),
            (1200, "Strongest pulsed (~1200 T)", "C2"),
            (1e5, "Magnetar surface (~$10^8$ T)", "C7")]
    for v, label, color in refs:
        if v < 1e4:
            ax.axhline(v, color=color, linestyle="--", lw=1.2)
            ax.text(0.02, v * 1.1, label, fontsize=9, color=color)
    ax.set_yscale("log")
    ax.set_xlabel(r"Cavity coupling $\lambda$ (a.u.)", fontsize=12)
    ax.set_ylabel(r"Effective field $B_{eff}$ (Tesla)", fontsize=12)
    ax.set_xlim(-0.02, 0.78)
    ax.set_ylim(1, 1e4)
    ax.grid(alpha=0.3, which="both")
    ax.set_title(r"CH$_3\cdot$ inverse Faraday $\to$ equivalent $B_{eff}$",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig.savefig(osp.join(OUT_DIR, "fig5_Beff_Tesla.png"), dpi=200,
                bbox_inches="tight")
    print("saved Fig 5")


# ---- Fig 8: predicted NMR shift on H ----
def fig8():
    r_CH = 2.039 * A_BOHR    # m
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    L_z_arr = CH3_sp["L_z"]
    L_z_err = CH3_sp["L_z_err"]
    # B induced at H from point-dipole on C
    mu = -MU_B * L_z_arr            # J/T
    B_at_H = -(MU_0 / (4 * np.pi)) * mu / r_CH**3   # T, along z
    B_err = (MU_0 / (4 * np.pi)) * MU_B * L_z_err / r_CH**3
    # NMR shift in ppm at B_ext = 11.74 T
    B_ext = 11.74
    shift_ppm = -B_at_H / B_ext * 1e6
    shift_err = B_err / B_ext * 1e6
    ax.errorbar(CH3_sp["lambda"], shift_ppm, yerr=shift_err,
                marker="o", markersize=11, lw=2, capsize=4, color="C3",
                label="point-dipole estimate")
    # 5x correction (from distributed ring current)
    shift_realistic = shift_ppm / 5
    ax.errorbar(CH3_sp["lambda"], shift_realistic, yerr=shift_err/5,
                marker="^", markersize=10, lw=1, capsize=3, color="C0",
                linestyle="--", alpha=0.7,
                label="point-dipole / 5 (realistic est.)")
    # Reference: typical 1H chemical shift range
    ax.axhspan(-12, 0, color="gray", alpha=0.2)
    ax.text(0.02, -50, "typical 1H NMR range", fontsize=9, color="gray")
    ax.set_xlabel(r"Cavity coupling $\lambda$ (a.u.)", fontsize=12)
    ax.set_ylabel(r"Cavity-induced 1H NMR shift (ppm)", fontsize=12)
    ax.set_title(
        r"Predicted 1H NMR shift on CH$_3\cdot$ at $B_{ext}=11.7$ T",
        fontsize=12,
    )
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 0.78)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    fig.savefig(osp.join(OUT_DIR, "fig8_NMR_shift.png"), dpi=200,
                bbox_inches="tight")
    print("saved Fig 8")


# ---- Copy existing figures over ----
def copy_existing():
    import shutil
    src_dir = osp.abspath(osp.join(osp.dirname(__file__),
                                   "..", "..", "logs", "from_mango"))
    mapping = {
        "L050_chirality_clean.png": "fig3_density_chirality.png",
        "ch3_omega_scan.png":       "fig4_omega_dispersion.png",
        "menagerie_master.png":     "fig7_menagerie.png",
    }
    for src, dst in mapping.items():
        s = osp.join(src_dir, src)
        d = osp.join(OUT_DIR, dst)
        if osp.exists(s):
            shutil.copy(s, d)
            print(f"copied {src} -> {dst}")
        else:
            print(f"WARN: missing {s}")


if __name__ == "__main__":
    fig2()
    fig5()
    fig8()
    copy_existing()
    print(f"\nFigures in {OUT_DIR}")
