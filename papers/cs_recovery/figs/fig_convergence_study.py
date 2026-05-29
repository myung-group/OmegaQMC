"""Fig 5: BeH₂/cc-pVDZ — CI-vector convergence vs SR training duration.

The recovered C_0 and NO occupations approach the FCI reference more
slowly than the total energy does. This figure documents that gap:
training to longer durations yields meaningfully better CI vectors
even when the energy has already plateaued.

Data: papers/cs_recovery/data/beh2_iter{500,1000,2000,3500}_properties.json
plus the converged 5000-iter run in beh2_R1p330_cc-pvdz_HF_nn_no.json.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    iters = [500, 1000, 2000, 3500, 5000]
    C0_chat = []
    NO_occ2 = []
    NO_occ3 = []
    C0_FCI = None
    NO_occ2_FCI = None
    NO_occ3_FCI = None

    for it in iters:
        if it == 5000:
            d = json.load(open("papers/cs_recovery/data/beh2_R1p330_cc-pvdz_HF_nn_no.json"))
            C0_chat.append(d["c0_NN_NO"])
            if C0_FCI is None:
                C0_FCI = d["c0_FCI"]
            occs = d["nn_no_occupations"]
            fci_occs = d["fci_no_occupations"]
            NO_occ2.append(occs[1])
            NO_occ3.append(occs[2])
            if NO_occ2_FCI is None:
                NO_occ2_FCI = fci_occs[1]
                NO_occ3_FCI = fci_occs[2]
        else:
            d = json.load(open(f"papers/cs_recovery/data/beh2_iter{it}_properties.json"))
            C0_chat.append(d["chat"]["C0"])
            C0_FCI = d["fci"]["C0"]
            NO_occ2.append(d["chat"]["occupations"][1])
            NO_occ3.append(d["chat"]["occupations"][2])
            NO_occ2_FCI = d["fci"]["occupations"][1]
            NO_occ3_FCI = d["fci"]["occupations"][2]

    iters = np.array(iters)
    C0_chat = np.array(C0_chat)
    NO_occ2 = np.array(NO_occ2)
    NO_occ3 = np.array(NO_occ3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    color1 = "#3a3"
    color2 = "#c66"

    ax1.plot(iters, np.abs(C0_chat - C0_FCI), "o-", color=color1, markersize=8,
             markeredgecolor="black", markeredgewidth=0.6,
             label=r"$|C_0^{\widehat c} - C_0^{\rm FCI}|$")
    ax1.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("SR training iterations", fontsize=11)
    ax1.set_ylabel(r"$|C_0^{\widehat c} - C_0^{\rm FCI}|$", fontsize=11)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title(r"CI-vector convergence: reference-determinant amplitude",
                  fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(iters, np.abs(NO_occ2 - NO_occ2_FCI), "o-", color=color1, markersize=8,
             markeredgecolor="black", markeredgewidth=0.6, label="NO occ #2 (Be–H)")
    ax2.plot(iters, np.abs(NO_occ3 - NO_occ3_FCI), "s-", color=color2, markersize=8,
             markeredgecolor="black", markeredgewidth=0.6, label="NO occ #3 (Be–H)")
    ax2.set_xlabel("SR training iterations", fontsize=11)
    ax2.set_ylabel(r"$|n_p^{\widehat c} - n_p^{\rm FCI}|$", fontsize=11)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title(r"NN-NO occupation convergence on valence orbitals", fontsize=10)
    ax2.legend(fontsize=10, frameon=False, loc="best")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle(r"\textbf{BeH$_2$/cc-pVDZ: CI-vector convergence with SR training duration "
                 "(FermiNet+J, 1024 walkers)}", fontsize=11)
    plt.tight_layout()

    out_pdf = "papers/cs_recovery/figs/beh2_convergence_study.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")

    out_json = "papers/cs_recovery/data/beh2_convergence_study.json"
    data = {
        "iters": iters.tolist(),
        "C0_chat": C0_chat.tolist(),
        "C0_FCI": float(C0_FCI),
        "NO_occ2_chat": NO_occ2.tolist(),
        "NO_occ2_FCI": float(NO_occ2_FCI),
        "NO_occ3_chat": NO_occ3.tolist(),
        "NO_occ3_FCI": float(NO_occ3_FCI),
    }
    json.dump(data, open(out_json, "w"), indent=2)
    print(f"  wrote {out_json}")

    print("\nNumerical summary:")
    for k, it in enumerate(iters):
        print(f"  iter {it:>5}: C0_chat={C0_chat[k]:+.4f}  "
              f"|dC0|={abs(C0_chat[k]-C0_FCI):.4f}  "
              f"NO_occ2={NO_occ2[k]:.4f}  NO_occ3={NO_occ3[k]:.4f}")


if __name__ == "__main__":
    main()
