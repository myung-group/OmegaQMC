"""K_s sample-complexity scaling on H_4 linear / cc-pVDZ.

The §III.D union-bound argument predicts K_s ≳ O(log N_det · η⁻²)
for the per-coefficient recovery error to fall below threshold η.
Equivalently, the L_2 recovery error ‖ĉ − c_FCI‖₂ should fall as
K_s^(-1/2) until it saturates at the NN-FCI bias floor.

We subsample the existing H_4 walker bank (K_s_max = 25,600) at
K_s ∈ {200, 400, 800, 1600, 3200, 6400, 12800, 25600}, recover
ĉ at each, and plot ‖ĉ - c_FCI‖₂ vs K_s on log-log axes.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import jax
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[3]))
sys.path.insert(0, str(Path(__file__).parents[3] / "examples"))

from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import evaluate_orbitals_on_walkers, f_I_matrix
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.vmc_nn import get_vmc_nn_func
from run_cs_properties import build_mol
from run_cs_h4_scaling import evaluate_signed_psi


def main():
    cell_dir = Path("cs_h4_results/h4_linear_ferminet_jastrow_R1p000_cc-pvdz")
    prefix = cell_dir.name
    walkers_path = cell_dir / f"{prefix}_walkers.h5"
    checkpoint = cell_dir / f"{prefix}.chk.h5"

    print(f"=== K_s scaling on {prefix} ===")
    mol = build_mol("h4", 1.0, "cc-pvdz", "Angstrom", "linear")
    print(f"  mol: {mol.nao} AOs, nelec=(2,2)")

    fci_ref = compute_fci_reference(mol, n_alpha=2, n_beta=2,
                                    candidate_tol=1e-4)
    n_det = len(fci_ref["candidate_set"])
    c_fci = np.array([fci_ref["ci_dict"][k] for k in fci_ref["candidate_set"]])
    c_fci = c_fci / np.linalg.norm(c_fci)
    print(f"  candidate set size = {n_det}")

    walkers, _, _ = load_walker_bank(str(walkers_path))
    K_s_max = walkers.shape[0]
    print(f"  K_s_max from bank = {K_s_max}")

    init_key = jax.random.split(jax.random.key(42), 4)[1]
    ansatz = "OmegaQMC/psi/nn/conf/ferminet_jastrow.yaml"
    driver = get_vmc_nn_func(mol, ansatz, init_key, prefix=str(cell_dir / prefix))
    driver.load_checkpoint(str(checkpoint))
    log_psi, _, _, _ = make_nn_log_psi(ansatz, mol, init_key)
    psi_vals = evaluate_signed_psi(walkers, np.asarray(mol.atom_coords()),
                                    driver.params, log_psi, batch_size=2048)
    orb_vals = evaluate_orbitals_on_walkers(
        mol, walkers, fci_ref["no_coeff_ao"],
        convention="interleaved", n_alpha=2, n_beta=2,
    )
    f_I_full = np.asarray(f_I_matrix(orb_vals, fci_ref["candidate_set"], psi_vals, 2, 2))
    print(f"  f_I matrix shape = {f_I_full.shape}")

    K_s_sweep = [200, 400, 800, 1600, 3200, 6400, 12800, K_s_max]
    n_seeds = 5
    rng = np.random.default_rng(42)
    results = []
    for K_s in K_s_sweep:
        if K_s > K_s_max:
            continue
        errs_l2 = []
        errs_linf = []
        for s in range(n_seeds):
            idx = rng.choice(K_s_max, size=K_s, replace=False)
            c_raw = f_I_full[:, idx].mean(axis=1)
            c_hat = c_raw / np.linalg.norm(c_raw)
            if c_hat[0] * c_fci[0] < 0:
                c_hat = -c_hat
            errs_l2.append(float(np.linalg.norm(c_hat - c_fci)))
            errs_linf.append(float(np.max(np.abs(c_hat - c_fci))))
        results.append({
            "K_s": int(K_s),
            "err_l2_mean": float(np.mean(errs_l2)),
            "err_l2_std":  float(np.std(errs_l2)),
            "err_linf_mean": float(np.mean(errs_linf)),
            "err_linf_std":  float(np.std(errs_linf)),
        })
        print(f"  K_s={K_s:>5}: ‖ĉ-c_FCI‖_2 = {np.mean(errs_l2):.4f} "
              f"± {np.std(errs_l2):.4f}  (n_seeds={n_seeds})")

    Ks = np.array([r["K_s"] for r in results])
    l2_mean = np.array([r["err_l2_mean"] for r in results])
    l2_std = np.array([r["err_l2_std"] for r in results])

    log_K = np.log(Ks)
    log_e = np.log(l2_mean)
    # Fit only the early portion (where MC noise dominates, before bias floor)
    fit_mask = Ks <= 6400
    coeffs = np.polyfit(log_K[fit_mask], log_e[fit_mask], 1)
    slope, intercept = coeffs[0], coeffs[1]
    print(f"  power-law fit (K_s <= 6400): slope = {slope:.3f} "
          f"(predicted -0.5 for pure MC noise)")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(Ks, l2_mean, yerr=l2_std, fmt="o-", color="#3a3",
                markersize=7, capsize=3,
                label=fr"recovery error (mean ± std, $n_{{\rm seeds}}=5$)")
    Ks_fit = np.array([Ks.min(), 6400.0])
    ax.plot(Ks_fit, np.exp(intercept) * Ks_fit ** slope,
            "--", color="#c66", alpha=0.8,
            label=fr"power-law fit: slope = {slope:.2f}")
    Ks_pred = np.array([Ks.min(), Ks.max()])
    err_at_min = l2_mean[0] * (Ks.min() / Ks_pred) ** 0.5
    ax.plot(Ks_pred, err_at_min, ":", color="gray", alpha=0.7,
            label=r"$K_s^{-1/2}$ (CS theory)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Walker count $K_s$", fontsize=11)
    ax.set_ylabel(r"Recovery error $\|\widehat c - c^{\rm FCI}\|_2$",
                  fontsize=11)
    ax.set_title(r"\ce{H_4} linear / cc-pVDZ: sample-complexity scaling",
                  fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_pdf = "papers/cs_recovery/figs/h4_Ks_scaling.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")

    out_json = "papers/cs_recovery/data/h4_Ks_scaling.json"
    json.dump({"Ks_sweep": Ks.tolist(),
               "results": results,
               "fit_slope": slope,
               "n_det": n_det}, open(out_json, "w"), indent=2)
    print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
