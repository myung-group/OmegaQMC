"""λ-sweep demonstration: where Lasso soft-thresholding matters.

H₂O / aug-cc-pVDZ K=4 FermiNet+Jastrow ground-state recovery at
CAS(16,(4,4)). Loads c_eig from existing spec JSON (~1661 active-
space coefficients), applies soft_threshold for
λ ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05}, plots:

  (a) surviving fraction |{I : ĉ_I ≠ 0}| / |S| vs λ
  (b) magnitude histogram showing the noise floor and the
      noise-vs-signal separation

The CAS(16) candidate set after the candidate_tol filter has ~1661
determinants. The noise floor in the recovered c_eig sits around
~10⁻³; soft-thresholding above this floor zeros out the noise
without touching the signal, demonstrating where CS recovery bites.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[3]))
from OmegaQMC.cs.estimators import soft_threshold


def main():
    json_path = Path("papers/cs_recovery/data/h2o_pfau_k4_spec_aug-cc-pvdz_cas16.json")
    d = json.load(open(json_path))
    print(f"Loaded {json_path.name}")
    print(f"  K = {d['K']}, |c_eig| entries = {len(d['c_eig'])}")

    c_eig_ground = np.array(d["c_eig"][0])
    if abs(np.linalg.norm(c_eig_ground) - 1.0) > 1e-3:
        c_eig_ground = c_eig_ground / np.linalg.norm(c_eig_ground)
    n_det = len(c_eig_ground)
    print(f"  ground state |c_eig| stats: max={np.max(np.abs(c_eig_ground)):.4f}, "
          f"median={np.median(np.abs(c_eig_ground)):.4e}, "
          f"min={np.min(np.abs(c_eig_ground)):.4e}")

    ref_idx = int(np.argmax(np.abs(c_eig_ground)))
    print(f"  reference (largest-|c|) idx = {ref_idx} with c = {c_eig_ground[ref_idx]:+.4f}")

    abs_c_sorted = np.sort(np.abs(c_eig_ground))[::-1]
    sigma_est = float(np.median(np.abs(c_eig_ground[100:])))
    print(f"  noise σ estimate (median of |c_I| beyond top-100): {sigma_est:.4e}")

    lambdas = [0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05]
    results = []
    for lam in lambdas:
        c_thresh = soft_threshold(c_eig_ground.copy(), lam)
        c_thresh[ref_idx] = c_eig_ground[ref_idx]
        norm = float(np.linalg.norm(c_thresh))
        n_kept = int(np.sum(np.abs(c_thresh) > 0))
        residual_norm = float(np.linalg.norm(c_thresh - c_eig_ground))
        results.append({
            "lambda": lam,
            "n_kept": n_kept,
            "n_kept_frac": n_kept / n_det,
            "Csurv_l2": norm,
            "residual_l2": residual_norm,
            "C0_largest": float(c_thresh[ref_idx]),
        })
        print(f"  λ={lam:7.4f}: n_kept={n_kept:>5}/{n_det} "
              f"({100*n_kept/n_det:5.1f}%) "
              f"||c_surv||={norm:.4f} ||c_thresh - c_raw||={residual_norm:.4e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    lams = np.array([r["lambda"] for r in results])
    n_kept_frac_arr = np.array([r["n_kept_frac"] for r in results])

    ax1.semilogy(lams, n_kept_frac_arr, "o-", color="#c66",
                 markersize=7, markeredgecolor="black", markeredgewidth=0.6,
                 label="surviving fraction")
    ax1.axvline(sigma_est, color="gray", linestyle=":", alpha=0.7,
                label=fr"noise σ ≈ {sigma_est:.1e}")
    ax1.axvline(2*sigma_est, color="gray", linestyle="--", alpha=0.7,
                label=fr"2σ ≈ {2*sigma_est:.1e}")
    ax1.set_xlabel(r"Soft-threshold $\lambda$", fontsize=10)
    ax1.set_ylabel(r"$|\{I: \widehat c_I^{\rm Lasso}\neq 0\}|\,/\,|\mathcal{S}|$",
                   fontsize=10)
    ax1.set_title(f"Lasso compression rate ($|\\mathcal{{S}}|={n_det:,}$)",
                  fontsize=10)
    ax1.set_xscale("symlog", linthresh=1e-3)
    ax1.legend(fontsize=9, frameon=False, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.3)

    bins = np.logspace(-7, 0, 50)
    ax2.hist(np.abs(c_eig_ground), bins=bins, color="#bbb", edgecolor="black",
             linewidth=0.4, alpha=0.8, label=r"all $|\widehat c_I|$")
    ax2.axvline(sigma_est, color="gray", linestyle=":", alpha=0.7,
                label=fr"noise σ ≈ {sigma_est:.1e}")
    ax2.axvline(2*sigma_est, color="gray", linestyle="--", alpha=0.7,
                label=fr"recommended $\lambda \approx 2\sigma$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$|\widehat c_I|$", fontsize=10)
    ax2.set_ylabel("count", fontsize=10)
    ax2.set_title(r"Distribution of $|\widehat c_I|$ — noise vs signal gap",
                  fontsize=10)
    ax2.legend(fontsize=9, frameon=False, loc="upper left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_pdf = "papers/cs_recovery/figs/h2o_lambda_sweep.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"  wrote {out_pdf}")

    out_json = "papers/cs_recovery/data/h2o_lambda_sweep.json"
    json.dump({"lambdas": lams.tolist(),
               "results": results,
               "candidate_set_size": n_det,
               "noise_sigma_estimate": sigma_est}, open(out_json, "w"),
              indent=2)
    print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
