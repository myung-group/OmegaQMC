"""S-T inversion scan driver — Phase 2o (Weber Fig 1b analog).

Sweeps (R, lambda, spin) for H2 under an electric chiral cavity and
collects E ± sigma_E and <n_photon> per run. Reuses run_qed_vmc.py
as a subprocess for each grid point.

Output:
    logs/st_inversion_scan/scan.csv
    logs/st_inversion_scan/<run_id>/*  (per-run HDF5 + run.log)

Usage:
    python scripts/run_st_inversion_scan.py \\
        --R 1.4,2.0,3.0,4.0,5.0,6.0 \\
        --lam 0.0,0.05,0.1,0.2 \\
        --spin singlet,triplet \\
        --budget pilot
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from datetime import datetime

import h5py
import yaml


def _budget_overrides(budget: str) -> dict:
    """Return (train, eval) sub-dicts for a named budget."""
    if budget == "pilot":
        return dict(
            train=dict(
                num_iters=50, num_walkers=32, num_steps_per_block=30,
                num_blocks_equil=5, num_steps_decorr=5, mc_timestep=0.1,
                lr=0.05, damping=1e-3, cg_maxiter=100,
                max_param_change=0.5, jac_batch_size=16, verbose=0,
            ),
            eval=dict(
                enabled=True, num_walkers=64, num_steps_per_block=30,
                num_blocks=10, num_blocks_equil=5, mc_timestep=0.1, verbose=0,
            ),
        )
    if budget == "medium":
        return dict(
            train=dict(
                num_iters=500, num_walkers=256, num_steps_per_block=30,
                num_blocks_equil=5, num_steps_decorr=5, mc_timestep=0.1,
                lr=0.05, damping=1e-3, cg_maxiter=100,
                max_param_change=0.5, jac_batch_size=64, verbose=0,
            ),
            eval=dict(
                enabled=True, num_walkers=512, num_steps_per_block=30,
                num_blocks=30, num_blocks_equil=10, mc_timestep=0.1, verbose=0,
            ),
        )
    raise ValueError(f"unknown budget: {budget}")


def _build_yaml(out_path: str, project: str, R: float, lam: float,
                spin: str, budget: str, seed: int):
    """Write a per-run YAML by overriding the chiral pilot template."""
    cfg = {
        "project": project,
        "seed": seed,
        "system": {"type": "h2", "R_bohr": R, "spin": spin},
        "cavity": {
            "omega": 0.5,
            "lambda": lam,
            "polarization":   [1.0, 0.0, 0.0],   # eps_x perp to z (H-H axis)
            "polarization_y": [0.0, 1.0, 0.0],   # eps_y perp to z
            "chiral_handedness": 1,               # sigma+
        },
        "ansatz": {"type": "OmegaQMC/psi/nn/conf/ferminet_jastrow_complex.yaml"},
        "optimizer": {
            "arch": "tang_native",
            "complex_psi": True,
            "alpha_init": 0.0,
            "alpha_train": False,
            "nph_max": 4,
            **_budget_overrides(budget),
        },
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _parse_results_h5(h5_path: str) -> dict:
    with h5py.File(h5_path, "r") as f:
        g = f["eval"]
        out = {
            "E_mean": float(g.attrs["E_mean"]),
            "E_serr": float(g.attrs["E_serr"]),
            "n_photon_mean": float(g.attrs["n_photon_mean"]),
            "acceptance_r": float(g.attrs["acceptance_r"]),
            "acceptance_n": float(g.attrs["acceptance_n"]),
            "eval_elapsed_s": float(g.attrs["elapsed_s"]),
        }
        if "l_z_mean" in g.attrs:
            out["l_z_mean"] = float(g.attrs["l_z_mean"])
            out["l_z_serr"] = float(g.attrs["l_z_serr"])
        gt = f["train"]
        out["train_elapsed_s"] = float(gt.attrs["elapsed_s"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", default="1.4,2.0,3.0,4.0,5.0,6.0",
                    help="comma-sep R_bohr values")
    ap.add_argument("--lam", default="0.0,0.05,0.1,0.2",
                    help="comma-sep lambda values")
    ap.add_argument("--spin", default="singlet,triplet",
                    help="comma-sep spin states")
    ap.add_argument("--budget", default="pilot",
                    choices=["pilot", "medium"])
    ap.add_argument("--out", default="logs/st_inversion_scan",
                    help="output directory")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip run if results.h5 already exists")
    args = ap.parse_args()

    Rs = [float(x) for x in args.R.split(",")]
    lams = [float(x) for x in args.lam.split(",")]
    spins = [x.strip() for x in args.spin.split(",")]

    os.makedirs(args.out, exist_ok=True)
    csv_path = osp.join(args.out, "scan.csv")
    new_csv = not osp.exists(csv_path)
    csv_fh = open(csv_path, "a", newline="")
    writer = csv.writer(csv_fh)
    if new_csv:
        writer.writerow([
            "run_id", "R_bohr", "lambda", "spin",
            "E_mean", "E_serr", "n_photon_mean",
            "l_z_mean", "l_z_serr",
            "train_elapsed_s", "eval_elapsed_s",
        ])
        csv_fh.flush()

    total = len(Rs) * len(lams) * len(spins)
    t_scan_start = time.time()
    print(f"[scan] {total} runs, budget={args.budget}, "
          f"out={args.out}", flush=True)
    print(f"[scan] R={Rs}", flush=True)
    print(f"[scan] lam={lams}", flush=True)
    print(f"[scan] spin={spins}", flush=True)

    idx = 0
    for spin in spins:
        for R in Rs:
            for lam in lams:
                idx += 1
                run_id = (f"st_R{R:.2f}_L{lam:.3f}_{spin}"
                          .replace(".", "p"))
                run_dir = osp.join(args.out, run_id)
                os.makedirs(run_dir, exist_ok=True)
                yaml_path = osp.join(run_dir, "config.yaml")
                # Let the inner driver write to its default logs/<project>/.
                # We just collect the resulting h5 here.
                project = run_id
                _build_yaml(
                    yaml_path, project=project,
                    R=R, lam=lam, spin=spin, budget=args.budget,
                    seed=args.seed,
                )
                inner_log_dir = osp.join("logs", project)
                h5_path = osp.join(
                    inner_log_dir, f"{project}.results.h5",
                )

                t0 = time.time()
                tag = f"[{idx}/{total}]"
                print(f"{tag} {run_id} ...", flush=True)

                if args.skip_existing and osp.exists(h5_path):
                    print(f"{tag} (skipped — h5 exists)", flush=True)
                else:
                    stdout_path = osp.join(run_dir, "stdout.log")
                    with open(stdout_path, "w") as logf:
                        rc = subprocess.run(
                            [sys.executable,
                             "scripts/run_qed_vmc.py", yaml_path],
                            stdout=logf, stderr=subprocess.STDOUT,
                        ).returncode
                    if rc != 0:
                        print(f"{tag} FAILED rc={rc}, "
                              f"see {stdout_path}", flush=True)
                        continue

                try:
                    res = _parse_results_h5(h5_path)
                except Exception as e:
                    print(f"{tag} parse-failed: {e}", flush=True)
                    continue

                elapsed = time.time() - t0
                lz = res.get("l_z_mean", float("nan"))
                lz_serr = res.get("l_z_serr", float("nan"))
                writer.writerow([
                    run_id, R, lam, spin,
                    res["E_mean"], res["E_serr"],
                    res["n_photon_mean"],
                    lz, lz_serr,
                    res["train_elapsed_s"], res["eval_elapsed_s"],
                ])
                csv_fh.flush()
                print(f"{tag} E={res['E_mean']:+.4f} "
                      f"± {res['E_serr']:.4f} Ha, "
                      f"<n>={res['n_photon_mean']:.3f}, "
                      f"<L_z>={lz:+.4f} (elapsed {elapsed:.0f}s)",
                      flush=True)

    csv_fh.close()
    dt = time.time() - t_scan_start
    print(f"[scan] done in {dt/60:.1f} min, "
          f"csv: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
