"""Focused S-T inversion test at R = 2.0 Angstrom = 3.7795 Bohr.

Generates 4 YAMLs from the template (S/T x lambda=0/0.5) and runs each
through scripts/run_qed_vmc.py, collecting E +- sigma and <L_z>.

Output:
    logs/st_focused_R200A/scan.csv
"""
from __future__ import annotations

import csv
import os
import os.path as osp
import subprocess
import sys
import time

import h5py
import yaml

TEMPLATE = "inputs/qed_h2/_diag/h2_st_inversion_R200A.yaml"
OUT_DIR = "logs/st_focused_R200A"
LAMBDAS = [0.0, 0.5]
SPINS = ["singlet", "triplet"]


def _build_yaml(out_path, project, spin, lam):
    with open(TEMPLATE) as f:
        cfg = yaml.safe_load(f)
    cfg["project"] = project
    cfg["system"]["spin"] = spin
    cfg["cavity"]["lambda"] = lam
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _parse_h5(p):
    with h5py.File(p, "r") as f:
        g = f["eval"]
        out = dict(
            E_mean=float(g.attrs["E_mean"]),
            E_serr=float(g.attrs["E_serr"]),
            n_photon=float(g.attrs["n_photon_mean"]),
            train_s=float(f["train"].attrs["elapsed_s"]),
            eval_s=float(g.attrs["elapsed_s"]),
        )
        if "l_z_mean" in g.attrs:
            out["l_z_mean"] = float(g.attrs["l_z_mean"])
            out["l_z_serr"] = float(g.attrs["l_z_serr"])
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = osp.join(OUT_DIR, "scan.csv")
    new = not osp.exists(csv_path)
    with open(csv_path, "a", newline="") as csv_fh:
        w = csv.writer(csv_fh)
        if new:
            w.writerow([
                "spin", "lambda", "E_mean", "E_serr",
                "n_photon", "l_z_mean", "l_z_serr",
                "train_s", "eval_s",
            ])
        idx = 0
        total = len(LAMBDAS) * len(SPINS)
        for spin in SPINS:
            for lam in LAMBDAS:
                idx += 1
                run_id = (
                    f"st_focused_R200A_{spin}_L{lam:.2f}"
                    .replace(".", "p")
                )
                run_dir = osp.join(OUT_DIR, run_id)
                os.makedirs(run_dir, exist_ok=True)
                yaml_path = osp.join(run_dir, "config.yaml")
                _build_yaml(yaml_path, run_id, spin, lam)
                inner_dir = osp.join("logs", run_id)
                h5_path = osp.join(inner_dir, f"{run_id}.results.h5")

                t0 = time.time()
                print(f"[{idx}/{total}] {run_id}", flush=True)
                stdout_path = osp.join(run_dir, "stdout.log")
                with open(stdout_path, "w") as lf:
                    rc = subprocess.run(
                        [sys.executable,
                         "scripts/run_qed_vmc.py", yaml_path],
                        stdout=lf, stderr=subprocess.STDOUT,
                    ).returncode
                if rc != 0:
                    print(f"  FAILED rc={rc}", flush=True)
                    continue
                r = _parse_h5(h5_path)
                lz = r.get("l_z_mean", float("nan"))
                lzs = r.get("l_z_serr", float("nan"))
                w.writerow([
                    spin, lam, r["E_mean"], r["E_serr"], r["n_photon"],
                    lz, lzs, r["train_s"], r["eval_s"],
                ])
                csv_fh.flush()
                dt = time.time() - t0
                print(f"  E={r['E_mean']:+.4f} ± {r['E_serr']:.4f} Ha, "
                      f"<n>={r['n_photon']:.3f}, "
                      f"<L_z>={lz:+.4f}, dt={dt:.0f}s",
                      flush=True)

        # Summary
        print(f"\nresults in {csv_path}", flush=True)


if __name__ == "__main__":
    main()
