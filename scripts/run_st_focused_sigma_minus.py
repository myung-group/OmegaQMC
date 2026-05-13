"""Sigma-minus mirror of run_st_focused.py — S, T at R=2.0 A, lambda=0.5.

Tests time-reversal partner of the sigma+ run. Expected:
  E(sigma-) = E(sigma+)  (within MC noise)
  <L_z>(sigma-) = - <L_z>(sigma+)  (sign flip)
for both spin sectors.

Output: logs/st_focused_R200A_sm/scan.csv (append).
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
OUT_DIR = "logs/st_focused_R200A_sm"
LAMBDAS = [0.5]
SPINS = ["singlet", "triplet"]
HAND = -1   # sigma-


def _build_yaml(out_path, project, spin, lam):
    with open(TEMPLATE) as f:
        cfg = yaml.safe_load(f)
    cfg["project"] = project
    cfg["system"]["spin"] = spin
    cfg["cavity"]["lambda"] = lam
    cfg["cavity"]["chiral_handedness"] = HAND
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _parse_h5(p):
    with h5py.File(p, "r") as f:
        g = f["eval"]
        out = dict(
            E_mean=float(g.attrs["E_mean"]),
            E_serr=float(g.attrs["E_serr"]),
            n_photon=float(g.attrs["n_photon_mean"]),
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
                "spin", "lambda", "hand", "E_mean", "E_serr",
                "n_photon", "l_z_mean", "l_z_serr",
            ])
        idx = 0
        total = len(LAMBDAS) * len(SPINS)
        for spin in SPINS:
            for lam in LAMBDAS:
                idx += 1
                run_id = (
                    f"st_focused_R200A_sm_{spin}_L{lam:.2f}"
                    .replace(".", "p")
                )
                run_dir = osp.join(OUT_DIR, run_id)
                os.makedirs(run_dir, exist_ok=True)
                yaml_path = osp.join(run_dir, "config.yaml")
                _build_yaml(yaml_path, run_id, spin, lam)
                inner_dir = osp.join("logs", run_id)
                h5_path = osp.join(inner_dir, f"{run_id}.results.h5")

                t0 = time.time()
                print(f"[{idx}/{total}] {run_id} (sigma-)", flush=True)
                if osp.exists(h5_path):
                    print("  (skip — h5 already exists)", flush=True)
                else:
                    stdout_path = osp.join(run_dir, "stdout.log")
                    with open(stdout_path, "w") as lf:
                        proc = subprocess.Popen(
                            [sys.executable, "-u",
                             "scripts/run_qed_vmc.py", yaml_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        for line in proc.stdout:
                            print(line, end="", flush=True)
                            lf.write(line)
                            lf.flush()
                        rc = proc.wait()
                    if rc != 0:
                        print(f"  FAILED rc={rc}", flush=True)
                        continue
                r = _parse_h5(h5_path)
                lz = r.get("l_z_mean", float("nan"))
                lzs = r.get("l_z_serr", float("nan"))
                w.writerow([
                    spin, lam, HAND, r["E_mean"], r["E_serr"],
                    r["n_photon"], lz, lzs,
                ])
                csv_fh.flush()
                dt = time.time() - t0
                print(f"  E={r['E_mean']:+.4f} ± {r['E_serr']:.4f} Ha, "
                      f"<n>={r['n_photon']:.3f}, "
                      f"<L_z>={lz:+.4f}, dt={dt:.0f}s",
                      flush=True)

        print(f"\nresults in {csv_path}", flush=True)


if __name__ == "__main__":
    main()
