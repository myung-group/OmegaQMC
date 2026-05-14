"""CH3 chiral cavity scan: lambda x handedness on planar methyl radical.

Sweeps lambda and handedness, collects E +- sigma_E, <L_z>, <n_photon>.
Reuses scripts/run_qed_vmc.py per grid point, streaming per-iter output.

Output:
    logs/ch3_chiral_scan/scan.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import os.path as osp
import subprocess
import sys
import time

import h5py
import yaml

TEMPLATE = "inputs/qed_h2/_diag/h2_st_inversion_R200A.yaml"  # noqa: F841 — unused, kept for ref


def _build_yaml(out_path, project, lam, hand):
    with open("inputs/qed_ch3/ch3_chiral_template.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["project"] = project
    cfg["cavity"]["lambda"] = lam
    cfg["cavity"]["chiral_handedness"] = hand
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", default="0.05,0.10,0.20,0.30",
                    help="comma-sep lambda values")
    ap.add_argument("--hand", default="1",
                    help="comma-sep handedness values (1=sigma+, -1=sigma-)")
    ap.add_argument("--out", default="logs/ch3_chiral_scan")
    args = ap.parse_args()

    lams = [float(x) for x in args.lam.split(",")]
    hands = [int(x) for x in args.hand.split(",")]

    os.makedirs(args.out, exist_ok=True)
    csv_path = osp.join(args.out, "scan.csv")
    new = not osp.exists(csv_path)
    with open(csv_path, "a", newline="") as csv_fh:
        w = csv.writer(csv_fh)
        if new:
            w.writerow([
                "lambda", "hand", "E_mean", "E_serr",
                "n_photon", "l_z_mean", "l_z_serr",
                "train_s", "eval_s",
            ])
        total = len(lams) * len(hands)
        idx = 0
        for hand in hands:
            hand_tag = "p" if hand == 1 else "m"
            for lam in lams:
                idx += 1
                run_id = (
                    f"ch3_chiral_L{lam:.2f}_s{hand_tag}"
                    .replace(".", "p")
                )
                run_dir = osp.join(args.out, run_id)
                os.makedirs(run_dir, exist_ok=True)
                yaml_path = osp.join(run_dir, "config.yaml")
                _build_yaml(yaml_path, run_id, lam, hand)
                inner_dir = osp.join("logs", run_id)
                h5_path = osp.join(inner_dir, f"{run_id}.results.h5")

                t0 = time.time()
                print(f"[{idx}/{total}] {run_id}", flush=True)
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
                    lam, hand, r["E_mean"], r["E_serr"],
                    r["n_photon"], lz, lzs,
                    r["train_s"], r["eval_s"],
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
