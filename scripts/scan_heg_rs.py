"""Scan HEG VMC across rs values and compare to published NN-VMC.

For each rs in ``--rs``, the script:
  1. Reads the base YAML,
  2. Overrides ``system.rs`` and ``project`` → ``<base>_rs<value>``,
  3. Writes ``runs/<new_project>/input.yaml``, and
  4. Invokes ``scripts/run_heg_psiformer.py`` on it.

At the end (or when invoked with ``--collect``), reads every
``runs/<project>/summary.json`` and prints a comparison table against
Cassella 2023 (FermiNet) and DMC benchmarks for unpolarised N=14.

Examples:

    # Full scan (runs jobs sequentially):
    python scripts/scan_heg_rs.py input_heg.yaml --rs 1 2 5 10 20

    # Just generate YAMLs (to launch yourself, e.g. on SLURM):
    python scripts/scan_heg_rs.py input_heg.yaml --rs 1 2 5 10 \\
        --generate-only

    # Tabulate existing runs without re-running:
    python scripts/scan_heg_rs.py input_heg.yaml --rs 1 2 5 10 \\
        --collect

The reference numbers are hard-coded in this file (see
``BENCHMARKS_N14_UNPOL``); edit that dict to add other system sizes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


# -------------------------------------------------------------------
# Published benchmarks — N=14 unpolarized, Γ point, cubic cell.
# From Cassella 2023 Table I (FermiNet); DMC col from Fraser 1996 /
# subsequent refinements. All values in Ha per electron.
# -------------------------------------------------------------------

BENCHMARKS_N14_UNPOL = {
    # rs:  (E_HF/N,   E_FermiNet/N, err,     E_DMC/N,  err)
    1.0:  (0.5878,    0.52985,      3e-5,    0.5300,   2e-4),
    2.0:  (0.02304,  -0.01881,      1e-5,   -0.0188,   1e-4),
    5.0:  (-0.0754,  -0.08049,      1e-5,   -0.08049,  4e-5),
    10.0: (-0.05236, -0.05429,      1e-5,   -0.05429,  2e-5),
    20.0: (-0.0312,  -0.03180,      1e-5,   -0.03180,  1e-5),
}


def _generate_yaml(base_cfg, rs, run_dir: Path, base_project: str):
    """Write a per-rs YAML derived from base_cfg into run_dir."""
    cfg = deepcopy(base_cfg)
    cfg.setdefault('system', {})
    cfg['system']['rs'] = float(rs)
    cfg['project'] = f"{base_project}_rs{rs}"

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / 'input.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def _run_one(yaml_path: Path):
    """Invoke the HEG runner on one YAML file, streaming its output."""
    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / 'scripts' / 'run_heg_psiformer.py'
    cmd = [sys.executable, str(runner), str(yaml_path)]
    print(f"\n{'=' * 70}")
    print(f"[scan] running: {' '.join(cmd)}")
    print(f"{'=' * 70}\n", flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def _collect(project_names):
    """Load summary.json for every requested project, return list of dicts."""
    rows = []
    for proj in project_names:
        summary = Path('runs') / proj / 'summary.json'
        if not summary.is_file():
            rows.append({'project': proj, 'missing': True})
            continue
        with open(summary) as f:
            rows.append({'project': proj, **json.load(f)})
    return rows


def _print_table(rows, pol='unpolarized', system_N=14):
    """Pretty-print a rs-scan comparison table."""
    benchmarks = (BENCHMARKS_N14_UNPOL
                  if pol == 'unpolarized' and system_N == 14
                  else {})

    print("\n" + "=" * 92)
    print(f" HEG rs-scan — N={system_N}, {pol}   (energies in Ha/elec)")
    print("=" * 92)
    print(f"{'rs':>5}  {'E_VMC (ours)':>16}  {'E_HF':>10}  "
          f"{'E_FermiNet':>12}  {'E_DMC':>10}  "
          f"{'ΔE vs DMC':>11}  {'corr%':>7}")
    print("-" * 92)

    for row in rows:
        if row.get('missing'):
            print(f"{row['project']:<30}  (summary.json not found)")
            continue

        sys_info = row.get('system', {})
        rs = sys_info.get('rs')
        e_vmc = row.get('e_vmc_ha')
        e_vmc_err = row.get('e_vmc_serr_ha', 0.0)
        e_hf = row.get('e_hf_ha')

        bench = benchmarks.get(round(float(rs) * 100) / 100)
        if bench is None:
            e_nn, e_nn_err, e_dmc, e_dmc_err = (None,) * 4
        else:
            _, e_nn, e_nn_err, e_dmc, e_dmc_err = bench

        e_nn_str = (f"{e_nn:+.5f}" if e_nn is not None else '    —')
        e_dmc_str = (f"{e_dmc:+.5f}" if e_dmc is not None else '    —')

        # Fraction of DMC correlation recovered.
        if e_dmc is not None and e_hf is not None:
            corr_dmc = e_dmc - e_hf
            corr_vmc = e_vmc - e_hf
            pct = 100.0 * corr_vmc / corr_dmc if corr_dmc else 0.0
            delta = (e_vmc - e_dmc) * 1e3   # mHa
            delta_str = f"{delta:+7.3f} mHa"
            pct_str = f"{pct:5.1f}%"
        else:
            delta_str = '     —'
            pct_str = '    —'

        print(f"{rs:>5.1f}  {e_vmc:+11.5f}±{e_vmc_err:.5f}  "
              f"{e_hf:+10.5f}  {e_nn_str:>12}  {e_dmc_str:>10}  "
              f"{delta_str:>11}  {pct_str:>7}")

    print("=" * 92)
    if benchmarks:
        print(" Benchmark sources:")
        print("   E_FermiNet — Cassella et al. 2023, PRL 130, 036401 "
              "(Table I, N=14 paramagnetic)")
        print("   E_DMC      — Fraser 1996 / subsequent refinements "
              "(cited in Cassella Table I)")
    print("=" * 92)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('base_yaml', type=Path,
                   help='Base input YAML (project + all settings)')
    p.add_argument('--rs', type=float, nargs='+', required=True,
                   help='List of rs values to scan, e.g. --rs 1 2 5 10')
    p.add_argument('--generate-only', action='store_true',
                   help='Write per-rs YAMLs but do not run')
    p.add_argument('--collect', action='store_true',
                   help='Do not generate or run — just tabulate '
                        'existing runs/<project>_rs<value>/summary.json')
    p.add_argument('--continue-on-error', action='store_true',
                   help='Keep scanning even if a single rs run fails')
    args = p.parse_args()

    if not args.base_yaml.is_file():
        print(f"error: {args.base_yaml} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.base_yaml) as f:
        base_cfg = yaml.safe_load(f)

    base_project = base_cfg.get('project') or args.base_yaml.stem
    project_names = [f"{base_project}_rs{rs}" for rs in args.rs]

    # --- Collect-only path ---
    if args.collect:
        rows = _collect(project_names)
        _print_table(
            rows,
            pol=base_cfg.get('system', {}).get('polarization',
                                               'unpolarized'),
            system_N=base_cfg.get('system', {}).get('N', 14),
        )
        return

    # --- Generate YAMLs ---
    yaml_paths = []
    for rs in args.rs:
        run_dir = Path('runs') / f"{base_project}_rs{rs}"
        path = _generate_yaml(base_cfg, rs, run_dir, base_project)
        yaml_paths.append(path)
        print(f"[scan] wrote {path}")

    if args.generate_only:
        print("\n[scan] --generate-only: done.")
        print("To run each:")
        for path in yaml_paths:
            print(f"  python scripts/run_heg_psiformer.py {path}")
        return

    # --- Run sequentially ---
    failed = []
    for rs, path in zip(args.rs, yaml_paths):
        rc = _run_one(path)
        if rc != 0:
            failed.append((rs, path, rc))
            if not args.continue_on_error:
                print(f"[scan] rs={rs} failed (rc={rc}); stopping. "
                      f"Use --continue-on-error to keep going.",
                      file=sys.stderr)
                break

    # --- Tabulate ---
    rows = _collect(project_names)
    _print_table(
        rows,
        pol=base_cfg.get('system', {}).get('polarization',
                                           'unpolarized'),
        system_N=base_cfg.get('system', {}).get('N', 14),
    )

    if failed:
        print("\n[scan] some rs points failed:")
        for rs, path, rc in failed:
            print(f"  rs={rs}  rc={rc}  input={path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
