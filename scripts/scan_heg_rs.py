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

    # Full Cassella 2023 Table I scan (rs = 0.5, 1, 2, 5):
    python scripts/scan_heg_rs.py input_heg.yaml

    # Custom rs list:
    python scripts/scan_heg_rs.py input_heg.yaml --rs 1 2 5

    # Just generate YAMLs (to launch yourself, e.g. on SLURM):
    python scripts/scan_heg_rs.py input_heg.yaml --generate-only

    # Tabulate existing runs without re-running:
    python scripts/scan_heg_rs.py input_heg.yaml --collect

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
# Published benchmarks — Cassella et al. 2023 Table I
# 3D HEG, N=14 unpolarized (paramagnetic), Γ point, cubic cell.
# Table I reports TOTAL correlation energies in Hartree (not per
# electron); we divide by N=14 here for direct comparison with our
# ``e_corr_vmc_ha`` which is per-electron.
#
# Columns (all correlation energies, Ha/elec, negative):
#   SJB-VMC           Slater-Jastrow-backflow VMC (CASINO)
#   SJB-DMC           same, propagated to DMC
#   FermiNet n=1      single-det FermiNet VMC
#   FermiNet n=16     16-det FermiNet VMC (Cassella's best)
#   i-FCIQMC CBS      basis-set-limit i-FCIQMC (the ground-truth
#                     reference; all other methods compare against this)
# -------------------------------------------------------------------

_N14 = 14.0

def _per_e(x):
    return x / _N14

BENCHMARKS_N14_UNPOL = {
    # rs:   SJB-VMC         SJB-DMC         FN(n=1)         FN(n=16)        iFCIQMC-CBS
    0.5:  (_per_e(-0.58624), _per_e(-0.58778), _per_e(-0.58895), _per_e(-0.59094), _per_e(-0.5969)),
    1.0:  (_per_e(-0.5254),  _per_e(-0.5254),  _per_e(-0.52568), _per_e(-0.52692), _per_e(-0.5325)),
    2.0:  (_per_e(-0.437),   _per_e(-0.4385),  _per_e(-0.43881), _per_e(-0.44053), _per_e(-0.4447)),
    5.0:  (_per_e(-0.30339), _per_e(-0.30474), _per_e(-0.30468), _per_e(-0.30495), _per_e(-0.306)),
}

_BENCH_COLS = ('SJB-VMC', 'SJB-DMC', 'FN(n=1)', 'FN(n=16)', 'iFCIQMC-CBS')


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
    """Pretty-print a rs-scan comparison table.

    Compares our per-electron correlation energy (``e_corr_vmc_ha`` =
    ``E_VMC/N - E_HF/N``) against every column of Cassella 2023
    Table I, plus the fraction of the i-FCIQMC CBS correlation our
    VMC recovers.
    """
    benchmarks = (BENCHMARKS_N14_UNPOL
                  if pol == 'unpolarized' and system_N == 14
                  else {})

    print("\n" + "=" * 108)
    print(f" HEG rs-scan — N={system_N}, {pol}"
          f"   (correlation energy in mHa / electron, negative)")
    print("=" * 108)
    hdr = (f"{'rs':>4}  {'ours':>10}  "
           + "  ".join(f"{c:>10}" for c in _BENCH_COLS)
           + f"  {'Δ(ours-iFCIQMC)':>16}  {'%iFCIQMC':>9}")
    print(hdr)
    print("-" * 108)

    for row in rows:
        if row.get('missing'):
            print(f"  {row['project']:<30}  (summary.json not found)")
            continue

        sys_info = row.get('system', {})
        rs = sys_info.get('rs')
        e_corr = row.get('e_corr_vmc_ha')       # E_VMC/N - E_HF/N
        e_corr_err = row.get('e_vmc_serr_ha', 0.0)

        # Round rs to avoid 2.0 vs 2.00000001 key miss.
        rs_key = round(float(rs), 3)
        bench = benchmarks.get(rs_key)

        # Correlation-energy columns in mHa/e for readability.
        our_mha = e_corr * 1e3
        our_mha_err = e_corr_err * 1e3
        our_str = f"{our_mha:+8.3f}±{our_mha_err:.2f}"

        if bench is None:
            bench_strs = ['     —'] * len(_BENCH_COLS)
            delta_str = '     —'
            pct_str = '    —'
        else:
            bench_mha = [v * 1e3 for v in bench]
            bench_strs = [f"{v:+9.3f}" for v in bench_mha]
            iFCIQMC = bench_mha[-1]              # CBS column
            delta_mha = our_mha - iFCIQMC
            pct = 100.0 * our_mha / iFCIQMC if iFCIQMC else 0.0
            delta_str = f"{delta_mha:+9.3f} mHa"
            pct_str = f"{pct:6.1f}%"

        print(f"{rs:>4.1f}  {our_str:>10}  "
              + "  ".join(f"{s:>10}" for s in bench_strs)
              + f"  {delta_str:>16}  {pct_str:>9}")

    print("=" * 108)
    if benchmarks:
        print(" Benchmark source: Cassella et al. 2023, "
              "PRL 130, 036401, Table I (divided by N=14 → per-electron)")
        print("   SJB-VMC / SJB-DMC : Slater-Jastrow-backflow "
              "CASINO VMC / DMC")
        print("   FN(n=1), FN(n=16) : Cassella's FermiNet, "
              "single- and 16-det")
        print("   iFCIQMC-CBS       : initiator FCIQMC, "
              "basis-set-limit extrapolated — ground-truth reference")
    print("=" * 108)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('base_yaml', type=Path,
                   help='Base input YAML (project + all settings)')
    p.add_argument('--rs', type=float, nargs='+',
                   default=[0.5, 1.0, 2.0, 5.0],
                   help='List of rs values to scan. '
                        'Default: 0.5 1 2 5 (matches Cassella 2023 '
                        'Table I). Example: --rs 1 2 5')
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
