#!/bin/bash
# Sequential local CPU rs scan for the 2D HEG fluid phase.
#
# Runs 5 PsiFormer VMC calculations at N=10 unpolarized, varying rs:
#   rs = 1, 2, 5, 10, 20
#
# Each takes ~7 min on CPU.  Total wall-clock: ~35 min.
# Outputs per-run directories under runs/ and a summary log
# rs_scan.log in the project root.
#
# Usage:  bash scripts/run_rs_scan_local.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export JAX_PLATFORMS=cpu

LOG=rs_scan.log
: > "$LOG"

YAMLS=(
  inputs/2dheg/heg2d_rs1_N10_unpol_500iter.yaml
  inputs/2dheg/heg2d_rs2_N10_unpol_500iter.yaml
  inputs/2dheg/heg2d_rs5_N10_unpol_500iter.yaml
  inputs/2dheg/heg2d_rs10_N10_unpol_500iter.yaml
  inputs/2dheg/heg2d_rs20_N10_unpol_500iter.yaml
)

echo "==== rs scan: $(date -Iseconds) ====" | tee -a "$LOG"

t_total_start=$(date +%s)

for yaml in "${YAMLS[@]}"; do
    project=$(basename "$yaml" .yaml)
    echo "" | tee -a "$LOG"
    echo "[run] $project ($(date -Iseconds))" | tee -a "$LOG"
    t0=$(date +%s)
    python scripts/run_heg_psiformer.py "$yaml" 2>&1 \
        | tee -a "$LOG" | tail -25 | tee -a "$LOG.tail"
    t1=$(date +%s)
    echo "[done] $project: $((t1 - t0)) s" | tee -a "$LOG"
done

t_total_end=$(date +%s)
echo "" | tee -a "$LOG"
echo "==== total: $((t_total_end - t_total_start)) s ====" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Per-run summaries:" | tee -a "$LOG"
for yaml in "${YAMLS[@]}"; do
    project=$(basename "$yaml" .yaml)
    if [[ -f "runs/$project/summary.json" ]]; then
        rs=$(python -c "import json; d=json.load(open('runs/$project/summary.json')); print(d['system']['rs'])")
        e=$(python -c "import json; d=json.load(open('runs/$project/summary.json')); print(f\"{d['e_vmc_ha']:.6f} +/- {d['e_vmc_serr_ha']:.6f}\")")
        echo "  rs=$rs  E/N = $e Ha/elec  (-> runs/$project/summary.json)" | tee -a "$LOG"
    fi
done
