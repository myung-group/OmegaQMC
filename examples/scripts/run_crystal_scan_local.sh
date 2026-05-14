#!/bin/bash
# Sequential local CPU run: fluid vs crystal sectors at low density.
#
# Tests the energy crossing E_fluid(rs) vs E_crystal(rs) at rs values
# bracketing the Drummond-Needs 2009 fluid-AFM-crystal boundary
# (rs_c ~= 31).  Settings match the rs scan (light: N=10,
# embedding_dim=32, n_det=4, 500 SR iters).
#
# 8 runs total: fluid + crystal at rs = 25, 30, 35, 40.
# Total wall-clock: ~60 min on CPU.

set -euo pipefail
cd "$(dirname "$0")/.."

export JAX_PLATFORMS=cpu

LOG=crystal_scan.log
: > "$LOG"

# Run order: fluid first (cheaper), then crystal.  The runs are
# independent so order doesn't matter for correctness; running all
# fluids first means we get partial fluid-energy data before any
# crystal data is available — useful for early peeking.
YAMLS=(
  inputs/2dheg/heg2d_rs25_N10_unpol_fluid.yaml
  inputs/2dheg/heg2d_rs30_N10_unpol_fluid.yaml
  inputs/2dheg/heg2d_rs35_N10_unpol_fluid.yaml
  inputs/2dheg/heg2d_rs40_N10_unpol_fluid.yaml
  inputs/2dheg/heg2d_rs25_N10_unpol_crystal.yaml
  inputs/2dheg/heg2d_rs30_N10_unpol_crystal.yaml
  inputs/2dheg/heg2d_rs35_N10_unpol_crystal.yaml
  inputs/2dheg/heg2d_rs40_N10_unpol_crystal.yaml
)

echo "==== fluid-vs-crystal scan: $(date -Iseconds) ====" | tee -a "$LOG"

t_total_start=$(date +%s)

for yaml in "${YAMLS[@]}"; do
    project=$(basename "$yaml" .yaml)
    echo "" | tee -a "$LOG"
    echo "[run] $project ($(date -Iseconds))" | tee -a "$LOG"
    t0=$(date +%s)
    python scripts/run_heg_psiformer.py "$yaml" 2>&1 | tee -a "$LOG" > /dev/null
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
        echo "  $project rs=$rs  E/N = $e Ha/elec" | tee -a "$LOG"
    fi
done
