#!/bin/bash
# Run the N=50 fluid + crystal pair sequentially on GPU.
#
# Total wall-clock: ~25-30 min on a single A100/4070 Ti.
# Each run produces:
#   runs/heg2d_rs50_N50_unpol_<sector>/{train.log, summary.json, *.chk.h5}
#
# After both finish, generate side-by-side density + S(k) plots.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==== N=50 fluid+crystal pair: $(date -Iseconds) ===="

for sector in fluid crystal; do
    yaml="inputs/2dheg/heg2d_rs50_N50_unpol_${sector}.yaml"
    project="heg2d_rs50_N50_unpol_${sector}"
    rm -rf "runs/$project"
    echo ""
    echo "[$(date -Iseconds)] running $sector ..."
    t0=$(date +%s)
    python scripts/run_heg_psiformer.py "$yaml"
    t1=$(date +%s)
    echo "[$(date -Iseconds)] $sector done in $((t1 - t0))s"
done

echo ""
echo "==== Generating comparison plots ===="
python scripts/plot_sk_2d.py \
    runs/heg2d_rs50_N50_unpol_fluid/summary.json \
    runs/heg2d_rs50_N50_unpol_crystal/summary.json \
    --out sk_N50_fluid_vs_crystal.png

python scripts/plot_density_2d.py \
    runs/heg2d_rs50_N50_unpol_fluid \
    runs/heg2d_rs50_N50_unpol_crystal \
    --n-walkers 256 --n-equil 500 --n-sample 200 --decorr 5 \
    --out density_N50_fluid_vs_crystal.png

echo ""
echo "==== DONE: $(date -Iseconds) ===="
echo "Outputs: sk_N50_fluid_vs_crystal.png + density_N50_fluid_vs_crystal.png"
echo ""
echo "Energy comparison:"
python3 -c "
import json
for sector in ['fluid', 'crystal']:
    p = f'runs/heg2d_rs50_N50_unpol_{sector}/summary.json'
    s = json.load(open(p))
    print(f'  {sector}: E/N = {s[\"e_vmc_ha\"]*1000:+.4f} +/- {s[\"e_vmc_serr_ha\"]*1000:.4f} mHa')
"
