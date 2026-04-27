#!/bin/bash
# rs scan at N=50: scan rs from 60 -> 20, running fluid (GPU 1) and
# crystal (GPU 2) in parallel for each rs.
#
# Total wall-clock estimate: ~40 min on 2x A100.
# Run this from inside the OmegaQMC repo root on frodo:
#     conda activate base
#     bash scripts/run_rs_scan_n50.sh
#
# Per-run output: runs/<project>/{train.log, summary.json, *.chk.h5}
# Per-rs scan log: runs/rs_scan_n50_master.log

set -euo pipefail
cd "$(dirname "$0")/.."

# rs values in scan order (high density first per user request).
RS_VALUES=(60 50 45 40 38 35 30 20)

LOG=runs/rs_scan_n50_master.log
mkdir -p runs
: > "$LOG"

echo "==== rs scan @ N=50 starts: $(date -Iseconds) ====" | tee -a "$LOG"
t_total_start=$(date +%s)

for rs in "${RS_VALUES[@]}"; do
    fluid_yaml=inputs/2dheg/rs_scan_n50/rs_scan_n50_rs${rs}_fluid.yaml
    crystal_yaml=inputs/2dheg/rs_scan_n50/rs_scan_n50_rs${rs}_crystal.yaml
    fluid_proj=rs_scan_n50_rs${rs}_fluid
    crystal_proj=rs_scan_n50_rs${rs}_crystal

    rm -rf "runs/${fluid_proj}" "runs/${crystal_proj}" 2>/dev/null

    echo "" | tee -a "$LOG"
    echo "[$(date -Iseconds)] === rs=${rs}: launching fluid (GPU 1) + crystal (GPU 2) in parallel ===" | tee -a "$LOG"
    t0=$(date +%s)

    CUDA_VISIBLE_DEVICES=1 python scripts/run_heg_psiformer.py "$fluid_yaml" \
        > "runs/${fluid_proj}.stdout" 2>&1 &
    F_PID=$!

    CUDA_VISIBLE_DEVICES=2 python scripts/run_heg_psiformer.py "$crystal_yaml" \
        > "runs/${crystal_proj}.stdout" 2>&1 &
    C_PID=$!

    echo "  Fluid PID:   $F_PID  (GPU 1) -> ${fluid_proj}" | tee -a "$LOG"
    echo "  Crystal PID: $C_PID  (GPU 2) -> ${crystal_proj}" | tee -a "$LOG"

    wait $F_PID
    F_RC=$?
    wait $C_PID
    C_RC=$?

    t1=$(date +%s)
    dt=$((t1 - t0))
    echo "[$(date -Iseconds)] rs=${rs}: fluid_rc=$F_RC, crystal_rc=$C_RC, took ${dt}s" | tee -a "$LOG"

    if [[ -f "runs/${fluid_proj}/summary.json" ]] && [[ -f "runs/${crystal_proj}/summary.json" ]]; then
        python3 -c "
import json
sf = json.load(open('runs/${fluid_proj}/summary.json'))
sc = json.load(open('runs/${crystal_proj}/summary.json'))
ef, sef = sf['e_vmc_ha']*1000, sf['e_vmc_serr_ha']*1000
ec, sec = sc['e_vmc_ha']*1000, sc['e_vmc_serr_ha']*1000
delta = ec - ef
print(f'  rs=${rs}: E_fluid = {ef:+.4f} +- {sef:.4f} mHa, E_crystal = {ec:+.4f} +- {sec:.4f} mHa, delta = {delta:+.4f} mHa')
" | tee -a "$LOG"
    fi
done

t_total_end=$(date +%s)
total=$((t_total_end - t_total_start))
echo "" | tee -a "$LOG"
echo "==== rs scan finished: $(date -Iseconds) (total ${total}s, ~$((total/60)) min) ====" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Per-rs final summary:" | tee -a "$LOG"
python3 -c "
import json, os
print(f'{\"rs\":>4} | {\"E_fluid (mHa)\":>20} | {\"E_crystal (mHa)\":>20} | {\"delta (mHa)\":>14} | {\"Sk_max_xtl\":>10}')
print('-' * 84)
for rs in [60, 50, 45, 40, 38, 35, 30, 20]:
    pf = f'runs/rs_scan_n50_rs{rs}_fluid/summary.json'
    pc = f'runs/rs_scan_n50_rs{rs}_crystal/summary.json'
    if not (os.path.exists(pf) and os.path.exists(pc)):
        continue
    sf = json.load(open(pf))
    sc = json.load(open(pc))
    ef = sf['e_vmc_ha']*1000; sef = sf['e_vmc_serr_ha']*1000
    ec = sc['e_vmc_ha']*1000; sec = sc['e_vmc_serr_ha']*1000
    sk_max = max(sc.get('observables', {}).get('sk_triangular_bragg', {}).get('mean', [0]))
    delta = ec - ef
    print(f'{rs:>4} | {ef:>+11.4f} +- {sef:.4f} | {ec:>+11.4f} +- {sec:.4f} | {delta:>+10.4f}    | {sk_max:>10.3f}')
" | tee -a "$LOG"
