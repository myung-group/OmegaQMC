#!/bin/bash
# Quick validation: 3 short runs at N=10 to confirm the 2D pipeline
# converges before launching the long N=58 production runs.
#
# Each run takes ~30 min on a single A100; total wall-clock ~30 min
# if run in parallel.

set -euo pipefail
cd "$(dirname "$0")/../.."

INPUTS=(
  inputs/2dheg/heg2d_rs1_N10_unpol_quick.yaml
  inputs/2dheg/heg2d_rs2_N10_unpol_quick.yaml
  inputs/2dheg/heg2d_rs5_N10_unpol_quick.yaml
)

mkdir -p logs

for yaml in "${INPUTS[@]}"; do
    project=$(basename "$yaml" .yaml)
    sbatch --job-name="$project" --time=2:00:00 \
        scripts/slurm_2dheg/run_one.sh "$yaml"
done
