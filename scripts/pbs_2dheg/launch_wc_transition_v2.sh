#!/bin/bash
# Submit Fix-A WC transition test: 4 jobs at N=18 unpolarized.
#
# Usage:  bash scripts/pbs_2dheg/launch_wc_transition_v2.sh
#
# Submits 4 PBS jobs:
#   * fluid + crystal at rs=20 (control: fluid should win)
#   * fluid + crystal at rs=40 (test: crystal should win)
#
# Each takes ~30 min on one A100.  Frodo has 3 A100s, so PBS will
# schedule them with ~30 min wall-clock if all GPUs are free.

set -euo pipefail
cd "$(dirname "$0")/../.."

INPUTS=(
  inputs/2dheg/heg2d_rs20_N18_unpol_fluid_v2.yaml
  inputs/2dheg/heg2d_rs20_N18_unpol_crystal_v2.yaml
  inputs/2dheg/heg2d_rs40_N18_unpol_fluid_v2.yaml
  inputs/2dheg/heg2d_rs40_N18_unpol_crystal_v2.yaml
)

mkdir -p logs

for yaml in "${INPUTS[@]}"; do
    project=$(basename "$yaml" .yaml)
    echo "Submitting $project ..."
    qsub -v YAML="$yaml" -N "$project" scripts/pbs_2dheg/run_one.pbs
done

echo ""
echo "Submitted ${#INPUTS[@]} jobs.  Status: qstat -u cwmyung"
