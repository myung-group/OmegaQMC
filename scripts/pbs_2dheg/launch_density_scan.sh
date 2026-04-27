#!/bin/bash
# Phase 0 density scan on frodo: 9 jobs (5 unpol + 4 pol).
#
# Usage:  bash scripts/pbs_2dheg/launch_density_scan.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

INPUTS=(
  inputs/2dheg/heg2d_rs1_N58_unpol.yaml
  inputs/2dheg/heg2d_rs2_N58_unpol.yaml
  inputs/2dheg/heg2d_rs5_N58_unpol.yaml
  inputs/2dheg/heg2d_rs10_N58_unpol.yaml
  inputs/2dheg/heg2d_rs20_N58_unpol.yaml
  inputs/2dheg/heg2d_rs1_N57_pol.yaml
  inputs/2dheg/heg2d_rs2_N57_pol.yaml
  inputs/2dheg/heg2d_rs5_N57_pol.yaml
  inputs/2dheg/heg2d_rs10_N57_pol.yaml
)

mkdir -p logs

for yaml in "${INPUTS[@]}"; do
    project=$(basename "$yaml" .yaml)
    echo "Submitting $project ..."
    qsub -v YAML="$yaml" -N "$project" scripts/pbs_2dheg/run_one.pbs
done

echo ""
echo "Submitted ${#INPUTS[@]} jobs.  Status: qstat -u cwmyung"
