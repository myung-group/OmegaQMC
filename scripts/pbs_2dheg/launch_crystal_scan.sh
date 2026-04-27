#!/bin/bash
# Phase 2 Wigner-crystal phase boundary scan on frodo: 10 jobs at N=18.

set -euo pipefail
cd "$(dirname "$0")/../.."

INPUTS=(
  inputs/2dheg/heg2d_rs25_N18_fluid.yaml
  inputs/2dheg/heg2d_rs30_N18_fluid.yaml
  inputs/2dheg/heg2d_rs32_N18_fluid.yaml
  inputs/2dheg/heg2d_rs35_N18_fluid.yaml
  inputs/2dheg/heg2d_rs40_N18_fluid.yaml
  inputs/2dheg/heg2d_rs25_N18_crystal_AF.yaml
  inputs/2dheg/heg2d_rs30_N18_crystal_AF.yaml
  inputs/2dheg/heg2d_rs32_N18_crystal_AF.yaml
  inputs/2dheg/heg2d_rs35_N18_crystal_AF.yaml
  inputs/2dheg/heg2d_rs40_N18_crystal_AF.yaml
)

mkdir -p logs

for yaml in "${INPUTS[@]}"; do
    project=$(basename "$yaml" .yaml)
    echo "Submitting $project ..."
    qsub -v YAML="$yaml" -N "$project" scripts/pbs_2dheg/run_one.pbs
done

echo ""
echo "Submitted ${#INPUTS[@]} jobs.  Status: qstat -u cwmyung"
