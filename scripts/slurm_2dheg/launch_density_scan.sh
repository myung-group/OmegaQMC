#!/bin/bash
# Launch the 2D HEG density-scan benchmark suite (Phase 0 + Phase 1
# of the cavity-2DEG project).
#
# Submits one job per (rs, polarization) combination on the frodo
# A100 cluster.  Each job writes runs/<project>/{train.log,
# summary.json, <project>.chk.h5}.
#
# Usage:
#   bash scripts/slurm_2dheg/launch_density_scan.sh
#
# Reference targets (Attaccalite 2002, FN-DMC backflow, Ha/elec):
#   rs=1   unpol -0.20372(4)   pol +0.13109(4)
#   rs=2   unpol -0.25721(3)   pol -0.19359(2)
#   rs=5   unpol -0.149518(9)  pol -0.143610(7)
#   rs=10  unpol -0.085427(6)  pol -0.084584(2)
#   rs=20  unpol -0.046385(6)  (pol value also in benchmarks file)

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
    sbatch --job-name="$project" scripts/slurm_2dheg/run_one.sh "$yaml"
done

echo "Submitted ${#INPUTS[@]} jobs.  Check status with: squeue -u $USER"
