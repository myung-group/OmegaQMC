#!/bin/bash
# Phase 2: Wigner-crystal phase boundary scan.
#
# For each rs in {25, 30, 32, 35, 40} launch BOTH the fluid-sector
# PsiFormer and the crystal-sector PsiFormer.  The crystal sector
# uses GaussianLocalizedEnvelope2D with a triangular Bravais lattice
# and AFM (Neel) spin pattern.
#
# After convergence, compare E_fluid(rs) and E_crystal(rs).  The
# phase boundary r_s^c is where the two curves cross.  Drummond-Needs
# 2009 reports r_s^c = 31(1) for the unpolarized 2D HEG; we expect
# the crystal to win at rs=35,40 and lose at rs=25,30.

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
    sbatch --job-name="$project" scripts/slurm_2dheg/run_one.sh "$yaml"
done

echo "Submitted ${#INPUTS[@]} jobs.  Phase 2 scan running."
