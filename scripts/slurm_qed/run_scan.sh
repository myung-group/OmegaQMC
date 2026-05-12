#!/bin/bash
#SBATCH --job-name=qed_st_scan
#SBATCH --partition=kisti-grace
#SBATCH --gres=gpu:GH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Driver: full S-T inversion scan (Phase 2o, Weber Fig 1b analog).
# Runs scripts/run_st_inversion_scan.py through one GH200 allocation.
#
# Usage:
#   sbatch scripts/slurm_qed/run_scan.sh [extra args to scan driver]
# Example:
#   sbatch scripts/slurm_qed/run_scan.sh --budget pilot \
#          --R 1.4,2.0,3.0,4.0,5.0,6.0 --lam 0.0,0.05,0.1,0.2

set -eo pipefail

mkdir -p logs

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_CUDNN_DETERMINISTIC=1

cd "$(dirname "$0")/../.."

source ~/.bashrc
conda activate omegaqmc

echo "==== S-T inversion scan ===="
echo "Host:   $(hostname)"
echo "Date:   $(date -Iseconds)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse HEAD)"
echo "============================="

python scripts/run_st_inversion_scan.py "$@"
