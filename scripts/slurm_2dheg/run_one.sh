#!/bin/bash
#SBATCH --job-name=heg2d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Single 2D HEG run on one A100.
#
# Usage:
#   sbatch scripts/slurm_2dheg/run_one.sh inputs/2dheg/heg2d_rs2_N58_unpol.yaml
#
# The script's job name is overridden via --job-name on the sbatch
# command line for clearer log filenames, e.g.:
#   sbatch --job-name=heg2d_rs2_N58 scripts/slurm_2dheg/run_one.sh inputs/2dheg/heg2d_rs2_N58_unpol.yaml

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <yaml-input>"
    exit 1
fi

YAML="$1"
mkdir -p logs

# Reproducible cuDNN; deterministic XLA op ordering.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_CUDNN_DETERMINISTIC=1

cd "$(dirname "$0")/../.."     # project root

source ~/.bashrc
conda activate bam

echo "==== 2D HEG run: $YAML ===="
echo "Host: $(hostname)"
echo "Date: $(date -Iseconds)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
nvidia-smi
echo "============================="

python scripts/run_heg_psiformer.py "$YAML"
