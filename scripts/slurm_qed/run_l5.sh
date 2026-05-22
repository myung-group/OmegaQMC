#!/bin/bash
#SBATCH --job-name=l5_qed
#SBATCH --partition=kisti-grace
#SBATCH --gres=gpu:GH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# L5/L6/L7 QED-NN-VMC run on one GH200.
#
# Usage:
#   sbatch --job-name=<name> \
#          --output=runs/<project>.slurm.out \
#          --error=runs/<project>.slurm.err \
#          scripts/slurm_qed/run_l5.sh inputs/2dheg_qed/<config>.yaml

set -eo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 <yaml-input>"; exit 1; fi
YAML="$1"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_CUDNN_DETERMINISTIC=1

cd "$(dirname "$0")/../.."

source ~/.bashrc
conda activate omegaqmc

echo "=== $(basename "$YAML" .yaml) — $(date) ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"

python scripts/run_qed_l5_heg.py "$YAML"
