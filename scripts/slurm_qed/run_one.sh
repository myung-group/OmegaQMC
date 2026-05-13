#!/bin/bash
#SBATCH --job-name=qed_vmc
#SBATCH --partition=kisti-grace
#SBATCH --gres=gpu:GH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Single QED-NN-VMC run on one GH200 (mango cluster).
#
# Usage:
#   sbatch scripts/slurm_qed/run_one.sh inputs/qed_h2/<config>.yaml
#
# Override job name for cleaner logs:
#   sbatch --job-name=qed_h2_pilot scripts/slurm_qed/run_one.sh \
#          inputs/qed_h2/h2_decoupling_pilot.yaml

# NOTE: -u disabled because mango's /etc/bashrc references unbound vars
set -eo pipefail

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
conda activate omegaqmc

echo "==== QED-NN-VMC run: $YAML ===="
echo "Host:  $(hostname)"
echo "Date:  $(date -Iseconds)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Branch: $(git -C "$(dirname "$0")/../.." rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
nvidia-smi
echo "=================================="

python scripts/run_qed_vmc.py "$YAML"
