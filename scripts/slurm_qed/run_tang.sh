#!/bin/bash
#SBATCH --job-name=l8v2_tang
#SBATCH --partition=kisti-grace
#SBATCH --gres=gpu:GH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# L8 V2 (Tang-architecture) cavity-QED HEG run on one GH200.
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
python scripts/run_qed_tang_heg.py "$YAML"
