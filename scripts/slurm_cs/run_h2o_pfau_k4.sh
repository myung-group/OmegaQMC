#!/bin/bash
#SBATCH --job-name=h2o_pfau_k4
#SBATCH --partition=kisti-grace
#SBATCH --gres=gpu:GH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# H2O Pfau-NES K=4 training on one GH200.
# Per-iter cost scales as K^3 (det) + K (NN evals per walker) so K=4
# is roughly 16x slower per iter than K=2 on the same walker count.
# We compensate by reducing iters and walker count.

export SLURM_MPI_TYPE=none
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")/../.."
fi
source ~/.bashrc || true
conda activate omegaqmc

set -eo pipefail
export PYTHONPATH="$PWD:$PYTHONPATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

BASIS=${BASIS:-cc-pvdz}
K=${K:-4}
ITERS=${ITERS:-400}
WALKERS=${WALKERS:-96}
LR=${LR:-0.01}
SEED=${SEED:-77}
INIT_PERT=${INIT_PERT:-0.5}
INIT_RANDOM_STATES=${INIT_RANDOM_STATES:-2}

echo "=== H2O Pfau-NES K=$K, $BASIS — $(date) ==="
echo "  ITERS=$ITERS WALKERS=$WALKERS LR=$LR SEED=$SEED"
echo "  INIT_PERT=$INIT_PERT INIT_RANDOM_STATES=$INIT_RANDOM_STATES"

# Pre-flight: ensure H2O ground state checkpoint exists
GS_CKPT="cs_h2o_pfau_gs/h2o_gs_${BASIS}.chk.h5"
if [[ ! -f "$GS_CKPT" ]]; then
    echo ">>> GS checkpoint missing, training first"
    python examples/run_h2o_groundstate_only.py \
        --basis "$BASIS" \
        --ansatz examples/inputs/psiformer_small.yaml \
        --out-dir cs_h2o_pfau_gs \
        --gs-iters 1500 \
        --seed 11
fi

echo ""
echo ">>> Step 1: K=$K training"
python examples/run_h2o_pfau_kgeneral_train.py \
    --basis "$BASIS" \
    --K "$K" \
    --iters "$ITERS" \
    --walkers "$WALKERS" \
    --lr "$LR" \
    --seed "$SEED" \
    --init-perturbation "$INIT_PERT" \
    --init-random-states "$INIT_RANDOM_STATES" \
    --gs-source-dir cs_h2o_pfau_gs \
    --out-dir cs_h2o_pfau_k${K}_results

echo "=== done — $(date) ==="
