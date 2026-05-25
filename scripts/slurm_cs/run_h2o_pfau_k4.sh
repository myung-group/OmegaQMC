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

ANSATZ=${ANSATZ:-examples/inputs/psiformer_small.yaml}
# ANSATZ can be a YAML path OR a built-in config name (paulinet,
# ferminet, ferminet_jastrow, ferminet_jastrow_complex, deeperwin,
# psiformer). Production-grade FermiNet + Jastrow + backflow:
# ANSATZ=ferminet_jastrow.

# Suffix for output directories so different ansatz runs don't collide
ANSATZ_TAG=$(basename "$ANSATZ" .yaml | sed 's|/|_|g')
OUT_TAG=${OUT_TAG:-${ANSATZ_TAG}}

echo "=== H2O Pfau-NES K=$K, $BASIS, ANSATZ=$ANSATZ — $(date) ==="
echo "  ITERS=$ITERS WALKERS=$WALKERS LR=$LR SEED=$SEED"
echo "  INIT_PERT=$INIT_PERT INIT_RANDOM_STATES=$INIT_RANDOM_STATES"
echo "  OUT_TAG=$OUT_TAG"

# Pre-flight: ensure H2O ground state checkpoint for this ansatz exists
GS_DIR="cs_h2o_pfau_gs_${OUT_TAG}"
GS_CKPT="${GS_DIR}/h2o_gs_${BASIS}.chk.h5"
GS_ITERS=${GS_ITERS:-1500}
if [[ ! -f "$GS_CKPT" ]]; then
    echo ">>> GS checkpoint missing, training first ($GS_ITERS iters)"
    python examples/run_h2o_groundstate_only.py \
        --basis "$BASIS" \
        --ansatz "$ANSATZ" \
        --out-dir "$GS_DIR" \
        --gs-iters "$GS_ITERS" \
        --seed 11
fi

echo ""
echo ">>> Step 1: K=$K training"
python examples/run_h2o_pfau_kgeneral_train.py \
    --basis "$BASIS" \
    --ansatz "$ANSATZ" \
    --K "$K" \
    --iters "$ITERS" \
    --walkers "$WALKERS" \
    --lr "$LR" \
    --seed "$SEED" \
    --init-perturbation "$INIT_PERT" \
    --init-random-states "$INIT_RANDOM_STATES" \
    --gs-source-dir "$GS_DIR" \
    --out-dir cs_h2o_pfau_k${K}_${OUT_TAG}_results

echo "=== done — $(date) ==="
