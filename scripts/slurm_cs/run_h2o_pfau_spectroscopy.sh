#!/bin/bash
#SBATCH --job-name=h2o_pfau_spec
#SBATCH --partition=kisti-grace
#SBATCH --gres=gpu:GH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# H2O Pfau-NES + CS-recovery + spectroscopy on one GH200.
#
# H2O is the right test molecule for a clean dipole-allowed-transition
# demo: C2v point group has no inversion symmetry, so the
# gerade-trap that affects H2/cc-pVDZ does not apply. The lowest
# singlet excitation is the A 1B1 state around ~7.4 eV, dipole-
# allowed along the C2 axis.
#
# nelec = 10 in STO-3G (7 AOs, FCI = 441 dets, tractable). Pfau-NES
# K=2 with 10 electrons + 2 states = (2, 10, 3) = 60-dim joint walker
# space; expect ~10x slower per-iter than H2.
#
# Steps:
#  1. Train ground state via SR (run_h2o_groundstate_only.py)
#  2. Train Pfau-NES K=2 + spectroscopy (run_h2o_pfau_nes_spectroscopy.py)

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
CAS_NCAS=${CAS_NCAS:-8}
CAS_NELECAS=${CAS_NELECAS:-4,4}
ANSATZ=${ANSATZ:-examples/inputs/psiformer_small.yaml}
GS_ITERS=${GS_ITERS:-1500}
PFAU_ITERS=${PFAU_ITERS:-800}
PFAU_WALKERS=${PFAU_WALKERS:-256}
PFAU_LR=${PFAU_LR:-0.01}
PFAU_DAMPING=${PFAU_DAMPING:-1e-3}
PFAU_CG=${PFAU_CG:-100}
PFAU_DECORR=${PFAU_DECORR:-10}
SEED=${SEED:-77}
INIT_FROM_GROUND=${INIT_FROM_GROUND:-1}
INIT_STATE2_RANDOM=${INIT_STATE2_RANDOM:-1}
INIT_PERTURBATION=${INIT_PERTURBATION:-0.5}
OUT_DIR=${OUT_DIR:-cs_h2o_pfau_spec_results}

echo "=== H2O Pfau-NES + CS spectroscopy on $(hostname) — $(date) ==="
echo "  BASIS=$BASIS  ansatz=$ANSATZ"
echo "  GS_ITERS=$GS_ITERS  PFAU_ITERS=$PFAU_ITERS"

# Step 1: GS bootstrap
echo ""
echo ">>> Step 1: Train H2O ground state ($GS_ITERS SR iters)"
python examples/run_h2o_groundstate_only.py \
    --basis "$BASIS" --ansatz "$ANSATZ" \
    --out-dir cs_h2o_pfau_gs \
    --gs-iters "$GS_ITERS" \
    --seed 11

# Step 2: Pfau-NES K=2 + spectroscopy
INIT_ARG=""
if [[ "$INIT_FROM_GROUND" == "1" ]]; then
    INIT_ARG="--init-from-ground --init-perturbation $INIT_PERTURBATION"
    if [[ "$INIT_STATE2_RANDOM" == "1" ]]; then
        INIT_ARG="$INIT_ARG --init-state2-random"
    fi
fi

echo ""
echo ">>> Step 2: H2O Pfau-NES K=2 + spectroscopy ($PFAU_ITERS iters)"
CAS_ARG=""
if [[ -n "$CAS_NCAS" ]]; then
    CAS_ARG="--cas-ncas $CAS_NCAS --cas-nelecas $CAS_NELECAS"
fi
python examples/run_h2o_pfau_nes_spectroscopy.py \
    --basis "$BASIS" --ansatz "$ANSATZ" \
    --out-dir "$OUT_DIR" \
    --pfau-iters "$PFAU_ITERS" \
    --num-walkers "$PFAU_WALKERS" \
    --lr "$PFAU_LR" \
    --damping "$PFAU_DAMPING" \
    --cg-maxiter "$PFAU_CG" \
    --num-steps-decorr "$PFAU_DECORR" \
    $CAS_ARG \
    $INIT_ARG \
    --gs-source-dir cs_h2o_pfau_gs \
    --seed "$SEED"

echo ""
echo "=== done — $(date) ==="
