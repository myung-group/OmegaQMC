#!/bin/bash
#SBATCH --job-name=h2_pfau_spec
#SBATCH --partition=kisti-grace
#SBATCH --gres=gpu:GH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# H2 Pfau-NES + CS-recovery spectroscopy demo on one GH200.
#
# Steps:
#  1. Train ground state via SR (~10 min) using run_h2_nesvmc.py with
#     --skip-es-train (only does GS training + walker sampling)
#  2. Train Pfau-NES K=2 + run spectroscopy via
#     run_h2_pfau_nes_spectroscopy.py with --init-from-ground
#
# Output:
#  cs_h2_nesvmc_results/   GS checkpoint + walker bank
#  cs_h2_pfau_spec_results/  Pfau-NES checkpoints + walker banks +
#                            transition properties + NEVPT2 JSON
#
# Usage:
#   sbatch --output=runs/h2_pfau_spec.slurm.out \
#          --error=runs/h2_pfau_spec.slurm.err \
#          scripts/slurm_cs/run_h2_pfau_spectroscopy.sh

# kisti-grace SLURM has a broken default MPI plugin ("mpi/mpi/pmix")
# that causes any implicit srun to fail before user code runs. Skip
# the plugin entirely; we don't use MPI for the single-GPU NN-VMC code.
export SLURM_MPI_TYPE=none

# Under SLURM, $0 is /var/spool/slurm/slurmd/jobNNNNN/slurm_script
# (a copy of the script). Use SLURM_SUBMIT_DIR (the directory sbatch
# was invoked from) instead, falling back to dirname $0/../.. for
# direct bash invocations outside SLURM.
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")/../.."
fi

# Source bashrc and activate conda *before* enabling pipefail/errexit:
# on kisti-grace mango the .bashrc / conda init returns a nonzero
# status in non-interactive shells (harmless), which would otherwise
# kill the job silently under set -e.
source ~/.bashrc || true
conda activate omegaqmc

set -eo pipefail

# The omegaqmc conda env has OmegaQMC pip-installed (editable) from
# the main worktree at ~/Workspace/OmegaQMC, which does NOT contain
# our compressed-sensing branch's new modules. Prepend our worktree
# to PYTHONPATH so Python picks up vmcopt_nn_pfau, cs.transition,
# cs.mrpt updates, etc., before falling back to the editable install.
export PYTHONPATH="$PWD:$PYTHONPATH"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Force unbuffered stdout/stderr so SLURM .out captures iter progress
# in real time instead of holding it in the block buffer for the whole
# run (Python defaults to block-buffered when stdout is a regular file).
export PYTHONUNBUFFERED=1

R=${R:-2.5}
BASIS=${BASIS:-cc-pvdz}
ANSATZ=${ANSATZ:-examples/inputs/psiformer_small.yaml}
GS_ITERS=${GS_ITERS:-600}
PFAU_ITERS=${PFAU_ITERS:-800}
PFAU_WALKERS=${PFAU_WALKERS:-256}
PFAU_SAMPLE_BLOCKS=${PFAU_SAMPLE_BLOCKS:-2}
PFAU_BATCH=${PFAU_BATCH:-256}
PFAU_EPOCHS=${PFAU_EPOCHS:-3}
SEED=${SEED:-77}

echo "=== H2 Pfau-NES + CS spectroscopy on $(hostname) — $(date) ==="
echo "  R=$R  BASIS=$BASIS  ansatz=$ANSATZ"
echo "  GS_ITERS=$GS_ITERS  PFAU_ITERS=$PFAU_ITERS"

# Step 1: GS bootstrap (uses the minimal GS-only driver to avoid
# run_h2_nesvmc.py's mandatory excited-state sampling step that
# would fail without a pre-existing ES checkpoint).
echo ""
echo ">>> Step 1: Train ground state ($GS_ITERS SR iters)"
python examples/run_h2_groundstate_only.py \
    --R "$R" --basis "$BASIS" --ansatz "$ANSATZ" \
    --out-dir cs_h2_nesvmc_results \
    --gs-iters "$GS_ITERS" \
    --seed 11

# Step 2: Pfau-NES K=2 + spectroscopy
echo ""
echo ">>> Step 2: Pfau-NES K=2 training + CS spectroscopy ($PFAU_ITERS iters)"
python examples/run_h2_pfau_nes_spectroscopy.py \
    --R "$R" --basis "$BASIS" --ansatz "$ANSATZ" \
    --pfau-iters "$PFAU_ITERS" \
    --num-walkers "$PFAU_WALKERS" \
    --num-sample-blocks "$PFAU_SAMPLE_BLOCKS" \
    --batch-size "$PFAU_BATCH" \
    --num-epochs "$PFAU_EPOCHS" \
    --init-from-ground \
    --gs-source-dir cs_h2_nesvmc_results \
    --seed "$SEED"

echo ""
echo "=== done — $(date) ==="
