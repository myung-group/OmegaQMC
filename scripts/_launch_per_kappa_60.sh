#!/usr/bin/env bash
# Launch 60-twist per-κ fine-tune across 5 slurm + 1 login GPU.
# Usage: scripts/_launch_per_kappa_60.sh <yaml> <chkpt> <project> [iters]
set -e
YAML="${1:-inputs/2dheg_qed/l5_weber_fig1b_v11_on.yaml}"
CHKPT="${2:-runs/l5_weber_fig1b_v11_on/l5_weber_fig1b_v11_on.chk.npz}"
PROJECT="${3:-v11_perk}"
ITERS="${4:-100}"

N_TOTAL=60
CHUNK=10
WALKERS="${WALKERS:-1024}"
EQUIL="${EQUIL:-30}"
DECORR="${DECORR:-15}"
SLURM_TIME="${SLURM_TIME:-02:00:00}"

cd /home/cwmyung/Workspace/OmegaQMC

echo "=== 60-twist per-κ fine-tune ==="
echo "  yaml : $YAML"
echo "  chkpt: $CHKPT"
echo "  iters per κ: $ITERS"
echo "  output: runs/${PROJECT}_c{0..5}/"
echo ""

for c in 0 1 2 3 4; do
    s=$((c * CHUNK)); e=$((s + CHUNK))
    JOB=$(sbatch --job-name=perk_c${c} \
        --partition=kisti-grace --gres=gpu:GH200:1 \
        --cpus-per-task=8 --mem=128G --time=${SLURM_TIME} \
        --output=runs/${PROJECT}_c${c}.slurm.out \
        --error=runs/${PROJECT}_c${c}.slurm.err \
        --wrap "export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform TF_CUDNN_DETERMINISTIC=1; cd /home/cwmyung/Workspace/OmegaQMC; source ~/.bashrc; conda activate omegaqmc; python scripts/train_per_kappa.py ${YAML} --chkpt ${CHKPT} --n-twists ${N_TOTAL} --twist-start ${s} --twist-end ${e} --iters ${ITERS} --equil-steps ${EQUIL} --walkers ${WALKERS} --mcmc-decorr-steps ${DECORR} --chunk-tag c${c} --out runs/${PROJECT}" \
        | awk '{print $NF}')
    echo "  c${c} (twists [${s}:${e})) → slurm $JOB"
done

s=50; e=60
echo "  c5 (twists [${s}:${e})) → login (nohup background)"
mkdir -p runs/${PROJECT}_c5
nohup bash -c "
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_CUDNN_DETERMINISTIC=1
source ~/.bashrc
conda activate omegaqmc
cd /home/cwmyung/Workspace/OmegaQMC
python scripts/train_per_kappa.py ${YAML} \
    --chkpt ${CHKPT} --n-twists ${N_TOTAL} \
    --twist-start ${s} --twist-end ${e} \
    --iters ${ITERS} --equil-steps ${EQUIL} \
    --walkers ${WALKERS} --mcmc-decorr-steps ${DECORR} \
    --chunk-tag c5 --out runs/${PROJECT}
" > runs/${PROJECT}_c5.login.out 2>&1 &
LOGIN_PID=$!
echo "    login pid: $LOGIN_PID"

echo ""
echo "Aggregate later: python scripts/_aggregate_per_kappa.py runs/${PROJECT}"
