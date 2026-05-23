#!/usr/bin/env bash
# Launch 60-twist TABC across 5 Slurm GPU nodes + 1 login GPU.
# Each chunk handles 10 contiguous Halton indices.
# Usage: scripts/_launch_tabc_60_parallel.sh <yaml> <chkpt> <project-out-dir>
set -e
YAML="${1:-inputs/2dheg_qed/l5_weber_fig1b_v11_on.yaml}"
CHKPT="${2:-runs/l5_weber_fig1b_v11_on/l5_weber_fig1b_v11_on.chk.npz}"
PROJECT="${3:-tabc_v11_g050}"

N_TOTAL=60
CHUNK=10
WALKERS=1024
BLOCKS=20
EQUIL=5
STEPS=10

cd /home/cwmyung/Workspace/OmegaQMC

echo "=== Launching 60-twist TABC: 5 slurm chunks + 1 login chunk ==="
echo "  yaml : $YAML"
echo "  chkpt: $CHKPT"
echo "  output: runs/${PROJECT}_c{0..5}/"
echo ""

# Slurm chunks: c0..c4 → twists [0:10), [10:20), [20:30), [30:40), [40:50)
for c in 0 1 2 3 4; do
    s=$((c * CHUNK))
    e=$((s + CHUNK))
    JOB=$(sbatch --job-name=tabc_c${c} \
        --partition=kisti-grace --gres=gpu:GH200:1 \
        --cpus-per-task=8 --mem=128G --time=01:00:00 \
        --output=runs/${PROJECT}_c${c}.slurm.out \
        --error=runs/${PROJECT}_c${c}.slurm.err \
        --wrap "export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform TF_CUDNN_DETERMINISTIC=1; cd /home/cwmyung/Workspace/OmegaQMC; source ~/.bashrc; conda activate omegaqmc; python scripts/run_qed_l5_tabc.py ${YAML} --chkpt ${CHKPT} --n-twists ${N_TOTAL} --twist-start ${s} --twist-end ${e} --chunk-tag c${c} --walkers ${WALKERS} --blocks ${BLOCKS} --equil-blocks ${EQUIL} --steps-per-block ${STEPS} --out-dir runs/${PROJECT}_c${c}" \
        | awk '{print $NF}')
    echo "  c${c} (twists [${s}:${e})) → slurm $JOB"
done

# Login-node chunk: c5 → twists [50:60)
s=50; e=60
echo ""
echo "  c5 (twists [${s}:${e})) → login node (nohup background)"
mkdir -p runs/${PROJECT}_c5
nohup bash -c "
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_CUDNN_DETERMINISTIC=1
source ~/.bashrc
conda activate omegaqmc
cd /home/cwmyung/Workspace/OmegaQMC
python scripts/run_qed_l5_tabc.py ${YAML} \
    --chkpt ${CHKPT} \
    --n-twists ${N_TOTAL} \
    --twist-start ${s} --twist-end ${e} \
    --chunk-tag c5 \
    --walkers ${WALKERS} --blocks ${BLOCKS} \
    --equil-blocks ${EQUIL} --steps-per-block ${STEPS} \
    --out-dir runs/${PROJECT}_c5
" > runs/${PROJECT}_c5.login.out 2>&1 &
LOGIN_PID=$!
echo "    login pid: $LOGIN_PID"

echo ""
echo "Aggregate later with: python scripts/_aggregate_tabc_chunks.py runs/${PROJECT}"
