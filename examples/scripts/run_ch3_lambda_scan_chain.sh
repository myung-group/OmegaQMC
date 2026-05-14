#!/bin/bash
# Chain: CH3 sigma- (lambda=0.5) → sigma+ warm-start at lambda=0.7
#                                  → sigma+ warm-start at lambda=0.3.
# Sequential since they share one GH200.
set -eo pipefail

# Important: load full bashrc so conda activate works under nohup.
source ~/.bashrc
conda activate omegaqmc

cd ~/Workspace/OmegaQMC
mkdir -p logs

run_one () {
    YAML="$1"
    NAME="$(basename "$YAML" .yaml)"
    echo "[chain] $(date -Iseconds) starting $NAME"
    python -u scripts/run_qed_vmc.py "$YAML" 2>&1 \
        | tee -a "logs/ch3_lambda_chain.log"
    echo "[chain] $(date -Iseconds) done $NAME"
}

run_one inputs/qed_ch3/ch3_L050_sigma_minus.yaml
run_one inputs/qed_ch3/ch3_L070_warmstart.yaml
run_one inputs/qed_ch3/ch3_L030_warmstart.yaml

echo "[chain] $(date -Iseconds) ALL RUNS COMPLETE"
