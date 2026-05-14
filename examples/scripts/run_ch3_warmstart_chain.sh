#!/bin/bash
# Chain: wait for lambda=0.10 seed pickle, then run lambda=0.20 warm-start,
# then lambda=0.30 warm-start. All output streamed to a single log.
set -eo pipefail

SEED_PKL="logs/qed_ch3_chiral_L010_warmstart_seed/qed_ch3_chiral_L010_warmstart_seed.params.pkl"
CHAIN_LOG="logs/ch3_warmstart_chain.log"

cd "$(dirname "$0")/.."
source ~/.bashrc
conda activate omegaqmc

echo "[chain] $(date -Iseconds) waiting for seed pickle: $SEED_PKL"
until [ -f "$SEED_PKL" ]; do
  sleep 30
done
echo "[chain] $(date -Iseconds) seed pickle found, starting lambda=0.20"

python -u scripts/run_qed_vmc.py inputs/qed_ch3/ch3_chiral_L020_warmstart.yaml \
  2>&1 | tee -a "$CHAIN_LOG"

echo "[chain] $(date -Iseconds) lambda=0.20 done, starting lambda=0.30"

python -u scripts/run_qed_vmc.py inputs/qed_ch3/ch3_chiral_L030_warmstart.yaml \
  2>&1 | tee -a "$CHAIN_LOG"

echo "[chain] $(date -Iseconds) all warm-start runs complete"
