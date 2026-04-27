#!/usr/bin/env bash
# Launcher for the 3D HEG VMC validation run.
#
# Default configuration: unpolarized N=14 closed-shell at rs=2, Adam-VMC
# training with a small Jastrow, followed by a higher-statistics eval
# pass that reports correlation-energy recovery vs Perdew-Zunger.
#
# Tweak arguments below for larger systems (N=38, 54), different rs,
# polarization, or longer training.  Run `python
# scripts/run_heg_psiformer.py --help` for the full option list.
#
# Wall-clock scaling (single A100-class GPU): ~45 s of warm-up XLA
# compile, then ~0.5 s/iter for N=14 at the shown walker count, plus
# ~0.6 s/block during eval.

set -eu

RS=${RS:-2.0}
N=${N:-14}
POL=${POL:-unpolarized}
ITERS=${ITERS:-500}
LR=${LR:-3e-3}
OPT_WALKERS=${OPT_WALKERS:-128}
EVAL_WALKERS=${EVAL_WALKERS:-256}
EVAL_BLOCKS=${EVAL_BLOCKS:-40}
EVAL_EQUIL=${EVAL_EQUIL:-20}
STEPS_PER_BLOCK=${STEPS_PER_BLOCK:-30}
SEED=${SEED:-42}

python scripts/run_heg_psiformer.py \
    --rs "$RS" --N "$N" --polarization "$POL" \
    --iters "$ITERS" --lr "$LR" \
    --opt-walkers "$OPT_WALKERS" \
    --eval-walkers "$EVAL_WALKERS" \
    --eval-blocks "$EVAL_BLOCKS" \
    --eval-equil-blocks "$EVAL_EQUIL" \
    --steps-per-block "$STEPS_PER_BLOCK" \
    --seed "$SEED" \
    ${PSIFORMER:+--psiformer} \
    "$@"
