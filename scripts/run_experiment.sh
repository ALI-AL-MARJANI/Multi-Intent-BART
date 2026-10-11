#!/bin/sh
# Train, evaluate and OOD-score one config for the given seeds.
#   sh scripts/run_experiment.sh configs/mixatis_bart-base.yaml 13 42 87
# Set PYTHON to use another interpreter; add RESUME=1 to continue interrupted runs.
set -e
CONFIG=$1; shift
PYTHON=${PYTHON:-python}
export TOKENIZERS_PARALLELISM=false
for SEED in "$@"; do
    NAME=$(basename "$CONFIG" .yaml)
    $PYTHON scripts/train.py --config "$CONFIG" --seed "$SEED" ${RESUME:+--resume}
    $PYTHON scripts/evaluate.py --config "$CONFIG" --seed "$SEED" --split test
    $PYTHON scripts/score_ood.py --config "$CONFIG" --seed "$SEED"
    $PYTHON scripts/evaluate_ood.py --scores_dir "results/ood/$NAME/seed$SEED"
done
