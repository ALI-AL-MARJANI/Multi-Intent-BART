#!/bin/sh
# Train, evaluate and OOD-score one config for the given seeds.
#   sh scripts/run_experiment.sh configs/mixatis_bart-base.yaml 13 42 87
# Re-running the same command continues: finished trainings are skipped and
# interrupted ones resume from last.pt. Set PYTHON to use another interpreter.
set -e
CONFIG=$1; shift
PYTHON=${PYTHON:-python}
NAME=$(basename "$CONFIG" .yaml)
export TOKENIZERS_PARALLELISM=false
for SEED in "$@"; do
    # a crashed training (e.g. out of memory) is resumed, at most 5 times
    TRIES=0
    until $PYTHON scripts/train.py --config "$CONFIG" --seed "$SEED" --resume; do
        TRIES=$((TRIES + 1))
        [ "$TRIES" -lt 5 ] || exit 1
        echo "training failed; resuming (attempt $TRIES)"
    done
    $PYTHON scripts/evaluate.py --config "$CONFIG" --seed "$SEED" --split test
    $PYTHON scripts/score_ood.py --config "$CONFIG" --seed "$SEED"
    $PYTHON scripts/evaluate_ood.py --scores_dir "results/ood/$NAME/seed$SEED"
done
