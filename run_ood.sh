#!/bin/sh
# Usage: sh run_ood.sh [mixsnips|mixatis]
DATASET=${1:-mixsnips}
export PYTHONPATH=/root/myhomedir/bucket/Multi-Intent-BART/src
export USE_TF=0
export TOKENIZERS_PARALLELISM=false

CHECKPOINT=checkpoints/${DATASET}/best.pt
CONFIG=configs/${DATASET}_t4.yaml
CALIBRATOR=checkpoints/${DATASET}/ood_calibrator.pkl

echo "=== Step 1: OOD Calibration ==="
/opt/conda/bin/python scripts/calibrate_ood.py --checkpoint "$CHECKPOINT" --config "$CONFIG" --output "$CALIBRATOR"

echo "=== Step 2: OOD Evaluation ==="
/opt/conda/bin/python scripts/evaluate_ood.py --checkpoint "$CHECKPOINT" --config "$CONFIG" --calibrator "$CALIBRATOR"
