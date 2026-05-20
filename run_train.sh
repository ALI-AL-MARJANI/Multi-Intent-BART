#!/bin/sh
# Usage: sh run_train.sh [mixsnips|mixatis]
DATASET=${1:-mixsnips}
export PYTHONPATH=/root/myhomedir/bucket/Multi-Intent-BART/src
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
CONFIG=configs/${DATASET}_t4.yaml
LOG=logs/train_${DATASET}.log
echo "Starting training: $CONFIG -> $LOG"
nohup /opt/conda/bin/python scripts/train.py --config "$CONFIG" >"$LOG" 2>&1 &
echo "PID: $!"
echo "Logs: tail -f $LOG"
