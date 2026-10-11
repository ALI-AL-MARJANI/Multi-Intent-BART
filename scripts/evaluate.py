#!/usr/bin/env python3
"""Evaluate a trained checkpoint with greedy decoding.

    python scripts/evaluate.py --config configs/mixatis_bart-base.yaml --seed 13

Writes ``results/nlu/<experiment>/seed<seed>_<split>.json`` (metrics, bootstrap
intervals, error analysis, environment) and a ``_predictions.jsonl`` file with
one gold/predicted frame per utterance.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from gemis.experiment import load_experiment, load_splits, make_loader, run_name
from gemis.models.gemis import GEMISModel
from gemis.training.evaluation import predict_frames
from gemis.training.metrics import (
    bootstrap_ci,
    compute_metrics,
    error_analysis,
    intent_correct,
    overall_correct,
)
from gemis.utils.env import collect_env
from gemis.utils.io import write_json
from gemis.utils.training_utils import resolve_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: best.pt of the run")
    parser.add_argument("--tag", default="", help="suffix for the output files (e.g. _debug)")
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    set_seed(args.seed)
    device = resolve_device(cfg["training"]["device"])
    name = run_name(cfg, args.seed)
    checkpoint_path = args.checkpoint or Path(cfg["output"]["checkpoint_dir"]) / name / "best.pt"

    model, tokenizer, checkpoint = GEMISModel.from_checkpoint(checkpoint_path)
    model.to(device)
    samples = load_splits(cfg)[args.split]
    loader = make_loader(
        samples, model, tokenizer, cfg["training"]["eval_batch_size"], cfg["data"]["max_source_length"]
    )

    started = time.time()
    golds, preds, words = predict_frames(model, loader, device, cfg["training"]["max_new_tokens"])
    seconds = time.time() - started
    metrics = compute_metrics(golds, preds)

    out_dir = Path(cfg["output"]["results_dir"]) / cfg["experiment_name"]
    stem = f"seed{args.seed}_{args.split}{args.tag}"
    predictions_path = out_dir / f"{stem}_predictions.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as f:
        for gold, pred, utterance in zip(golds, preds, words):
            row = {
                "text": " ".join(utterance),
                "gold": gold.to_json(),
                "pred": pred.to_json(),
                "intent_correct": intent_correct(gold, pred),
                "overall_correct": overall_correct(gold, pred),
            }
            f.write(json.dumps(row) + "\n")

    write_json(
        {
            "experiment": cfg["experiment_name"],
            "seed": args.seed,
            "split": args.split,
            "n_utterances": len(golds),
            "metrics": metrics,
            "bootstrap_ci95": bootstrap_ci(golds, preds),
            "error_analysis": error_analysis(golds, preds),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint["best_epoch"],
            "decode_seconds": round(seconds, 1),
            "utterances_per_second": round(len(golds) / seconds, 1),
            "predictions_file": predictions_path.name,
            "config": cfg,
            "env": collect_env(device),
        },
        out_dir / f"{stem}.json",
    )
    logger.info(
        "%s %s: slot F1 %.2f | intent acc %.2f | overall acc %.2f (n=%d)",
        name,
        args.split,
        metrics["slot_f1"],
        metrics["intent_accuracy"],
        metrics["overall_accuracy"],
        len(golds),
    )


if __name__ == "__main__":
    main()
