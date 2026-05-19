#!/usr/bin/env python3
"""Run OOD calibration: score in-distribution samples and save a ConformalCalibrator.

Usage:
    USE_TF=0 python scripts/calibrate_ood.py \\
        --checkpoint checkpoints/mixsnips/best.pt \\
        --config     configs/mixsnips.yaml \\
        --output     checkpoints/mixsnips/ood_calibrator.pkl \\
        --alpha      0.05
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizerFast

from gemis.data.collator import GEMISDataCollator
from gemis.data.dataset import GEMISDataset
from gemis.models.gemis import GEMISModel
from gemis.ood.calibrator import ConformalCalibrator
from gemis.ood.scorer import OODScorer
from gemis.utils.io import load_config
from gemis.utils.training_utils import collect_labels, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate OOD detector for GEMIS")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config",     required=True, type=Path)
    parser.add_argument("--output",     required=True, type=Path, help="Where to save calibrator .pkl")
    parser.add_argument("--split",      default="dev", choices=["dev", "test"])
    parser.add_argument("--alpha",      type=float, default=0.05, help="Target FPR (default 5%%)")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    # ── model ─────────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    all_paths = [Path(data_cfg[k]) for k in ("train_path", "dev_path", "test_path")]
    intent_labels, slot_labels = collect_labels(all_paths)

    backbone = cfg["model"]["backbone"]
    tokenizer = BartTokenizerFast.from_pretrained(backbone)
    model = GEMISModel(
        backbone_name=backbone,
        intent_labels=intent_labels,
        slot_labels=slot_labels,
        tokenizer=tokenizer,
        mlp_hidden=cfg["model"]["mlp_hidden"],
    )

    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # ── dataloader ────────────────────────────────────────────────────────────
    split_path = data_cfg[f"{args.split}_path"]
    ds = GEMISDataset(
        split_path, tokenizer,
        max_source_length=data_cfg["max_source_length"],
        max_target_length=data_cfg["max_target_length"],
    )
    collator = GEMISDataCollator(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # ── collect non-conformity scores ─────────────────────────────────────────
    all_scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            words_batch = batch.pop("words", None)
            batch = {k: v.to(device) for k, v in batch.items()}

            result = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                input_words_batch=words_batch,
                return_scores=True,
            )
            scores = OODScorer.score_batch(result["intent_entropies"])
            all_scores.extend(scores)

    logger.info("Collected %d calibration scores", len(all_scores))

    # ── fit and save calibrator ───────────────────────────────────────────────
    calibrator = ConformalCalibrator(alpha=args.alpha)
    calibrator.fit(all_scores)
    calibrator.save(args.output)

    summary = calibrator.summary()
    logger.info("Calibration summary:")
    for k, v in summary.items():
        logger.info("  %-30s: %.4f", k, v)

    logger.info(
        "Threshold at α=%.2f: %.4f  (scores above this → OOD)",
        args.alpha, calibrator.threshold(),
    )
    logger.info("Calibrator saved → %s", args.output)


if __name__ == "__main__":
    main()
