#!/usr/bin/env python3
"""Evaluate a trained GEMIS checkpoint on a test set.

Usage:
    python scripts/evaluate.py \\
        --checkpoint checkpoints/mixatis/best.pt \\
        --config     configs/mixatis.yaml
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
from gemis.training.metrics import Frame, compute_metrics, parse_target_sequence  # noqa: F401
from gemis.utils.io import load_config
from gemis.utils.training_utils import collect_labels, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GEMIS checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config",     required=True, type=Path)
    parser.add_argument("--split",      default="test", choices=["dev", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    split_path = data_cfg[f"{args.split}_path"]

    # ── collect labels ────────────────────────────────────────────────────────
    all_data_paths = [
        Path(data_cfg["train_path"]),
        Path(data_cfg["dev_path"]),
        Path(data_cfg["test_path"]),
    ]
    intent_labels, slot_labels = collect_labels(all_data_paths)

    # ── tokenizer + model ─────────────────────────────────────────────────────
    backbone = cfg["model"]["backbone"]
    tokenizer = BartTokenizerFast.from_pretrained(backbone)
    model = GEMISModel(
        backbone_name=backbone,
        intent_labels=intent_labels,
        slot_labels=slot_labels,
        tokenizer=tokenizer,
        mlp_hidden=cfg["model"]["mlp_hidden"],
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # ── dataloader ────────────────────────────────────────────────────────────
    ds = GEMISDataset(
        split_path, tokenizer,
        max_source_length=data_cfg["max_source_length"],
        max_target_length=data_cfg["max_target_length"],
    )
    collator = GEMISDataCollator(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                        shuffle=False, collate_fn=collator)

    # ── inference ────────────────────────────────────────────────────────────
    golds: list[Frame] = []
    preds: list[Frame] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            for i, gen_ids in enumerate(generated):
                gold_ids = [t for t in batch["labels"][i].tolist() if t != -100]
                golds.append(parse_target_sequence(gold_ids, tokenizer))
                preds.append(parse_target_sequence(gen_ids, tokenizer))

    # ── report ────────────────────────────────────────────────────────────────
    metrics = compute_metrics(golds, preds)
    print(f"\n{'='*40}")
    print(f"  Dataset : {cfg['experiment_name']} ({args.split})")
    print(f"  Slot F1 : {metrics['slot_f1']:.2f}")
    print(f"  Intent Accuracy  : {metrics['intent_accuracy']:.2f}")
    print(f"  Overall Accuracy : {metrics['overall_accuracy']:.2f}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
