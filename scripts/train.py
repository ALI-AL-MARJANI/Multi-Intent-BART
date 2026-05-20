#!/usr/bin/env python3
"""GEMIS training entry point.

Usage:
    python scripts/train.py --config configs/mixatis.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import BartTokenizerFast

from gemis.data.collator import GEMISDataCollator
from gemis.data.dataset import GEMISDataset
from gemis.models.gemis import GEMISModel
from gemis.training.trainer import GEMISTrainer
from gemis.utils.io import load_config
from gemis.utils.training_utils import collect_labels, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GEMIS")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    logger.info("=== GEMIS Training — %s ===", cfg["experiment_name"])

    # ── collect labels from training data ─────────────────────────────────────
    data_cfg = cfg["data"]
    all_data_paths = [
        Path(data_cfg["train_path"]),
        Path(data_cfg["dev_path"]),
        Path(data_cfg["test_path"]),
    ]
    intent_labels, slot_labels = collect_labels(all_data_paths)
    logger.info("Found %d intent types, %d slot types", len(intent_labels), len(slot_labels))

    # ── tokenizer ─────────────────────────────────────────────────────────────
    backbone = cfg["model"]["backbone"]
    tokenizer = BartTokenizerFast.from_pretrained(backbone)
    # Vocabulary extension happens inside GEMISModel.__init__

    # ── model ─────────────────────────────────────────────────────────────────
    model = GEMISModel(
        backbone_name=backbone,
        intent_labels=intent_labels,
        slot_labels=slot_labels,
        tokenizer=tokenizer,
        mlp_hidden=cfg["model"]["mlp_hidden"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # ── datasets & loaders ────────────────────────────────────────────────────
    train_ds = GEMISDataset(
        data_cfg["train_path"], tokenizer,
        max_source_length=data_cfg["max_source_length"],
        max_target_length=data_cfg["max_target_length"],
    )
    dev_ds = GEMISDataset(
        data_cfg["dev_path"], tokenizer,
        max_source_length=data_cfg["max_source_length"],
        max_target_length=data_cfg["max_target_length"],
    )

    collator = GEMISDataCollator(tokenizer=tokenizer)
    train_cfg = cfg["training"]

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = GEMISTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        learning_rate=train_cfg["learning_rate"],
        num_epochs=train_cfg["max_epochs"],
        warmup_steps=train_cfg["warmup_steps"],
        output_dir=cfg["output"]["checkpoint_dir"],
        device=train_cfg["device"],
        gradient_clip=train_cfg["gradient_clip"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
    )

    trainer.train()
    logger.info("Training complete. Best overall accuracy: %.2f", trainer.best_overall_acc)


if __name__ == "__main__":
    main()
