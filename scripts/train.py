#!/usr/bin/env python3
"""Train GEMIS for one (config, seed) pair.

    python scripts/train.py --config configs/mixatis_bart-base.yaml --seed 13

Writes ``checkpoints/<experiment>/seed<seed>/{best,last}.pt`` and
``results/nlu/<experiment>/seed<seed>_train.json`` (config, environment,
per-epoch dev metrics).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from gemis.experiment import build_model, load_experiment, load_splits, make_loader, run_name
from gemis.training.trainer import GEMISTrainer
from gemis.utils.env import collect_env
from gemis.utils.io import write_json
from gemis.utils.training_utils import resolve_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resume", action="store_true", help="continue from last.pt")
    parser.add_argument("--max_train_samples", type=int, default=None, help="debugging only")
    parser.add_argument("--max_epochs", type=int, default=None, help="debugging only")
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    train_cfg, data_cfg = cfg["training"], cfg["data"]
    if args.max_epochs is not None:
        train_cfg["max_epochs"] = args.max_epochs
    debug = args.max_train_samples is not None or args.max_epochs is not None

    set_seed(args.seed)
    device = resolve_device(train_cfg["device"])
    name = run_name(cfg, args.seed)
    logger.info("run %s on %s", name, device)

    splits = load_splits(cfg)
    if args.max_train_samples is not None:
        splits["train"] = splits["train"][: args.max_train_samples]
    model, tokenizer = build_model(cfg, splits["train"])
    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(
        "%d train / %d dev utterances, %d intents, %d slot types, %.1fM parameters",
        len(splits["train"]),
        len(splits["dev"]),
        len(model.label_space.intents),
        len(model.label_space.slots),
        n_parameters / 1e6,
    )

    train_loader = make_loader(
        splits["train"],
        model,
        tokenizer,
        train_cfg["batch_size"],
        data_cfg["max_source_length"],
        train=True,
    )
    dev_loader = make_loader(
        splits["dev"], model, tokenizer, train_cfg["eval_batch_size"], data_cfg["max_source_length"]
    )

    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"]) / name
    trainer = GEMISTrainer(
        model,
        train_loader,
        dev_loader,
        checkpoint_dir,
        device,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        max_epochs=train_cfg["max_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        gradient_clip=train_cfg["gradient_clip"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        early_stopping_patience=train_cfg["early_stopping_patience"],
        mixed_precision=train_cfg["mixed_precision"],
        max_new_tokens=train_cfg["max_new_tokens"],
        run_config={"config": cfg, "seed": args.seed},
        show_progress=sys.stderr.isatty(),
    )
    if args.resume:
        trainer.resume(checkpoint_dir / "last.pt")

    started = time.time()
    history = trainer.train()
    suffix = "_debug" if debug else ""
    out_path = Path(cfg["output"]["results_dir"]) / f"{name}_train{suffix}.json"
    write_json(
        {
            "experiment": cfg["experiment_name"],
            "seed": args.seed,
            "debug_run": debug,
            "config": cfg,
            "env": collect_env(device),
            "n_parameters": n_parameters,
            "n_train": len(splits["train"]),
            "n_dev": len(splits["dev"]),
            "dropped_spans_train": train_loader.dataset.n_dropped_spans,
            "pointer_alpha": model.pointer.alpha.item(),
            "best_epoch": trainer.best_epoch,
            "best_dev_overall_accuracy": trainer.best_metric,
            "train_seconds": round(time.time() - started, 1),
            "history": history,
        },
        out_path,
    )
    logger.info("best dev overall accuracy %.2f at epoch %d -> %s", trainer.best_metric, trainer.best_epoch, out_path)


if __name__ == "__main__":
    main()
