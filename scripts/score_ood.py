#!/usr/bin/env python3
"""Compute every OOD score for in-distribution and OOD utterances.

    python scripts/score_ood.py --config configs/mixsnips_bart-base.yaml --seed 13

Writes ``results/ood/<experiment>/seed<seed>/scores.csv`` with one row per
utterance (set, text, predicted intents, six scores) and ``meta.json``.

Sets: ``id_dev`` and ``id_test`` (in-distribution), ``far_clinc`` (CLINC150
out-of-scope test queries) and, for a held-out-intent config, ``near_holdout``
(test utterances carrying an intent removed from training).
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np

from gemis.experiment import load_experiment, load_splits, make_loader, run_name
from gemis.models.gemis import GEMISModel
from gemis.ood.data import load_ood_utterances
from gemis.ood.scores import DECODER_SCORES, FEATURE_SCORES, FeatureScorer, score_loader
from gemis.utils.env import collect_env
from gemis.utils.io import load_raw_dataset, write_json
from gemis.utils.training_utils import only_holdout, resolve_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("score_ood")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out_root", type=Path, default=Path("results/ood"))
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    ood_cfg = cfg["ood"]
    set_seed(args.seed)
    device = resolve_device(cfg["training"]["device"])
    name = run_name(cfg, args.seed)
    checkpoint_path = args.checkpoint or Path(cfg["output"]["checkpoint_dir"]) / name / "best.pt"
    model, tokenizer, _ = GEMISModel.from_checkpoint(checkpoint_path)
    model.to(device)

    splits = load_splits(cfg)
    holdout = cfg["data"].get("holdout_intents") or []
    sets = {
        "id_dev": splits["dev"],
        "id_test": splits["test"],
        "far_clinc": load_ood_utterances(ood_cfg["far_ood_path"]),
    }
    if holdout:
        sets["near_holdout"] = only_holdout(load_raw_dataset(cfg["data"]["test_path"]), holdout)

    def run(samples):
        loader = make_loader(
            samples,
            model,
            tokenizer,
            cfg["training"]["eval_batch_size"],
            cfg["data"]["max_source_length"],
        )
        return score_loader(model, loader, device, cfg["training"]["max_new_tokens"])

    started = time.time()
    rng = np.random.default_rng(args.seed)
    fit_samples = splits["train"]
    if len(fit_samples) > ood_cfg["feature_bank_size"]:
        keep = rng.choice(len(fit_samples), ood_cfg["feature_bank_size"], replace=False)
        fit_samples = [fit_samples[i] for i in sorted(keep)]
    logger.info("encoding %d training utterances for the feature bank", len(fit_samples))
    scorer = FeatureScorer(k=ood_cfg["knn_k"]).fit(run(fit_samples)["features"])

    out_dir = args.out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["set", "index", "text", "predicted_intents", *DECODER_SCORES, *FEATURE_SCORES]
    sizes = {}
    with open(out_dir / "scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for set_name, samples in sets.items():
            result = run(samples)
            scores = {**result["scores"], **scorer.score(result["features"])}
            sizes[set_name] = len(samples)
            for i, (text, frame) in enumerate(zip(result["texts"], result["frames"])):
                writer.writerow(
                    [set_name, i, text, "#".join(sorted(frame.intents))]
                    + [f"{scores[name][i]:.6f}" for name in (*DECODER_SCORES, *FEATURE_SCORES)]
                )
            logger.info("scored %-13s n=%d", set_name, len(samples))

    write_json(
        {
            "experiment": cfg["experiment_name"],
            "seed": args.seed,
            "checkpoint": str(checkpoint_path),
            "set_sizes": sizes,
            "holdout_intents": holdout,
            "feature_bank_size": len(fit_samples),
            "seconds": round(time.time() - started, 1),
            "config": cfg,
            "env": collect_env(device),
        },
        out_dir / "meta.json",
    )


if __name__ == "__main__":
    main()
