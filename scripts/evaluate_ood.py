#!/usr/bin/env python3
"""Evaluate the OOD detector: AUROC, FPR@95TPR, AUPR, calibration plot.

This script:
1. Scores in-distribution (test split) samples   → label 0
2. Scores OOD (synthetic / external) utterances  → label 1
3. Computes standard OOD detection metrics.
4. Saves a calibration curve plot.

Usage:
    USE_TF=0 python scripts/evaluate_ood.py \\
        --checkpoint  checkpoints/mixsnips/best.pt \\
        --config      configs/mixsnips.yaml \\
        --calibrator  checkpoints/mixsnips/ood_calibrator.pkl \\
        --ood_file    data/ood/snips_ood.txt \\
        --alpha       0.05 \\
        --plot        results/ood_calibration.png
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
from gemis.ood.synthetic import SyntheticOODGenerator
from gemis.utils.io import load_config
from gemis.utils.training_utils import collect_labels, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── metric helpers ────────────────────────────────────────────────────────────

def auroc(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC without sklearn (pure Python, O(n log n))."""
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    return auc / (n_pos * n_neg)


def fpr_at_tpr(labels: list[int], scores: list[float], tpr_target: float = 0.95) -> float:
    """FPR when TPR >= tpr_target (standard OOD metric)."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = fp = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        if n_pos > 0 and tp / n_pos >= tpr_target:
            return fp / n_neg if n_neg > 0 else float("nan")
    return 1.0


def aupr(labels: list[int], scores: list[float]) -> float:
    """Area under precision-recall curve (trapezoidal rule)."""
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")
    tp = fp = 0
    precisions = []
    recalls = []
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    # trapezoidal integration
    area = 0.0
    for i in range(1, len(recalls)):
        area += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2
    return area


# ── OOD scoring for raw text ──────────────────────────────────────────────────

def score_raw_utterances(
    utterances: list[str],
    model: GEMISModel,
    tokenizer,
    device: torch.device,
    batch_size: int = 16,
) -> list[float]:
    """Tokenize and score raw text utterances (no labels needed)."""
    scores = []
    for i in range(0, len(utterances), batch_size):
        batch_utts = utterances[i: i + batch_size]
        enc = tokenizer(
            batch_utts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            result = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_scores=True,
            )
        scores.extend(OODScorer.score_batch(result["intent_entropies"]))

    return scores


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GEMIS OOD detection")
    parser.add_argument("--checkpoint",  required=True, type=Path)
    parser.add_argument("--config",      required=True, type=Path)
    parser.add_argument("--calibrator",  required=True, type=Path, help="Path to .pkl calibrator")
    parser.add_argument("--ood_file",    type=Path, default=None,
                        help="Text file with one OOD utterance per line (auto-generated if absent)")
    parser.add_argument("--n_ood",       type=int, default=200, help="OOD utterances to use")
    parser.add_argument("--alpha",       type=float, default=0.05)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--plot",        type=Path, default=None, help="Save calibration plot to path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    device = torch.device("cpu")

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
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # ── calibrator ────────────────────────────────────────────────────────────
    calibrator = ConformalCalibrator.load(args.calibrator)
    threshold = calibrator.threshold(args.alpha)
    logger.info("OOD threshold at α=%.2f: %.4f", args.alpha, threshold)

    # ── in-distribution scores (test split) ───────────────────────────────────
    test_ds = GEMISDataset(
        data_cfg["test_path"], tokenizer,
        max_source_length=data_cfg["max_source_length"],
        max_target_length=data_cfg["max_target_length"],
    )
    collator = GEMISDataCollator(tokenizer=tokenizer)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    in_scores: list[float] = []
    with torch.no_grad():
        for batch in test_loader:
            words_batch = batch.pop("words", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            result = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                input_words_batch=words_batch,
                return_scores=True,
            )
            in_scores.extend(OODScorer.score_batch(result["intent_entropies"]))

    logger.info("In-distribution scores collected: %d", len(in_scores))

    # ── OOD utterances ────────────────────────────────────────────────────────
    if args.ood_file and args.ood_file.exists():
        ood_utterances = SyntheticOODGenerator.load(args.ood_file)
        logger.info("Loaded %d OOD utterances from %s", len(ood_utterances), args.ood_file)
    else:
        logger.info("Generating synthetic OOD utterances (n=%d) …", args.n_ood)
        gen = SyntheticOODGenerator()
        ood_utterances = gen.generate(n=args.n_ood)
        if args.ood_file:
            SyntheticOODGenerator.save(ood_utterances, args.ood_file)

    ood_utterances = ood_utterances[: args.n_ood]
    ood_scores = score_raw_utterances(ood_utterances, model, tokenizer, device, args.batch_size)
    logger.info("OOD scores collected: %d", len(ood_scores))

    # ── metrics ───────────────────────────────────────────────────────────────
    # label 0 = in-domain, label 1 = OOD
    all_labels = [0] * len(in_scores) + [1] * len(ood_scores)
    all_scores_combined = in_scores + ood_scores

    auc  = auroc(all_labels, all_scores_combined)
    fpr  = fpr_at_tpr(all_labels, all_scores_combined, tpr_target=0.95)
    aupr_score = aupr(all_labels, all_scores_combined)

    # accuracy at chosen alpha
    preds = [1 if s > threshold else 0 for s in all_scores_combined]
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, all_labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, all_labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, all_labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, all_labels))
    tpr_actual = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr_actual = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"  OOD Detection Results  (α = {args.alpha})")
    print(f"{'='*50}")
    print(f"  In-distribution samples : {len(in_scores)}")
    print(f"  OOD samples             : {len(ood_scores)}")
    print(f"  {'─'*44}")
    print(f"  AUROC                   : {auc:.4f}")
    print(f"  FPR@95TPR               : {fpr:.4f}  (lower is better)")
    print(f"  AUPR                    : {aupr_score:.4f}")
    print(f"  {'─'*44}")
    print(f"  Threshold               : {threshold:.4f}")
    print(f"  TPR (OOD recall)        : {tpr_actual:.4f}")
    print(f"  FPR (false alarm rate)  : {fpr_actual:.4f}")
    print(f"{'='*50}\n")

    # ── calibration plot ──────────────────────────────────────────────────────
    if args.plot:
        _save_calibration_plot(
            in_scores, ood_scores, threshold, args.alpha, args.plot
        )
        logger.info("Calibration plot saved → %s", args.plot)


def _save_calibration_plot(
    in_scores: list[float],
    ood_scores: list[float],
    threshold: float,
    alpha: float,
    path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # histogram
        ax = axes[0]
        ax.hist(in_scores,  bins=40, alpha=0.6, label="In-domain",  color="steelblue")
        ax.hist(ood_scores, bins=40, alpha=0.6, label="OOD",        color="tomato")
        ax.axvline(threshold, color="black", linestyle="--",
                   label=f"Threshold (α={alpha}): {threshold:.3f}")
        ax.set_xlabel("Non-conformity score (entropy)")
        ax.set_ylabel("Count")
        ax.set_title("Score distributions: In-domain vs OOD")
        ax.legend()

        # empirical CDF
        ax2 = axes[1]
        for scores, label, color in [
            (in_scores, "In-domain", "steelblue"),
            (ood_scores, "OOD", "tomato"),
        ]:
            s = sorted(scores)
            n = len(s)
            ax2.plot(s, [i / n for i in range(n)], label=label, color=color)
        ax2.axvline(threshold, color="black", linestyle="--", label=f"Threshold (α={alpha})")
        ax2.set_xlabel("Non-conformity score")
        ax2.set_ylabel("CDF")
        ax2.set_title("Empirical CDF")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
