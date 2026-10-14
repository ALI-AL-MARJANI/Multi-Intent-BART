"""Threshold-free OOD detection metrics (OOD is the positive class)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, tpr: float = 0.95) -> float:
    """False-positive rate at the first threshold reaching the target recall on OOD."""
    fprs, tprs, _ = roc_curve(labels, scores)
    return float(fprs[np.searchsorted(tprs, tpr, side="left")])


def detection_metrics(in_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    """AUROC, AUPR with OOD as positive (``aupr_out``) and ID as positive (``aupr_in``), FPR@95TPR."""
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([in_scores, ood_scores])
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr_out": float(average_precision_score(labels, scores)),
        "aupr_in": float(average_precision_score(1 - labels, -scores)),
        "fpr_at_95tpr": fpr_at_tpr(labels, scores),
    }


def bootstrap_detection_metrics(
    in_scores: np.ndarray,
    ood_scores: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, list[float]]:
    """Percentile bootstrap intervals, resampling ID and OOD utterances separately."""
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n_resamples):
        metrics = detection_metrics(rng.choice(in_scores, len(in_scores)), rng.choice(ood_scores, len(ood_scores)))
        for name, value in metrics.items():
            samples.setdefault(name, []).append(value)
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in samples.items()}
