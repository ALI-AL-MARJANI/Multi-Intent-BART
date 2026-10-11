"""Split-conformal outlier detection.

Given calibration scores ``s_1..s_n`` of in-distribution utterances and a
test score ``s``, the conformal p-value is

    p(s) = (1 + #{i : s_i >= s}) / (n + 1).

Flagging ``s`` as OOD when ``p(s) <= alpha`` is the same as ``s > q`` with
``q`` the ``ceil((n + 1)(1 - alpha))``-th smallest calibration score. If the
test utterance is in-distribution and exchangeable with the calibration set,
``P(flagged) <= alpha`` (Vovk et al. 2005; Bates et al. 2023).

The guarantee is *marginal*: it averages over the draw of the calibration
set. For one fixed calibration set and continuous scores, the false-alarm
rate is ``Beta(n + 1 - k, k)`` distributed with ``k = ceil((n + 1)(1 - alpha))``;
``expected_fpr`` returns its mean and standard deviation. It says nothing
about how many OOD utterances are detected.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from gemis.utils.io import read_json, write_json


def conformal_threshold(calibration_scores: np.ndarray, alpha: float) -> float:
    """Scores strictly above the returned value are flagged as OOD."""
    scores = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    n = len(scores)
    if n == 0:
        raise ValueError("empty calibration set")
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(scores[k - 1])


def conformal_p_values(calibration_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    cal = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    n_geq = len(cal) - np.searchsorted(cal, np.asarray(test_scores, dtype=np.float64), side="left")
    return (1 + n_geq) / (len(cal) + 1)


def expected_fpr(n_calibration: int, alpha: float) -> dict[str, float]:
    """Mean and std of the false-alarm rate over calibration draws (continuous scores)."""
    k = math.ceil((n_calibration + 1) * (1 - alpha))
    if k > n_calibration:
        return {"mean": 0.0, "std": 0.0}
    a, b = n_calibration + 1 - k, k
    return {
        "mean": a / (a + b),
        "std": math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))),
    }


def validity_experiment(
    in_distribution_scores: np.ndarray,
    alphas: tuple[float, ...] = (0.01, 0.05, 0.1),
    n_splits: int = 1000,
    calibration_fraction: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Empirical false-alarm rate over random calibration / test splits.

    A valid detector has a mean false-alarm rate at or just below ``alpha``.
    """
    scores = np.asarray(in_distribution_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_cal = int(round(calibration_fraction * len(scores)))
    rates: dict[float, list[float]] = {alpha: [] for alpha in alphas}
    for _ in range(n_splits):
        permutation = rng.permutation(len(scores))
        cal, test = scores[permutation[:n_cal]], scores[permutation[n_cal:]]
        for alpha in alphas:
            rates[alpha].append(float((test > conformal_threshold(cal, alpha)).mean()))
    return {
        "n_splits": n_splits,
        "n_calibration": n_cal,
        "n_test": len(scores) - n_cal,
        "per_alpha": {
            str(alpha): {
                "mean_fpr": float(np.mean(values)),
                "std_fpr": float(np.std(values)),
                "p05_fpr": float(np.percentile(values, 5)),
                "p95_fpr": float(np.percentile(values, 95)),
                "theory": expected_fpr(n_cal, alpha),
                "fpr_per_split": values,
            }
            for alpha, values in rates.items()
        },
    }


class ConformalCalibrator:
    """Stores calibration scores; thresholds and p-values are derived on demand."""

    def __init__(self, scores: np.ndarray, score_name: str = "") -> None:
        self.scores = np.sort(np.asarray(scores, dtype=np.float64))
        self.score_name = score_name

    def threshold(self, alpha: float) -> float:
        return conformal_threshold(self.scores, alpha)

    def is_ood(self, scores: np.ndarray, alpha: float) -> np.ndarray:
        return np.asarray(scores) > self.threshold(alpha)

    def p_values(self, scores: np.ndarray) -> np.ndarray:
        return conformal_p_values(self.scores, scores)

    def save(self, path: str | Path) -> None:
        write_json({"score_name": self.score_name, "scores": self.scores.tolist()}, path)

    @classmethod
    def load(cls, path: str | Path) -> ConformalCalibrator:
        data = read_json(path)
        return cls(np.array(data["scores"]), data["score_name"])
