"""ConformalCalibrator: distribution-free OOD detection via conformal prediction.

How conformal prediction works here (simple version)
-----------------------------------------------------
1. Collect non-conformity scores on a *calibration set* of IN-DISTRIBUTION
   samples (e.g. the dev split).  These scores are the "reference".

2. At test time, given a new sample's score s:
       p-value = |{i : s_i >= s}| / n_cal          (fraction of cal scores >= s)
   If p-value < α, the sample is declared OOD.

   Equivalently, we reject if  s > q_{1-α}  where q_{1-α} is the (1-α) empirical
   quantile of the calibration scores.

3. Guarantee (Vovk et al., 2005):
       P(in-domain sample flagged as OOD) ≤ α
   This holds *without any distributional assumptions* — only exchangeability.

Finite-sample correction
------------------------
The exact conformal quantile uses  ceil((n+1)(1-α)) / n  to account for the
test point, keeping the coverage guarantee tight even for small calibration sets.

Usage
-----
    calibrator = ConformalCalibrator()
    calibrator.fit(in_domain_scores)        # list of floats from OODScorer
    calibrator.save("calibrator.pkl")

    # later
    calibrator = ConformalCalibrator.load("calibrator.pkl")
    is_ood = calibrator.predict(new_score, alpha=0.05)
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path


class ConformalCalibrator:
    """Calibrated OOD detector using split conformal prediction.

    Args:
        alpha: default significance level (false-positive rate).  Can be
               overridden at prediction time.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._cal_scores: list[float] = []
        self._sorted: list[float] = []

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit(self, scores: list[float]) -> "ConformalCalibrator":
        """Store calibration scores from in-distribution samples.

        Args:
            scores: non-conformity scores for the calibration set.
                    Larger score = more OOD-like.
        """
        if not scores:
            raise ValueError("Calibration set is empty.")
        self._cal_scores = list(scores)
        self._sorted = sorted(scores)
        return self

    # ── threshold ─────────────────────────────────────────────────────────────

    def threshold(self, alpha: float | None = None) -> float:
        """Return the conformal quantile used as the OOD decision boundary.

        Any test score ABOVE this threshold is declared OOD.

        Uses the finite-sample corrected quantile:
            q = sorted_scores[ ceil((n+1)(1-α)) - 1 ]
        clipped to the last element if the index exceeds the array length
        (which means α is too small for the calibration set size — we fall
        back to "always in-domain" by returning +inf in that case).
        """
        self._check_fitted()
        alpha = alpha if alpha is not None else self.alpha
        n = len(self._sorted)
        # finite-sample conformal quantile index
        idx = math.ceil((n + 1) * (1 - alpha)) - 1
        if idx >= n:
            return float("inf")  # calibration set too small for this alpha
        return self._sorted[idx]

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(self, score: float, alpha: float | None = None) -> bool:
        """Return True if the sample should be flagged as OOD.

        Args:
            score: non-conformity score from OODScorer.
            alpha: significance level (overrides self.alpha if provided).
        """
        return score > self.threshold(alpha)

    def predict_batch(
        self,
        scores: list[float],
        alpha: float | None = None,
    ) -> list[bool]:
        """Vectorised prediction for a list of scores."""
        thr = self.threshold(alpha)
        return [s > thr for s in scores]

    def p_value(self, score: float) -> float:
        """Marginal p-value: fraction of calibration scores >= score.

        A small p-value means the test score is unusually high compared to
        in-distribution scores → evidence of OOD.
        """
        self._check_fitted()
        n = len(self._sorted)
        # count how many calibration scores are >= score (including self by +1)
        n_geq = sum(1 for s in self._sorted if s >= score) + 1
        return n_geq / (n + 1)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"cal_scores": self._cal_scores, "alpha": self.alpha}, f)

    @classmethod
    def load(cls, path: str | Path) -> "ConformalCalibrator":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(alpha=data["alpha"])
        obj.fit(data["cal_scores"])
        return obj

    # ── diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a human-readable summary of the calibration set."""
        self._check_fitted()
        n = len(self._sorted)
        return {
            "n_calibration": n,
            "score_min":   self._sorted[0],
            "score_p25":   self._sorted[n // 4],
            "score_median": self._sorted[n // 2],
            "score_p75":   self._sorted[3 * n // 4],
            "score_max":   self._sorted[-1],
            "threshold_alpha_005": self.threshold(0.05),
            "threshold_alpha_010": self.threshold(0.10),
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._sorted:
            raise RuntimeError("ConformalCalibrator has not been fitted yet. Call fit() first.")
