#!/usr/bin/env python3
"""OOD detection metrics and conformal validity from a ``scores.csv`` file.

    python scripts/evaluate_ood.py --scores_dir results/ood/mixsnips_bart-base/seed13

CPU only. Writes ``metrics.json`` and ``calibrator.json`` next to the scores.

For every score and OOD set: AUROC, AUPR-in, AUPR-out, FPR@95TPR with 95%
bootstrap intervals. ``negative_length`` (minus the number of words) is added
as a trivial baseline: it flags short utterances and uses no model at all. Conformal part, for every score and alpha:

* ``dev_calibrated``: threshold from ``id_dev``, false-alarm rate on ``id_test``
  and detection rate on each OOD set. The dev set also selects the checkpoint,
  so exchangeability with the test set is not guaranteed.
* ``test_split``: 1000 random calibration / evaluation splits of ``id_test``,
  which are exchangeable by construction; the mean false-alarm rate should
  match the theoretical value.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from gemis.ood.calibrator import ConformalCalibrator, conformal_threshold, expected_fpr, validity_experiment
from gemis.ood.metrics import bootstrap_detection_metrics, detection_metrics
from gemis.ood.scores import DECODER_SCORES, FEATURE_SCORES
from gemis.utils.io import write_json

MODEL_SCORES = (*DECODER_SCORES, *FEATURE_SCORES)
SCORES = (*MODEL_SCORES, "negative_length")
ALPHAS = (0.01, 0.05, 0.1)


def read_scores(path: Path) -> dict[str, dict[str, np.ndarray]]:
    sets: dict[str, dict[str, list[float]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target = sets.setdefault(row["set"], {name: [] for name in SCORES})
            for name in MODEL_SCORES:
                target[name].append(float(row[name]))
            target["negative_length"].append(-float(len(row["text"].split())))
    return {s: {n: np.array(v) for n, v in scores.items()} for s, scores in sets.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores_dir", required=True, type=Path)
    parser.add_argument("--n_splits", type=int, default=1000)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--calibrator_score", default="intent_entropy")
    args = parser.parse_args()

    sets = read_scores(args.scores_dir / "scores.csv")
    ood_sets = [name for name in sets if not name.startswith("id_")]
    dev, test = sets["id_dev"], sets["id_test"]

    detection: dict[str, dict] = {}
    conformal: dict[str, dict] = {}
    for score in SCORES:
        detection[score] = {}
        for ood in ood_sets:
            detection[score][ood] = {
                "point": detection_metrics(test[score], sets[ood][score]),
                "ci95": bootstrap_detection_metrics(test[score], sets[ood][score], n_resamples=args.n_bootstrap),
            }

        ks = ks_2samp(dev[score], test[score])
        dev_calibrated = {}
        for alpha in ALPHAS:
            threshold = conformal_threshold(dev[score], alpha)
            dev_calibrated[str(alpha)] = {
                "threshold": threshold,
                "fpr_id_test": float((test[score] > threshold).mean()),
                "theory": expected_fpr(len(dev[score]), alpha),
                "tpr": {ood: float((sets[ood][score] > threshold).mean()) for ood in ood_sets},
            }
        conformal[score] = {
            "dev_vs_test_ks": {"statistic": float(ks.statistic), "p_value": float(ks.pvalue)},
            "share_of_tied_test_scores": float(1 - len(np.unique(test[score])) / len(test[score])),
            "dev_calibrated": dev_calibrated,
            "test_split": validity_experiment(test[score], ALPHAS, n_splits=args.n_splits),
        }

    ConformalCalibrator(dev[args.calibrator_score], args.calibrator_score).save(args.scores_dir / "calibrator.json")
    write_json(
        {
            "set_sizes": {name: len(next(iter(scores.values()))) for name, scores in sets.items()},
            "alphas": list(ALPHAS),
            "detection": detection,
            "conformal": conformal,
        },
        args.scores_dir / "metrics.json",
    )

    for ood in ood_sets:
        print(f"\n{ood} vs id_test  (n_ood={len(sets[ood][SCORES[0]])}, n_id={len(test[SCORES[0]])})")
        print(f"  {'score':<16} {'AUROC':>7} {'AUPR-out':>9} {'AUPR-in':>8} {'FPR@95':>7}")
        for score in SCORES:
            m = detection[score][ood]["point"]
            row = f"{m['auroc']:>7.4f} {m['aupr_out']:>9.4f} {m['aupr_in']:>8.4f} {m['fpr_at_95tpr']:>7.4f}"
            print(f"  {score:<16} {row}")
    print("\nconformal false-alarm rate on in-distribution test data (target: alpha)")
    print(f"  {'score':<16} {'alpha':>5} {'dev-calibrated':>15} {'test-split mean +- std':>24}")
    for score in SCORES:
        for alpha in ALPHAS:
            d = conformal[score]["dev_calibrated"][str(alpha)]
            s = conformal[score]["test_split"]["per_alpha"][str(alpha)]
            print(f"  {score:<16} {alpha:>5} {d['fpr_id_test']:>15.4f} {s['mean_fpr']:>15.4f} +- {s['std_fpr']:.4f}")


if __name__ == "__main__":
    main()
