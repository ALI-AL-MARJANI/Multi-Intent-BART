"""Metrics of the multi-intent SLU literature (AGIF, GL-GIN, GEMIS).

* **Slot F1**: micro-averaged F1 over exact ``(start, end, type)`` spans. On
  well-formed BIO sequences this equals the conlleval / seqeval chunk F1 used
  by the baselines (checked in ``tests/test_metrics.py``).
* **Intent accuracy**: share of utterances whose predicted intent *set* equals
  the gold set.
* **Overall accuracy**: share of utterances with both the intent set and the
  full slot set correct.
"""

from __future__ import annotations

from collections import Counter
from math import comb
from typing import Any

import numpy as np

from gemis.data.dataset import Frame


def slot_counts(gold: Frame, pred: Frame) -> tuple[int, int, int]:
    """(true positives, false positives, false negatives) for one utterance."""
    tp = len(gold.slots & pred.slots)
    return tp, len(pred.slots) - tp, len(gold.slots) - tp


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def slot_f1(golds: list[Frame], preds: list[Frame]) -> float:
    counts = np.array([slot_counts(g, p) for g, p in zip(golds, preds)]).reshape(-1, 3)
    return _f1(*counts.sum(axis=0).tolist())


def intent_correct(gold: Frame, pred: Frame) -> bool:
    return gold.intents == pred.intents


def overall_correct(gold: Frame, pred: Frame) -> bool:
    return gold.intents == pred.intents and gold.slots == pred.slots


def compute_metrics(golds: list[Frame], preds: list[Frame]) -> dict[str, float]:
    """Slot F1, intent accuracy and overall accuracy, in percent."""
    if len(golds) != len(preds):
        raise ValueError("gold and predicted frames differ in number")
    n = max(len(golds), 1)
    return {
        "slot_f1": 100 * slot_f1(golds, preds),
        "intent_accuracy": 100 * sum(map(intent_correct, golds, preds)) / n,
        "overall_accuracy": 100 * sum(map(overall_correct, golds, preds)) / n,
    }


def bootstrap_ci(
    golds: list[Frame],
    preds: list[Frame],
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, list[float]]:
    """Percentile bootstrap interval over utterances for the three metrics."""
    rng = np.random.default_rng(seed)
    counts = np.array([slot_counts(g, p) for g, p in zip(golds, preds)])
    intent = np.array([intent_correct(g, p) for g, p in zip(golds, preds)], dtype=float)
    overall = np.array([overall_correct(g, p) for g, p in zip(golds, preds)], dtype=float)
    n = len(golds)
    samples: dict[str, list[float]] = {"slot_f1": [], "intent_accuracy": [], "overall_accuracy": []}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        samples["slot_f1"].append(100 * _f1(*counts[idx].sum(axis=0).tolist()))
        samples["intent_accuracy"].append(100 * intent[idx].mean())
        samples["overall_accuracy"].append(100 * overall[idx].mean())
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in samples.items()}


def mcnemar_exact(correct_a: list[bool], correct_b: list[bool]) -> dict[str, float]:
    """Exact two-sided McNemar test on paired per-utterance correctness."""
    only_a = sum(a and not b for a, b in zip(correct_a, correct_b))
    only_b = sum(b and not a for a, b in zip(correct_a, correct_b))
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return {"only_a": only_a, "only_b": only_b, "p_value": min(1.0, 2 * tail)}


def error_analysis(golds: list[Frame], preds: list[Frame]) -> dict[str, Any]:
    """Where the errors are: intent-count confusion and per-slot-type scores."""
    count_confusion: Counter[str] = Counter()
    per_slot: dict[str, list[int]] = {}
    missed_intents: Counter[str] = Counter()
    spurious_intents: Counter[str] = Counter()
    boundary_errors = 0  # predicted span has the right type and overlaps a gold span

    for gold, pred in zip(golds, preds):
        count_confusion[f"gold={len(gold.intents)},pred={len(pred.intents)}"] += 1
        missed_intents.update(gold.intents - pred.intents)
        spurious_intents.update(pred.intents - gold.intents)
        for span in gold.slots & pred.slots:
            per_slot.setdefault(span[2], [0, 0, 0])[0] += 1
        for span in pred.slots - gold.slots:
            per_slot.setdefault(span[2], [0, 0, 0])[1] += 1
            if any(g[2] == span[2] and g[0] < span[1] and span[0] < g[1] for g in gold.slots):
                boundary_errors += 1
        for span in gold.slots - pred.slots:
            per_slot.setdefault(span[2], [0, 0, 0])[2] += 1

    slots = {
        name: {"tp": tp, "fp": fp, "fn": fn, "f1": round(100 * _f1(tp, fp, fn), 2)}
        for name, (tp, fp, fn) in per_slot.items()
    }
    worst = sorted(slots.items(), key=lambda kv: -(kv[1]["fp"] + kv[1]["fn"]))[:15]
    return {
        "intent_count_confusion": dict(sorted(count_confusion.items())),
        "most_missed_intents": dict(missed_intents.most_common(10)),
        "most_spurious_intents": dict(spurious_intents.most_common(10)),
        "slot_boundary_errors": boundary_errors,
        "slot_types_with_most_errors": dict(worst),
    }
