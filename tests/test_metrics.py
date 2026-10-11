from __future__ import annotations

import random

import pytest

from gemis.data.dataset import Frame, bio_to_spans
from gemis.data.decoding import items_to_frame
from gemis.data.labels import Item
from gemis.training.metrics import (
    bootstrap_ci,
    compute_metrics,
    error_analysis,
    mcnemar_exact,
    slot_f1,
)


def frame(intents, slots):
    return Frame(frozenset(intents), frozenset(slots))


def test_metrics_on_a_hand_computed_example():
    golds = [frame(["a", "b"], [(0, 1, "x"), (2, 4, "y")]), frame(["a"], [(1, 2, "x")])]
    preds = [frame(["b", "a"], [(0, 1, "x"), (2, 3, "y")]), frame(["a"], [(1, 2, "x")])]
    metrics = compute_metrics(golds, preds)
    # tp=2, fp=1, fn=1 -> F1 = 4/6
    assert metrics["slot_f1"] == pytest.approx(100 * 2 / 3)
    assert metrics["intent_accuracy"] == 100.0  # intent order is irrelevant
    assert metrics["overall_accuracy"] == 50.0


def test_empty_predictions():
    golds = [frame(["a"], [(0, 1, "x")])]
    preds = [frame([], [])]
    assert compute_metrics(golds, preds) == {"slot_f1": 0.0, "intent_accuracy": 0.0, "overall_accuracy": 0.0}


def test_span_f1_matches_seqeval():
    """Span-set F1 equals the conlleval chunk F1 the baselines report."""
    seqeval = pytest.importorskip("seqeval.metrics")
    rng = random.Random(0)
    types = ["a", "b", "c"]

    def random_tags(n):
        tags, prev = [], "O"
        for _ in range(n):
            choice = rng.random()
            if choice < 0.4:
                tag = "O"
            elif choice < 0.7 or prev == "O":
                tag = "B-" + rng.choice(types)
            else:
                tag = "I-" + prev[2:]
            tags.append(tag)
            prev = tag
        return tags

    gold_tags = [random_tags(rng.randint(3, 15)) for _ in range(200)]
    pred_tags = [[t if rng.random() < 0.8 else rng.choice(["O", "B-a", "B-b"]) for t in s] for s in gold_tags]
    golds = [frame([], bio_to_spans(t)) for t in gold_tags]
    preds = [frame([], bio_to_spans(t)) for t in pred_tags]
    assert slot_f1(golds, preds) == pytest.approx(seqeval.f1_score(gold_tags, pred_tags))


def test_malformed_sequences_are_parsed_leniently():
    items = [
        Item("intent", "a"),
        Item("pos", 2),
        Item("pos", 5),
        Item("slot", "x"),
        Item("pos", 7),
        Item("slot", "y"),  # incomplete triple
        Item("pos", 4),
        Item("pos", 4),
        Item("slot", "z"),  # empty span
        Item("pos", -1),
        Item("pos", 3),
        Item("slot", "z"),  # pointer on <s>
        Item("intent", "late"),
    ]
    assert items_to_frame(items) == frame(["a"], [(2, 5, "x")])


def test_mcnemar():
    assert mcnemar_exact([True] * 10, [True] * 10)["p_value"] == 1.0
    result = mcnemar_exact([True] * 20, [False] * 20)
    assert result["only_a"] == 20 and result["p_value"] < 1e-5


def test_bootstrap_interval_contains_point_estimate():
    rng = random.Random(1)
    golds = [frame(["a"], [(0, 1, "x")]) for _ in range(200)]
    preds = [g if rng.random() < 0.7 else frame(["b"], []) for g in golds]
    point = compute_metrics(golds, preds)
    for name, (lo, hi) in bootstrap_ci(golds, preds, n_resamples=200).items():
        assert lo <= point[name] <= hi


def test_error_analysis_counts():
    golds = [frame(["a", "b"], [(0, 2, "x")])]
    preds = [frame(["a"], [(0, 1, "x")])]
    report = error_analysis(golds, preds)
    assert report["intent_count_confusion"] == {"gold=2,pred=1": 1}
    assert report["most_missed_intents"] == {"b": 1}
    assert report["slot_boundary_errors"] == 1
