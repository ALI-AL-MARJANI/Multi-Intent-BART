from __future__ import annotations

import numpy as np
import pytest
import torch

from gemis.experiment import make_loader
from gemis.ood.calibrator import (
    ConformalCalibrator,
    conformal_p_values,
    conformal_threshold,
    expected_fpr,
    validity_experiment,
)
from gemis.ood.data import load_ood_utterances, normalise_utterance
from gemis.ood.metrics import bootstrap_detection_metrics, detection_metrics
from gemis.ood.scores import DECODER_SCORES, FeatureScorer, score_loader


def test_threshold_is_the_finite_sample_quantile():
    scores = np.arange(1, 20)  # n = 19
    assert conformal_threshold(scores, 0.05) == 19  # ceil(20 * 0.95) = 19th smallest
    assert conformal_threshold(scores, 0.10) == 18
    assert conformal_threshold(scores, 0.01) == float("inf")  # too few points for alpha
    with pytest.raises(ValueError):
        conformal_threshold(np.array([]), 0.05)


def test_p_values_agree_with_the_threshold():
    rng = np.random.default_rng(0)
    cal, test = rng.normal(size=500), rng.normal(size=2000)
    for alpha in (0.01, 0.05, 0.1):
        flagged = test > conformal_threshold(cal, alpha)
        assert np.array_equal(flagged, conformal_p_values(cal, test) <= alpha)


def test_false_alarm_rate_is_controlled_on_exchangeable_data():
    scores = np.random.default_rng(0).gamma(2.0, size=4000)
    result = validity_experiment(scores, alphas=(0.05, 0.1), n_splits=300)
    for alpha in (0.05, 0.1):
        stats = result["per_alpha"][str(alpha)]
        assert stats["mean_fpr"] <= alpha + 0.002
        assert stats["mean_fpr"] == pytest.approx(stats["theory"]["mean"], abs=0.003)


def test_expected_fpr_matches_simulation():
    rng = np.random.default_rng(1)
    n, alpha = 200, 0.1
    rates = [1 - np.sort(rng.uniform(size=n))[int(np.ceil((n + 1) * (1 - alpha))) - 1] for _ in range(4000)]
    theory = expected_fpr(n, alpha)
    assert np.mean(rates) == pytest.approx(theory["mean"], abs=0.002)
    assert np.std(rates) == pytest.approx(theory["std"], abs=0.002)


def test_calibrator_roundtrip(tmp_path):
    calibrator = ConformalCalibrator(np.random.default_rng(0).normal(size=100), "intent_entropy")
    calibrator.save(tmp_path / "cal.json")
    loaded = ConformalCalibrator.load(tmp_path / "cal.json")
    assert loaded.score_name == "intent_entropy"
    assert loaded.threshold(0.05) == calibrator.threshold(0.05)
    assert loaded.is_ood(np.array([-10.0, 10.0]), 0.05).tolist() == [False, True]


def test_detection_metrics():
    perfect = detection_metrics(np.array([0.0, 0.1, 0.2]), np.array([0.9, 1.0]))
    assert perfect == {"auroc": 1.0, "aupr_out": 1.0, "aupr_in": 1.0, "fpr_at_95tpr": 0.0}
    rng = np.random.default_rng(0)
    chance = detection_metrics(rng.normal(size=5000), rng.normal(size=5000))
    assert chance["auroc"] == pytest.approx(0.5, abs=0.02)
    assert chance["fpr_at_95tpr"] == pytest.approx(0.95, abs=0.02)
    interval = bootstrap_detection_metrics(rng.normal(size=300), rng.normal(1, size=300), n_resamples=100)
    assert interval["auroc"][0] < interval["auroc"][1]


def test_feature_scorer_ranks_far_points_higher():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(500, 8))
    scorer = FeatureScorer(k=5).fit(train)
    near = scorer.score(rng.normal(size=(100, 8)))
    far = scorer.score(rng.normal(size=(100, 8)) + np.array([6.0] + [0.0] * 7))
    assert far["mahalanobis"].mean() > 2 * near["mahalanobis"].mean()
    assert far["knn"].mean() > near["knn"].mean()
    own = scorer.score(train[:10], exclude_self=True)
    assert (own["knn"] > 0).all()


def test_normalisation_and_deduplication(tmp_path):
    assert normalise_utterance("What's the Dow, today?") == ["what's", "the", "dow", ",", "today"]
    path = tmp_path / "ood.txt"
    path.write_text("How are you?\nhow are you\n\nTell me a joke.\n")
    utterances = load_ood_utterances(path)
    assert [u.words for u in utterances] == [["how", "are", "you"], ["tell", "me", "a", "joke"]]
    assert all(set(u.tags) == {"O"} for u in utterances)


def test_scores_are_identical_alone_and_in_a_batch(samples, model_and_tokenizer):
    """Calibration and evaluation use one scoring path; batching must not change a score."""
    model, tokenizer = model_and_tokenizer
    device = torch.device("cpu")
    batched = score_loader(model, make_loader(samples, model, tokenizer, 12, 128), device, 12)
    single = score_loader(model, make_loader(samples, model, tokenizer, 1, 128), device, 12)
    assert set(batched["scores"]) == set(DECODER_SCORES)
    for name in DECODER_SCORES:
        assert np.isfinite(batched["scores"][name]).all()
        assert np.allclose(batched["scores"][name], single["scores"][name], atol=1e-3), name
    assert np.allclose(batched["features"], single["features"], atol=1e-4)
    assert batched["frames"] == single["frames"]
    assert len(batched["texts"]) == len(samples)
