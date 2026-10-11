from __future__ import annotations

from collections import Counter

from gemis.data.construct import MultiIntentConstructor
from gemis.data.dataset import bio_to_spans


def single_intent(samples):
    return [s for s in samples if len(s.intents) == 1]


def test_merge_keeps_tags_aligned(samples):
    base = single_intent(samples)
    constructor = MultiIntentConstructor(lambda a, b: 1.0, connectors=("and then",))
    merged = constructor.merge(base[0], base[1])
    assert merged.words == base[0].words + ["and", "then"] + base[1].words
    assert merged.intents == base[0].intents + base[1].intents
    offset = len(base[0].words) + 2
    shifted = [(s + offset, e + offset, t) for s, e, t in bio_to_spans(base[1].tags)]
    assert bio_to_spans(merged.tags) == bio_to_spans(base[0].tags) + shifted


def test_intent_count_distribution_and_distinct_intents(samples):
    base = single_intent(samples) * 100
    constructed = MultiIntentConstructor(lambda a, b: 1.0, seed=0).construct(base)
    assert len(constructed) == len(base)
    assert all(len(set(s.intents)) == len(s.intents) for s in constructed)
    share = Counter(len(s.intents) for s in constructed)
    assert abs(share[1] / len(base) - 0.3) < 0.06
    assert abs(share[2] / len(base) - 0.5) < 0.06


def test_threshold_blocks_concatenation(samples):
    base = single_intent(samples)
    constructed = MultiIntentConstructor(lambda a, b: 0.0, tau=0.5).construct(base)
    assert [s.words for s in constructed] == [s.words for s in base]
