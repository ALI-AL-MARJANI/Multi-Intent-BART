from __future__ import annotations

import pytest

from gemis.data.alignment import encode_source, position_to_word
from tests.conftest import build_tokenizer


def _check(words, tokenizer):
    source = encode_source(words, tokenizer)
    assert source.n_words == len(words)
    assert source.word_starts[-1] == len(source.input_ids) - 1
    assert source.input_ids[source.word_starts[-1]] == tokenizer.eos_token_id
    for k, word in enumerate(words):
        start, end = source.word_starts[k], source.word_starts[k + 1]
        # the subwords between two consecutive word starts spell exactly that word
        assert tokenizer.decode(source.input_ids[start:end]).strip() == word
        for position in range(start, end):
            assert position_to_word(position, source.word_starts) == k
    return source


def test_every_word_start_is_the_first_subword(samples):
    tokenizer = build_tokenizer(samples)
    n_multi = 0
    for sample in samples:
        source = _check(sample.words, tokenizer)
        n_multi += len(source.input_ids) - 2 > len(sample.words)
    assert n_multi > 0, "the toy tokenizer should split some words into several subwords"


def test_special_positions(samples):
    tokenizer = build_tokenizer(samples)
    source = encode_source(samples[0].words, tokenizer)
    assert position_to_word(0, source.word_starts) == -1  # <s>
    assert position_to_word(source.word_starts[-1], source.word_starts) == source.n_words  # </s>
    assert position_to_word(source.word_starts[-1] + 3, source.word_starts) == -1  # padding


def test_truncation_drops_trailing_words(samples):
    tokenizer = build_tokenizer(samples)
    words = samples[8].words
    source = encode_source(words, tokenizer, max_length=12)
    assert len(source.input_ids) == 12
    assert 0 < source.n_words < len(words)


def test_real_bart_tokenizer(samples):
    """BART's BPE treats a leading space as part of the token: the original bug."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-base", local_files_only=True)
    except OSError:
        pytest.skip("facebook/bart-base tokenizer not in the local cache")
    for sample in samples:
        _check(sample.words, tokenizer)
    _check("flights from pittsburgh to baltimore between 10 am and 2 pm".split(), tokenizer)
