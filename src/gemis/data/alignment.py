"""Word-to-subword alignment computed on the actual encoder input.

BPE tokenizers split ``"baltimore"`` and ``" baltimore"`` differently, so the
position of a word's first subword must be read off the tokenization of the
whole sentence. Counting subwords word by word gives wrong positions.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from transformers import PreTrainedTokenizerFast


@dataclass
class EncodedSource:
    input_ids: list[int]
    # word_starts[k] = encoder position of the first subword of word k.
    # One extra entry is appended: the position of the final </s>, which is the
    # pointer target for an exclusive span end that falls after the last word.
    word_starts: list[int]

    @property
    def n_words(self) -> int:
        return len(self.word_starts) - 1


def encode_source(
    words: list[str],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 128,
) -> EncodedSource:
    """Tokenize ``" ".join(words)`` and locate the first subword of every word.

    If the sentence is truncated, words that lost their first subword are
    dropped from ``word_starts``.
    """
    text = " ".join(words)
    enc = tokenizer(text, max_length=max_length, truncation=True, return_offsets_mapping=True)
    input_ids: list[int] = enc["input_ids"]

    char_starts: list[int] = []
    cursor = 0
    for word in words:
        char_starts.append(cursor)
        cursor += len(word) + 1

    word_starts: list[int] = []
    for pos, (start, end) in enumerate(enc["offset_mapping"]):
        if start == end:  # special token
            continue
        if text[start] == " ":  # some tokenizers keep the leading space in the offset
            start += 1
        word = bisect_right(char_starts, start) - 1
        if word == len(word_starts):
            word_starts.append(pos)

    word_starts.append(len(input_ids) - 1)
    return EncodedSource(input_ids=input_ids, word_starts=word_starts)


def position_to_word(position: int, word_starts: list[int]) -> int:
    """Word index for an encoder position.

    The final ``</s>`` maps to ``n_words`` (a valid exclusive end). Positions
    before the first word (``<s>``) and padding map to -1.
    """
    if position < word_starts[0] or position > word_starts[-1]:
        return -1
    return bisect_right(word_starts, position) - 1
