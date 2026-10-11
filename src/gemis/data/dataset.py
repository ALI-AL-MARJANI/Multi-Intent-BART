"""GEMISDataset: AGIF utterances -> encoder inputs and pointer/label targets.

Target sequence (paper section 3.1)::

    intent_1 ... intent_p  start_1 end_1 slot_1  ...  start_q end_q slot_q  </s>

``start``/``end`` are word indices, 0-based, end exclusive: in
``please play got the time and add my hands to travelling playlist`` the slot
``got the time`` is ``2 5 track``. A word index is predicted by pointing at the
first subword of that word; ``end == n_words`` points at the final ``</s>``.

The decoder input at a pointer step is the pointed wordpiece, as in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from gemis.data.alignment import encode_source
from gemis.data.labels import LabelSpace
from gemis.utils.io import RawSampleIO


@dataclass(frozen=True)
class Frame:
    """Semantic frame: a set of intents and (start, end_exclusive, type) spans."""

    intents: frozenset[str]
    slots: frozenset[tuple[int, int, str]]

    def to_json(self) -> dict[str, Any]:
        return {"intents": sorted(self.intents), "slots": sorted(self.slots)}


def bio_to_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """BIO tags -> (start, end_exclusive, type) spans, with conlleval chunking.

    A chunk starts at ``B-x``, or at ``I-x`` when the previous tag is ``O`` or
    has another type.
    """
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    current: str | None = None
    for i, tag in enumerate(tags):
        prefix, _, slot_type = tag.partition("-")
        inside = prefix == "I" and current == slot_type
        if start is not None and not inside:
            spans.append((start, i, current))  # type: ignore[arg-type]
            start, current = None, None
        if prefix == "B" or (prefix == "I" and not inside):
            start, current = i, slot_type
    if start is not None:
        spans.append((start, len(tags), current))  # type: ignore[arg-type]
    return spans


def gold_frame(sample: RawSampleIO) -> Frame:
    return Frame(frozenset(sample.intents), frozenset(bio_to_spans(sample.tags)))


class GEMISDataset(Dataset):
    """Pre-tokenized utterances.

    Args:
        samples: parsed utterances.
        tokenizer: tokenizer already extended with the label tokens.
        label_space: class inventory of the model.
        label_token_ids: tokenizer id of every label class (decoder inputs).
        decoder_start_token_id: first decoder input token.
        max_source_length: encoder truncation length, in subwords.
        use_pointer: if False, positions are ``<pos:k>`` classes (ablation).
        with_targets: build ``labels`` / ``decoder_input_ids``. Disable for
            evaluation sets, which may contain labels unseen in training.
    """

    def __init__(
        self,
        samples: list[RawSampleIO],
        tokenizer: PreTrainedTokenizerFast,
        label_space: LabelSpace,
        label_token_ids: list[int],
        decoder_start_token_id: int,
        max_source_length: int = 128,
        use_pointer: bool = True,
        with_targets: bool = True,
    ) -> None:
        self.label_space = label_space
        self.label_token_ids = label_token_ids
        self.decoder_start_token_id = decoder_start_token_id
        self.use_pointer = use_pointer
        self.n_dropped_spans = 0
        self.items: list[dict[str, Any]] = []

        for index, sample in enumerate(samples):
            source = encode_source(sample.words, tokenizer, max_source_length)
            item: dict[str, Any] = {
                "index": index,
                "input_ids": source.input_ids,
                "word_starts": source.word_starts,
                "words": sample.words,
                "gold": gold_frame(sample),
            }
            if with_targets:
                item.update(self._build_targets(sample, source.input_ids, source.word_starts))
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]

    def _build_targets(self, sample: RawSampleIO, input_ids: list[int], word_starts: list[int]) -> dict[str, list[int]]:
        space = self.label_space
        n_labels = len(space)
        n_words = len(word_starts) - 1

        labels: list[int] = []  # class ids in [0, L + N)
        fed: list[int] = []  # the token the decoder reads after emitting each class

        def emit_label(class_id: int) -> None:
            labels.append(class_id)
            fed.append(self.label_token_ids[class_id])

        def emit_position(word_index: int) -> None:
            if self.use_pointer:
                position = word_starts[word_index]
                labels.append(n_labels + position)
                fed.append(input_ids[position])
            else:
                emit_label(space.pos_id(word_index))

        for intent in sample.intents:
            emit_label(space.intent_id(intent))
        for start, end, slot_type in bio_to_spans(sample.tags):
            if end > n_words:  # span lost to truncation
                self.n_dropped_spans += 1
                continue
            emit_position(start)
            emit_position(end)
            emit_label(space.slot_id(slot_type))
        labels.append(space.eos_id)

        return {
            "labels": labels,
            "decoder_input_ids": [self.decoder_start_token_id] + fed,
        }
