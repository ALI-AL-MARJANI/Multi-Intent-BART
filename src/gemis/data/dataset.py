"""GEMISDataset: converts raw AGIF-format files into seq2seq samples for BART.

Target sequence format (paper §3.1):
    <BOS> <intent:X> <intent:Y> ... start_1 end_1 <slot:Z> ... <EOS>

Position indices are 0-indexed from the first word of the utterance, and
end is EXCLUSIVE (Python-slice style).  Example for
  "please play Got The Time and add My Hands to travelling playlist":
    words[2:5] = "Got The Time" → target has "2 5 <slot:track>"
    words[7:9] = "My Hands"     → target has "7 9 <slot:entity_name>"

During training (teacher forcing), the decoder receives the WORDPIECE that
corresponds to each position integer, not the integer token itself.  During
inference, the model generates integer tokens and we convert them back to
wordpieces for the next decoder step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from gemis.utils.io import RawSampleIO, load_raw_dataset

# ── typed containers ─────────────────────────────────────────────────────────

@dataclass
class SlotSpan:
    """Word-level slot span.  end is EXCLUSIVE (words[start:end] = slot value)."""
    start: int
    end: int          # exclusive
    slot_type: str


@dataclass
class RawSample:
    words: list[str]
    intents: list[str]
    slot_spans: list[SlotSpan]


# ── BIO → span conversion (exclusive end) ────────────────────────────────────

def bio_to_spans(words: list[str], bio_tags: list[str]) -> list[SlotSpan]:
    """Convert BIO-tagged token list into SlotSpan objects with EXCLUSIVE end."""
    spans: list[SlotSpan] = []
    start: Optional[int] = None
    current_type: Optional[str] = None

    for i, tag in enumerate(bio_tags):
        if tag.startswith("B-"):
            if start is not None:
                spans.append(SlotSpan(start, i, current_type))   # i is exclusive end
            start = i
            current_type = tag[2:]
        elif tag.startswith("I-"):
            pass
        else:
            if start is not None:
                spans.append(SlotSpan(start, i, current_type))   # i is exclusive end
                start = None
                current_type = None

    if start is not None:
        spans.append(SlotSpan(start, len(words), current_type))  # len is exclusive end

    return spans


# ── target sequence builder ───────────────────────────────────────────────────

def build_target_tokens(
    intents: list[str],
    slot_spans: list[SlotSpan],
    tokenizer: PreTrainedTokenizerFast,
) -> list[int]:
    """Encode the structured target sequence as BART token ids.

    Format: [BOS] <intent:X> ... start_1 end_1 <slot:Z> ... [EOS]

    Special tokens (intents / slots) are encoded token-by-token so the
    tokenizer does not insert unwanted Ġ-space tokens between them.
    Position integers are encoded with a leading space (" 2") so they receive
    the Ġ-prefix BPE token; decode(token_id).strip() == "2" for clean
    round-tripping in parse_target_sequence.
    """
    ids: list[int] = [tokenizer.bos_token_id]

    for intent in intents:
        ids += tokenizer.encode(f"<intent:{intent}>", add_special_tokens=False)

    for span in slot_spans:
        for pos_int in (span.start, span.end):
            ids += tokenizer.encode(f" {pos_int}", add_special_tokens=False)
        ids += tokenizer.encode(f"<slot:{span.slot_type}>", add_special_tokens=False)

    ids.append(tokenizer.eos_token_id)
    return ids


# ── word-index → absolute encoder subword position ───────────────────────────

def word_to_encoder_pos(
    word_idx: int,
    words: list[str],
    tokenizer: PreTrainedTokenizerFast,
) -> int:
    """Return the absolute encoder position of the first subword of word_idx.

    Encoder absolute positions:
      0 → BOS (<s>)
      1 → first subword of word 0
      ...
      pos_after_last_word → EOS (</s>)

    word_idx == len(words) is valid: it means "one past the last word", i.e.
    the exclusive end of a span that ends at the final word.  We map it to
    the EOS token position so the pointer network has a valid target.
    """
    pos = 1  # skip BOS
    for i, word in enumerate(words):
        if i == word_idx:
            return pos
        pos += len(tokenizer.encode(word, add_special_tokens=False))
    # pos is now the position of EOS (one past the last word's subwords)
    if word_idx == len(words):
        return pos   # valid: EOS token
    return -1


def encoder_pos_to_word_idx(
    enc_pos: int,
    words: list[str],
    tokenizer: PreTrainedTokenizerFast,
) -> int:
    """Inverse of word_to_encoder_pos: abs encoder position → word index.

    Returns -1 if enc_pos falls on BOS, EOS, or padding.
    """
    if enc_pos <= 0:
        return -1
    pos = 1
    for i, word in enumerate(words):
        n = len(tokenizer.encode(word, add_special_tokens=False))
        if pos <= enc_pos < pos + n:
            return i
        pos += n
    return -1


# ── dataset ───────────────────────────────────────────────────────────────────

class GEMISDataset(Dataset):
    """Torch Dataset wrapping raw AGIF files, yielding seq2seq tensors."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
        max_source_length: int = 128,
        max_target_length: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        raw_io: list[RawSampleIO] = load_raw_dataset(path)
        self.samples: list[RawSample] = [
            RawSample(
                words=[t.word for t in r.tokens],
                intents=r.intents,
                slot_spans=bio_to_spans(
                    [t.word for t in r.tokens],
                    [t.bio_tag for t in r.tokens],
                ),
            )
            for r in raw_io
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        tok = self.tokenizer

        # ── encode source ─────────────────────────────────────────────────────
        enc = tok(
            " ".join(sample.words),
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
        )
        input_ids: list[int] = enc["input_ids"]

        # ── build full target token sequence ──────────────────────────────────
        full_target = build_target_tokens(sample.intents, sample.slot_spans, tok)
        full_target = full_target[: self.max_target_length + 1]  # keep room for BOS

        # Standard seq2seq split:
        #   labels           = full_target[1:]        (BOS removed, EOS kept)
        #   decoder_input    = full_target[:-1]        (EOS removed, BOS kept)
        # … but positions in decoder_input are replaced by the actual wordpieces.
        labels: list[int] = full_target[1:]

        # ── decoder_input_ids: replace position integers with wordpieces ───────
        decoder_input_ids: list[int] = self._build_decoder_inputs(
            full_target[:-1], sample.words, input_ids
        )

        # ── pointer_targets: for each label step, encoder abs pos or -1 ───────
        pointer_targets: list[int] = self._build_pointer_targets(
            labels, sample.words, input_ids
        )

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
            "pointer_targets": pointer_targets,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _is_position_token(self, token_id: int) -> tuple[bool, int]:
        """Return (True, word_idx) if token_id decodes to an integer, else (False, -1)."""
        try:
            word_idx = int(self.tokenizer.decode([token_id]).strip())
            return True, word_idx
        except ValueError:
            return False, -1

    def _build_decoder_inputs(
        self,
        shifted: list[int],   # full_target[:-1] = [BOS, tok1, ..., tokN]
        words: list[str],
        input_ids: list[int],
    ) -> list[int]:
        """Replace position integer tokens in the decoder input with wordpieces.

        This implements the paper's rule: "previously predicted position indices
        will be converted to the corresponding wordpieces to be fed into the
        decoder."
        """
        tok = self.tokenizer
        result: list[int] = [shifted[0]]  # BOS is always kept as-is

        for tid in shifted[1:]:           # tok1, ..., tokN
            is_pos, word_idx = self._is_position_token(tid)
            if is_pos:
                enc_pos = word_to_encoder_pos(word_idx, words, tok)
                if 0 < enc_pos < len(input_ids):
                    result.append(input_ids[enc_pos])
                else:
                    result.append(tok.unk_token_id or 3)
            else:
                result.append(tid)

        return result

    def _build_pointer_targets(
        self,
        labels: list[int],    # full_target[1:] = [tok1, ..., tokN, EOS]
        words: list[str],
        input_ids: list[int],
    ) -> list[int]:
        """For each label step: encoder absolute position (if pointer step) else -1."""
        tok = self.tokenizer
        result: list[int] = []

        for tid in labels:
            is_pos, word_idx = self._is_position_token(tid)
            if is_pos:
                enc_pos = word_to_encoder_pos(word_idx, words, tok)
                result.append(enc_pos if 0 < enc_pos < len(input_ids) else -1)
            else:
                result.append(-1)

        return result
