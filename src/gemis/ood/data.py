"""OOD utterance files: one utterance per line."""

from __future__ import annotations

import re
from pathlib import Path

from gemis.utils.io import RawSampleIO, RawToken

OOS_INTENT = "oos"
_DROP = re.compile(r"[?!.\"]")


def normalise_utterance(text: str) -> list[str]:
    """Lowercase, drop sentence punctuation, split commas off as tokens.

    MixATIS / MixSNIPS are lowercased with no sentence punctuation. Without
    this step a detector could separate OOD text by its formatting alone.
    """
    text = _DROP.sub(" ", text.lower()).replace(",", " , ")
    return text.split()


def load_ood_utterances(path: str | Path) -> list[RawSampleIO]:
    """Read, normalise and deduplicate; every word is tagged ``O``."""
    seen: set[tuple[str, ...]] = set()
    samples: list[RawSampleIO] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            words = tuple(normalise_utterance(line))
            if not words or words in seen:
                continue
            seen.add(words)
            samples.append(RawSampleIO(tokens=[RawToken(w, "O") for w in words], intents=[OOS_INTENT]))
    return samples
