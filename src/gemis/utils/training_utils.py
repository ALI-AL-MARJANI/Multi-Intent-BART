"""Shared utilities for training and evaluation scripts."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from gemis.utils.io import load_raw_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_labels(paths: list[Path]) -> tuple[list[str], list[str]]:
    """Scan dataset files and return sorted lists of intent and slot labels."""
    intents: set[str] = set()
    slots: set[str] = set()

    for path in paths:
        if not path.exists():
            continue
        for sample in load_raw_dataset(path):
            intents.update(sample.intents)
            for tok in sample.tokens:
                tag = tok.bio_tag
                if tag.startswith("B-") or tag.startswith("I-"):
                    slots.add(tag[2:])

    return sorted(intents), sorted(slots)
