"""Seeding, device selection and label collection."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from gemis.utils.io import RawSampleIO, load_raw_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` picks cuda, then mps, then cpu; an explicit name is used as is."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def release_memory(device: torch.device) -> None:
    """Return cached blocks to the system. The MPS allocator otherwise grows without bound."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def drop_holdout(samples: list[RawSampleIO], holdout: Iterable[str]) -> list[RawSampleIO]:
    """Remove every utterance that carries at least one held-out intent."""
    held = set(holdout)
    return [s for s in samples if not held.intersection(s.intents)]


def only_holdout(samples: list[RawSampleIO], holdout: Iterable[str]) -> list[RawSampleIO]:
    """Keep the utterances that carry at least one held-out intent."""
    held = set(holdout)
    return [s for s in samples if held.intersection(s.intents)]


def collect_labels(samples: list[RawSampleIO]) -> tuple[list[str], list[str]]:
    """Sorted intent names and slot types that occur in ``samples``."""
    intents: set[str] = set()
    slots: set[str] = set()
    for sample in samples:
        intents.update(sample.intents)
        for tag in sample.tags:
            if tag[:2] in ("B-", "I-"):
                slots.add(tag[2:])
    return sorted(intents), sorted(slots)


def load_split(path: str | Path, holdout: Iterable[str] = ()) -> list[RawSampleIO]:
    """Load one split, dropping utterances with held-out intents."""
    return drop_holdout(load_raw_dataset(path), holdout)
