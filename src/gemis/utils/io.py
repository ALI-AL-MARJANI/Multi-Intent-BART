"""File I/O: YAML configs and the AGIF text format used by MixATIS / MixSNIPS."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class RawToken:
    word: str
    bio_tag: str


@dataclass
class RawSampleIO:
    """One utterance parsed from an AGIF-format file."""

    tokens: list[RawToken] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)

    @property
    def words(self) -> list[str]:
        return [t.word for t in self.tokens]

    @property
    def tags(self) -> list[str]:
        return [t.bio_tag for t in self.tokens]


def load_raw_dataset(path: str | Path) -> list[RawSampleIO]:
    """Parse an AGIF-format file.

    Each utterance is a block of ``word BIO-tag`` lines followed by one
    ``intent1#intent2`` line; blocks are separated by a blank line.
    """
    samples: list[RawSampleIO] = []
    tokens: list[RawToken] = []
    intents: list[str] = []

    def flush() -> None:
        nonlocal tokens, intents
        if tokens and intents:
            samples.append(RawSampleIO(tokens=tokens, intents=intents))
        tokens, intents = [], []

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.split()
            if not parts:
                flush()
            elif len(parts) == 1:
                intents = parts[0].split("#")
            elif len(parts) == 2:
                tokens.append(RawToken(word=parts[0], bio_tag=parts[1]))
            else:
                raise ValueError(f"{path}: unexpected line {raw_line!r}")
    flush()
    return samples


def write_dataset(samples: list[RawSampleIO], path: str | Path) -> None:
    """Write samples in AGIF format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            for token in sample.tokens:
                f.write(f"{token.word} {token.bio_tag}\n")
            f.write("#".join(sample.intents) + "\n\n")
