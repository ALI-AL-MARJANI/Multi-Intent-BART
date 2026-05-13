"""File I/O helpers and YAML config loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class RawToken:
    word: str
    bio_tag: str


@dataclass
class RawSampleIO:
    """Parsed utterance from AGIF-format text files."""
    tokens: list[RawToken] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)


def load_raw_dataset(path: str | Path) -> list[RawSampleIO]:
    """Parse AGIF-format tab-separated files into RawSampleIO objects.

    Format per utterance (blank-line separated):
        word<TAB>BIO-tag
        ...
        intent1#intent2#...   (no BIO column)
    """
    samples: list[RawSampleIO] = []
    current_tokens: list[RawToken] = []
    current_intents: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line == "":
                if current_tokens and current_intents:
                    samples.append(
                        RawSampleIO(tokens=current_tokens, intents=current_intents)
                    )
                current_tokens = []
                current_intents = []
                continue

            # AGIF data is space-separated (not tab-separated)
            parts = line.split()
            if len(parts) == 1:
                # intent line: "intent1#intent2#..."
                current_intents = parts[0].split("#")
            elif len(parts) == 2:
                current_tokens.append(RawToken(word=parts[0], bio_tag=parts[1]))

    # flush last sample if file doesn't end with blank line
    if current_tokens and current_intents:
        samples.append(RawSampleIO(tokens=current_tokens, intents=current_intents))

    return samples


def write_dataset(samples: list[RawSampleIO], path: str | Path) -> None:
    """Write samples back to AGIF format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            for token in sample.tokens:
                f.write(f"{token.word} {token.bio_tag}\n")
            f.write("#".join(sample.intents) + "\n")
            f.write("\n")
