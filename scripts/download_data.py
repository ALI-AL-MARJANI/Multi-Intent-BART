#!/usr/bin/env python3
"""Download the datasets into ``data/raw/<name>/{train,dev,test}.txt`` (AGIF format).

    python scripts/download_data.py --dataset mixatis mixsnips

* ``mixatis`` / ``mixsnips``: the cleaned multi-intent sets released with AGIF
  (Qin et al., 2020), https://github.com/LooperXX/AGIF.
* ``atis`` / ``snips``: the single-intent sets as distributed with Slot-Gated
  SLU (Goo et al., 2018), https://github.com/MiuLab/SlotGated-SLU, converted
  to AGIF format. They are only needed for ``scripts/construct_dataset.py``.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

AGIF = "https://raw.githubusercontent.com/LooperXX/AGIF/master/data"
SLOT_GATED = "https://raw.githubusercontent.com/MiuLab/SlotGated-SLU/master/data"
AGIF_DIRS = {"mixatis": "MixATIS_clean", "mixsnips": "MixSNIPS_clean"}
SPLITS = {"train": "train", "dev": "valid", "test": "test"}


def fetch(url: str) -> str:
    print(f"  {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def download_agif(name: str, out_dir: Path) -> None:
    for split in SPLITS:
        (out_dir / f"{split}.txt").write_text(fetch(f"{AGIF}/{AGIF_DIRS[name]}/{split}.txt"), encoding="utf-8")


def download_single_intent(name: str, out_dir: Path) -> None:
    for split, remote in SPLITS.items():
        base = f"{SLOT_GATED}/{name}/{remote}"
        words = fetch(f"{base}/seq.in").splitlines()
        tags = fetch(f"{base}/seq.out").splitlines()
        intents = fetch(f"{base}/label").splitlines()
        if not len(words) == len(tags) == len(intents):
            raise ValueError(f"{name}/{split}: files differ in length")
        with open(out_dir / f"{split}.txt", "w", encoding="utf-8") as f:
            for sentence, tag_line, intent in zip(words, tags, intents):
                tokens, labels = sentence.split(), tag_line.split()
                if len(tokens) != len(labels):
                    raise ValueError(f"{name}/{split}: token/tag mismatch in {sentence!r}")
                for token, label in zip(tokens, labels):
                    f.write(f"{token} {label}\n")
                f.write(intent.strip() + "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="+", required=True, choices=["mixatis", "mixsnips", "atis", "snips"])
    parser.add_argument("--output", default=Path("data/raw"), type=Path)
    args = parser.parse_args()
    for name in args.dataset:
        out_dir = args.output / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (download_agif if name in AGIF_DIRS else download_single_intent)(name, out_dir)
        print(f"{name} -> {out_dir}")


if __name__ == "__main__":
    main()
