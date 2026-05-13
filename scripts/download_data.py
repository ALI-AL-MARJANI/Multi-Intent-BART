#!/usr/bin/env python3
"""Download ATIS, SNIPS, MixATIS, or MixSNIPS datasets.

MixATIS / MixSNIPS come from the AGIF repository (LooperXX/AGIF).
ATIS / SNIPS are pulled via the sz128 GitHub repo or HuggingFace.

Usage:
    python scripts/download_data.py --dataset mixatis --output data/raw/
    python scripts/download_data.py --dataset mixsnips --output data/raw/
    python scripts/download_data.py --dataset atis    --output data/raw/
    python scripts/download_data.py --dataset snips   --output data/raw/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

AGIF_BASE = "https://raw.githubusercontent.com/LooperXX/AGIF/master/data"

DATASET_URLS: dict[str, dict[str, str]] = {
    "mixatis": {
        "train.txt": f"{AGIF_BASE}/MixATIS_clean/train.txt",
        "dev.txt":   f"{AGIF_BASE}/MixATIS_clean/dev.txt",
        "test.txt":  f"{AGIF_BASE}/MixATIS_clean/test.txt",
    },
    "mixsnips": {
        "train.txt": f"{AGIF_BASE}/MixSNIPS_clean/train.txt",
        "dev.txt":   f"{AGIF_BASE}/MixSNIPS_clean/dev.txt",
        "test.txt":  f"{AGIF_BASE}/MixSNIPS_clean/test.txt",
    },
}


def download_file(url: str, dest: Path) -> None:
    print(f"  Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def download_agif_dataset(name: str, output_dir: Path) -> None:
    urls = DATASET_URLS[name]
    target_dir = output_dir / name
    for filename, url in urls.items():
        download_file(url, target_dir / filename)
    print(f"✓ {name} saved to {target_dir}")


def download_snips_hf(output_dir: Path) -> None:
    """Download SNIPS via HuggingFace datasets and convert to AGIF format."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package not installed. Run: pip install datasets")
        sys.exit(1)

    print("Downloading SNIPS from HuggingFace (DeepPavlov/snips)...")
    ds = load_dataset("DeepPavlov/snips")

    target_dir = output_dir / "snips"
    target_dir.mkdir(parents=True, exist_ok=True)

    split_map = {"train": "train.txt", "validation": "dev.txt", "test": "test.txt"}
    for hf_split, fname in split_map.items():
        if hf_split not in ds:
            continue
        out_path = target_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            for item in ds[hf_split]:
                tokens = item["tokens"]
                tags = item["ner_tags_str"] if "ner_tags_str" in item else item.get("slot_tags", ["O"] * len(tokens))
                intent = item.get("intent", "unknown")
                for token, tag in zip(tokens, tags):
                    f.write(f"{token}\t{tag}\n")
                f.write(f"{intent}\n\n")
    print(f"✓ SNIPS saved to {target_dir}")


def download_atis_github(output_dir: Path) -> None:
    """Clone the sz128 repo and copy ATIS files."""
    tmp = Path("/tmp/atis_tmp")
    if not tmp.exists():
        print("Cloning sz128/slot_filling_and_intent_detection_of_SLU...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/sz128/slot_filling_and_intent_detection_of_SLU.git",
             str(tmp)],
            check=True,
        )
    atis_src = tmp / "data" / "atis"
    target_dir = output_dir / "atis"
    shutil.copytree(atis_src, target_dir, dirs_exist_ok=True)
    print(f"✓ ATIS saved to {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SLU datasets")
    parser.add_argument(
        "--dataset",
        choices=["mixatis", "mixsnips", "atis", "snips"],
        required=True,
    )
    parser.add_argument("--output", default="data/raw/", type=Path)
    args = parser.parse_args()

    if args.dataset in ("mixatis", "mixsnips"):
        download_agif_dataset(args.dataset, args.output)
    elif args.dataset == "snips":
        download_snips_hf(args.output)
    elif args.dataset == "atis":
        download_atis_github(args.output)


if __name__ == "__main__":
    main()
