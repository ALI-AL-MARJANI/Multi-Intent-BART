#!/usr/bin/env python3
"""Run Algorithm 1 (NSP-based multi-intent construction) to create MultiATIS/MultiSNIPS.

Usage:
    python scripts/construct_dataset.py \\
        --input  data/raw/atis/train.txt \\
        --output data/processed/multiatis/ \\
        --tau    0.5 \\
        --bert   bert-base-uncased \\
        --device cpu
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gemis.data.construct import MultiIntentConstructor
from gemis.utils.io import load_raw_dataset, write_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct multi-intent dataset via BERT NSP")
    parser.add_argument("--input",  required=True, type=Path, help="Input single-intent .txt file")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--split",  default="train", choices=["train", "dev", "test"])
    parser.add_argument("--tau",    type=float, default=0.5, help="NSP coherence threshold")
    parser.add_argument("--bert",   default="bert-base-uncased", help="BERT model for NSP")
    parser.add_argument("--device", default="cpu", help="torch device (cpu / cuda)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading base dataset from %s", args.input)
    dataset = load_raw_dataset(args.input)
    logger.info("Loaded %d samples", len(dataset))

    constructor = MultiIntentConstructor(
        tau=args.tau,
        bert_model=args.bert,
        device=args.device,
        seed=args.seed,
    )

    multi_dataset = constructor.construct(dataset)

    out_path = args.output / f"{args.split}.txt"
    write_dataset(multi_dataset, out_path)
    logger.info("Written %d samples to %s", len(multi_dataset), out_path)

    # quick stats
    n_intents = [len(s.intents) for s in multi_dataset]
    for k in [1, 2, 3]:
        count = sum(n == k for n in n_intents)
        logger.info("  %d intent(s): %d samples (%.1f%%)", k, count, 100 * count / len(multi_dataset))


if __name__ == "__main__":
    main()
