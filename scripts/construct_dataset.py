#!/usr/bin/env python3
"""Build a multi-intent split from a single-intent one (Algorithm 1 of the paper).

    python scripts/download_data.py --dataset snips
    python scripts/construct_dataset.py --input data/raw/snips/train.txt \\
        --output data/processed/multisnips/train.txt --tau 0.5

The result is not the paper's MultiATIS / MultiSNIPS (see ``gemis/data/construct.py``).
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from gemis.data.construct import MultiIntentConstructor, bert_nsp_scorer
from gemis.utils.io import load_raw_dataset, write_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("construct_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tau", type=float, default=0.5, help="NSP threshold (not reported in the paper)")
    parser.add_argument("--bert", default="bert-base-uncased")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_raw_dataset(args.input)
    constructor = MultiIntentConstructor(bert_nsp_scorer(args.bert, args.device), tau=args.tau, seed=args.seed)
    multi = constructor.construct(dataset)
    write_dataset(multi, args.output)

    counts = Counter(len(s.intents) for s in multi)
    for k in sorted(counts):
        logger.info("%d intent(s): %d utterances (%.1f%%)", k, counts[k], 100 * counts[k] / len(multi))
    logger.info("wrote %d utterances to %s", len(multi), args.output)


if __name__ == "__main__":
    main()
