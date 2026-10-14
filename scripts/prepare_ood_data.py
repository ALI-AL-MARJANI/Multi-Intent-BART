#!/usr/bin/env python3
"""Build the far-OOD evaluation set from CLINC150 (Larson et al., 2019).

    python scripts/prepare_ood_data.py

Writes ``data/ood/clinc150_oos_test.txt``: the 1,000 out-of-scope test queries
of the ``data_full`` release, deduplicated, one per line. Source:
https://github.com/clinc/oos-eval (CC BY 3.0).
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("data/ood/clinc150_oos_test.txt"))
    args = parser.parse_args()

    with urllib.request.urlopen(URL, timeout=60) as response:
        data = json.load(response)
    queries = list(dict.fromkeys(text.strip() for text, label in data["oos_test"] if label == "oos"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(queries) + "\n", encoding="utf-8")
    print(f"{len(queries)} unique out-of-scope queries (of {len(data['oos_test'])}) -> {args.output}")


if __name__ == "__main__":
    main()
