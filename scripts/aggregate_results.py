#!/usr/bin/env python3
"""Aggregate per-seed result files into the tables shown in the README.

    python scripts/aggregate_results.py

Reads ``results/nlu/<experiment>/seed*_test.json`` and
``results/ood/<experiment>/seed*/metrics.json``; writes ``summary.json`` and
``summary.md`` in ``results/nlu`` and ``results/ood``. Experiments that have a
config but no result yet are listed as ``TBD (run pending)``. Published
numbers come from ``results/reference/published.json`` and stay in their own
columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from gemis.data.dataset import Frame, bio_to_spans
from gemis.experiment import load_experiment, load_splits
from gemis.ood.calibrator import conformal_threshold
from gemis.training.metrics import compute_metrics, mcnemar_exact
from gemis.utils.io import read_json, write_json
from gemis.utils.training_utils import collect_labels

METRICS = ("slot_f1", "intent_accuracy", "overall_accuracy")
TBD = "TBD (run pending)"


def mean_std(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def fmt(stat: dict[str, Any] | None, digits: int = 2) -> str:
    if stat is None:
        return TBD
    if stat["std"] is None:
        return f"{stat['mean']:.{digits}f} (1 seed)"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def paper_key(cfg: dict[str, Any]) -> str | None:
    """Key of the comparable row of the paper, if there is one."""
    model = cfg["model"]
    if not model["use_pointer"] or model["label_init"] != "subword_mean":
        return None
    if cfg["data"].get("holdout_intents"):
        return None
    key = model["backbone"].split("/")[-1]
    return key if model["use_aoa"] else f"{key}_no-aoa"


def correctness(path: Path) -> list[bool]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["overall_correct"] for line in f]


def unseen_label_mask(cfg: dict[str, Any]) -> list[bool] | None:
    """True for the test utterances with an intent or slot type absent from the training split.

    Returns None when the dataset files are not on disk.
    """
    try:
        splits = load_splits(cfg)
    except FileNotFoundError:
        return None
    intents, slots = collect_labels(splits["train"])
    known = set(intents) | set(slots)
    return [bool(({*s.intents} | {t for _, _, t in bio_to_spans(s.tags)}) - known) for s in splits["test"]]


def subset_metrics(path: Path, keep: list[bool]) -> dict[str, float]:
    """NLU metrics recomputed from a predictions file on the utterances where ``keep`` is true."""
    golds, preds = [], []
    with open(path, encoding="utf-8") as f:
        for line, kept in zip(f, keep, strict=True):
            if kept:
                row = json.loads(line)
                golds.append(Frame(frozenset(row["gold"]["intents"]), frozenset(map(tuple, row["gold"]["slots"]))))
                preds.append(Frame(frozenset(row["pred"]["intents"]), frozenset(map(tuple, row["pred"]["slots"]))))
    return compute_metrics(golds, preds)


def aggregate_nlu(config_dir: Path, results_dir: Path, reference: dict[str, Any]) -> None:
    rows = []
    for config_path in sorted(p for p in config_dir.glob("*.yaml") if not p.name.startswith("_")):
        cfg = load_experiment(config_path)
        name = cfg["experiment_name"]
        files = sorted((results_dir / name).glob("seed*_test.json"))
        runs = [read_json(f) for f in files]
        key = paper_key(cfg)
        paper = reference["gemis"]["results"].get(cfg["dataset"], {}).get(key) if key else None
        row: dict[str, Any] = {
            "experiment": name,
            "dataset": cfg["dataset"],
            "seeds": [r["seed"] for r in runs],
            "n_test": runs[0]["n_utterances"] if runs else None,
            "ours": {m: mean_std([r["metrics"][m] for r in runs]) for m in METRICS} if runs else None,
            "paper": paper,
            "source_files": [str(f) for f in files],
        }
        if runs and paper:
            row["delta"] = {m: row["ours"][m]["mean"] - paper[m] for m in METRICS}
        if runs:
            train_files = [results_dir / name / f"seed{r['seed']}_train.json" for r in runs]
            train_runs = [read_json(f) for f in train_files if f.exists()]
            if len(train_runs) == len(runs):
                row["dev_overall_accuracy_at_selected_epoch"] = mean_std(
                    [t["best_dev_overall_accuracy"] for t in train_runs]
                )
                row["selected_epochs"] = [t["best_epoch"] for t in train_runs]
            unseen = unseen_label_mask(cfg)
            if unseen is not None and len(unseen) == row["n_test"]:
                seen = [not u for u in unseen]
                per_seed = [subset_metrics(results_dir / name / r["predictions_file"], seen) for r in runs]
                row["n_test_with_unseen_label"] = sum(unseen)
                row["seen_labels_only"] = {m: mean_std([s[m] for s in per_seed]) for m in METRICS}
        rows.append(row)

    # paired McNemar tests on overall correctness: each ablation against its full model, per seed
    by_name = {row["experiment"]: row for row in rows}
    tests = []
    for row in rows:
        parent = row["experiment"].rsplit("_", 1)[0]
        if parent not in by_name or parent == row["experiment"]:
            continue
        if row["n_test"] != by_name[parent]["n_test"]:  # another test set (held-out intents): not paired
            continue
        for seed in sorted(set(row["seeds"]) & set(by_name[parent]["seeds"])):
            a = correctness(results_dir / parent / f"seed{seed}_test_predictions.jsonl")
            b = correctness(results_dir / row["experiment"] / f"seed{seed}_test_predictions.jsonl")
            tests.append({"full": parent, "variant": row["experiment"], "seed": seed, **mcnemar_exact(a, b)})

    write_json({"experiments": rows, "mcnemar_overall_accuracy": tests}, results_dir / "summary.json")

    lines = [
        "<!-- generated by scripts/aggregate_results.py; do not edit -->",
        "",
        "Test-set results. *Ours*: mean ± sample standard deviation over seeds. "
        "*Paper*: Li & Zhu (2026), Tables 3 and 5, number of runs not reported.",
        "",
        "| Experiment | Seeds | Slot F1 (ours) | Paper | Intent acc. (ours) | Paper | Overall acc. (ours) | Paper |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = [row["experiment"], str(len(row["seeds"])) if row["seeds"] else "0"]
        for metric in METRICS:
            cells.append(fmt(row["ours"][metric]) if row["ours"] else TBD)
            cells.append(f"{row['paper'][metric]:.1f}" if row["paper"] else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    detail = [row for row in rows if "seen_labels_only" in row and "dev_overall_accuracy_at_selected_epoch" in row]
    if detail:
        lines += [
            "",
            "Dev vs. test, and the test utterances whose intents and slot types all occur in the training split "
            "(the others cannot be fully correct with a training-split label inventory):",
            "",
            "| Experiment | Dev overall acc. (selected epoch) | Test overall acc. "
            "| Test utterances with an unseen label "
            "| Slot F1, seen labels only | Intent acc., seen labels only | Overall acc., seen labels only |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in detail:
            seen = row["seen_labels_only"]
            lines.append(
                f"| {row['experiment']} | {fmt(row['dev_overall_accuracy_at_selected_epoch'])} "
                f"| {fmt(row['ours']['overall_accuracy'])} | {row['n_test_with_unseen_label']} of {row['n_test']} "
                f"| {fmt(seen['slot_f1'])} | {fmt(seen['intent_accuracy'])} | {fmt(seen['overall_accuracy'])} |"
            )
    if tests:
        lines += [
            "",
            "Exact McNemar test on per-utterance overall correctness, ablation vs. full model (same seed):",
            "",
            "| Full model | Variant | Seed | Only full correct | Only variant correct | p-value |",
            "|---|---|---|---|---|---|",
        ]
        for t in tests:
            lines.append(
                f"| {t['full']} | {t['variant']} | {t['seed']} | {t['only_a']} | {t['only_b']} | {t['p_value']:.3g} |"
            )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def read_id_scores(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Scores of the in-distribution sets of a ``scores.csv`` file, in file order."""
    sets: dict[str, dict[str, list[float]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["set"].startswith("id_"):
                target = sets.setdefault(row["set"], {})
                for name, value in row.items():
                    if name not in ("set", "index", "text", "predicted_intents"):
                        target.setdefault(name, []).append(float(value))
    return {s: {n: np.array(v) for n, v in scores.items()} for s, scores in sets.items()}


def false_alarms_by_label_status(exp_dir: Path, config_dir: Path, alpha: float = 0.05) -> dict[str, Any] | None:
    """Dev-calibrated false-alarm rate, split by whether the test utterance has a label unseen in training."""
    config_path = config_dir / f"{exp_dir.name}.yaml"
    unseen = unseen_label_mask(load_experiment(config_path)) if config_path.exists() else None
    if not unseen or not any(unseen):
        return None
    mask = np.array(unseen)
    rates: dict[str, dict[str, list[float]]] = {}
    for scores_path in sorted(exp_dir.glob("seed*/scores.csv")):
        sets = read_id_scores(scores_path)
        if len(next(iter(sets["id_test"].values()))) != len(mask):
            return None
        for score, test in sets["id_test"].items():
            flagged = test > conformal_threshold(sets["id_dev"][score], alpha)
            entry = rates.setdefault(score, {"unseen_label": [], "seen_labels_only": []})
            entry["unseen_label"].append(float(flagged[mask].mean()))
            entry["seen_labels_only"].append(float(flagged[~mask].mean()))
    return {
        "alpha": alpha,
        "n_unseen_label": int(mask.sum()),
        "n_seen_labels_only": int((~mask).sum()),
        "fpr_dev_calibrated": {s: {k: mean_std(v) for k, v in r.items()} for s, r in rates.items()},
    }


def aggregate_ood(results_dir: Path, config_dir: Path) -> None:
    experiments = []
    for exp_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()) if results_dir.exists() else []:
        files = sorted(exp_dir.glob("seed*/metrics.json"))
        if not files:
            continue
        runs = [read_json(f) for f in files]
        detection: dict[str, Any] = {}
        for ood in runs[0]["detection"][next(iter(runs[0]["detection"]))]:
            detection[ood] = {
                score: {
                    metric: mean_std([r["detection"][score][ood]["point"][metric] for r in runs])
                    for metric in ("auroc", "aupr_out", "aupr_in", "fpr_at_95tpr")
                }
                for score in runs[0]["detection"]
            }
        conformal = {
            score: {
                alpha: {
                    "fpr_dev_calibrated": mean_std(
                        [r["conformal"][score]["dev_calibrated"][alpha]["fpr_id_test"] for r in runs]
                    ),
                    "fpr_test_split": mean_std(
                        [r["conformal"][score]["test_split"]["per_alpha"][alpha]["mean_fpr"] for r in runs]
                    ),
                    "tpr_dev_calibrated": {
                        ood: mean_std([r["conformal"][score]["dev_calibrated"][alpha]["tpr"][ood] for r in runs])
                        for ood in detection
                    },
                }
                for alpha in (str(a) for a in runs[0]["alphas"])
            }
            for score in runs[0]["conformal"]
        }
        experiments.append(
            {
                "experiment": exp_dir.name,
                "seeds": [f.parent.name for f in files],
                "set_sizes": runs[0]["set_sizes"],
                "detection": detection,
                "conformal": conformal,
                "false_alarms_by_label_status": false_alarms_by_label_status(exp_dir, config_dir),
                "source_files": [str(f) for f in files],
            }
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    write_json({"experiments": experiments}, results_dir / "summary.json")
    lines = ["<!-- generated by scripts/aggregate_results.py; do not edit -->", ""]
    if not experiments:
        lines.append(TBD)
    for exp in experiments:
        n_seeds = len(exp["seeds"])
        for ood, scores in exp["detection"].items():
            lines += [
                f"**{exp['experiment']}**: `{ood}` (n={exp['set_sizes'][ood]}) vs. `id_test` "
                f"(n={exp['set_sizes']['id_test']}), {n_seeds} seed(s). OOD is the positive class.",
                "",
                "| Score | AUROC ↑ | AUPR-out ↑ | AUPR-in ↑ | FPR@95TPR ↓ |",
                "|---|---|---|---|---|",
            ]
            for score, m in scores.items():
                lines.append(
                    f"| {score} | {fmt(m['auroc'], 4)} | {fmt(m['aupr_out'], 4)} | "
                    f"{fmt(m['aupr_in'], 4)} | {fmt(m['fpr_at_95tpr'], 4)} |"
                )
            lines.append("")
        lines += [
            f"**{exp['experiment']}**: conformal false-alarm rate on `id_test` (target: alpha).",
            "",
            "| Score | alpha | Calibrated on dev | Random splits of test (mean of 1000) |",
            "|---|---|---|---|",
        ]
        for score, per_alpha in exp["conformal"].items():
            for alpha, c in per_alpha.items():
                lines.append(
                    f"| {score} | {alpha} | {fmt(c['fpr_dev_calibrated'], 4)} | {fmt(c['fpr_test_split'], 4)} |"
                )
        lines.append("")
        by_status = exp["false_alarms_by_label_status"]
        if by_status:
            lines += [
                f"**{exp['experiment']}**: dev-calibrated false-alarm rate at alpha = {by_status['alpha']}, by whether "
                "the `id_test` utterance carries an intent or slot type that never occurs in the training split.",
                "",
                f"| Score | With an unseen label (n={by_status['n_unseen_label']}) "
                f"| Seen labels only (n={by_status['n_seen_labels_only']}) |",
                "|---|---|---|",
            ]
            for score, r in by_status["fpr_dev_calibrated"].items():
                lines.append(f"| {score} | {fmt(r['unseen_label'], 4)} | {fmt(r['seen_labels_only'], 4)} |")
            lines.append("")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    reference = read_json(args.results / "reference" / "published.json")
    aggregate_nlu(args.configs, args.results / "nlu", reference)
    print()
    aggregate_ood(args.results / "ood", args.configs)


if __name__ == "__main__":
    main()
