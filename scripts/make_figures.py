#!/usr/bin/env python3
"""Draw every figure of the README from the files in ``results/``.

    python scripts/make_figures.py

Writes PNGs to ``results/figures/``. Nothing is drawn for runs that do not exist.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from gemis.utils.io import read_json  # noqa: E402

SCORES = ("intent_entropy", "first_step_msp", "sequence_nll", "intent_energy", "mahalanobis", "knn")
ID_COLOR, OOD_COLORS = "#4c72b0", {"far_clinc": "#c44e52", "near_holdout": "#dd8452"}


def read_scores(path: Path) -> dict[str, dict[str, np.ndarray]]:
    sets: dict[str, dict[str, list[float]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target = sets.setdefault(row["set"], {name: [] for name in SCORES})
            for name in SCORES:
                target[name].append(float(row[name]))
    return {s: {n: np.array(v) for n, v in scores.items()} for s, scores in sets.items()}


def training_curves(results: Path, out: Path) -> None:
    for exp_dir in sorted(p for p in (results / "nlu").glob("*") if p.is_dir()):
        files = sorted(exp_dir.glob("seed*_train.json"))
        if not files:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for f in files:
            run = read_json(f)
            epochs = [h["epoch"] for h in run["history"]]
            axes[0].plot(epochs, [h["train_loss"] for h in run["history"]], label=f"seed {run['seed']}")
            axes[1].plot(epochs, [h["dev_overall_accuracy"] for h in run["history"]], label=f"seed {run['seed']}")
        axes[0].set(xlabel="epoch", ylabel="training loss", yscale="log")
        axes[1].set(xlabel="epoch", ylabel="dev overall accuracy (%)")
        axes[1].legend(frameon=False)
        fig.suptitle(exp_dir.name, fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"training_{exp_dir.name}.png", dpi=150)
        plt.close(fig)


def ood_figures(run_dir: Path, out: Path) -> None:
    sets = read_scores(run_dir / "scores.csv")
    metrics = read_json(run_dir / "metrics.json")
    tag = f"{run_dir.parent.name}_{run_dir.name}"
    ood_sets = [name for name in sets if not name.startswith("id_")]

    # 1. score distributions of the conformal score
    fig, ax = plt.subplots(figsize=(6, 3.4))
    score = "intent_entropy"
    values = np.concatenate([sets[name][score] for name in ("id_test", *ood_sets)])
    bins = np.linspace(values.min(), np.percentile(values, 99.5), 50)
    ax.hist(sets["id_test"][score], bins=bins, alpha=0.6, density=True, color=ID_COLOR, label="in-distribution test")
    for name in ood_sets:
        ax.hist(sets[name][score], bins=bins, alpha=0.6, density=True, color=OOD_COLORS.get(name), label=name)
    threshold = metrics["conformal"][score]["dev_calibrated"]["0.05"]["threshold"]
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label="conformal threshold, alpha = 0.05")
    ax.set(xlabel="mean entropy at intent steps (nats)", ylabel="density", yscale="log")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"ood_scores_{tag}.png", dpi=150)
    plt.close(fig)

    # 2. ROC curves of all scores
    fig, axes = plt.subplots(1, len(ood_sets), figsize=(4.6 * len(ood_sets), 4), squeeze=False)
    for ax, name in zip(axes[0], ood_sets):
        for score in SCORES:
            labels = np.r_[np.zeros(len(sets["id_test"][score])), np.ones(len(sets[name][score]))]
            fpr, tpr, _ = roc_curve(labels, np.r_[sets["id_test"][score], sets[name][score]])
            auroc = metrics["detection"][score][name]["point"]["auroc"]
            ax.plot(fpr, tpr, label=f"{score} ({auroc:.3f})", linewidth=1.2)
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
        ax.set(xlabel="false-positive rate", ylabel="true-positive rate", title=f"{name} vs. id_test")
        ax.legend(frameon=False, fontsize=7, title="score (AUROC)", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"ood_roc_{tag}.png", dpi=150)
    plt.close(fig)

    # 3. conformal validity: false-alarm rate over random calibration/test splits
    alphas = [str(a) for a in metrics["alphas"]]
    fig, axes = plt.subplots(1, len(alphas), figsize=(3.6 * len(alphas), 3.2))
    for ax, alpha in zip(axes, alphas):
        stats = metrics["conformal"]["intent_entropy"]["test_split"]["per_alpha"][alpha]
        ax.hist(stats["fpr_per_split"], bins=30, color=ID_COLOR, alpha=0.8)
        ax.axvline(float(alpha), color="black", linestyle="--", linewidth=1, label="alpha")
        ax.axvline(stats["mean_fpr"], color="#c44e52", linewidth=1, label="mean")
        dev = metrics["conformal"]["intent_entropy"]["dev_calibrated"][alpha]["fpr_id_test"]
        ax.axvline(dev, color="#55a868", linestyle="-.", linewidth=1, label="calibrated on dev")
        ax.set(xlabel="false-alarm rate", title=f"alpha = {alpha}")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel(f"random splits (of {metrics['conformal']['intent_entropy']['test_split']['n_splits']})")
    fig.tight_layout()
    fig.savefig(out / f"conformal_validity_{tag}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    training_curves(args.results, out)
    for run_dir in sorted((args.results / "ood").glob("*/seed*")):
        if (run_dir / "metrics.json").exists():
            ood_figures(run_dir, out)
    print(f"figures -> {out}: {len(list(out.glob('*.png')))} files")


if __name__ == "__main__":
    main()
