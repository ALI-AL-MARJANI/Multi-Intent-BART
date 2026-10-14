from __future__ import annotations

from pathlib import Path

import pytest

from gemis.experiment import load_experiment

CONFIGS = sorted(p for p in Path("configs").glob("*.yaml") if not p.name.startswith("_"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_is_complete(path):
    cfg = load_experiment(path)
    assert cfg["experiment_name"] == path.stem
    assert "base" not in cfg
    for section, keys in {
        "model": ["backbone", "mlp_hidden", "use_aoa", "use_pointer", "label_init"],
        "data": ["train_path", "dev_path", "test_path", "max_source_length", "holdout_intents"],
        "training": ["learning_rate", "batch_size", "max_epochs", "early_stopping_patience", "device"],
        "ood": ["far_ood_path", "feature_bank_size", "knn_k"],
        "output": ["checkpoint_dir", "results_dir"],
    }.items():
        assert set(keys) <= set(cfg[section]), (section, path)
    assert len(cfg["seeds"]) >= 3


def test_at_least_the_paper_setting_and_ablations_exist():
    names = {p.stem for p in CONFIGS}
    assert {"mixatis_bart-large", "mixsnips_bart-large", "mixatis_bart-base_no-aoa"} <= names
