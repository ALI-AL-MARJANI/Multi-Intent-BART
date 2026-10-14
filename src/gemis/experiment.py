"""Build tokenizer, model and data loaders from an experiment config."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from gemis.data.collator import GEMISDataCollator
from gemis.data.dataset import GEMISDataset
from gemis.data.labels import LabelSpace
from gemis.models.gemis import GEMISModel
from gemis.utils.io import RawSampleIO, load_config
from gemis.utils.training_utils import collect_labels, load_split


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving ``base: other.yaml`` inheritance."""
    path = Path(path)
    cfg = load_config(path)
    base = cfg.pop("base", None)
    if base is not None:
        cfg = _merge(load_experiment(path.parent / base), cfg)
    return cfg


def run_name(cfg: dict[str, Any], seed: int) -> str:
    return f"{cfg['experiment_name']}/seed{seed}"


def build_label_space(train: list[RawSampleIO], cfg: dict[str, Any]) -> LabelSpace:
    """Label inventory from the training split only."""
    intents, slots = collect_labels(train)
    n_positions = 0
    if not cfg["model"]["use_pointer"]:
        n_positions = cfg["data"]["max_position_classes"]
    return LabelSpace(intents, slots, n_positions)


def build_model(cfg: dict[str, Any], train: list[RawSampleIO]) -> tuple[GEMISModel, PreTrainedTokenizerFast]:
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["backbone"])
    model = GEMISModel(
        tokenizer=tokenizer,
        label_space=build_label_space(train, cfg),
        backbone=model_cfg["backbone"],
        mlp_hidden=model_cfg["mlp_hidden"],
        use_aoa=model_cfg["use_aoa"],
        use_pointer=model_cfg["use_pointer"],
        label_init=model_cfg["label_init"],
    )
    return model, tokenizer


def make_loader(
    samples: list[RawSampleIO],
    model: GEMISModel,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    max_source_length: int,
    train: bool = False,
    pad_to_multiple_of: int = 1,
) -> DataLoader:
    dataset = GEMISDataset(
        samples,
        tokenizer,
        model.label_space,
        model.label_token_ids.tolist(),
        model.decoder_start_token_id,
        max_source_length=max_source_length,
        use_pointer=model.use_pointer,
        with_targets=train,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=GEMISDataCollator(pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=pad_to_multiple_of),
    )


def load_splits(cfg: dict[str, Any]) -> dict[str, list[RawSampleIO]]:
    """Train / dev / test with held-out-intent utterances removed."""
    data = cfg["data"]
    holdout = data.get("holdout_intents") or []
    return {name: load_split(data[f"{name}_path"], holdout) for name in ("train", "dev", "test")}
