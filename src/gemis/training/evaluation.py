"""Greedy-decoding evaluation loop shared by the trainer and the scripts."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from gemis.data.dataset import Frame
from gemis.data.decoding import decode_frame
from gemis.models.gemis import GEMISModel
from gemis.utils.training_utils import release_memory


@torch.no_grad()
def predict_frames(
    model: GEMISModel,
    loader: DataLoader,
    device: torch.device,
    max_new_tokens: int = 96,
) -> tuple[list[Frame], list[Frame], list[list[str]]]:
    """Decode every utterance of ``loader``. Returns (gold, predicted, words)."""
    model.eval()
    golds: list[Frame] = []
    preds: list[Frame] = []
    words: list[list[str]] = []
    for batch in loader:
        output = model.generate(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
        )
        for sequence, word_starts in zip(output.sequences, batch["word_starts"]):
            preds.append(decode_frame(sequence, model.label_space, word_starts))
        golds.extend(batch["gold"])
        words.extend(batch["words"])
        release_memory(device)
    return golds, preds, words
