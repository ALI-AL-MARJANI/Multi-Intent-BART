"""Batch collation: pads tensors, passes Python-side metadata through."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_PASSTHROUGH = ("index", "word_starts", "words", "gold")


@dataclass
class GEMISDataCollator:
    pad_token_id: int
    label_pad_id: int = -100
    # Widths are rounded up to a multiple of this value. Padding is masked, so
    # outputs do not change; fewer distinct tensor shapes keeps the MPS graph
    # cache small.
    pad_to_multiple_of: int = 1

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = self._pad([f["input_ids"] for f in features], self.pad_token_id)
        lengths = torch.tensor([len(f["input_ids"]) for f in features])
        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": (torch.arange(input_ids.size(1))[None, :] < lengths[:, None]).long(),
        }
        if "labels" in features[0]:
            batch["labels"] = self._pad([f["labels"] for f in features], self.label_pad_id)
            batch["decoder_input_ids"] = self._pad([f["decoder_input_ids"] for f in features], self.pad_token_id)
        for key in _PASSTHROUGH:
            batch[key] = [f[key] for f in features]
        return batch

    def _pad(self, sequences: list[list[int]], value: int) -> torch.Tensor:
        width = max(len(s) for s in sequences)
        width = -(-width // self.pad_to_multiple_of) * self.pad_to_multiple_of
        return torch.tensor([s + [value] * (width - len(s)) for s in sequences], dtype=torch.long)


TENSOR_KEYS = ("input_ids", "attention_mask", "labels", "decoder_input_ids")
