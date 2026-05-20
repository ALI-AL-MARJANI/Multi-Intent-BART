"""DataCollator for GEMIS: pads variable-length sequences in a batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast


@dataclass
class GEMISDataCollator:
    tokenizer: PreTrainedTokenizerFast
    label_pad_token_id: int = -100  # ignored by CrossEntropyLoss

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id

        batch: dict[str, Any] = {
            "input_ids":        self._pad([f["input_ids"] for f in features], pad_id),
            "attention_mask":   self._pad([f["attention_mask"] for f in features], 0),
            "labels":           self._pad([f["labels"] for f in features], self.label_pad_token_id),
            "decoder_input_ids": self._pad([f["decoder_input_ids"] for f in features], pad_id),
            "pointer_targets":  self._pad([f["pointer_targets"] for f in features], -1),
        }

        # words is a list[list[str]] — kept as Python lists for word-boundary decoding
        if "words" in features[0]:
            batch["words"] = [f["words"] for f in features]

        return batch

    @staticmethod
    def _pad(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(s) for s in sequences)
        padded = [s + [pad_value] * (max_len - len(s)) for s in sequences]
        return torch.tensor(padded, dtype=torch.long)
