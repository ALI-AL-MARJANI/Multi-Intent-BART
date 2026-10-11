"""Out-of-distribution scores for a generative SLU model.

Every score is oriented so that **higher means more likely OOD**.

Decoder scores (from one greedy decoding pass):

* ``intent_entropy``: mean entropy of the output distribution over the steps
  that emit an intent; the first step if no intent was emitted.
* ``first_step_msp``: 1 - max probability at the first step (maximum softmax
  probability, Hendrycks & Gimpel 2017).
* ``sequence_nll``: length-normalised negative log-likelihood of the greedy
  output (Ren et al. 2023 use the analogous perplexity score).
* ``intent_energy``: -logsumexp of the intent logits at the first step
  (energy score, Liu et al. 2020, with T = 1).

Encoder-feature scores (need training-set features):

* ``mahalanobis``: Mahalanobis distance of the mean-pooled encoder state to
  the training distribution. A single Gaussian is used because utterances are
  multi-label; this is the class-agnostic variant of Lee et al. (2018).
* ``knn``: distance to the k-th nearest training feature after L2
  normalisation (Sun et al. 2022).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from gemis.data.decoding import decode_frame
from gemis.data.labels import LabelSpace
from gemis.models.gemis import GEMISModel, GenerationOutput

DECODER_SCORES = ("intent_entropy", "first_step_msp", "sequence_nll", "intent_energy")
FEATURE_SCORES = ("mahalanobis", "knn")


def decoder_scores(output: GenerationOutput, label_space: LabelSpace) -> dict[str, np.ndarray]:
    """The four decoder scores for every utterance of a decoded batch."""
    intent_entropy = []
    for sequence, entropies in zip(output.sequences, output.entropies):
        n_intents = 0
        while n_intents < len(sequence) and label_space.is_intent(sequence[n_intents]):
            n_intents += 1
        intent_entropy.append(float(np.mean(entropies[: max(n_intents, 1)])))

    intent_logits = output.first_step_label_logits[:, label_space.intent_ids]
    return {
        "intent_entropy": np.array(intent_entropy),
        "first_step_msp": 1.0 - output.first_step_max_prob.numpy(),
        "sequence_nll": np.array([-float(np.mean(lp)) for lp in output.log_probs]),
        "intent_energy": -torch.logsumexp(intent_logits, dim=-1).numpy(),
    }


class FeatureScorer:
    """Mahalanobis and k-NN scores against a bank of training features."""

    def __init__(self, k: int = 10, shrinkage: float = 1e-3) -> None:
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, features: np.ndarray) -> FeatureScorer:
        features = features.astype(np.float64)
        self.mean = features.mean(axis=0)
        centered = features - self.mean
        cov = centered.T @ centered / max(len(features) - 1, 1)
        cov += self.shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        self.precision = np.linalg.pinv(cov)
        self.bank = torch.from_numpy(self._normalise(features)).float()
        return self

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)

    def score(self, features: np.ndarray, exclude_self: bool = False) -> dict[str, np.ndarray]:
        """``exclude_self`` skips the nearest neighbour (use when scoring the bank itself)."""
        centered = features.astype(np.float64) - self.mean
        mahalanobis = np.sqrt(np.einsum("nd,de,ne->n", centered, self.precision, centered).clip(min=0))

        queries = torch.from_numpy(self._normalise(features.astype(np.float64))).float()
        k = min(self.k + int(exclude_self), len(self.bank))
        knn = []
        for chunk in queries.split(1024):
            distances = torch.cdist(chunk, self.bank)
            knn.append(distances.topk(k, dim=1, largest=False).values[:, -1])
        return {"mahalanobis": mahalanobis, "knn": torch.cat(knn).numpy().astype(np.float64)}


@torch.no_grad()
def score_loader(
    model: GEMISModel,
    loader: DataLoader,
    device: torch.device,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Decode every utterance once and collect scores, features and predictions.

    This is the only scoring path: calibration, in-distribution and OOD sets
    all go through it, so their scores are comparable by construction.
    """
    model.eval()
    scores: dict[str, list[np.ndarray]] = {name: [] for name in DECODER_SCORES}
    features, frames, texts = [], [], []
    for batch in loader:
        output = model.generate(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
        )
        for name, values in decoder_scores(output, model.label_space).items():
            scores[name].append(values)
        features.append(output.encoder_mean.numpy())
        for sequence, word_starts in zip(output.sequences, batch["word_starts"]):
            frames.append(decode_frame(sequence, model.label_space, word_starts))
        texts.extend(" ".join(words) for words in batch["words"])
    return {
        "scores": {name: np.concatenate(values) for name, values in scores.items()},
        "features": np.concatenate(features),
        "frames": frames,
        "texts": texts,
    }
