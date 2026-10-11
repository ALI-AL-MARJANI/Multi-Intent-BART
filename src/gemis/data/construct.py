"""Algorithm 1 of Li & Zhu (2026): multi-intent utterances from single-intent data.

For each single-intent utterance, a target number of intents ``n`` is drawn
from {1, 2, 3} with probabilities (0.3, 0.5, 0.2). Candidate utterances with
an intent not yet present are appended while BERT's next-sentence-prediction
head scores the concatenation above a threshold ``tau``.

The paper does not report ``tau`` or the connector between utterances, and its
MultiATIS / MultiSNIPS files are not public. Datasets built with this module
are therefore **not** the paper's datasets, and results on them are not
comparable with the paper's Table 4.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable

from gemis.utils.io import RawSampleIO, RawToken

logger = logging.getLogger(__name__)

NspScorer = Callable[[str, str], float]


def bert_nsp_scorer(model_name: str = "bert-base-uncased", device: str = "cpu") -> NspScorer:
    """P(second sentence follows the first) under BERT's NSP head."""
    import torch
    from transformers import BertForNextSentencePrediction, BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForNextSentencePrediction.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def score(first: str, second: str) -> float:
        inputs = tokenizer(first, second, return_tensors="pt", truncation=True, max_length=128).to(device)
        return torch.softmax(model(**inputs).logits, dim=-1)[0, 0].item()  # index 0 = "is next"

    return score


class MultiIntentConstructor:
    """Builds utterances with one to three intents.

    Args:
        nsp_scorer: coherence score of ``(current utterance, candidate)``.
        tau: a candidate is appended when its score exceeds this threshold.
        connectors: word sequences inserted between two utterances (tagged ``O``).
        max_retries: candidates tried before giving up on adding an intent.
        seed: seed of the sampling.
    """

    INTENT_COUNTS = (1, 2, 3)
    INTENT_PROBS = (0.3, 0.5, 0.2)

    def __init__(
        self,
        nsp_scorer: NspScorer,
        tau: float = 0.5,
        connectors: tuple[str, ...] = ("and", "and then"),
        max_retries: int = 20,
        seed: int = 42,
    ) -> None:
        self.nsp_scorer = nsp_scorer
        self.tau = tau
        self.connectors = connectors
        self.max_retries = max_retries
        self.rng = random.Random(seed)

    def construct(self, dataset: list[RawSampleIO]) -> list[RawSampleIO]:
        results = []
        for sample in dataset:
            target_n = self.rng.choices(self.INTENT_COUNTS, weights=self.INTENT_PROBS, k=1)[0]
            results.append(self._grow(sample, target_n, dataset))
        logger.info("constructed %d utterances (tau=%.2f)", len(results), self.tau)
        return results

    def _grow(self, base: RawSampleIO, target_n: int, pool: list[RawSampleIO]) -> RawSampleIO:
        current = RawSampleIO(tokens=list(base.tokens), intents=list(base.intents))
        while len(current.intents) < target_n:
            for _ in range(self.max_retries):
                candidate = self.rng.choice(pool)
                if set(candidate.intents) & set(current.intents):
                    continue
                if self.nsp_scorer(" ".join(current.words), " ".join(candidate.words)) > self.tau:
                    current = self.merge(current, candidate)
                    break
            else:
                break  # no coherent candidate found; keep the current number of intents
        return current

    def merge(self, first: RawSampleIO, second: RawSampleIO) -> RawSampleIO:
        connector = [RawToken(word, "O") for word in self.rng.choice(self.connectors).split()]
        return RawSampleIO(
            tokens=list(first.tokens) + connector + list(second.tokens),
            intents=list(dict.fromkeys(first.intents + second.intents)),
        )
