"""Algorithm 1: NSP-based Multi-Intent Dataset Construction.

Reference: Li & Zhu (2026), §4 and Algorithm 1.

The algorithm:
  1. For each base (single-intent) utterance u_m, sample a target intent
     count n ∈ {1, 2, 3} with probabilities (0.3, 0.5, 0.2).
  2. Until |intents(u_m)| == n:
       a. Sample a candidate utterance u_c with a *different* intent class.
       b. Compute P_NSP(u_m || u_c) using BERT's Next Sentence Prediction head.
       c. If P_NSP > τ, concatenate: u_m ← u_m || u_c (merge tokens + BIO tags).
  3. Collect all constructed multi-intent utterances.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

import torch
from transformers import BertForNextSentencePrediction, BertTokenizerFast

from gemis.utils.io import RawSampleIO, RawToken

logger = logging.getLogger(__name__)


class MultiIntentConstructor:
    """Constructs multi-intent utterances from a single-intent dataset.

    Args:
        tau:          NSP coherence threshold (default 0.5).
        bert_model:   HuggingFace model id for BERT NSP.
        max_retries:  how many candidates to try before skipping concatenation.
        device:       torch device for BERT inference.
        seed:         random seed for reproducibility.
    """

    INTENT_PROBS = [0.3, 0.5, 0.2]   # P(n=1), P(n=2), P(n=3)
    INTENT_COUNTS = [1, 2, 3]

    def __init__(
        self,
        tau: float = 0.5,
        bert_model: str = "bert-base-uncased",
        max_retries: int = 20,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.tau = tau
        self.max_retries = max_retries
        self.rng = random.Random(seed)

        logger.info("Loading BERT NSP model: %s", bert_model)
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model)
        self.bert_model = BertForNextSentencePrediction.from_pretrained(bert_model)
        self.device = torch.device(device)
        self.bert_model.to(self.device)
        self.bert_model.eval()

    # ── public API ─────────────────────────────────────────────────────────────

    def construct(self, dataset: list[RawSampleIO]) -> list[RawSampleIO]:
        """Build and return a multi-intent dataset.

        Args:
            dataset: list of single-intent RawSampleIO objects.

        Returns:
            New list of RawSampleIO objects with 1–3 intents each.
        """
        # index by intent for sampling
        intent_to_samples: dict[str, list[RawSampleIO]] = defaultdict(list)
        for sample in dataset:
            for intent in sample.intents:
                intent_to_samples[intent].append(sample)

        all_intents = list(intent_to_samples.keys())
        results: list[RawSampleIO] = []

        for sample in dataset:
            target_n = self.rng.choices(
                self.INTENT_COUNTS, weights=self.INTENT_PROBS, k=1
            )[0]

            constructed = self._grow(sample, target_n, intent_to_samples, all_intents)
            results.append(constructed)

        logger.info(
            "Constructed %d multi-intent samples (τ=%.2f)", len(results), self.tau
        )
        return results

    # ── internal ───────────────────────────────────────────────────────────────

    def _grow(
        self,
        base: RawSampleIO,
        target_n: int,
        intent_to_samples: dict[str, list[RawSampleIO]],
        all_intents: list[str],
    ) -> RawSampleIO:
        """Iteratively concatenate utterances until intent count reaches target_n."""
        current = RawSampleIO(
            tokens=list(base.tokens),
            intents=list(base.intents),
        )

        while len(current.intents) < target_n:
            # sample a candidate with a new intent
            current_intent_set = set(current.intents)
            other_intents = [i for i in all_intents if i not in current_intent_set]

            if not other_intents:
                break  # can't grow further

            # Build filtered candidate set Û_s (different intent from current)
            candidate_pool: list[RawSampleIO] = []
            for intent in other_intents:
                candidate_pool.extend(intent_to_samples[intent])

            # Shuffle and try up to max_retries candidates (Algorithm 1: first match wins)
            self.rng.shuffle(candidate_pool)
            concatenated = False
            for candidate in candidate_pool[: self.max_retries]:
                nsp_score = self._nsp_score(current, candidate)
                if nsp_score > self.tau:
                    current = self._merge(current, candidate)
                    concatenated = True
                    break

            if not concatenated:
                break  # no coherent candidate found — keep current intent count

        return current

    def _nsp_score(self, sent_a: RawSampleIO, sent_b: RawSampleIO) -> float:
        """Return P(is_next=True) from BERT NSP for the concatenation of two utterances."""
        text_a = " ".join(t.word for t in sent_a.tokens)
        text_b = " ".join(t.word for t in sent_b.tokens)

        inputs = self.bert_tokenizer(
            text_a, text_b,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            logits = self.bert_model(**inputs).logits  # (1, 2): [IsNext, NotNext]

        p_is_next = torch.softmax(logits, dim=-1)[0, 0].item()
        return p_is_next

    def _merge(self, a: RawSampleIO, b: RawSampleIO) -> RawSampleIO:
        """Concatenate two utterances with a natural connector word.

        The connector is sampled from ["and", "and then"] which mirrors the
        natural connectors observed in Table 1 of the paper.
        """
        connector = self.rng.choice(["and", "and then"])
        sep_tokens = [RawToken(word=w, bio_tag="O") for w in connector.split()]
        merged_tokens = list(a.tokens) + sep_tokens + list(b.tokens)
        merged_intents = list(dict.fromkeys(a.intents + b.intents))
        return RawSampleIO(tokens=merged_tokens, intents=merged_intents)
