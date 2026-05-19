"""OODScorer: computes a non-conformity score from GEMIS decoder entropy.

The key idea
------------
In a generative seq2seq model like GEMIS, there is no single classification
logit vector to read uncertainty from. Instead, we hook into the *decoding
process* itself: at every step where the model predicts an intent token, we
measure how uncertain it is (high entropy = model is spread across many tokens
= likely OOD).

Non-conformity score s(x)
-------------------------
For a given utterance x, let T be the set of decode steps where the model
predicted an intent token. At each step t ∈ T:

    H_t = -sum_v  p_tv * log(p_tv + ε)      (Shannon entropy over N+L logits)

The final score is:

    s(x) = mean(H_t for t in T)              if T is non-empty
         = H_max                              fallback when no intent predicted
                                              (signals deep OOD)

Higher score → model is more uncertain → sample more likely to be OOD.

Usage
-----
    result = model.generate(..., return_scores=True)
    score  = OODScorer.score(result["intent_entropies"][0])
"""

from __future__ import annotations

import math


class OODScorer:
    """Stateless utility that converts per-step intent entropies to a scalar score."""

    # Fallback score when no intent token was generated at all.
    # A very high entropy (log of a large vocab) signals the model produced
    # something completely outside the expected format — strongly OOD.
    NO_INTENT_SCORE: float = math.log(50_000)  # ≈ ln(vocab_size)

    @staticmethod
    def score(intent_entropies: list[float]) -> float:
        """Compute the non-conformity score for one sample.

        Args:
            intent_entropies: list of per-step entropy values collected at each
                intent-position decode step (from generate(return_scores=True)).
                Empty if no intent token was generated.

        Returns:
            Scalar non-conformity score ≥ 0.  Higher = more uncertain = more OOD.
        """
        if not intent_entropies:
            return OODScorer.NO_INTENT_SCORE
        return sum(intent_entropies) / len(intent_entropies)

    @staticmethod
    def score_batch(intent_entropies_batch: list[list[float]]) -> list[float]:
        """Vectorised version over a full batch."""
        return [OODScorer.score(e) for e in intent_entropies_batch]
