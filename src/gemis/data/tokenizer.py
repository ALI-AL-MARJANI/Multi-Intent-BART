"""Vocabulary extension utilities for GEMIS.

Adds intent and slot-type tokens to the BART tokenizer, then initializes
their embeddings as the mean of the token's constituent subwords — following
the paper's prescription to avoid random cold-start embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerFast


def extend_tokenizer_vocab(
    tokenizer: PreTrainedTokenizerFast,
    intent_labels: list[str],
    slot_labels: list[str],
) -> tuple[list[str], list[str]]:
    """Add intent and slot tokens to the tokenizer.

    Returns the lists of new intent tokens and new slot tokens actually added
    (tokens already present in the vocab are skipped).
    """
    new_tokens = []
    new_intent_tokens: list[str] = []
    new_slot_tokens: list[str] = []

    for label in intent_labels:
        tok = f"<intent:{label}>"
        if tok not in tokenizer.get_vocab():
            new_tokens.append(tok)
            new_intent_tokens.append(tok)

    for label in slot_labels:
        tok = f"<slot:{label}>"
        if tok not in tokenizer.get_vocab():
            new_tokens.append(tok)
            new_slot_tokens.append(tok)

    if new_tokens:
        tokenizer.add_tokens(new_tokens)

    return new_intent_tokens, new_slot_tokens


def initialize_new_token_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    new_tokens: list[str],
) -> None:
    """Initialize new token embeddings as mean of their subword pieces.

    This function mutates the model's embedding table in-place.  Must be called
    AFTER `model.resize_token_embeddings(len(tokenizer))`.
    """
    model.resize_token_embeddings(len(tokenizer))

    embed_weight: nn.Parameter = model.get_input_embeddings().weight

    with torch.no_grad():
        for token in new_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)

            # Tokenize the raw label text (strip the <intent:> / <slot:> wrapper)
            if token.startswith("<intent:") or token.startswith("<slot:"):
                inner = token.split(":", 1)[1].rstrip(">")
            else:
                inner = token

            subword_ids: list[int] = tokenizer.encode(
                inner, add_special_tokens=False
            )

            if subword_ids:
                mean_vec = embed_weight[subword_ids].mean(dim=0)
                embed_weight[token_id] = mean_vec
            # else: leave the randomly-initialized embedding as-is
