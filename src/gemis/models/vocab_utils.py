"""Vocabulary utilities — convenience wrappers used by GEMISModel.__init__."""

from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizerFast

from gemis.data.tokenizer import extend_tokenizer_vocab, initialize_new_token_embeddings


def setup_vocab(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    intent_labels: list[str],
    slot_labels: list[str],
) -> tuple[list[str], list[str]]:
    """Extend vocab, resize embeddings, and init new embeddings from subwords.

    Returns:
        new_intent_tokens: list of added intent token strings.
        new_slot_tokens:   list of added slot token strings.
    """
    new_intent_tokens, new_slot_tokens = extend_tokenizer_vocab(
        tokenizer, intent_labels, slot_labels
    )
    # Always resize and init, even if extend_tokenizer_vocab added nothing new.
    # This handles the case where the tokenizer was already extended before the
    # model was created (e.g., a notebook cell that calls extend_tokenizer_vocab
    # standalone), which would leave all_new empty and skip the resize — causing
    # an IndexError when decoder_input_ids contains intent/slot token IDs that
    # exceed the original embedding table size.
    all_tokens = (
        [f"<intent:{label}>" for label in intent_labels]
        + [f"<slot:{label}>" for label in slot_labels]
    )
    initialize_new_token_embeddings(model, tokenizer, all_tokens)
    return new_intent_tokens, new_slot_tokens
