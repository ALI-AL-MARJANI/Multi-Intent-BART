"""Vocabulary extension: one tokenizer token per label class."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from gemis.data.labels import LabelSpace


def add_label_tokens(tokenizer: PreTrainedTokenizerFast, label_space: LabelSpace) -> list[int]:
    """Add the label tokens to the tokenizer and return their ids in class order."""
    tokens = label_space.token_strings()
    vocab = tokenizer.get_vocab()
    new_tokens = [tok for tok in tokens if tok not in vocab]
    if new_tokens:
        tokenizer.add_tokens(new_tokens, special_tokens=True)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if tokenizer.unk_token_id in ids:
        raise ValueError("a label token resolved to <unk>")
    return ids


def init_label_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    label_space: LabelSpace,
    label_token_ids: list[int],
    mode: str = "subword_mean",
) -> None:
    """Resize the embedding table and initialise the label rows.

    ``subword_mean``: mean of the embeddings of the label text's subwords
    (``atis_flight`` -> mean of its BPE pieces). ``random``: keep the random
    initialisation of ``resize_token_embeddings`` (ablation).
    """
    if mode not in ("subword_mean", "random"):
        raise ValueError(f"unknown label_init {mode!r}")
    n_original = model.get_input_embeddings().weight.size(0)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    if mode == "random":
        return
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        for token_id, text in zip(label_token_ids, label_space.init_texts()):
            if text is None or token_id < n_original:
                continue
            pieces = [i for i in tokenizer.encode(text.replace("_", " "), add_special_tokens=False) if i < n_original]
            if pieces:
                weight[token_id] = weight[pieces].mean(dim=0)
