"""Shared fixtures: a toy dataset, a tiny BPE tokenizer and a tiny random BART.

Nothing here downloads data or weights, so the suite runs offline on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import BartConfig, PreTrainedTokenizerFast

from gemis.data.labels import LabelSpace
from gemis.experiment import make_loader
from gemis.models.gemis import GEMISModel
from gemis.utils.io import load_raw_dataset
from gemis.utils.training_utils import collect_labels

FIXTURE = Path(__file__).parent / "fixtures" / "toy.txt"


@pytest.fixture(scope="session")
def samples():
    return load_raw_dataset(FIXTURE)


def build_tokenizer(samples) -> PreTrainedTokenizerFast:
    """Byte-level BPE with a small vocabulary, so most words split into several pieces."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=330,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator([" ".join(s.words) for s in samples], trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def build_model(samples, use_aoa=True, use_pointer=True, label_init="subword_mean", seed=0):
    torch.manual_seed(seed)
    tokenizer = build_tokenizer(samples)
    intents, slots = collect_labels(samples)
    config = BartConfig(
        vocab_size=len(tokenizer),
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        max_position_embeddings=128,
        dropout=0.0,
        attention_dropout=0.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
    )
    model = GEMISModel(
        tokenizer=tokenizer,
        label_space=LabelSpace(intents, slots, n_positions=0 if use_pointer else 40),
        bart_config=config,
        mlp_hidden=32,
        use_aoa=use_aoa,
        use_pointer=use_pointer,
        label_init=label_init,
    )
    return model, tokenizer


@pytest.fixture()
def model_and_tokenizer(samples):
    return build_model(samples)


@pytest.fixture()
def train_loader(samples, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    torch.manual_seed(0)
    return make_loader(samples, model, tokenizer, batch_size=4, max_source_length=128, train=True)
