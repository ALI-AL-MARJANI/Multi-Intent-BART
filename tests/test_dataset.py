from __future__ import annotations

import pytest

from gemis.data.dataset import bio_to_spans, gold_frame
from gemis.data.decoding import decode_frame
from gemis.data.labels import LabelSpace
from gemis.experiment import make_loader
from gemis.utils.io import load_raw_dataset, write_dataset
from tests.conftest import build_model


def test_bio_to_spans_exclusive_end():
    tags = ["O", "B-a", "I-a", "O", "B-b"]
    assert bio_to_spans(tags) == [(1, 3, "a"), (4, 5, "b")]


def test_bio_to_spans_conlleval_edge_cases():
    assert bio_to_spans(["B-a", "B-a"]) == [(0, 1, "a"), (1, 2, "a")]
    assert bio_to_spans(["I-a", "I-a"]) == [(0, 2, "a")]  # orphan I- opens a chunk
    assert bio_to_spans(["B-a", "I-b"]) == [(0, 1, "a"), (1, 2, "b")]  # type change splits
    assert bio_to_spans([]) == []


def test_label_space_roundtrip():
    space = LabelSpace(["i1", "i2"], ["s1"], n_positions=3)
    assert len(space) == 1 + 2 + 1 + 3
    assert len(space.token_strings()) == len(space) == len(space.init_texts())
    assert space.decode(space.eos_id).kind == "eos"
    assert space.decode(space.intent_id("i2")).value == "i2"
    assert space.decode(space.slot_id("s1")).value == "s1"
    assert space.decode(space.pos_id(2)).value == 2
    assert space.is_intent(space.intent_id("i1")) and not space.is_intent(space.slot_id("s1"))
    with pytest.raises(KeyError):
        space.pos_id(3)


@pytest.mark.parametrize("use_pointer", [True, False])
def test_gold_targets_decode_back_to_the_gold_frame(samples, use_pointer):
    """Oracle check: a model that outputs the training targets scores 100%."""
    model, tokenizer = build_model(samples, use_pointer=use_pointer)
    loader = make_loader(samples, model, tokenizer, batch_size=3, max_source_length=128, train=True)
    n_labels = len(model.label_space)
    for item, sample in zip(loader.dataset.items, samples):
        labels = item["labels"]
        assert labels[-1] == model.label_space.eos_id
        assert len(item["decoder_input_ids"]) == len(labels)
        assert decode_frame(labels[:-1], model.label_space, item["word_starts"]) == gold_frame(sample)
        for step, class_id in enumerate(labels[:-1]):
            fed = item["decoder_input_ids"][step + 1]
            if class_id >= n_labels:  # pointer step: the decoder reads the pointed wordpiece
                assert use_pointer
                assert fed == item["input_ids"][class_id - n_labels]
                assert class_id - n_labels in item["word_starts"]
            else:
                assert fed == model.label_token_ids[class_id].item()
    assert loader.dataset.n_dropped_spans == 0


def test_collator_pads_and_masks(samples, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    loader = make_loader(samples, model, tokenizer, batch_size=4, max_source_length=128, train=True)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert (batch["input_ids"][batch["attention_mask"] == 0] == tokenizer.pad_token_id).all()
    assert (batch["labels"] == -100).sum() == (batch["decoder_input_ids"] == tokenizer.pad_token_id).sum()
    # pointer targets never point at padding
    n_labels = len(model.label_space)
    lengths = batch["attention_mask"].sum(1, keepdim=True)
    pointers = batch["labels"] - n_labels
    assert (pointers < lengths).all()


def test_eval_dataset_tolerates_unseen_labels(samples):
    model, tokenizer = build_model(samples[:2])  # label space from two utterances only
    loader = make_loader(samples, model, tokenizer, batch_size=4, max_source_length=128)
    batch = next(iter(loader))
    assert "labels" not in batch and len(batch["gold"]) == 4


def test_agif_roundtrip(samples, tmp_path):
    write_dataset(samples, tmp_path / "out.txt")
    again = load_raw_dataset(tmp_path / "out.txt")
    assert [(s.words, s.tags, s.intents) for s in again] == [(s.words, s.tags, s.intents) for s in samples]
