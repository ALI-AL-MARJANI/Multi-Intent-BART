from __future__ import annotations

import copy

import pytest
import torch

from gemis.data.dataset import gold_frame
from gemis.experiment import make_loader
from gemis.models.bart_layer import GEMISDecoderLayer
from gemis.models.gemis import GEMISModel
from gemis.training.evaluation import predict_frames
from gemis.training.metrics import compute_metrics
from gemis.training.trainer import GEMISTrainer
from tests.conftest import build_model, build_tokenizer


def test_forward_shapes_and_loss(model_and_tokenizer, train_loader):
    model, _ = model_and_tokenizer
    batch = next(iter(train_loader))
    out = model(batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"], batch["labels"])
    b, t = batch["labels"].shape
    assert out["logits"].shape == (b, t, len(model.label_space) + batch["input_ids"].size(1))
    assert torch.isfinite(out["loss"])


def test_padded_positions_cannot_be_pointed_at(model_and_tokenizer, train_loader):
    model, _ = model_and_tokenizer
    batch = next(iter(train_loader))
    logits = model(batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"])["logits"]
    probs = logits.softmax(-1)[:, :, len(model.label_space) :]
    assert probs[(batch["attention_mask"] == 0)[:, None, :].expand_as(probs)].max() == 0


def test_outputs_do_not_depend_on_batch_padding(samples, model_and_tokenizer):
    """Scoring an utterance alone or inside a padded batch gives the same logits."""
    model, tokenizer = model_and_tokenizer
    model.eval()
    short, long = sorted(samples, key=lambda s: len(s.words))[0], max(samples, key=lambda s: len(s.words))
    alone = next(iter(make_loader([short], model, tokenizer, 1, 128, train=True)))
    padded = next(iter(make_loader([short, long], model, tokenizer, 2, 128, train=True)))
    t, n = alone["decoder_input_ids"].shape[1], alone["input_ids"].shape[1]
    a = model(alone["input_ids"], alone["attention_mask"], alone["decoder_input_ids"])["logits"][0]
    b = model(padded["input_ids"], padded["attention_mask"], padded["decoder_input_ids"][:, :t])["logits"][0]
    width = len(model.label_space) + n
    assert torch.allclose(a, b[:, :width], atol=1e-4)


def test_aoa_layer_without_sam_equals_the_stock_bart_layer(samples):
    """Re-classing a pretrained layer must not change it apart from the AoA bias."""
    stock, tokenizer = build_model(samples, use_aoa=False)
    aoa = copy.deepcopy(stock)
    for layer in aoa.bart.decoder.layers:
        GEMISDecoderLayer.convert(layer, 0.0)
        layer.encoder_attn.use_sam = False
    stock.eval(), aoa.eval()
    batch = next(iter(make_loader(samples, stock, tokenizer, 4, 128, train=True)))
    args = (batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"])
    assert torch.allclose(stock(*args)["logits"], aoa(*args)["logits"], atol=1e-4)

    for layer in aoa.bart.decoder.layers:
        layer.encoder_attn.use_sam = True
    assert not torch.allclose(stock(*args)["logits"], aoa(*args)["logits"], atol=1e-6)


def test_aoa_adds_no_parameters(samples):
    plain, _ = build_model(samples, use_aoa=False)
    aoa, _ = build_model(samples, use_aoa=True)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(plain) == count(aoa)
    assert all(isinstance(layer, GEMISDecoderLayer) for layer in aoa.bart.decoder.layers)


def test_label_embedding_init(samples):
    mean_model, tokenizer = build_model(samples, label_init="subword_mean")
    embed = mean_model.bart.get_input_embeddings().weight
    space = mean_model.label_space
    token_id = mean_model.label_token_ids[space.intent_id("GetWeather")]
    pieces = tokenizer.encode("GetWeather", add_special_tokens=False)
    assert torch.allclose(embed[token_id], embed[pieces].mean(0), atol=1e-6)
    random_model, _ = build_model(samples, label_init="random")
    assert not torch.allclose(random_model.bart.get_input_embeddings().weight[token_id], embed[token_id])


def test_generate_is_well_formed(samples, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    model.eval()
    batch = next(iter(make_loader(samples, model, tokenizer, 5, 128)))
    out = model.generate(batch["input_ids"], batch["attention_mask"], max_new_tokens=9)
    width = len(model.label_space) + batch["input_ids"].size(1)
    for sequence, log_probs, entropies in zip(out.sequences, out.log_probs, out.entropies):
        assert len(sequence) <= 9 and all(0 < c < width for c in sequence)
        assert len(log_probs) == len(entropies) and len(log_probs) in (len(sequence), len(sequence) + 1)
        assert all(lp <= 0 for lp in log_probs) and all(e >= 0 for e in entropies)
    assert out.first_step_label_logits.shape == (5, len(model.label_space))
    assert out.encoder_mean.shape == (5, 64)


@pytest.mark.parametrize("use_pointer", [True, False])
def test_overfits_the_toy_set_and_reloads(samples, tmp_path, use_pointer):
    """End to end: training reaches 100% on 12 utterances; the checkpoint reproduces it."""
    model, tokenizer = build_model(samples, use_pointer=use_pointer)
    torch.manual_seed(0)
    train = make_loader(samples, model, tokenizer, 4, 128, train=True)
    dev = make_loader(samples, model, tokenizer, 12, 128)
    trainer = GEMISTrainer(
        model,
        train,
        dev,
        tmp_path,
        torch.device("cpu"),
        learning_rate=3e-3,
        weight_decay=0.0,
        max_epochs=60,
        warmup_ratio=0.05,
        early_stopping_patience=0,
        max_new_tokens=40,
        show_progress=False,
    )
    history = trainer.train()
    assert trainer.best_metric == 100.0, history[-1]

    reloaded, _, checkpoint = GEMISModel.from_checkpoint(tmp_path / "best.pt", tokenizer=build_tokenizer(samples))
    golds, preds, _ = predict_frames(reloaded, dev, torch.device("cpu"), max_new_tokens=40)
    assert golds == [gold_frame(s) for s in samples]
    assert compute_metrics(golds, preds)["overall_accuracy"] == 100.0
    assert checkpoint["optimizer_state_dict"] is None


def test_resume_restores_counters(samples, tmp_path):
    def make_trainer(max_epochs):
        model, tokenizer = build_model(samples)
        train = make_loader(samples, model, tokenizer, 4, 128, train=True)
        dev = make_loader(samples, model, tokenizer, 12, 128)
        return GEMISTrainer(
            model,
            train,
            dev,
            tmp_path,
            torch.device("cpu"),
            max_epochs=max_epochs,
            max_new_tokens=8,
            show_progress=False,
        )

    make_trainer(1).train()
    trainer = make_trainer(2)
    trainer.resume(tmp_path / "last.pt")
    assert trainer.epoch == 1 and len(trainer.history) == 1
    assert [r["epoch"] for r in trainer.train()] == [1, 2]
