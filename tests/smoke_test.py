#!/usr/bin/env python3
"""End-to-end smoke test: real data → dataset → model → forward → generate → metrics."""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path

import torch
from transformers import BartTokenizerFast

from gemis.data.collator import GEMISDataCollator
from gemis.data.dataset import GEMISDataset, bio_to_spans, build_target_tokens, word_to_encoder_pos
from gemis.data.tokenizer import extend_tokenizer_vocab
from gemis.models.aoa import AoACrossAttention
from gemis.models.bart_layer import GEMISDecoderLayer
from gemis.models.gemis import GEMISModel
from gemis.training.metrics import compute_metrics, parse_target_sequence
from gemis.utils.io import load_raw_dataset
from gemis.utils.training_utils import collect_labels

TRAIN_PATH = Path("data/raw/mixatis/train.txt")
assert TRAIN_PATH.exists(), "Run: python scripts/download_data.py --dataset mixatis first"

print("=" * 60)
print("GEMIS SMOKE TEST")
print("=" * 60)

# ── 1. load raw data & inspect ────────────────────────────────────────────────
print("\n[1] Loading raw data …")
raw = load_raw_dataset(TRAIN_PATH)
sample = raw[0]
words = [t.word for t in sample.tokens]
bio   = [t.bio_tag for t in sample.tokens]
spans = bio_to_spans(words, bio)

print(f"   Words    : {words}")
print(f"   Intents  : {sample.intents}")
print(f"   BIO tags : {bio}")
print(f"   Spans    : {spans}")
for sp in spans:
    print(f"     [{sp.start}:{sp.end}] ({sp.slot_type}) = {words[sp.start:sp.end]}")

# ── 2. tokenizer + vocab extension ───────────────────────────────────────────
print("\n[2] Extending BART-base tokenizer …")
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
intent_labels, slot_labels = collect_labels([TRAIN_PATH])
print(f"   intents={len(intent_labels)}, slots={len(slot_labels)}")

new_i, new_s = extend_tokenizer_vocab(tokenizer, intent_labels, slot_labels)
print(f"   Added {len(new_i)} intent tokens, {len(new_s)} slot tokens")
print(f"   New vocab size: {len(tokenizer)}")

# ── 3. target sequence round-trip ─────────────────────────────────────────────
print("\n[3] Target sequence round-trip …")
target_ids = build_target_tokens(sample.intents, spans, tokenizer)
decoded = tokenizer.decode(target_ids, skip_special_tokens=False)
print(f"   Encoded  : {target_ids[:20]} ... ({len(target_ids)} tokens)")
print(f"   Decoded  : {decoded[:120]}")

# verify position round-trip
for sp in spans:
    enc_start = word_to_encoder_pos(sp.start, words, tokenizer)
    enc_end   = word_to_encoder_pos(sp.end, words, tokenizer) if sp.end < len(words) else -1
    print(f"   Span [{sp.start}:{sp.end}] → enc positions [{enc_start},{enc_end}]")

# ── 4. dataset & collator ────────────────────────────────────────────────────
print("\n[4] Building GEMISDataset (first 8 samples) …")
# write a tiny temp file with 8 samples
tmp = Path(tempfile.mkdtemp())
shutil.copy(TRAIN_PATH, tmp / "train.txt")

ds = GEMISDataset(TRAIN_PATH, tokenizer, max_source_length=128, max_target_length=64)
item = ds[0]
print(f"   input_ids shape       : {len(item['input_ids'])}")
print(f"   labels shape          : {len(item['labels'])}")
print(f"   decoder_input_ids len : {len(item['decoder_input_ids'])}")
print(f"   pointer_targets len   : {len(item['pointer_targets'])}")
print(f"   labels == decoder_input_ids length: {len(item['labels'])==len(item['decoder_input_ids'])}")

# check alignment
n_ptr = sum(1 for p in item['pointer_targets'] if p >= 0)
print(f"   pointer steps in target: {n_ptr}")

# collate a batch
collator = GEMISDataCollator(tokenizer=tokenizer)
batch = collator([ds[i] for i in range(4)])
print(f"   batch input_ids  : {batch['input_ids'].shape}")
print(f"   batch labels     : {batch['labels'].shape}")

# ── 5. model instantiation ────────────────────────────────────────────────────
print("\n[5] Instantiating GEMISModel (bart-base) …")
model = GEMISModel(
    backbone_name="facebook/bart-base",
    intent_labels=intent_labels,
    slot_labels=slot_labels,
    tokenizer=tokenizer,
    mlp_hidden=256,
)
model.eval()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Trainable params: {n_params:,}")

# verify AoA injection
decoder_layers = model.bart.model.decoder.layers
aoa_ok = all(isinstance(layer, GEMISDecoderLayer) for layer in decoder_layers)
enc_ok = all(isinstance(layer.encoder_attn, AoACrossAttention) for layer in decoder_layers)
print(f"   All layers → GEMISDecoderLayer : {aoa_ok}")
print(f"   All encoder_attn → AoA         : {enc_ok}")

# ── 6. forward pass ───────────────────────────────────────────────────────────
print("\n[6] Forward pass …")
batch_dev = {k: v for k, v in batch.items()}
with torch.no_grad():
    out = model(
        input_ids=batch_dev['input_ids'],
        attention_mask=batch_dev['attention_mask'],
        decoder_input_ids=batch_dev['decoder_input_ids'],
        labels=batch_dev['labels'],
        pointer_targets=batch_dev['pointer_targets'],
    )
print(f"   logits shape : {out['logits'].shape}")
print(f"   loss         : {out['loss'].item():.4f}")

N = batch_dev['input_ids'].shape[1]
n_vocab = model._vocab_size
expected_NL = N + n_vocab
assert out['logits'].shape[-1] == expected_NL, \
    f"Expected N+L={expected_NL}, got {out['logits'].shape[-1]}"
print(f"   logit dim = {N} (enc positions) + {n_vocab} (vocab) = {expected_NL} ✓")

# ── 7. greedy generation ──────────────────────────────────────────────────────
print("\n[7] Greedy generation (2 samples) …")
with torch.no_grad():
    generated = model.generate(
        input_ids=batch_dev['input_ids'][:2],
        attention_mask=batch_dev['attention_mask'][:2],
        max_new_tokens=40,
    )

for i, gen_ids in enumerate(generated):
    gen_decoded = tokenizer.decode(gen_ids, skip_special_tokens=False)
    gold_ids = [t for t in batch_dev['labels'][i].tolist() if t != -100]
    gold_decoded = tokenizer.decode(gold_ids, skip_special_tokens=False)
    print(f"   Sample {i}:")
    print(f"     Gold : {gold_decoded[:100]}")
    print(f"     Pred : {gen_decoded[:100]}")

# ── 8. metrics ────────────────────────────────────────────────────────────────
print("\n[8] Metrics on 2 samples …")
golds, preds = [], []
for i, gen_ids in enumerate(generated):
    gold_ids = [t for t in batch_dev['labels'][i].tolist() if t != -100]
    golds.append(parse_target_sequence(gold_ids, tokenizer))
    preds.append(parse_target_sequence(gen_ids, tokenizer))

metrics = compute_metrics(golds, preds)
print(f"   slot_f1         : {metrics['slot_f1']:.2f}")
print(f"   intent_accuracy : {metrics['intent_accuracy']:.2f}")
print(f"   overall_accuracy: {metrics['overall_accuracy']:.2f}")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED ✓")
print("=" * 60)
