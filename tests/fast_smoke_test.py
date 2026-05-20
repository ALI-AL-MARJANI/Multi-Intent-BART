#!/usr/bin/env python3
"""Fast smoke test using a tiny random-weight BART — no model download required.

Tests the full pipeline: data → tokenizer → model (tiny) → forward → generate → metrics.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path

import torch
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizerFast

import gemis.models.gemis as _gm
from gemis.data.collator import GEMISDataCollator
from gemis.data.dataset import GEMISDataset, bio_to_spans, build_target_tokens, word_to_encoder_pos
from gemis.data.tokenizer import extend_tokenizer_vocab
from gemis.models.aoa import AoACrossAttention
from gemis.models.bart_layer import GEMISDecoderLayer
from gemis.models.pointer import PointerNetwork
from gemis.models.vocab_utils import setup_vocab
from gemis.training.metrics import compute_metrics, parse_target_sequence
from gemis.utils.io import load_raw_dataset
from gemis.utils.training_utils import collect_labels

TRAIN_PATH = Path("data/raw/mixatis/train.txt")
assert TRAIN_PATH.exists(), "Run download_data.py --dataset mixatis first"

print("=" * 60)
print("GEMIS FAST SMOKE TEST (tiny random-weight model)")
print("=" * 60)

# ── 1. data pipeline ──────────────────────────────────────────────────────────
print("\n[1] Data pipeline …")
raw = load_raw_dataset(TRAIN_PATH)
s0 = raw[0]
words = [t.word for t in s0.tokens]
tags  = [t.bio_tag  for t in s0.tokens]
spans = bio_to_spans(words, tags)

print(f"   Sample: {' '.join(words[:8])}…  intents={s0.intents}")
for sp in spans:
    print(f"   span [{sp.start}:{sp.end}) {sp.slot_type} = {words[sp.start:sp.end]}")

# invariant: exclusive end, all spans properly formed
for raw_s in raw[:200]:
    ws = [t.word for t in raw_s.tokens]
    ts = [t.bio_tag for t in raw_s.tokens]
    for sp in bio_to_spans(ws, ts):
        assert 0 <= sp.start < sp.end <= len(ws), f"Bad span {sp}"
        assert ts[sp.start].startswith("B-")
        for i in range(sp.start+1, sp.end):
            assert ts[i].startswith("I-")
print("   Span invariant OK (200 samples) ✓")

# ── 2. tokenizer ──────────────────────────────────────────────────────────────
print("\n[2] Tokenizer …")
# Use a cached tokenizer if available, otherwise fall back to a tiny GPT2-style
try:
    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    print("   Loaded from cache ✓")
except Exception:
    # Fall back to gpt2 tokenizer as a compatible substitute for testing
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.bos_token = tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    print("   Using GPT2 fallback tokenizer")

intent_labels, slot_labels = collect_labels([TRAIN_PATH])
print(f"   {len(intent_labels)} intents, {len(slot_labels)} slots")

new_i, new_s = extend_tokenizer_vocab(tokenizer, intent_labels, slot_labels)
print(f"   Added {len(new_i)} intent tokens + {len(new_s)} slot tokens")
print(f"   Final vocab size: {len(tokenizer)}")

# ── 3. target sequence round-trip ─────────────────────────────────────────────
print("\n[3] Target sequence round-trip …")
tgt = build_target_tokens(s0.intents, spans, tokenizer)
decoded = tokenizer.decode(tgt, skip_special_tokens=False)
print(f"   Decoded: {decoded[:100]}")

for sp in spans:
    enc_start = word_to_encoder_pos(sp.start, words, tokenizer)
    enc_end   = word_to_encoder_pos(sp.end,   words, tokenizer) if sp.end < len(words) else -1
    print(f"   [{sp.start}:{sp.end}) → enc [{enc_start},{enc_end}]")

# verify decoder_input_ids have wordpieces at position steps
enc = tokenizer(" ".join(words), return_tensors=None)
# manually build one item
class _Item:
    def __init__(self, tokenizer, words, intents, slot_spans, input_ids):
        self.tok = tokenizer
        self.words = words
        self.intents = intents
        self.slot_spans = slot_spans
        self.input_ids = input_ids

full_tgt = build_target_tokens(s0.intents, spans, tokenizer)
labels = full_tgt[1:]
input_ids_list = tokenizer(" ".join(words), add_special_tokens=True)["input_ids"]

# Check pointer_targets alignment
ptr_gt = []
for tid in labels:
    try:
        widx = int(tokenizer.decode([tid]).strip())
        ep = word_to_encoder_pos(widx, words, tokenizer)
        ptr_gt.append(ep)
    except ValueError:
        ptr_gt.append(-1)

n_ptr = sum(1 for p in ptr_gt if p >= 0)
print(f"   pointer steps: {n_ptr} (expected {2*len(spans)} for {len(spans)} spans)")
assert n_ptr == 2 * len(spans), f"Expected {2*len(spans)} pointer steps, got {n_ptr}"
print("   Pointer target count ✓")

# ── 4. tiny model (random weights, no download) ───────────────────────────────
print("\n[4] Tiny GEMIS model (random weights) …")
tiny_cfg = BartConfig(
    vocab_size=len(tokenizer),
    d_model=64,
    encoder_layers=1,
    decoder_layers=1,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    encoder_ffn_dim=128,
    decoder_ffn_dim=128,
    max_position_embeddings=256,
    forced_bos_token_id=tokenizer.bos_token_id,
    forced_eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id or 1,
)

# Build a tiny BartForConditionalGeneration with this config (random weights)
bart_tiny = BartForConditionalGeneration(tiny_cfg)

# Now monkey-patch GEMISModel to accept a pre-built bart
_orig_init = _gm.GEMISModel.__init__

def _patched_init(self, backbone_name, intent_labels, slot_labels, tokenizer, mlp_hidden=512):
    nn.Module.__init__(self)  # must call before assigning submodules
    self.tokenizer = tokenizer
    self.intent_labels = intent_labels
    self.slot_labels = slot_labels
    self.bart = bart_tiny
    config = bart_tiny.config
    setup_vocab(self.bart, tokenizer, intent_labels, slot_labels)
    self.bart.lm_head = nn.Linear(config.d_model, len(tokenizer), bias=False)
    self._inject_aoa(config)
    self.pointer = PointerNetwork(
        hidden_size=config.d_model,
        vocab_size=len(tokenizer),
        mlp_hidden=mlp_hidden,
    )
    self._vocab_size = len(tokenizer)

_gm.GEMISModel.__init__ = _patched_init

model = _gm.GEMISModel(
    backbone_name="(tiny)",
    intent_labels=intent_labels,
    slot_labels=slot_labels,
    tokenizer=tokenizer,
    mlp_hidden=32,
)
_gm.GEMISModel.__init__ = _orig_init  # restore

model.eval()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Trainable params: {n_params:,}")

aoa_ok = all(isinstance(layer, GEMISDecoderLayer) for layer in model.bart.model.decoder.layers)
enc_ok = all(isinstance(layer.encoder_attn, AoACrossAttention) for layer in model.bart.model.decoder.layers)
print(f"   GEMISDecoderLayer injected: {aoa_ok} ✓")
print(f"   AoACrossAttention injected: {enc_ok} ✓")

# ── 5. dataset + forward pass ─────────────────────────────────────────────────
print("\n[5] Forward pass …")
ds = GEMISDataset(TRAIN_PATH, tokenizer, max_source_length=64, max_target_length=32)
collator = GEMISDataCollator(tokenizer=tokenizer)
items = [ds[i] for i in range(4)]
batch = collator(items)

words_batch = batch.pop("words", None)
with torch.no_grad():
    out = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        decoder_input_ids=batch['decoder_input_ids'],
        labels=batch['labels'],
        pointer_targets=batch['pointer_targets'],
    )

B, T, NL = out['logits'].shape
N = batch['input_ids'].shape[1]
L = len(tokenizer)
print(f"   logits: {B}×{T}×{NL}  (N={N} enc + L={L} vocab = {N+L}) ✓")
print(f"   loss:   {out['loss'].item():.4f}")
assert NL == N + L, f"logit dim mismatch: {NL} != {N}+{L}={N+L}"

# ── 6. greedy generation ──────────────────────────────────────────────────────
print("\n[6] Greedy generation …")
with torch.no_grad():
    gen = model.generate(
        input_ids=batch['input_ids'][:2],
        attention_mask=batch['attention_mask'][:2],
        max_new_tokens=20,
        input_words_batch=words_batch[:2] if words_batch else None,
    )

for i, g in enumerate(gen):
    gold_ids = [t for t in batch['labels'][i].tolist() if t != -100]
    gold_str = tokenizer.decode(gold_ids, skip_special_tokens=False)[:80]
    pred_str = tokenizer.decode(g, skip_special_tokens=False)[:80]
    print(f"   [{i}] gold: {gold_str}")
    print(f"   [{i}] pred: {pred_str}")

# ── 7. metrics ────────────────────────────────────────────────────────────────
print("\n[7] Metrics …")
golds, preds = [], []
for i, g in enumerate(gen):
    gold_ids = [t for t in batch['labels'][i].tolist() if t != -100]
    golds.append(parse_target_sequence(gold_ids, tokenizer))
    preds.append(parse_target_sequence(g, tokenizer))

metrics = compute_metrics(golds, preds)
# With random weights, exact match is 0 — that's expected
print(f"   slot_f1:          {metrics['slot_f1']:.2f}  (0.00 expected with random weights)")
print(f"   intent_accuracy:  {metrics['intent_accuracy']:.2f}")
print(f"   overall_accuracy: {metrics['overall_accuracy']:.2f}")

# Verify gold frames parse correctly
for i, fr in enumerate(golds):
    print(f"   gold frame [{i}]: intents={set(fr.intents)}, spans={fr.slot_spans}")

print("\n" + "=" * 60)
print("ALL FAST SMOKE TESTS PASSED ✓")
print("=" * 60)
