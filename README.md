# GEMIS — Generative Model for Joint Multiple Intent Detection and Slot Filling

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/Transformers-4.40%2B-FFD21E?logo=huggingface" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

> **Full PyTorch implementation** of  
> *"A Generative Model for Joint Multiple Intent Detection and Slot Filling"*  
> Li & Zhu (2026) — [arXiv:2602.08322](https://arxiv.org/abs/2602.08322)

---

## Overview

Real users rarely express a single intent per utterance.  
GEMIS handles the realistic multi-intent setting by recasting the entire SLU task as **one sequence-to-sequence problem** on top of BART:

```
Input  : "show me flights from Boston and list cities served by United"
Target : <intent:atis_flight> <intent:atis_city> 4 5 <slot:fromloc.city_name>
           11 13 <slot:airline_name>
```

Two architectural contributions on top of BART:

| Component | What it does |
|---|---|
| **Pointer Network** | Predicts slot-value spans by pointing to encoder positions rather than copying tokens into a fixed vocabulary |
| **Attention-over-Attention (AoA)** | Replaces the BART decoder's cross-attention with `softmax(Q·K^T + SAM·CAM/√d_k)`, linking slot generation to already-predicted intents |

The paper also introduces **MultiATIS** and **MultiSNIPS** — two new datasets built with BERT's NSP head to produce *coherent* multi-intent utterances, in contrast to the random concatenations in MixATIS/MixSNIPS.

---

## Architecture

```
Input utterance
      │
      ▼
┌──────────────────────┐
│    BART Encoder      │  ← facebook/bart-large (extended vocab)
└──────────┬───────────┘
           │  H_enc  (B, N, D)
           ▼
┌─────────────────────────────────────────────────────────┐
│                  Modified BART Decoder                  │
│                                                         │
│  ┌──────────────┐  SAM   ┌─────────────────────────┐   │
│  │  Self-Attn   │───────►│  Attention-over-Attn    │   │
│  │ (eager mode) │        │  A = softmax(QK^T        │   │
│  └──────────────┘        │      + SAM·CAM / √d_k)  │   │
│                          └───────────┬─────────────┘   │
│                              h_dec  (B, T, D)           │
└──────────────────────────────────────┬──────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │     Pointer Network      │
                          │  Ĥ_e = α·MLP(H_e)       │
                          │       + (1-α)·E_x        │
                          │  logits ∈ ℝ^{N + |V|}    │
                          └──────────────────────────┘
```

### Target sequence format

```
[BOS] <intent:X> <intent:Y>  start₁ end₁ <slot:Z>  start₂ end₂ <slot:W>  [EOS]
```

Position indices are **0-indexed**, **end-exclusive** (Python-slice convention).  
Example: `words[2:5]` = "Got The Time" → target has `2 5 <slot:track>`.

---

## Results

Reported numbers from the paper (BART-large):

| Dataset    | Slot F1 | Intent Acc | Overall Acc |
|------------|---------|------------|-------------|
| MixATIS    | 89.2    | 81.4       | 53.4        |
| MixSNIPS   | 97.4    | 98.1       | 87.4        |
| MultiATIS  | 95.4    | 94.1       | 71.5        |
| MultiSNIPS | 98.1    | 98.8       | 91.5        |

---

## Repository Structure

```
.
├── src/gemis/
│   ├── data/
│   │   ├── dataset.py        # GEMISDataset: BIO → seq2seq samples + pointer targets
│   │   ├── construct.py      # Algorithm 1: BERT NSP-based dataset construction
│   │   ├── tokenizer.py      # Vocabulary extension + subword-mean embedding init
│   │   └── collator.py       # DataCollator (padding, label masking)
│   ├── models/
│   │   ├── gemis.py          # Main model: BART + AoA injection + Pointer Network
│   │   ├── aoa.py            # AoACrossAttention layer
│   │   ├── bart_layer.py     # GEMISDecoderLayer (threads SAM weights into AoA)
│   │   └── pointer.py        # PointerNetwork head (N+L logits, learnable α)
│   ├── training/
│   │   ├── trainer.py        # Native PyTorch training loop (AdamW + warmup)
│   │   └── metrics.py        # Slot F1, Intent Accuracy, Overall Accuracy
│   └── utils/
│       ├── io.py              # File I/O, YAML config loading
│       └── training_utils.py  # set_seed, collect_labels
├── scripts/
│   ├── train.py              # Training entry point
│   ├── evaluate.py           # Evaluation on test set
│   ├── download_data.py      # Pull MixATIS / MixSNIPS / ATIS / SNIPS
│   └── construct_dataset.py  # Run Algorithm 1 → MultiATIS / MultiSNIPS
├── configs/
│   ├── mixatis.yaml
│   ├── mixsnips.yaml
│   ├── multiatis.yaml
│   └── multisnips.yaml
├── notebooks/
│   ├── 01_EDA_and_Data_Construction.ipynb
│   ├── 02_Model_Training_Tutorial.ipynb
│   └── 03_Attention_Visualization.ipynb
├── tests/
│   └── fast_smoke_test.py    # End-to-end test (no model download needed)
├── run.sh                    # Convenience wrapper (sets USE_TF=0)
└── Makefile                  # Common commands
```

---

## Installation

```bash
git clone https://github.com/ALI-AL-MARJANI/Multi-Intent-BART.git
cd Multi-Intent-BART

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

---

## Quick Start

All commands use the `run.sh` wrapper which sets the required environment
variables. Alternatively, prefix any command with `USE_TF=0`.

```bash
# 1. Download datasets
./run.sh scripts/download_data.py --dataset mixatis  --output data/raw/
./run.sh scripts/download_data.py --dataset mixsnips --output data/raw/

# 2. Train
./run.sh scripts/train.py --config configs/mixatis.yaml

# 3. Evaluate
./run.sh scripts/evaluate.py \
    --checkpoint checkpoints/mixatis/best.pt \
    --config     configs/mixatis.yaml
```

Or use the `Makefile` shortcuts:

```bash
make data-mix        # download MixATIS + MixSNIPS
make train-mixatis   # train on MixATIS
make eval-mixatis    # evaluate best checkpoint
make smoke           # run the fast smoke test
```

---

## Data Preparation

### Option A — MixATIS / MixSNIPS (existing datasets, ready to use)

```bash
./run.sh scripts/download_data.py --dataset mixatis  --output data/raw/
./run.sh scripts/download_data.py --dataset mixsnips --output data/raw/
```

Sizes: MixATIS (13,162 / 759 / 828 train/dev/test), MixSNIPS (39,776 / 2,198 / 2,199).

### Option B — Construct MultiATIS / MultiSNIPS (Algorithm 1)

```bash
# 1. Download base single-intent datasets
./run.sh scripts/download_data.py --dataset atis  --output data/raw/
./run.sh scripts/download_data.py --dataset snips --output data/raw/

# 2. Run NSP-based construction
./run.sh scripts/construct_dataset.py \
    --input  data/raw/atis/train.txt \
    --output data/processed/multiatis/ \
    --tau    0.5 \
    --bert   bert-base-uncased
```

**Algorithm 1 in brief**: for each source utterance, sample target intent count
`n ∈ {1,2,3}` with probabilities `(0.3, 0.5, 0.2)`. Iteratively concatenate a
candidate utterance if `P_NSP(u_m ‖ u_c) > τ` (BERT Next Sentence Prediction),
joining with "and" or "and then".

---

## Configuration

Key options (see `configs/mixatis.yaml` for a complete example):

```yaml
model:
  backbone: facebook/bart-large   # or facebook/bart-base
  mlp_hidden: 512

training:
  learning_rate: 2e-5
  batch_size: 16
  max_epochs: 30
  warmup_steps: 200
  gradient_clip: 1.0
  device: cuda
  seed: 42
```

---

## Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **Slot F1** | Micro-averaged F1 over exact slot spans `(start, end_exclusive, type)` |
| **Intent Accuracy** | Fraction of utterances where the predicted intent *set* equals the gold intent set (exact match) |
| **Overall Accuracy** | Exact match of the full semantic frame (intents **and** all slots) |

---

## Notebooks

| Notebook | Content |
|----------|---------|
| `01_EDA_and_Data_Construction` | Dataset statistics; live BERT NSP demo showing coherent vs. random pairs |
| `02_Model_Training_Tutorial` | Step-by-step: build model, forward pass, 10-step training, greedy decode |
| `03_Attention_Visualization` | Extract SAM / AoA weights; heatmaps showing how intents guide slot attention |

---

## Key Implementation Notes

- **Transformers ≥ 4.40 API**: `BartAttention.forward()` returns 2 values;
  `Cache` is updated in-place. `GEMISDecoderLayer` handles both the new
  self-attention API and the legacy interface for `AoACrossAttention`.
- **Eager attention**: decoder self-attention is forced to `eager` mode so
  that weight tensors (needed for the AoA bias) are always returned.
- **Position indices**: 0-indexed, end-exclusive.
- **Teacher forcing**: position-integer tokens in `decoder_input_ids` are
  replaced by the corresponding wordpieces (paper §3.1).

---

## Citation

```bibtex
@article{li2026gemis,
  title   = {A Generative Model for Joint Multiple Intent Detection and Slot Filling},
  author  = {Li, Liz and Zhu, Wei},
  journal = {arXiv preprint arXiv:2602.08322},
  year    = {2026}
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).
