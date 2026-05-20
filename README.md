# GEMIS — Generative Multi-Intent NLU with Conformal OOD Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface" />
  <img src="https://img.shields.io/badge/Conformal_Prediction-OOD-blueviolet" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<p align="center">
  <b>Seq2seq multi-intent NLU that knows what it doesn't know.</b><br/>
  Joint intent detection + slot filling over BART, extended with the first conformal OOD detector for a generative NLU model.
</p>

---

> Based on: *"A Generative Model for Joint Multiple Intent Detection and Slot Filling"* — Li & Zhu (2026) · [arXiv:2602.08322](https://arxiv.org/abs/2602.08322)

---

## Highlights

- **Generative NLU** — single seq2seq pass for multiple intents + all slot spans, no pipeline
- **Attention-over-Attention (AoA)** — decoder cross-attention reweighted by which past intents are relevant
- **Pointer Network** — predicts word-level position indices directly, no boundary post-processing
- **Conformal OOD detection** — distribution-free guarantee on false-alarm rate, no tuning required

---

## How OOD detection works

All prior OOD work in NLU assumes a discriminative classifier (BERT + softmax). GEMIS is generative — there is no single classification logit. Instead, we hook into the decoder itself.

```
User says: "what is the speed of light?"   ← outside SNIPS/ATIS domains
                    ↓
GEMIS decoder spreads probability across many intent tokens
                    ↓
OODScorer: s(x) = mean entropy over intent-position steps   → high
                    ↓
ConformalCalibrator: s(x) > q_{0.95} of calibration scores? → YES
                    ↓
model abstains with a guaranteed false-alarm rate ≤ 5%
```

**Guarantee (Vovk et al., 2005):**
```
P(in-domain sample wrongly flagged as OOD) ≤ α
```
Holds with no distributional assumptions — only exchangeability of the calibration set.

---

## Results

### NLU Performance (BART-large, paper results)

| Dataset | Slot F1 | Intent Acc | Overall Acc |
|---------|---------|------------|-------------|
| MixATIS | 89.2 | 81.4 | 53.4 |
| MixSNIPS | 97.4 | 98.1 | 87.4 |
| MultiATIS | 95.4 | 94.1 | 71.5 |
| MultiSNIPS | 98.1 | 98.8 | 91.5 |

### OOD Detection (MixSNIPS, BART-large)

Evaluated on the MixSNIPS test set (2 199 in-distribution samples) against 200 synthetic OOD utterances covering topics outside the SNIPS/ATIS domains (science, sport, cooking, history, programming).

| Metric | Value | |
|--------|-------|-|
| **AUROC** | **0.9751** | ↑ higher is better · random baseline = 0.50 |
| **FPR@95TPR** | **0.0991** | ↓ lower is better · only 10% false alarms at 95% OOD recall |
| **AUPR** | **0.8306** | ↑ higher is better |
| TPR (OOD recall) | 0.9450 | 189 / 200 OOD samples correctly flagged |
| FPR (false-alarm) | 0.0837 | ≈ 184 / 2199 in-domain samples wrongly flagged |
| Threshold (α = 0.05) | 0.0555 | conformal quantile on dev set |

---

## Architecture

```
Input utterance (tokenized)
        │
        ▼
┌─────────────────────┐
│    BART Encoder     │  ← facebook/bart-large  (vocab extended with intent/slot tokens)
└──────────┬──────────┘
           │  H_enc  (B, N, D)
           ▼
┌──────────────────────────────────────────────────────────┐
│                  Modified BART Decoder                   │
│                                                          │
│  ┌──────────────┐  SAM   ┌──────────────────────────┐   │
│  │  Self-Attn   │───────►│  Attention-over-Attention │   │
│  │ (eager mode) │        │  A = softmax(QKᵀ          │   │
│  └──────────────┘        │      + SAM·CAM / √d_k)   │   │
│                          └──────────┬────────────────┘   │
│                              h_dec  (B, T, D)             │
└──────────────────────────────────────┬───────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │     Pointer Network      │
                          │  Ĥ_e = α·MLP(H_e)        │
                          │       + (1−α)·E_x         │
                          │  logits ∈ ℝ^{N + |V|}     │
                          └──────────────────────────┘
                                       │
                     ┌─────────────────▼──────────────────┐
                     │   OOD Module (inference-time only)  │
                     │                                     │
                     │  H_t = −Σ p·log p  at intent steps │
                     │  s(x) = mean(H_t)                   │
                     │  flag ⟺ s(x) > q_{1−α}             │
                     └─────────────────────────────────────┘
```

### Target sequence format

```
[BOS] <intent:PlayMusic> <intent:AddToPlaylist>  2 5 <slot:track>  7 9 <slot:entity_name>  [EOS]
```

Position indices are **0-indexed** and **end-exclusive** (Python-slice convention).

---

## Installation

```bash
git clone https://github.com/ALI-AL-MARJANI/Multi-Intent-BART.git
cd Multi-Intent-BART

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

> **macOS / systems with broken TensorFlow:** always prefix commands with `USE_TF=0`.

**Optional — local LLM for richer synthetic OOD generation:**
```bash
pip install ollama
ollama pull llama3.2:1b   # requires ollama CLI from https://ollama.com
```

---

## Quick Start

### 1 · Download data

```bash
USE_TF=0 python scripts/download_data.py --dataset mixsnips --output data/raw/
USE_TF=0 python scripts/download_data.py --dataset mixatis  --output data/raw/
```

### 2 · Smoke test (no GPU required)

```bash
USE_TF=0 python tests/fast_smoke_test.py
```

### 3 · Train

```bash
USE_TF=0 python scripts/train.py --config configs/mixsnips.yaml
# Tesla T4 / 16 GB GPU:
USE_TF=0 python scripts/train.py --config configs/mixsnips_t4.yaml
```

### 4 · Evaluate NLU

```bash
USE_TF=0 python scripts/evaluate.py \
    --checkpoint checkpoints/mixsnips/best.pt \
    --config     configs/mixsnips.yaml
```

### 5 · Calibrate OOD detector

```bash
USE_TF=0 python scripts/calibrate_ood.py \
    --checkpoint checkpoints/mixsnips/best.pt \
    --config     configs/mixsnips.yaml \
    --output     checkpoints/mixsnips/ood_calibrator.pkl \
    --alpha      0.05
```

### 6 · Evaluate OOD detection

```bash
USE_TF=0 python scripts/evaluate_ood.py \
    --checkpoint checkpoints/mixsnips/best.pt \
    --config     configs/mixsnips.yaml \
    --calibrator checkpoints/mixsnips/ood_calibrator.pkl \
    --plot       results/ood_calibration.png
```

**One-liner (train already done):**
```bash
sh run_ood.sh mixsnips
```

---

## Python API

```python
import torch
from transformers import BartTokenizerFast
from gemis.models.gemis import GEMISModel
from gemis.ood import ConformalCalibrator, OODScorer

tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large")
model = GEMISModel(backbone_name="facebook/bart-large", ...)
model.load_state_dict(torch.load("checkpoints/mixsnips/best.pt")["model_state_dict"])
model.eval()

calibrator = ConformalCalibrator.load("checkpoints/mixsnips/ood_calibrator.pkl")

utterances = [
    "play some jazz music",          # in-domain
    "what is the speed of light?",   # OOD
]
enc = tokenizer(utterances, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    result = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        return_scores=True,
    )

for i, (gen_ids, entropies) in enumerate(
    zip(result["generated_ids"], result["intent_entropies"])
):
    score  = OODScorer.score(entropies)
    is_ood = calibrator.predict(score, alpha=0.05)
    print(f"[{'OOD' if is_ood else ' OK'}]  score={score:.3f}  →  "
          f"{tokenizer.decode(gen_ids, skip_special_tokens=False)[:60]}")
```

Expected output:
```
[ OK]  score=0.001  →  <intent:PlayMusic> 2 3 <slot:music_item></s>
[OOD]  score=0.312  →  <intent:GetWeather></s>
```

---

## Repository Structure

```
.
├── src/gemis/
│   ├── data/
│   │   ├── dataset.py          # GEMISDataset: BIO → seq2seq samples + pointer targets
│   │   ├── construct.py        # Algorithm 1: BERT NSP-based multi-intent construction
│   │   ├── tokenizer.py        # Vocabulary extension + subword-mean embedding init
│   │   └── collator.py         # DataCollator (padding, pointer targets, word lists)
│   ├── models/
│   │   ├── gemis.py            # GEMISModel + generate(return_scores=True)
│   │   ├── aoa.py              # AoACrossAttention layer
│   │   ├── bart_layer.py       # GEMISDecoderLayer (threads SAM into AoA)
│   │   └── pointer.py          # PointerNetwork head (N+L logits, learnable α)
│   ├── ood/
│   │   ├── scorer.py           # OODScorer: entropy-based non-conformity score
│   │   ├── calibrator.py       # ConformalCalibrator: fit / threshold / predict / save
│   │   └── synthetic.py        # SyntheticOODGenerator: ollama + hardcoded fallback
│   ├── training/
│   │   ├── trainer.py          # Training loop (AdamW, warmup, gradient accumulation)
│   │   └── metrics.py          # Slot F1, Intent Accuracy, Overall Accuracy
│   └── utils/
│       ├── io.py               # YAML config loader, AGIF file parser
│       └── training_utils.py   # set_seed, collect_labels
├── scripts/
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # NLU evaluation on test set
│   ├── calibrate_ood.py        # Fit ConformalCalibrator on dev set
│   ├── evaluate_ood.py         # AUROC / FPR@95TPR / AUPR + calibration plot
│   ├── download_data.py        # Pull MixATIS / MixSNIPS from AGIF repo
│   └── construct_dataset.py    # Run Algorithm 1 → MultiATIS / MultiSNIPS
├── configs/
│   ├── mixsnips.yaml           # MixSNIPS (standard)
│   ├── mixsnips_t4.yaml        # MixSNIPS (Tesla T4, batch=8 + grad_accum=2)
│   ├── mixatis.yaml
│   ├── mixatis_t4.yaml
│   ├── multiatis.yaml
│   └── multisnips.yaml
├── tests/
│   ├── fast_smoke_test.py      # Full pipeline test — no GPU, no download required
│   └── test_ood.py             # OOD module unit + integration tests (5 tests)
├── run_train.sh                # One-line training launcher
├── run_ood.sh                  # One-line OOD calibration + evaluation launcher
└── notebooks/
    ├── 01_EDA_and_Data_Construction.ipynb
    └── 02_Model_Training_Tutorial.ipynb
```

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{li2026gemis,
  title   = {A Generative Model for Joint Multiple Intent Detection and Slot Filling},
  author  = {Li, ... and Zhu, ...},
  journal = {arXiv preprint arXiv:2602.08322},
  year    = {2026}
}
```
