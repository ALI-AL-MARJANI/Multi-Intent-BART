# GEMIS reimplementation + conformal OOD detection

[![CI](https://github.com/ALI-AL-MARJANI/Multi-Intent-BART/actions/workflows/ci.yml/badge.svg)](https://github.com/ALI-AL-MARJANI/Multi-Intent-BART/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An independent reimplementation of GEMIS (Li & Zhu, 2026), a BART model that
generates the intents and slot spans of a multi-intent utterance as one
sequence, and a study of out-of-distribution (OOD) detection on top of it with
split-conformal calibration.

The paper has no public code. Part 1 measures how close this reimplementation
gets to the published numbers. Part 2 is not in the paper.

## TL;DR

<!-- TLDR:BEGIN -->
TBD (run pending)
<!-- TLDR:END -->

## Motivation

Multi-intent spoken language understanding is usually solved with a classifier
for intents and a tagger for slots. GEMIS casts both as generation with a
pointer network, which removes the fixed number of intents. A deployed
understanding model also has to notice utterances it was not trained for, and
OOD detection is mostly studied for discriminative intent classifiers. This
repository asks two questions: can the GEMIS results be reproduced from the
paper alone, and which OOD score works for such a generative model when the
false-alarm rate has to be controlled?

## Method

### Model

```
utterance ──► BART encoder ──► H_e ───────────────┐
                                │                 │ pointer scores over the N input positions
                                ▼                 ▼
 previous outputs ──► BART decoder with AoA ──► h_t ──► [ W·h_t ; H~·h_t ] ──► softmax over L + N classes
                                                          │
                                                          └ label scores over L = 1 + #intents + #slot types
```

**Target sequence.** `intent_1 … intent_p  start_1 end_1 slot_1 … start_q end_q slot_q </s>`.
For `please play got the time and add my hands to travelling playlist`:
`PlayMusic AddToPlaylist 2 5 track 7 9 entity_name 10 11 playlist`.
`start` and `end` are word indices (0-based, end exclusive). They are not
generated as tokens: the model points at the first subword of the word, or at
the final `</s>` for an end after the last word.

**Pointer head** (paper Eq. 5-6, `src/gemis/models/pointer.py`):

```
H~  = α · MLP(H_e) + (1 − α) · E_x
ŷ_t = softmax([ W h_t ; H~ h_t ])
```

`E_x` are the input token embeddings and `W` the embeddings of the label
tokens, which are initialised with the mean embedding of the label's subwords.

**Attention-over-attention** (paper Eq. 8-10, `src/gemis/models/aoa.py`),
which replaces the cross-attention of every decoder layer:

```
CAM = softmax(Q K_enc^T / √d_k)                    cross-attention map, T × N
SAM = softmax(Q K_dec^T / √d_k + causal mask)      self-attention map,  T × T
A   = softmax(Q K_enc^T / √d_k + SAM · CAM / √d_k)
```

AoA adds no parameters; the projections are the pretrained BART ones.

### Differences from the paper

The paper leaves several points open. Each choice below is a deviation or an
assumption, not a claim about what the authors did.

| Point | Paper | This repository |
|---|---|---|
| Position convention | Example `2 5 track` for "Got The Time"; inclusive/exclusive and word/subword level not stated | Word index, 0-based, exclusive end, pointer on the first subword |
| `α` in the pointer | Hyperparameter, value not reported | Learned scalar, `sigmoid`, initialised at 0.5 |
| First term of the AoA logits | Written without `1/√d_k` | Scaled, as the pretrained attention weights require |
| Pointer MLP size, warmup, weight decay, gradient clipping | Not reported | 512, 1% linear warmup, 0.01, 1.0 |
| Learning rate | Chosen in {1e-5, 2e-5, 3e-5} on dev | 2e-5, no search |
| Epochs | 30 | At most 30, early stopping after 5 epochs without dev improvement |
| Label inventory | Not stated | Training split only (labels unseen in training cannot be predicted) |
| Number of runs | Not reported | 3 seeds, mean ± standard deviation |
| MultiATIS / MultiSNIPS | Constructed with BERT NSP, threshold not reported, data not released | Not reproduced (see [Limitations](#limitations-and-threats-to-validity)) |

Since `SAM · CAM` is a convex combination of probability vectors, the AoA term
is bounded by `1/√d_k` (0.125 for BART). Its effect on the attention logits is
small by construction; the ablation below measures what it changes.

### OOD scores and conformal calibration

All scores come from one greedy decoding pass (`src/gemis/ood/scores.py`) and
are oriented so that higher means more likely OOD.

| Score | Definition | Origin |
|---|---|---|
| `intent_entropy` | Mean entropy of the output distribution over the steps that emit an intent | this repository |
| `first_step_msp` | 1 − maximum probability at the first step | Hendrycks & Gimpel (2017) |
| `sequence_nll` | Length-normalised negative log-likelihood of the greedy output | Ren et al. (2023) |
| `intent_energy` | −logsumexp of the intent logits at the first step | Liu et al. (2020) |
| `mahalanobis` | Mahalanobis distance of the mean encoder state to the training features (one Gaussian) | Lee et al. (2018), class-agnostic variant |
| `knn` | Distance to the 10th nearest training feature after L2 normalisation | Sun et al. (2022) |

A score is turned into a decision with split-conformal calibration
(`src/gemis/ood/calibrator.py`): given `n` in-distribution calibration scores,
an utterance is flagged when its score exceeds the
`⌈(n+1)(1−α)⌉`-th smallest of them. If calibration and test utterances are
exchangeable, the probability of flagging an in-distribution utterance is at
most `α` (Vovk et al., 2005; Bates et al., 2023). The guarantee is marginal
over the calibration draw and says nothing about the detection rate.

## Experimental protocol

**Datasets.** MixATIS and MixSNIPS, the cleaned versions released with AGIF
(Qin et al., 2020), with their official splits.

| Dataset | Train | Dev | Test | Intents | Slot types (train) |
|---|---|---|---|---|---|
| MixATIS | 13,162 | 759 | 828 | 17 | 74 |
| MixSNIPS | 39,776 | 2,198 | 2,199 | 7 | 39 |

113 of the 828 MixATIS test utterances contain an intent or slot type that
never occurs in the training split (`atis_day_name`, `flight`,
`stoploc.airport_code`, `booking_class`). A model whose labels come from the
training data cannot get them fully right, which caps overall accuracy at
86.4% on this test set.

**Metrics** (`src/gemis/training/metrics.py`), as in AGIF and GL-GIN:
slot F1 is the micro-averaged F1 over exact `(start, end, type)` spans, equal
to conlleval chunk F1 (tested against `seqeval`); intent accuracy is the share
of utterances whose predicted intent set equals the gold set; overall accuracy
requires both the intent set and all slots to be correct.

**Baselines.** Published AGIF and GL-GIN numbers as reference points; a BART
model without AoA, without the pointer network (positions generated as
classes) and with randomly initialised label embeddings as ablations.

**OOD evaluation sets.** In-distribution: the test split. Far-OOD: the 1,000
out-of-scope test queries of CLINC150 (Larson et al., 2019), lowercased and
stripped of sentence punctuation to match the in-distribution formatting.
Near-OOD: a model is retrained without any utterance containing
`AddToPlaylist` or `SearchScreeningEvent`; test utterances carrying one of
them are the OOD set. Metrics: AUROC, AUPR with OOD as positive (AUPR-out) and
with in-distribution as positive (AUPR-in), FPR at 95% TPR, each with a 95%
bootstrap interval, and the false-alarm rate at the conformal threshold for
`α ∈ {0.01, 0.05, 0.1}`.

**Conformal validity.** Two calibration schemes are compared. *Dev-calibrated*:
threshold from the dev split, false-alarm rate measured on the test split; the
dev split also selects the checkpoint, so exchangeability is not guaranteed.
*Random splits of test*: 1,000 random halves of the test split as calibration
and evaluation sets, exchangeable by construction.

**Hyperparameters.** [`configs/_base.yaml`](configs/_base.yaml) and one file
per experiment in [`configs/`](configs). Seeds: 13, 42, 87.

## Results

<!-- RESULTS:BEGIN -->
TBD (run pending)
<!-- RESULTS:END -->

## Limitations and threats to validity

* **This is not a faithful reproduction of the paper's numbers.** No code or
  hyperparameter file was released; the choices in "Differences from the
  paper" may differ from the authors'. A gap to the published numbers can come
  from those choices, from the missing learning-rate search, or from the
  backbone size.
* **MultiATIS and MultiSNIPS are not reproduced.** The paper's best-known
  numbers (e.g. 98.8 intent accuracy on MultiSNIPS) are on datasets the
  authors constructed and did not release. `scripts/construct_dataset.py`
  implements the published algorithm, but the threshold is unreported, so the
  output is a different dataset and no result on it is claimed.
* **Rows marked "TBD (run pending)" have not been run.** Nothing in this
  README is an expected or estimated value.
* **Seeds.** Three seeds give a rough standard deviation, not a tight
  interval. Rows with fewer seeds say so.
* **Far-OOD is an easy setting.** CLINC150 out-of-scope queries are about
  topics unrelated to flight booking or the SNIPS domains. High AUROC there
  does not transfer to utterances that are close to the training intents; the
  held-out-intent experiment is the relevant one for that.
* **The near-OOD set mixes known and unknown intents.** A held-out utterance
  in MixSNIPS usually also contains one or two in-distribution intents, so the
  task is to notice that *part* of the utterance is unsupported.
* **Conformal calibration controls false alarms only**, on average over
  calibration sets, and only if deployment data is exchangeable with the
  calibration data. It gives no guarantee on the share of OOD utterances
  detected, and none under distribution shift of the in-distribution data.
* **No latency study.** Decoding re-runs the decoder on the full prefix at
  each step (no key/value cache); `utterances_per_second` in the result files
  is indicative only.

## Reproduce

[`RUNBOOK.md`](RUNBOOK.md) lists the exact command, output files, hardware and
time for every table. In short:

```bash
pip install -e ".[dev]"
python scripts/download_data.py --dataset mixatis mixsnips
pytest                                                                # 50 tests, CPU, offline
sh scripts/run_experiment.sh configs/mixatis_bart-base.yaml 13 42 87  # train, evaluate, OOD-score
python scripts/aggregate_results.py && python scripts/make_figures.py
```

Using a trained checkpoint:

```python
import torch
from gemis.data.alignment import encode_source
from gemis.data.decoding import decode_frame
from gemis.models.gemis import GEMISModel

model, tokenizer, _ = GEMISModel.from_checkpoint("checkpoints/mixatis_bart-base/seed13/best.pt")
source = encode_source("show me flights from boston to denver and what is fare code h".split(), tokenizer)
ids = torch.tensor([source.input_ids])
output = model.eval().generate(ids, torch.ones_like(ids))
print(decode_frame(output.sequences[0], model.label_space, source.word_starts))
```

## Repository structure

```
configs/            one YAML per experiment, all inheriting from _base.yaml
data/ood/           CLINC150 out-of-scope test queries (far-OOD set)
docs/devlog.md      defects found in v0.1 and claims that were withdrawn
results/            committed outputs: per-seed JSON/CSV, summaries, figures, published reference numbers
scripts/            train, evaluate, score_ood, evaluate_ood, aggregate_results, make_figures, data preparation
src/gemis/
  data/             AGIF parsing, word-to-subword alignment, targets, decoding to frames, Algorithm 1
  models/           AoA cross-attention, decoder layer, pointer head, GEMISModel
  ood/              OOD scores, conformal calibration, detection metrics
  training/         trainer, evaluation loop, metrics, error analysis
tests/              50 offline tests on a toy dataset with a tiny random BART
```

The tests include: the word-to-subword alignment on BART's tokenizer, an
oracle check that the training targets decode back to the gold frames,
equivalence of the modified decoder layer with stock BART when the AoA term is
switched off, invariance of the outputs to batch padding, an end-to-end
overfit and checkpoint reload, span F1 against `seqeval`, and false-alarm
control of the conformal threshold on exchangeable scores.

## References

1. L. Li, W. Zhu. *A Generative Model for Joint Multiple Intent Detection and Slot Filling.* 2026. [arXiv:2602.08322](https://arxiv.org/abs/2602.08322)
2. M. Lewis et al. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* ACL 2020. [arXiv:1910.13461](https://arxiv.org/abs/1910.13461)
3. H. Yan et al. *A Unified Generative Framework for Various NER Subtasks.* ACL 2021. [arXiv:2106.01223](https://arxiv.org/abs/2106.01223) (pointer formulation on BART)
4. L. Qin et al. *AGIF: An Adaptive Graph-Interactive Framework for Joint Multiple Intent Detection and Slot Filling.* Findings of EMNLP 2020. [arXiv:2004.10087](https://arxiv.org/abs/2004.10087), [code and MixATIS / MixSNIPS](https://github.com/LooperXX/AGIF)
5. L. Qin et al. *GL-GIN: Fast and Accurate Non-Autoregressive Model for Joint Multiple Intent Detection and Slot Filling.* ACL 2021. [arXiv:2106.01925](https://arxiv.org/abs/2106.01925)
6. C.-W. Goo et al. *Slot-Gated Modeling for Joint Slot Filling and Intent Prediction.* NAACL 2018. [code and data](https://github.com/MiuLab/SlotGated-SLU)
7. S. Larson et al. *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction.* EMNLP 2019. [arXiv:1909.02027](https://arxiv.org/abs/1909.02027), [data](https://github.com/clinc/oos-eval)
8. D. Hendrycks, K. Gimpel. *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR 2017. [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)
9. W. Liu et al. *Energy-based Out-of-distribution Detection.* NeurIPS 2020. [arXiv:2010.03759](https://arxiv.org/abs/2010.03759)
10. K. Lee et al. *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS 2018. [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)
11. A. Podolskiy et al. *Revisiting Mahalanobis Distance for Transformer-Based Out-of-Domain Detection.* AAAI 2021. [arXiv:2101.03778](https://arxiv.org/abs/2101.03778)
12. Y. Sun et al. *Out-of-Distribution Detection with Deep Nearest Neighbors.* ICML 2022. [arXiv:2204.06507](https://arxiv.org/abs/2204.06507)
13. J. Ren et al. *Out-of-Distribution Detection and Selective Generation for Conditional Language Models.* ICLR 2023. [arXiv:2209.15558](https://arxiv.org/abs/2209.15558)
14. V. Vovk, A. Gammerman, G. Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
15. A. N. Angelopoulos, S. Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* 2021. [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
16. S. Bates et al. *Testing for Outliers with Conformal p-values.* Annals of Statistics 51(1), 2023. [arXiv:2104.08279](https://arxiv.org/abs/2104.08279)
17. F. den Hengst et al. *Conformal Intent Classification and Clarification for Fast and Accurate Intent Recognition.* Findings of NAACL 2024. [arXiv:2403.18973](https://arxiv.org/abs/2403.18973)
