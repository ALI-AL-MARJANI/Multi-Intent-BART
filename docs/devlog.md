# Development log

## v0.2.0 (October 2026): audit and rework

An audit of v0.1 (commit `9f05eac`) found defects that invalidate every number
the previous README reported. This entry records what was wrong and what
replaced it.

### Defects found in v0.1

| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | Pointer targets were computed by tokenizing each word separately, without the leading space BART's BPE depends on. | On MixATIS dev 20.2% of position targets pointed at the intended token, 64.2% at another token, 15.5% fell outside the input and silently became vocabulary targets (MixSNIPS dev: 30.6 / 54.8 / 14.5%). | `gemis.data.alignment.encode_source` reads positions from the offset mapping of the encoded sentence. Regression test: `tests/test_alignment.py`. |
| 2 | An exclusive span end after the last word pointed at `</s>`, which the decoder mapped back to "no word". | With gold targets as predictions, the v0.1 decoding scored 88.6-91.2 overall accuracy instead of 100. | `position_to_word` maps `</s>` to `n_words`. The oracle round trip is now 100% on all six splits (`tests/test_dataset.py`). |
| 3 | `scripts/evaluate.py` crashed on the first batch (`'list' object has no attribute 'to'`); the default decoding path without word lists scored 13-20 slot F1 with gold targets. | Reproduced locally. | One evaluation loop (`gemis.training.evaluation.predict_frames`) used by the trainer and the scripts. |
| 4 | The output layer scored the full 50k BART vocabulary; the paper scores `N` positions plus `L` intent/slot labels. An unused, untied 51M-parameter `lm_head` was saved in every checkpoint. | Code reading against paper Eq. 6. | `PointerHead` scores `L` labels and `N` positions. |
| 5 | Pointer scores of padded positions were not masked; the AoA cross-attention map was normalised before the padding mask. | Code reading. | Masks applied; `test_outputs_do_not_depend_on_batch_padding`. |
| 6 | The label inventory was collected from train, dev **and test**. | Code reading. | Train split only. Consequence: 113 of 828 MixATIS test utterances carry a label that never occurs in training and cannot be fully correct. |
| 7 | `pyproject.toml` named a build backend that does not exist; `pytest` collected zero tests; launch scripts hard-coded a remote path. | `pip install -e .` and `pytest` output. | Packaging fixed, pytest suite, CI. |

### Numbers removed from the README

| Removed claim | Reason |
|---|---|
| NLU table (89.2 / 81.4 / 53.4 on MixATIS, 97.4 / 98.1 / 87.4 on MixSNIPS, 95.4 / 94.1 / 71.5 on MultiATIS, 98.1 / 98.8 / 91.5 on MultiSNIPS) | These are the results of Li & Zhu (2026). They were never measured with this code. The MixATIS / MixSNIPS rows are now shown only in "Paper" columns; the MultiATIS / MultiSNIPS rows are dropped because the paper's constructed datasets are not public. |
| OOD AUROC 0.9751, FPR@95TPR 0.0991, AUPR 0.8306, TPR 0.9450, FPR 0.0837, threshold 0.0555 | Not reproducible: no result file, no committed OOD set, no checkpoint. Per commit `1faa48e` the model had been trained for one epoch (slot F1 41.7 on dev) with the defects above. The OOD set was drawn from a 90-sentence hard-coded list when no local LLM was available, so 200 "utterances" could contain at most 90 distinct ones, and they differed from the in-distribution data in casing and punctuation. |
| "guaranteed false-alarm rate ≤ 5%" next to a measured 8.37% | With about 2,200 calibration scores the false-alarm rate of a valid split-conformal detector at alpha = 0.05 has a standard deviation of about 0.5 points, so 8.37% was a violation, not noise. The cause cannot be established after the fact because the scores were not saved. The rework measures validity directly (`results/ood/*/metrics.json`, field `conformal`). |
| "the first conformal OOD detector for a generative NLU model" | No literature search supported it. Conformal methods for intent classification with out-of-scope handling exist (Hengst et al., 2024). |
| The "Expected output" block of the Python API example | It was written by hand, not produced by the code. |

### Design changes

See the "Differences from the paper" section of the README.

### Notes on the committed runs

* **`mixatis_bart-base`, seed 13.** Training crashed after epoch 6 (MPS out of
  memory) and was resumed. Between the crash and the resume, batch padding to
  a multiple of 8 (`data.pad_to_multiple_of`), periodic release of the MPS
  cache and the `code_sha256` field were added. Epochs 1-6, including the
  selected epoch 6, were therefore trained with the code of commit `4d54c01`,
  while the result files of this seed record the `code_sha256` of the code
  that ran epochs 7-11, the evaluation and the OOD scoring. Padding is masked
  and `tests/test_model.py::test_outputs_do_not_depend_on_batch_padding`
  checks that it does not change the outputs, but the recorded hash does not
  describe the first six epochs. The `resumed_from_epoch` field of
  `seed13_train.json` reads 11 instead of 6: `scripts/train.py` rewrote the
  file when the finished run was invoked again. The script now leaves the
  file of a finished run unchanged.
* **Seeds 13, 42 and 87 were all resumed at least once** (crash or machine
  shutdown), so none of them is bit-identical to an uninterrupted run; see
  RUNBOOK section 1.
* **Score precision.** `scores.csv` first stored scores with six decimals,
  which made up to 75% of the `first_step_msp` values of the test split tied.
  Scores are now written with nine significant digits and the three seeds
  were re-scored.

