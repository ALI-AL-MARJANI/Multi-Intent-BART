# Runbook

Exact commands to reproduce every table and figure of the README. All commands
run from the repository root.

## 0. Setup (5 minutes)

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10 - 3.12
pip install -e ".[dev]"                                 # or: pip install -r requirements-lock.txt && pip install -e . --no-deps
python scripts/download_data.py --dataset mixatis mixsnips   # 16 MB, from the AGIF repository
python scripts/prepare_ood_data.py                      # already committed in data/ood/, re-run to verify
pytest                                                  # 50 tests, about 10 s on CPU, no download
```

`requirements-lock.txt` lists the versions the committed results were produced
with (torch 2.8.0, transformers 4.57.1). On a machine where TensorFlow is
installed but broken, prefix commands with `USE_TF=0`.

The device is chosen automatically (`cuda`, then `mps`, then `cpu`); set
`training.device` in a config to override.

## 1. What each run produces

`sh scripts/run_experiment.sh <config> <seed> [<seed> ...]` runs four steps per seed:

| Step | Command | Output |
|---|---|---|
| Train | `python scripts/train.py --config C --seed S` | `checkpoints/<exp>/seed<S>/{best,last}.pt`, `results/nlu/<exp>/seed<S>_train.json` |
| Evaluate | `python scripts/evaluate.py --config C --seed S` | `results/nlu/<exp>/seed<S>_test.json`, `seed<S>_test_predictions.jsonl` |
| Score OOD | `python scripts/score_ood.py --config C --seed S` | `results/ood/<exp>/seed<S>/{scores.csv,meta.json}` |
| OOD metrics | `python scripts/evaluate_ood.py --scores_dir results/ood/<exp>/seed<S>` | `results/ood/<exp>/seed<S>/{metrics.json,calibrator.json}` |

An interrupted training run continues with `RESUME=1 sh scripts/run_experiment.sh ...`
(or `scripts/train.py --resume`), which restarts from `last.pt`.

Every JSON file records the merged config, the seed, the git SHA, a flag
telling whether `src/`, `scripts/` or `configs/` differed from that SHA, the
hardware and the library versions. **Commit code changes before launching a
run**, otherwise the result is flagged `git_dirty: true`.

After any run:

```bash
python scripts/aggregate_results.py   # results/nlu/summary.{json,md}, results/ood/summary.{json,md}
python scripts/make_figures.py        # results/figures/*.png
git add results && git commit -m "exp: <what was run>"
```

The README tables are copies of `results/nlu/summary.md` and `results/ood/summary.md`.

## 2. Runs, hardware and time

Seeds are 13, 42 and 87 everywhere (`seeds` in `configs/_base.yaml`).

Time per epoch on the Apple M4 (MPS, 16 GB, fp32) is measured; see the
`seconds` field of each `seed*_train.json`. Runs stop early after 5 epochs
without improvement of the dev overall accuracy, so the totals below are upper
bounds for 30 epochs. T4 times were **not measured** and are rough estimates.

| # | README table | Config | Seeds | Measured on M4 (MPS) | Estimate on a T4 16 GB |
|---|---|---|---|---|---|
| 1 | Main NLU, MixATIS, bart-base | `mixatis_bart-base` | 3 | 8.3 min/epoch, ≤ 4.2 h/seed | ≤ 1.5 h/seed |
| 2 | Main NLU, MixSNIPS, bart-base (+ far-OOD) | `mixsnips_bart-base` | 3 | not measured; ≈ 3x run 1 (3x more training data) | ≤ 4.5 h/seed |
| 3 | Ablations (MixATIS, bart-base) | `mixatis_bart-base_no-aoa`, `_no-pointer`, `_random-label-init` | 3 each | as run 1 | as run 1 |
| 4 | Near-OOD, held-out intents | `mixsnips_bart-base_holdout` | 3 | not measured; ≈ 2x run 1 | ≤ 3 h/seed |
| 5 | Paper setting, MixATIS, bart-large | `mixatis_bart-large` | 3 | not feasible in reasonable time | ≤ 5 h/seed |
| 6 | Paper setting, MixSNIPS, bart-large | `mixsnips_bart-large` | 3 | not feasible in reasonable time | ≤ 15 h/seed |

Commands:

```bash
sh scripts/run_experiment.sh configs/mixatis_bart-base.yaml 13 42 87                   # 1
sh scripts/run_experiment.sh configs/mixsnips_bart-base.yaml 13 42 87                  # 2
for v in no-aoa no-pointer random-label-init; do                                       # 3
  sh scripts/run_experiment.sh configs/mixatis_bart-base_$v.yaml 13 42 87
done
sh scripts/run_experiment.sh configs/mixsnips_bart-base_holdout.yaml 13 42 87          # 4
sh scripts/run_experiment.sh configs/mixatis_bart-large.yaml 13 42 87                  # 5
sh scripts/run_experiment.sh configs/mixsnips_bart-large.yaml 13 42 87                 # 6
```

Notes for the T4:

* bart-large with batch size 16 fits in 16 GB in fp32 for these sequence
  lengths (sources ≤ 92 subwords, targets ≤ 78 classes). If it does not, set
  `batch_size: 8` and `gradient_accumulation_steps: 2` in the config.
* `mixed_precision: true` enables fp16 autocast. This path is covered by no
  test and no committed result; check the first epochs for `nan` losses.
* On a shared JupyterHub, start runs with `nohup ... > logs/<name>.log 2>&1 &`.
* Budget for everything on a rented T4-class GPU: about 100 GPU-hours as an
  upper bound without early stopping. Runs 1-4 (bart-base) are about 40 hours
  of that and are sufficient for every table except the bart-large rows.

## 3. Checks to run on new results

1. `results/nlu/summary.md`: a difference between two rows that is smaller
   than their standard deviations is not a difference. The McNemar table under
   it gives a paired test per seed for each ablation.
2. `results/ood/summary.md`, conformal table: the "random splits of test"
   column must be at or just below alpha for every score. This is a property
   of the procedure, so a larger value means a bug. The "calibrated on dev"
   column is allowed to deviate; the `dev_vs_test_ks` field of `metrics.json`
   tests whether dev and test scores have the same distribution.
3. `seed*_test.json`, field `error_analysis`: intent-count confusion and the
   slot types with most errors.

## 4. Publishing a checkpoint (optional)

`best.pt` contains the weights and the architecture config and is loaded with
`GEMISModel.from_checkpoint(path)`.

```bash
huggingface-cli login
huggingface-cli upload <user>/gemis-mixsnips-bart-base \
    checkpoints/mixsnips_bart-base/seed13/best.pt best.pt
huggingface-cli upload <user>/gemis-mixsnips-bart-base \
    results/ood/mixsnips_bart-base/seed13/calibrator.json calibrator.json
```

## 5. Not reproduced

* **MultiATIS / MultiSNIPS (paper Table 4).** The paper's constructed datasets
  are not public and the NSP threshold is not reported.
  `scripts/construct_dataset.py` implements Algorithm 1, but its output is a
  different dataset; no result on it is reported.
* **Learning-rate search.** The paper selects the learning rate in
  {1e-5, 2e-5, 3e-5} on dev; every run here uses 2e-5.
