.PHONY: help install lint format type-check test smoke \
        data-mix data-atis data-snips data-multi \
        train-mixatis train-mixsnips train-multiatis train-multisnips \
        eval-mixatis eval-mixsnips eval-multiatis eval-multisnips \
        clean

PYTHON  := USE_TF=0 TOKENIZERS_PARALLELISM=false python3
SCRIPTS := scripts

help:
	@echo "GEMIS — available targets:"
	@echo ""
	@echo "  Setup"
	@echo "    install           Install package in editable mode with dev extras"
	@echo ""
	@echo "  Code quality"
	@echo "    lint              Run ruff linter"
	@echo "    format            Run black formatter"
	@echo "    type-check        Run mypy"
	@echo "    test              Run pytest suite"
	@echo "    smoke             Run fast end-to-end smoke test (no download needed)"
	@echo ""
	@echo "  Data"
	@echo "    data-mix          Download MixATIS + MixSNIPS"
	@echo "    data-atis         Download base ATIS"
	@echo "    data-snips        Download base SNIPS"
	@echo "    data-multi        Construct MultiATIS + MultiSNIPS via BERT NSP"
	@echo ""
	@echo "  Training"
	@echo "    train-mixatis     Train on MixATIS"
	@echo "    train-mixsnips    Train on MixSNIPS"
	@echo "    train-multiatis   Train on MultiATIS"
	@echo "    train-multisnips  Train on MultiSNIPS"
	@echo ""
	@echo "  Evaluation"
	@echo "    eval-mixatis      Evaluate best MixATIS checkpoint"
	@echo "    eval-mixsnips     Evaluate best MixSNIPS checkpoint"
	@echo "    eval-multiatis    Evaluate best MultiATIS checkpoint"
	@echo "    eval-multisnips   Evaluate best MultiSNIPS checkpoint"
	@echo ""
	@echo "    clean             Remove build artifacts and caches"

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/ notebooks/

type-check:
	mypy src/

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -u tests/fast_smoke_test.py

# ── Data ──────────────────────────────────────────────────────────────────────

data-mix:
	$(PYTHON) $(SCRIPTS)/download_data.py --dataset mixatis  --output data/raw/
	$(PYTHON) $(SCRIPTS)/download_data.py --dataset mixsnips --output data/raw/

data-atis:
	$(PYTHON) $(SCRIPTS)/download_data.py --dataset atis  --output data/raw/

data-snips:
	$(PYTHON) $(SCRIPTS)/download_data.py --dataset snips --output data/raw/

data-multi: data-atis data-snips
	$(PYTHON) $(SCRIPTS)/construct_dataset.py \
		--input  data/raw/atis/train.txt \
		--output data/processed/multiatis/ --split train --tau 0.5
	$(PYTHON) $(SCRIPTS)/construct_dataset.py \
		--input  data/raw/atis/dev.txt \
		--output data/processed/multiatis/ --split dev   --tau 0.5
	$(PYTHON) $(SCRIPTS)/construct_dataset.py \
		--input  data/raw/atis/test.txt \
		--output data/processed/multiatis/ --split test  --tau 0.5
	$(PYTHON) $(SCRIPTS)/construct_dataset.py \
		--input  data/raw/snips/train.txt \
		--output data/processed/multisnips/ --split train --tau 0.5
	$(PYTHON) $(SCRIPTS)/construct_dataset.py \
		--input  data/raw/snips/dev.txt \
		--output data/processed/multisnips/ --split dev   --tau 0.5
	$(PYTHON) $(SCRIPTS)/construct_dataset.py \
		--input  data/raw/snips/test.txt \
		--output data/processed/multisnips/ --split test  --tau 0.5

# ── Training ──────────────────────────────────────────────────────────────────

train-mixatis:
	$(PYTHON) $(SCRIPTS)/train.py --config configs/mixatis.yaml

train-mixsnips:
	$(PYTHON) $(SCRIPTS)/train.py --config configs/mixsnips.yaml

train-multiatis:
	$(PYTHON) $(SCRIPTS)/train.py --config configs/multiatis.yaml

train-multisnips:
	$(PYTHON) $(SCRIPTS)/train.py --config configs/multisnips.yaml

# ── Evaluation ────────────────────────────────────────────────────────────────

eval-mixatis:
	$(PYTHON) $(SCRIPTS)/evaluate.py \
		--checkpoint checkpoints/mixatis/best.pt \
		--config     configs/mixatis.yaml

eval-mixsnips:
	$(PYTHON) $(SCRIPTS)/evaluate.py \
		--checkpoint checkpoints/mixsnips/best.pt \
		--config     configs/mixsnips.yaml

eval-multiatis:
	$(PYTHON) $(SCRIPTS)/evaluate.py \
		--checkpoint checkpoints/multiatis/best.pt \
		--config     configs/multiatis.yaml

eval-multisnips:
	$(PYTHON) $(SCRIPTS)/evaluate.py \
		--checkpoint checkpoints/multisnips/best.pt \
		--config     configs/multisnips.yaml

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"       -delete 2>/dev/null || true
