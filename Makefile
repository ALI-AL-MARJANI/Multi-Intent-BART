.PHONY: help install lint format test data tables figures clean

PYTHON ?= python

help:
	@echo "install   editable install with dev extras"
	@echo "lint      ruff check + format check"
	@echo "format    ruff format"
	@echo "test      offline CPU test suite"
	@echo "data      download MixATIS, MixSNIPS and the CLINC150 out-of-scope set"
	@echo "tables    rebuild results/*/summary.{json,md} from the per-seed files and copy them into the README"
	@echo "figures   rebuild results/figures from results/"
	@echo "Training and evaluation commands are in RUNBOOK.md."

install:
	pip install -e ".[dev]"

lint:
	ruff check src scripts tests
	ruff format --check src scripts tests

format:
	ruff format src scripts tests

test:
	$(PYTHON) -m pytest

data:
	$(PYTHON) scripts/download_data.py --dataset mixatis mixsnips
	$(PYTHON) scripts/prepare_ood_data.py

tables:
	$(PYTHON) scripts/aggregate_results.py
	$(PYTHON) scripts/update_readme.py

figures:
	$(PYTHON) scripts/make_figures.py

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
