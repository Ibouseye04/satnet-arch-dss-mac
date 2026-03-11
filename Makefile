# SatNet Proxy Metric Workflow — convenience targets
# Usage: make <target>

SHELL := /bin/bash
PYTHON ?= python

# ── data generation ─────────────────────────────────────────────────
.PHONY: dataset
dataset:
	$(PYTHON) scripts/export_design_dataset.py --num-runs 500 --seed 42

# ── RF training ─────────────────────────────────────────────────────
.PHONY: train-rf-binary
train-rf-binary:
	$(PYTHON) scripts/train_design_risk_model.py --target-name partition_any

.PHONY: train-rf-gcc
train-rf-gcc:
	$(PYTHON) scripts/train_design_risk_model.py --target-name gcc_frac_min

# ── GNN training ────────────────────────────────────────────────────
.PHONY: train-gnn-binary
train-gnn-binary:
	$(PYTHON) scripts/train_gnn_model.py --target-name partition_any --epochs 20

.PHONY: train-gnn-gcc
train-gnn-gcc:
	$(PYTHON) scripts/train_gnn_model.py --target-name gcc_frac_min --epochs 20 \
		--output-model models/gnn_gcc_frac_min.pt

# ── analysis ────────────────────────────────────────────────────────
.PHONY: audit
audit:
	$(PYTHON) tools/analyze_dataset_targets.py data/tier1_design_runs.csv

# ── tests ───────────────────────────────────────────────────────────
.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v --ignore=tests/test_tier1_contract.py \
		--ignore=tests/test_tier1_guardrails.py

.PHONY: test-fast
test-fast:
	$(PYTHON) -m pytest tests/metrics/ tests/utils/ -v
