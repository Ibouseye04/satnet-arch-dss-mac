# SatNet Proxy Metric Workflow — convenience targets
# Usage: make <target>

SHELL := /bin/bash
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

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

# ── proxy validation ────────────────────────────────────────────────
.PHONY: proxy-validate
proxy-validate:
	$(PYTHON) tools/validate_proxy_rankings.py \
		models/rf_gcc_frac_min_predictions.csv \
		models/rf_partition_fraction_predictions.csv \
		--id-col config_hash \
		--score-col y_pred \
		--allow-partial \
		--output analysis/proxy_validation_report.json

# ── tests ───────────────────────────────────────────────────────────
.PHONY: test
test:
	@echo "Running pytest with known long-running Tier-1 contract/guardrail suites excluded."
	@echo "Run full coverage manually when needed: $(PYTHON) -m pytest tests/ -v"
	$(PYTHON) -m pytest tests/ -v --ignore=tests/test_tier1_contract.py \
		--ignore=tests/test_tier1_guardrails.py

.PHONY: test-fast
test-fast:
	$(PYTHON) -m pytest tests/metrics/ tests/utils/ -v
