# Proxy Metric Workflow

Copy-pasteable commands for the proxy-metric / instrumentation pipeline.

All commands assume you are in the repository root.

Install ML extras before RF/GNN training:

```bash
python -m pip install -e ".[ml]"
```

`[ml]` includes `torch-geometric-temporal`, required by the GNN path.

---

## A. Generate Dataset (Build Cache Source)

```bash
python scripts/export_design_dataset.py \
    --num-runs 500 \
    --seed 42 \
    --duration 10 \
    --step-seconds 60 \
    --output-dir data/
```

This produces `data/tier1_design_runs.csv` and `data/tier1_design_steps.csv`.

---

## B. Train RF on Binary Target (partition_any)

```bash
python scripts/train_design_risk_model.py \
    --target-name partition_any \
    --seed 42
```

Output: `models/design_risk_model_tier1.joblib`, `models/design_risk_model_tier1_metrics.json`
and `models/design_risk_model_tier1_predictions.csv`.

---

## C. Train RF on Continuous Target (gcc_frac_min)

```bash
python scripts/train_design_risk_model.py \
    --target-name gcc_frac_min \
    --seed 42
```

Output: `models/rf_gcc_frac_min.joblib`, `models/rf_gcc_frac_min_metrics.json`, `models/rf_gcc_frac_min_predictions.csv`

Experiment log appended to `experiments/rf_log.jsonl`.

---

## D. Train GNN on gcc_frac_min (Regression)

```bash
python scripts/train_gnn_model.py \
    --target-name gcc_frac_min \
    --epochs 20 \
    --lr 0.01 \
    --hidden-dim 64 \
    --output-model models/gnn_gcc_frac_min.pt \
    --data-dir data/
```

For caching graph sequences (speeds up subsequent runs):

```bash
python scripts/train_gnn_model.py \
    --target-name gcc_frac_min \
    --epochs 20 \
    --write-cache \
    --output-model models/gnn_gcc_frac_min.pt

# Second run loads from cache:
python scripts/train_gnn_model.py \
    --target-name gcc_frac_min \
    --epochs 20 \
    --use-cache \
    --output-model models/gnn_gcc_frac_min.pt
```

Output: `models/gnn_gcc_frac_min.pt`, `models/gnn_gcc_frac_min_predictions.csv`

---

## E. Train GNN on Binary Target (Classification, Backwards Compatible)

```bash
python scripts/train_gnn_model.py \
    --target-name partition_any \
    --epochs 20 \
    --output-model models/satellite_gnn.pt
```

---

## F. Audit Dataset Targets

```bash
python tools/analyze_dataset_targets.py data/tier1_design_runs.csv
```

Optional CSV output:

```bash
python tools/analyze_dataset_targets.py data/tier1_design_runs.csv \
    --output analysis/target_summary.csv \
    --output-corr analysis/target_correlations.csv
```

Reports: per-target summary stats, Spearman correlation matrix, duplicate design signatures.

---

## G. Validate Proxy Rankings

Compare RF predictions on two targets to see if a cheap proxy preserves ranking:

```bash
# Train both targets first
python scripts/train_design_risk_model.py --target-name gcc_frac_min --seed 42
python scripts/train_design_risk_model.py --target-name partition_fraction --seed 42

# Compare rankings
python tools/validate_proxy_rankings.py \
    models/rf_gcc_frac_min_predictions.csv \
    models/rf_partition_fraction_predictions.csv \
    --id-col config_hash \
    --score-col y_pred \
    --top-k 5,10,20 \
    --allow-partial \
    --output analysis/proxy_validation_report.json
```

Reports: Spearman rho, Kendall tau, pairwise ordering accuracy, top-k overlap, NDCG@k.

---

## H. Convert Experiment Log to CSV

```bash
PYTHONPATH=src python -c "
from satnet.utils.experiment_logger import jsonl_to_csv
jsonl_to_csv('experiments/rf_log.jsonl', 'experiments/rf_summary.csv')
"
```

---

## Available Resilience Targets

| Target | Type | Description |
|---|---|---|
| `partition_any` | classification | 1 if any timestep partitioned |
| `partition_fraction` | regression | Fraction of timesteps partitioned |
| `gcc_frac_min` | regression | Minimum GCC fraction across timesteps |
| `gcc_frac_mean` | regression | Mean GCC fraction across timesteps |
| `max_partition_streak` | regression | Longest consecutive partitioned streak |

The primary recommended target for continuous resilience severity is **`gcc_frac_min`**.
