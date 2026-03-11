# Proxy Metric Repository Map (Current Branch State)

This file is a factual map of the proxy-metric pipeline as currently implemented.
It is not a planning memo.

## Pipeline Overview

```
sample params -> rollout (SGP4 + temporal ISLs) -> aggregate resilience targets
-> train proxy models (RF/GNN) -> export predictions -> ranking validation
```

## Canonical Target Set (Trainable End-to-End)

- `partition_any` (classification)
- `partition_fraction` (regression)
- `gcc_frac_min` (regression)
- `gcc_frac_mean` (regression)
- `max_partition_streak` (regression)

`num_components_max` is intentionally not advertised as trainable in this branch.

## Key Runtime Entry Points

- Dataset export: `scripts/export_design_dataset.py`
- RF training: `scripts/train_design_risk_model.py`
- GNN training: `scripts/train_gnn_model.py`
- Ranking validation: `tools/validate_proxy_rankings.py`

## Cache Contract (GNN Dataset)

- Module: `src/satnet/models/gnn_dataset.py`
- Cache utilities: `src/satnet/utils/graph_cache.py`
- Payload mode: `target_agnostic` only
- Cached payload excludes target labels (`y`)
- `y` is always reconstructed from the active row and active `target_name`
- Required metadata:
  - `cache_schema_version`
  - `generator_fingerprint`
  - `payload_mode`
  - `generator_config`
- Invalid/missing metadata, stale schema versions, and legacy target-bound payloads fail loudly.

## Prediction Export Schema

RF and GNN prediction CSVs use the same core columns:

- `config_hash`
- `target_name`
- `task_type`
- `seed`
- `split`
- `sample_idx`
- `y_true`
- `y_pred`

Optional fields may include `run_id`, `model_type`, and `data_path`.

## Experiment Logging

- Logger: `src/satnet/utils/experiment_logger.py`
- RF and GNN training scripts both emit JSONL experiment records with:
  - run metadata (timestamp/git SHA/argv/hostname via logger)
  - model/target/task/seed/data path
  - cache flags
  - best/final metrics
  - model and prediction artifact paths
