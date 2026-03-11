# Proxy Metric Repository Map

Generated to anchor the proxy-metric / instrumentation work to the actual repo structure.

## Pipeline Overview

```
sample params → propagate (SGP4) → generate temporal graphs → inject failures
  → compute GCC metrics per step → aggregate labels → train ML (RF / GNN)
```

## Key Pipeline Files

| Stage | File | Entry Point |
|---|---|---|
| Orbital propagation & ISL topology | `src/satnet/network/hypatia_adapter.py` | `HypatiaAdapter.calculate_isls()` |
| Rollout config & runner | `src/satnet/simulation/tier1_rollout.py` | `run_tier1_rollout(cfg)` |
| Monte Carlo dataset generation | `src/satnet/simulation/monte_carlo.py` | `generate_tier1_temporal_dataset(cfg)` |
| Pure label functions | `src/satnet/metrics/labels.py` | `compute_gcc_frac()`, `compute_partitioned()`, `aggregate_partition_streaks()` |
| Risk binning | `src/satnet/metrics/risk_binning.py` | `bin_satellite_risk()` |
| RF training (Tier1 v1) | `src/satnet/models/risk_model.py` | `train_tier1_v1_risk_model()`, `train_tier1_v1_design_model()` |
| GNN model definition | `src/satnet/models/gnn_model.py` | `SatelliteGNN` (GCLSTM, binary classification only) |
| GNN dataset (on-the-fly) | `src/satnet/models/gnn_dataset.py` | `SatNetTemporalDataset` |

## Script Entry Points

| Script | Purpose |
|---|---|
| `scripts/export_design_dataset.py` | Generate `tier1_design_{runs,steps}.csv` via Monte Carlo |
| `scripts/train_design_risk_model.py` | Train RF classifier on design params → `partition_any` |
| `scripts/train_gnn_model.py` | Train GCLSTM on temporal graphs → `partition_any` |
| `scripts/train_risk_model.py` | Legacy RF on `failure_dataset.csv` |
| `scripts/export_failure_dataset.py` | Legacy failure dataset export |
| `scripts/failure_sweep.py` | Quick aggregate sweep |
| `scripts/score_risk_examples.py` | Legacy scoring |

## Where Labels Are Computed

1. **Per-step metrics**: `tier1_rollout.py:run_tier1_rollout()` lines 337-343 call pure functions from `metrics/labels.py`.
2. **Aggregation**: `tier1_rollout.py` lines 358-373 compute `gcc_frac_min`, `gcc_frac_mean`, `partition_fraction`, `partition_any`, `max_partition_streak` from step-level data and pack into `Tier1RolloutSummary`.
3. **`Tier1RolloutSummary`** already contains all five key continuous targets.
4. **`Tier1RunRow`** in `monte_carlo.py` already persists all five to CSV.

## Where Target Selection Happens

- `risk_model.py:TIER1_V1_LABEL_COLUMN = "partition_any"` — hardcoded default.
- `train_tier1_v1_risk_model()` accepts `label_column` param but only does binary classification.
- `train_design_risk_model.py` script has no `--target-name` CLI arg.
- `gnn_model.py:SatelliteGNN` has `out_channels=2` (binary only).
- `gnn_dataset.py` hardcodes `label = int(row["partition_any"])` and `y=torch.tensor([label], dtype=torch.long)`.
- `train_gnn_model.py` uses `CrossEntropyLoss` — classification only, no regression path.

## Where Cache Hook Should Be Inserted

- `gnn_dataset.py:SatNetTemporalDataset.get()` regenerates graphs on-the-fly every time.
- The natural cache insertion point is inside `get()` — after computing the cache key from the row config and before/after calling `HypatiaAdapter`.
- An alternative: cache at the `generate_tier1_temporal_dataset()` level in `monte_carlo.py`.

## Where Experiment Logging Should Be Inserted

- `scripts/train_design_risk_model.py:main()` — after training, metrics are written to JSON but no structured experiment log.
- `scripts/train_gnn_model.py:main()` — saves model checkpoint but no structured log.
- Neither script records timing, cache metrics, or target distributions.

## Current Config Structure

- `Tier1RolloutConfig` (frozen dataclass) — constellation + time + failure + labeling params.
- `Tier1MonteCarloConfig` (dataclass) — ranges for sampling, seed, num_runs.
- `RiskModelConfig` (dataclass) — test_size, random_state, n_estimators, max_depth.
- No global config file; params passed via dataclasses and argparse.

## Current Output / Artifact Conventions

```
data/
  tier1_design_runs.csv      # Monte Carlo run summaries
  tier1_design_steps.csv     # Per-step metrics
models/
  design_risk_model_tier1.joblib
  design_risk_model_tier1_metrics.json
  satellite_gnn.pt
```

## Risk Areas and Duplication

- `risk_model.py` has 4 separate train functions (`train_risk_model`, `train_tier1_risk_model`, `train_tier1_v1_risk_model`, `train_tier1_v1_design_model`) with near-identical structure. Adding regression should avoid duplicating again.
- `num_components_max` is not currently computed at the summary level (only per-step `num_components` exists).
- GNN dataset hardcodes `partition_any` label extraction — needs parameterization for target selection.
- No `--target-name` threading anywhere in scripts.

## Assumptions for Implementation

- The five canonical resilience targets (`partition_any`, `partition_fraction`, `gcc_frac_min`, `gcc_frac_mean`, `max_partition_streak`) are already computed correctly in `Tier1RolloutSummary` and persisted in `Tier1RunRow`. The new `resilience_targets.py` module will provide a standalone pure-function interface that can also work from step-level dicts, complementing the existing aggregation in the rollout runner.
- Cache will use `.pt` format for PyG graph sequences, keyed by a SHA256 of normalized config params.
- The artifact directory `artifacts/graph_cache/` will be used for cache storage.
