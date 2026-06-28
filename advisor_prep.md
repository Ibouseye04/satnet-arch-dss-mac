# Advisor Prep

Prepared: 2026-06-27 21:25 CDT

## Repository Status

Initial audit before edits:

- Current branch reported as `main`.
- `git fetch origin --prune` completed successfully.
- `main...origin/main` was `0 ahead / 0 behind`.
- No uncommitted changes, no untracked files, and no merge conflicts were present.

Current status after the sprint:

- Branch now reports as `docs/alex-tailscale-windows-ssh`.
- Reflog shows a checkout from `main` to `docs/alex-tailscale-windows-ssh` at `2026-06-27 21:20:19 -0500`; this was not performed by Codex.
- `HEAD...origin/main` is still `0 ahead / 0 behind`; no merge conflicts.
- Modified files are limited to smoke/evaluation/split improvements and tests.
- Generated smoke artifacts are under `artifacts/smoke/`.

## Architecture Overview

```
Layer 1: Physics
  Walker Delta config -> TLE generation -> SGP4/Keplerian propagation
  -> LOS + link budget -> time-indexed NetworkX graph G_t

Layer 2: Simulation
  G_t sequence -> persistent node/edge failures -> effective graph G'_t
  -> pure GCC metrics per timestep -> run-level resilience targets

Layer 3: ML
  Runs table -> Random Forest tabular baseline
  Runs table -> graph reconstruction -> complete temporal graph sequence -> GCLSTM
```

Evidence:

- Walker Delta, SGP4, link budget constants: `src/satnet/network/hypatia_adapter.py:95`, `src/satnet/network/hypatia_adapter.py:143`.
- LOS and +Grid ISL generation: `src/satnet/network/hypatia_adapter.py:707`, `src/satnet/network/hypatia_adapter.py:772`.
- Graph generation and iteration: `src/satnet/network/hypatia_adapter.py:1164`, `src/satnet/network/hypatia_adapter.py:1246`.
- Temporal rollout and failure semantics: `src/satnet/simulation/tier1_rollout.py:239`.
- Dataset generator and schema writer: `src/satnet/simulation/monte_carlo.py:172`, `src/satnet/simulation/monte_carlo.py:440`.
- Pure labels: `src/satnet/metrics/labels.py:15`.
- RF trainer: `src/satnet/models/risk_model.py:664`.
- GNN dataset and model: `src/satnet/models/gnn_dataset.py:61`, `src/satnet/models/gnn_model.py:29`.

## Implementation Status

| Stage | Status | Evidence |
|---|---|---|
| Constellation design | Implemented | Walker Delta config includes planes, sats/plane, inclination, altitude, phasing. |
| Physics simulation | Implemented | SGP4 path with deterministic J2000 fallback; Keplerian fallback exists. |
| LOS + link budget | Implemented | Earth obscuration plus optical/RF link budget. |
| Dynamic graph sequence | Implemented | `iter_graphs()` yields all cached time steps. |
| Failure injection | Implemented with v1 limitation | Persistent failures sampled at `t=0`; later-appearing edges are immune in v1. |
| Monte Carlo | Implemented | `generate_tier1_temporal_dataset()` writes runs and steps tables. |
| RF dataset/training/eval | Implemented and improved | Added run-level train/val/test split and explicit metrics/artifacts. |
| Temporal GNN | Partially implemented | Dataset/model/trainer exist; local run blocked by `torch_geometric_temporal`. |
| Visualization | Implemented for RF smoke | Confusion matrix and prediction-vs-actual PNGs generated. |

## Random Forest Status

Feature coverage:

- Present: `altitude_km`, `inclination_deg`, `num_planes`, `sats_per_plane`, `node_failure_prob`, `edge_failure_prob`, `duration_minutes`, `step_seconds`.
- Present but derived: `total_satellites`.
- Missing as dataset feature: `phasing_factor`; it exists in rollout config but is fixed/defaulted in current dataset exports.

Target coverage:

- Present: `gcc_frac_min`, `gcc_frac_mean`, `partition_fraction`, `max_partition_streak`, `partition_any`.
- Not implemented as a training target: `risk_tier`; risk binning exists separately in `src/satnet/metrics/risk_binning.py`.

Leakage controls added:

- `RiskModelConfig` now has `val_size`.
- `train_rf_model()` now uses grouped run-level splits, preserving all rows with a `run_id` in one split.
- Classification metrics now explicitly include accuracy, precision, recall, F1, ROC AUC, and confusion matrix.
- Regression metrics include MAE, RMSE, R2, Spearman, and Kendall.

Smoke RF results:

- Dataset: 24 runs, 48 steps, partition probability `0.792`.
- Classification target `partition_any`: test accuracy `0.833`, precision `1.000`, recall `0.800`, F1 `0.889`.
- Regression target `gcc_frac_min`: test MAE `0.234`, RMSE `0.258`, R2 `0.342`.
- Outputs: `artifacts/smoke/rf/*`.

## Temporal GNN Status

Correct design points:

- One dataset item is one complete simulation run: `SatNetTemporalDataset.get()` returns `List[Data]`.
- Graphs are reconstructed from run rows using `epoch_iso`, duration, step size, and failure realization JSON.
- Node features currently encode normalized plane index, normalized satellite index, and existence flag.
- Edge features encode normalized distance, margin, link type, and link mode.
- Model uses GCLSTM plus global mean pooling and a linear prediction head.
- Training script now creates train/validation/test splits over complete run indices.
- Checkpoint selection is now validation-based; held-out test is evaluated after restoring the best validation checkpoint.

Current blocker:

- Local GNN smoke failed because `torch_geometric_temporal` is unavailable.
- Attempting `pip install -e '.[ml]'` failed while building `torch-sparse`.
- This is an environment/package wheel issue, not a repository logic failure.

## Data Leakage Audit

- `run_id` is preserved in runs/steps CSVs and prediction CSVs.
- RF splitting now groups by `run_id`.
- GNN splitting is by complete dataset index, where each index is a full run sequence.
- Target-derived diagnostics such as `num_failed_nodes`, `num_failed_edges`, `failed_nodes_json`, and step-level GCC metrics are not RF input features.
- Failure probabilities are included as input features; this is not leakage if the prediction task is "risk under stated failure assumptions." Use design-only features for architecture-only risk.
- Test data is no longer used for GNN checkpoint selection.

## Execution Record

Successful:

- `/tmp/satnet-arch-dss-py312/bin/python -m pytest tests/ -q --ignore=tests/test_tier1_contract.py --ignore=tests/test_tier1_guardrails.py`
  - Result: `160 passed, 1 skipped, 1 warning`.
- `/tmp/satnet-arch-dss-py312/bin/python -m pytest tests/test_tier1_guardrails.py -q`
  - Result: `28 passed`.
- `/tmp/satnet-arch-dss-py312/bin/python -m pytest tests/test_tier1_contract.py -q`
  - Result: `8 passed`.
- `/tmp/satnet-arch-dss-py312/bin/python scripts/export_design_dataset.py --smoke --seed 42`
  - Runtime: `0.85s`; wrote 24 runs and 48 steps.
- `/tmp/satnet-arch-dss-py312/bin/python scripts/train_design_risk_model.py --smoke --target-name partition_any --seed 42`
  - Runtime: `17.1s`; produced model, metrics, predictions, feature CSV, confusion matrix, prediction plot.
- `/tmp/satnet-arch-dss-py312/bin/python scripts/train_design_risk_model.py --smoke --target-name gcc_frac_min --seed 42`
  - Runtime: `2.0s`; produced regression metrics and prediction plot.
- `/tmp/satnet-arch-dss-py312/bin/python tools/analyze_dataset_targets.py data/tier1_design_runs.csv --output artifacts/smoke/target_summary.csv --output-corr artifacts/smoke/target_correlations.csv`
  - Result: target summary and Spearman correlations generated.

Failed or blocked:

- `.venv` is Python `3.14.5`; installing RF dependencies there creates a NumPy/SciPy conflict with the project pin.
- GNN smoke failed with: `No module named 'torch_geometric_temporal'`.
- Installing `torch-geometric-temporal` failed at `torch-sparse` build preparation.

## Demo Procedure

Use a Python 3.11 or 3.12 environment, not the current `.venv` Python 3.14 environment.

```
python scripts/export_design_dataset.py --smoke --seed 42
python scripts/train_design_risk_model.py --smoke --target-name partition_any --seed 42
python scripts/train_design_risk_model.py --smoke --target-name gcc_frac_min --seed 42
python tools/analyze_dataset_targets.py data/tier1_design_runs.csv
```

For tomorrow, demonstrate RF artifacts and explain that the GNN path is architecturally implemented but blocked locally by optional PyG temporal dependency installation.

## Current Risks

- GNN dependency installation is the largest demo blocker.
- Smoke dataset is intentionally tiny; metrics prove pipeline execution, not model quality.
- `phasing_factor` is not sampled/exported as a current RF feature.
- Edge failure sampling from `t=0` edges only is documented but should be expanded later.
- The current branch changed externally during the sprint; confirm branch before committing.

## Advisor Talking Points

- The truth labels come from physics-based temporal simulation, not random graph generators.
- Labels are pure functions of graph state; they do not inspect failure parameters.
- RF is the interpretable baseline; GNN is the thesis model for temporal graph structure.
- Splits are by run, not timestep, preventing temporal leakage.
- Test data is held out for final evaluation, not checkpoint selection.
- Tonight's metrics are smoke-scale evidence; larger runs are the next experimental step.
