# satnet-arch-dss (Tier 1)

**Satellite Network Architecture Decision Support System (DSS)**

This repository is the **Tier 1** implementation: a physics-compliant, **time-evolving** LEO constellation connectivity simulator plus ML tooling for resilience-target prediction and proxy-metric analysis.

For the advisor-facing architecture narrative and methodology framing, see `README_THESIS.md`.

---

## Tier 1 guardrails (Phase 1)

- **No toy topology**: `satnet.network.topology` is not used in `src/`.
- **Temporal, not static**: rollouts evaluate connectivity over **t = 0..T**.
- **Satellite-to-satellite only**: no ground stations/gateways in Phase 1.
- **Deterministic**: explicit `seed` + fixed epoch (`DEFAULT_EPOCH_ISO`) for reproducibility.

## Current code state

- The canonical simulation path is:
  `HypatiaAdapter` -> `tier1_rollout` -> `monte_carlo` -> schema-validated runs/steps CSVs.
- Legacy static-snapshot code is quarantined under `src/satnet/legacy/`.
- Supported resilience targets are:
  `partition_any`, `partition_fraction`, `gcc_frac_min`, `gcc_frac_mean`, and `max_partition_streak`.
- `scripts/train_design_risk_model.py` trains RandomForest classification or regression models against those targets and writes prediction CSVs with stable `config_hash` join keys.
- `scripts/train_gnn_model.py` reconstructs temporal graph sequences from `tier1_design_runs.csv`, supports cache read/write, and writes both model checkpoints and prediction CSVs.
- Proxy-metric workflow docs live in:
  `docs/proxy_metric_workflow.md` and `docs/proxy_metric_repo_map.md`.

## Architecture (three layers)

- **Physics layer**: `src/satnet/network/hypatia_adapter.py`
- **Simulation layer**: `src/satnet/simulation/tier1_rollout.py` and `src/satnet/simulation/monte_carlo.py`
- **Metrics/labels layer (pure)**: `src/satnet/metrics/labels.py` and `src/satnet/metrics/resilience_targets.py`

## Project structure

```text
src/satnet/
  network/hypatia_adapter.py      # Tier 1 physics: SGP4 + ISLs + link budgets
  simulation/tier1_rollout.py     # Temporal rollout: failures + t=0..T evaluation
  simulation/monte_carlo.py       # Dataset generation + schema validation
  metrics/labels.py               # Pure GCC / partition labels
  metrics/resilience_targets.py   # Canonical aggregate target computation
  models/
    risk_model.py                 # RandomForest training helpers (classification + regression)
    gnn_dataset.py                # PyG dataset with graph reconstruction + cache support
    gnn_model.py                  # GCLSTM-based temporal model
  utils/
    experiment_logger.py          # JSONL experiment logging
    graph_cache.py                # Graph-sequence cache helpers
  legacy/                         # Quarantined Tier-0-era artifacts
scripts/
  export_design_dataset.py        # Writes data/tier1_design_{runs,steps}.csv
  export_failure_dataset.py       # Writes data/tier1_failure_{runs,steps}.csv
  failure_sweep.py                # Quick aggregate sweep (Tier 1 pipeline)
  train_design_risk_model.py      # RF training on supported resilience targets
  train_gnn_model.py              # Temporal GNN training on supported targets
tools/
  analyze_dataset_targets.py      # Dataset/target audits
  validate_proxy_rankings.py      # Cross-target ranking validation
docs/datasets/
  tier1_temporal_connectivity_v1_schema.md
```

## Installation

Requires **Python 3.11+**.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Core + dev tooling
python -m pip install -e '.[dev]'

# RF/proxy workflow
python -m pip install scikit-learn scipy joblib

# Optional: GNN stack
python -m pip install -e '.[ml]'

# Optional compatibility path for the same GNN stack
python -m pip install -r requirements_ml.txt
```

## Verify

```bash
./.venv/bin/python -m pytest
```

## Quickstart: run a single Tier 1 rollout

This runs a small constellation for a short duration and prints temporal connectivity metrics.

```bash
python - <<'PY'
from satnet.simulation.tier1_rollout import Tier1RolloutConfig, run_tier1_rollout

cfg = Tier1RolloutConfig(
    num_planes=2,
    sats_per_plane=3,
    duration_minutes=3,
    step_seconds=60,
    gcc_threshold=0.8,
    node_failure_prob=0.10,
    edge_failure_prob=0.10,
    seed=42,
)

steps, summary, failures = run_tier1_rollout(cfg)

print("=== Tier 1 Rollout Summary ===")
print(f"config_hash:        {summary.config_hash}")
print(f"num_steps:          {summary.num_steps}")
print(f"gcc_frac_min:       {summary.gcc_frac_min:.3f}")
print(f"gcc_frac_mean:      {summary.gcc_frac_mean:.3f}")
print(f"partition_any:      {summary.partition_any}")
print(f"partition_fraction: {summary.partition_fraction:.3f}")
print(f"failed_nodes:       {summary.num_failed_nodes}")
print(f"failed_edges:       {summary.num_failed_edges}")

print("\nFirst 3 steps:")
for s in steps[:3]:
    print(f"t={s.t:2d} | nodes={s.num_nodes:3d} | edges={s.num_edges:3d} | gcc_frac={s.gcc_frac:.3f} | partitioned={s.partitioned}")

print("\nFailure realization (for reconstruction):")
print(f"failed_nodes sample: {sorted(list(failures.failed_nodes))[:10]}")
print(f"failed_edges sample: {sorted(list(failures.failed_edges))[:5]}")
PY
```

## Generate datasets (Tier 1)

### Design dataset (canonical)

```bash
python scripts/export_design_dataset.py --num-runs 200 --seed 42
```

Outputs:

- `data/tier1_design_runs.csv`
- `data/tier1_design_steps.csv`

The runs table includes `config_hash`, `seed`, `epoch_iso`,
`failed_nodes_json`, and `failed_edges_json` so graph sequences can be
reconstructed deterministically for downstream ML.

### Failure dataset (same pipeline, different defaults)

```bash
python scripts/export_failure_dataset.py
```

Outputs:

- `data/tier1_failure_runs.csv`
- `data/tier1_failure_steps.csv`

## Train models

### RandomForest

```bash
python scripts/train_design_risk_model.py
```

Outputs:

- `models/design_risk_model_tier1.joblib`
- `models/design_risk_model_tier1_metrics.json`
- `models/design_risk_model_tier1_predictions.csv`
- `experiments/rf_log.jsonl`

The default CLI target is `partition_any`. For continuous targets, pass
`--target-name` with one of:
`partition_fraction`, `gcc_frac_min`, `gcc_frac_mean`, `max_partition_streak`.

### Thesis model: temporal GNN (optional)

```bash
python scripts/train_gnn_model.py --data-dir data/ --device auto
```

Output:

- `models/satellite_gnn.pt`
- `models/satellite_gnn_predictions.csv`
- `experiments/gnn_log.jsonl`

The GNN script also supports `--target-name`, `--use-cache`, `--write-cache`,
and `--cache-dir` for repeated graph-sequence experiments.

## Workflow references

- `docs/proxy_metric_workflow.md` contains copy-pasteable dataset, RF, GNN, audit, and proxy-validation commands.
- `docs/proxy_metric_repo_map.md` summarizes the current branch-level proxy-metric contracts and artifact formats.

## Legacy artifacts (kept for backward compatibility)

Some scripts/models still target the **legacy** `Data/failure_dataset.csv` format (pre–Tier 1). They are not part of the Tier 1 dataset contract:

- `scripts/train_risk_model.py`
- `scripts/score_risk_examples.py`
