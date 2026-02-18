# satnet-arch-dss (Tier 1)

**Satellite Network Architecture Decision Support System (DSS)**

This repository is the **Tier 1** refactor: a physics-compliant, **time-evolving** LEO constellation connectivity simulator plus ML baselines for estimating **partition risk** under persistent failures.

For the advisor-facing architecture narrative and methodology framing, see `README_THESIS.md`.

---

## Tier 1 guardrails (Phase 1)

- **No toy topology**: `satnet.network.topology` is not used in `src/`.
- **Temporal, not static**: rollouts evaluate connectivity over **t = 0..T**.
- **Satellite-to-satellite only**: no ground stations/gateways in Phase 1.
- **Deterministic**: explicit `seed` + fixed epoch (`DEFAULT_EPOCH_ISO`) for reproducibility.

## Current execution status (2026-02-17)

Phase 1 remains sat-to-sat by design, but active execution work is focused on
closing Phase 1 and staging Phase 2 correctly:

- Re-generated Tier 1 design dataset at 10k runs (`110k` temporal step rows)
  on remote hardware for a clean post-refactor baseline.
- Re-trained the Tier 1 RF baseline on the refreshed dataset and captured
  reproducible metrics artifacts.
- Started temporal GNN environment/training setup on remote hardware.
- Last completed GNN smoke run metrics: acc=0.6155, prec=0.6615, rec=0.8648,
  f1=0.7496 (TP=1151, FP=589, FN=180, TN=80).
- Current run status: `train_gnn_model.py` 20-epoch full training on
  `data/runs/2026-02-17_full_10k` is in progress on remote CPU.
- Continued external dataset onboarding as a supporting/validation track
  (not Phase 1 truth labels), to feed Phase 2 planning.

For the detailed nightly handoff and Phase 2 entry gates, see:

- `docs/advisor_meeting/2026-02-17_phase1_restart_execution_update.md`

## Architecture (three layers)

- **Physics layer**: `src/satnet/network/hypatia_adapter.py`
- **Simulation layer**: `src/satnet/simulation/tier1_rollout.py` and `src/satnet/simulation/monte_carlo.py`
- **Metrics/labels layer (pure)**: `src/satnet/metrics/labels.py`

## Project structure

```text
src/satnet/
  network/hypatia_adapter.py      # Tier 1 physics: SGP4 + ISLs + link budgets
  simulation/tier1_rollout.py     # Temporal rollout: failures + t=0..T evaluation
  simulation/monte_carlo.py       # Dataset generation + schema validation
  metrics/labels.py               # Pure GCC / partition labels
  models/
    risk_model.py                 # RandomForest baselines (Tier 1 + legacy)
    gnn_dataset.py                # PyG dataset (reconstruct graphs on-the-fly)
    gnn_model.py                  # GCLSTM thesis model
  legacy/                         # Quarantined Tier-0-era artifacts
scripts/
  export_design_dataset.py        # Writes data/tier1_design_{runs,steps}.csv
  export_failure_dataset.py       # Writes data/tier1_failure_{runs,steps}.csv
  failure_sweep.py                # Quick aggregate sweep (Tier 1 pipeline)
  train_design_risk_model.py      # Baseline RF on tier1_design_runs.csv
  train_gnn_model.py              # Temporal GNN training (optional)
docs/datasets/
  tier1_temporal_connectivity_v1_schema.md
```

## Installation

Requires **Python 3.11+**.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Core + dev (pytest)
python -m pip install -e '.[dev]'

# Baseline ML (RandomForest)
python -m pip install scikit-learn

# Optional: temporal GNN dependencies
python -m pip install -r requirements_ml.txt

# Optional: visualization helper
python -m pip install matplotlib
```

## Verify

```bash
pytest -q
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

### Failure dataset (same pipeline, different defaults)

```bash
python scripts/export_failure_dataset.py
```

Outputs:

- `data/tier1_failure_runs.csv`
- `data/tier1_failure_steps.csv`

## Train models

### Baseline: design-time RandomForest

```bash
python scripts/train_design_risk_model.py
```

Outputs:

- `models/design_risk_model_tier1.joblib`
- `models/design_risk_model_tier1_metrics.json`

### Thesis model: temporal GNN (optional)

```bash
python scripts/train_gnn_model.py --data-dir data/ --device auto
```

Output:

- `models/satellite_gnn.pt`

## Legacy artifacts (kept for backward compatibility)

Some scripts/models still target the **legacy** `Data/failure_dataset.csv` format (pre–Tier 1). They are not part of the Tier 1 dataset contract:

- `scripts/train_risk_model.py`
- `scripts/score_risk_examples.py`
