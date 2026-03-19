# SatNet Thesis Framework: Tier 1 Architecture

## Executive Summary

This repository is the **Tier 1** implementation of Alex’s dissertation framework for satellite network architecture risk analysis. It is **not** a toy model.

The core of the system is a **physics-compliant, time-evolving simulator** that:

- Propagates LEO constellations using **SGP4** orbital mechanics (via Hypatia-style workflows).
- Computes **inter-satellite links (ISLs)** using Earth-obscuration checks and **1550nm optical / Ka-band RF link budgets**.
- Evaluates **temporal network connectivity** over **t = 0..T** rather than relying on a single static snapshot.
- Produces **reproducible, schema-versioned datasets** for machine learning.

The result is an end-to-end pipeline intended to be **defense-ready**:

- Deterministic configuration hashes.
- Explicit random seeds.
- Pure (non-leaky) labeling functions.
- Separation between physics, simulation, and metrics layers.

---

## System Overview

At a high level, the thesis framework is a three-layer system:

- **Physics Layer (Deterministic):** generates time-evolving satellite graphs from orbital mechanics and link budgets.
- **Simulation Layer (Stochastic):** injects persistent failures and runs temporal rollouts.
- **Metrics / Labels Layer (Pure Math):** computes connectivity metrics and partition labels from graph state only.

The primary dataset artifact is:

- `data/tier1_design_runs.csv` (one row per constellation design + failure assumption + temporal label summary)
- `data/tier1_design_steps.csv` (one row per time step per run; detailed time-series metrics)

---

## Module Breakdown (The “Why”)

### Physics Engine (`src/satnet/network/hypatia_adapter.py`)

This is the authoritative **Tier 1 physics engine** responsible for generating the temporal network topology.

**What it does**

- Builds **Walker Delta** constellations (planes, sats/plane, inclination, altitude).
- Generates synthetic **TLEs** for each satellite.
- Propagates satellite positions over time using:
  - **SGP4** with the **WGS72** gravity model (preferred Tier 1 mode).
  - A simplified Keplerian fallback if SGP4 is unavailable.
- Computes time-indexed ISL graphs using:
  - **Geometry engine:** Earth obscuration (grazing-height style clearance) to reject blocked lines-of-sight.
  - **Link budget engine:** evaluates link viability using:
    - **Optical (1550nm)** free-space path loss and aperture gain.
    - **RF (Ka-band 28GHz)** free-space path loss with antenna gain and a rain margin.
  - A standard **“+Grid”** ISL pattern consistent with common LEO constellation designs:
    - intra-plane neighbor links
    - inter-plane partner links
    - seam link tagging

**Why this matters (Tier 1 vs. toy)**

- The ISL graph is not an arbitrary random graph; it is an outcome of orbital geometry and link physics.
- Connectivity is evaluated **over time**, so transient partitions and sustained degradations are observable.
- Link viability is not “distance < threshold” alone: it is constrained by **line-of-sight** and **link margin**.

---

### Data Factory (`scripts/export_design_dataset.py`)

This script is the canonical generator for the **Tier 1 temporal design dataset**.

**What it does**

- Samples constellation architectures across a design space:
  - Altitude: `--altitude-min .. --altitude-max` (default 300–1200 km)
  - Inclination: `--inclination-min .. --inclination-max` (default 30–98 deg)
  - Planes: `--planes-min .. --planes-max`
  - Sats/plane: `--sats-min .. --sats-max`
- Samples stochastic failure assumptions per run:
  - Node failure probability (persistent failures)
  - Edge failure probability (persistent failures sampled from **t=0 edges only**)
- Runs the Tier 1 temporal rollout via:
  - `src/satnet/simulation/monte_carlo.py` → `generate_tier1_temporal_dataset()`
  - `src/satnet/simulation/tier1_rollout.py` → `run_tier1_rollout()`

**Outputs**

- `data/tier1_design_runs.csv`
  - Design-time parameters (architecture + failure assumptions)
  - Temporal aggregate labels (e.g., `partition_any`, `gcc_frac_mean`, `max_partition_streak`)
  - `seed` and `config_hash` for reproducibility
  - `failed_nodes_json` and `failed_edges_json` for graph reconstruction
  - `schema_version` and `dataset_version` for dataset stability

- `data/tier1_design_steps.csv`
  - Per-step graph metrics (`num_components`, `gcc_frac`, `partitioned`, etc.)

**Why this matters**

- The dataset is **temporal**: it captures dynamic connectivity, not just a static “t=0” snapshot.
- Labels are **non-leaky**:
  - The runs table stores **design-time features** (and configured failure probabilities), while the label is derived strictly from observed connectivity outcomes.
  - Label computation is performed using **pure metric functions**.

---

### The Brain (`src/satnet/models/gnn_model.py`)

This module implements the dissertation’s deep learning model family:

- `SatelliteGNN`: a **GCLSTM** (Graph Convolutional LSTM) model used for
  both classification and regression targets.

**What it does**

- Consumes a **sequence of graph snapshots** (one graph per time step).
- Uses `torch_geometric_temporal.nn.recurrent.GCLSTM` to learn:
  - local structural features (graph convolution)
  - temporal dynamics (recurrent hidden state)
- Pools node embeddings to a graph-level embedding via **global mean pooling**.
- Produces either:
  - logits for binary classification (`partition_any`)
  - a scalar regression output for continuous resilience targets

**Why temporal recurrence captures “state decay”**

Static models see only a single graph and must infer risk from instantaneous structure. A recurrent temporal GNN can learn:

- degradation patterns (e.g., gradual loss of cross-plane connectivity)
- persistence (partition streaks)
- “near-miss” dynamics where the GCC fraction oscillates near the threshold

This is the correct inductive bias for constellation resilience, because resilience is fundamentally a **time-domain property**.

**How training data is provided**

The temporal GNN training set is not stored as precomputed PyG tensors by default.

- `src/satnet/models/gnn_dataset.py` defines `SatNetTemporalDataset`.
- It **regenerates temporal graphs on-the-fly** by reading `data/tier1_design_runs.csv`, instantiating `HypatiaAdapter`, and emitting `List[torch_geometric.data.Data]` (one per time step).

This design avoids ballooning storage/RAM for large numbers of runs (e.g., 2,000+).

The current dataset implementation also supports optional graph-sequence caching
via `src/satnet/utils/graph_cache.py` so repeated experiments can skip
reconstruction when the cache metadata matches.

---

## Metrics & Labeling (Pure, Non-Leaky)

Labeling lives in:

- `src/satnet/metrics/labels.py`

These are pure functions operating on a NetworkX graph, including:

- `compute_gcc_frac(G)`
- `compute_partitioned(gcc_frac, threshold)`
- `aggregate_partition_streaks([...])`

**Critical invariant:** labels are derived from **graph state only**.

---

## How to Reproduce Results (The “How”)

### 1) Environment Setup

Recommended installs:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install scikit-learn scipy joblib
python -m pip install -e ".[ml]"
python -m pip install -r requirements_ml.txt
```

Notes:

- If `sgp4` is not installed, the physics engine will fall back to a simplified Keplerian model.
- Dataset generation is computationally non-trivial; for large runs (e.g., 2,000) use a machine with adequate CPU time.

---

### 2) Generate the Tier 1 Dataset

From the repo root:

```bash
python scripts/export_design_dataset.py --num-runs 2000
```

For strict reproducibility (recommended):

```bash
python scripts/export_design_dataset.py --num-runs 2000 --seed 12345
```

Artifacts written:

- `data/tier1_design_runs.csv`
- `data/tier1_design_steps.csv`

---

### 3) Train Baseline (Random Forest)

The current training CLI supports all canonical resilience targets:

- `partition_any` (classification)
- `partition_fraction` (regression)
- `gcc_frac_min` (regression)
- `gcc_frac_mean` (regression)
- `max_partition_streak` (regression)

The default script path uses the Tier 1 runs table columns present in
`src/satnet/models/risk_model.py` and writes stable prediction exports keyed
by `config_hash`.

Run:

```bash
python scripts/train_design_risk_model.py
```

Artifacts written:

- `models/design_risk_model_tier1.joblib`
- `models/design_risk_model_tier1_metrics.json`
- `models/design_risk_model_tier1_predictions.csv`
- `experiments/rf_log.jsonl`

---

### 4) Train the Temporal GNN (Deep Learning)

Train the GCLSTM model on temporal graph sequences:

```bash
python scripts/train_gnn_model.py
```

Common options:

```bash
python scripts/train_gnn_model.py --epochs 20 --lr 0.01 --hidden-dim 64 --data-dir data/ --device auto
```

Artifacts written:

- `models/satellite_gnn.pt`
- `models/satellite_gnn_predictions.csv`
- `experiments/gnn_log.jsonl`

The GNN training CLI also supports `--target-name`, `--use-cache`,
`--write-cache`, and `--cache-dir`.

---

## What the Code Guarantees

- **Deterministic graph generation** through explicit seeds plus fixed epoch handling.
- **Temporal evaluation** over `t = 0..T`, not a single snapshot.
- **Schema-validated dataset export** for runs and steps tables.
- **Stable prediction joins** through required `config_hash` values in RF and GNN prediction CSVs.
- **Graph reconstruction contract** through `failed_nodes_json` and `failed_edges_json`.

---

## Defense-Ready Reproducibility Guarantees

This repo is designed to withstand dissertation defense scrutiny:

- **Temporal, not static:** rollouts evaluate connectivity at every time step.
- **Determinism:**
  - Explicit `seed` passed through dataset generation and rollouts.
  - A stable propagation epoch is used by default (`J2000.0` via `DEFAULT_EPOCH_ISO`).
  - Each rollout exports a `config_hash`.
- **Non-leaky labels:** labels are computed solely from graph state (not failure parameters).
- **Schema versioning:** dataset writers enforce required columns (`SCHEMA_VERSION`, `DATASET_VERSION`).

---

## Primary Files (Quick Reference)

- **Physics:** `src/satnet/network/hypatia_adapter.py`
- **Temporal rollout contract:** `src/satnet/simulation/tier1_rollout.py`
- **Monte Carlo dataset generator:** `src/satnet/simulation/monte_carlo.py`
- **Dataset export script:** `scripts/export_design_dataset.py`
- **Random Forest training:** `scripts/train_design_risk_model.py`
- **GNN dataset:** `src/satnet/models/gnn_dataset.py`
- **GNN model:** `src/satnet/models/gnn_model.py`
- **GNN training:** `scripts/train_gnn_model.py`
- **Pure labels:** `src/satnet/metrics/labels.py`
