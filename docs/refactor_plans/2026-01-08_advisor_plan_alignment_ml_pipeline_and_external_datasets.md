# Advisor Plan Alignment — ML Pipeline + External Dataset Ingestion (Phase 1: Sat-to-Sat)

**Date:** 2026-01-08  
**Status:** Draft (execution plan only; no code changes in this doc)  

## Purpose

Align the repository implementation with the advisor-facing methodology deck (Data Collection → Preprocessing → Feature Selection → Model Dev → Tuning → Evaluation), while preserving Tier 1 guardrails:

- **NO TOY TOPOLOGY** (do not use `satnet.network.topology`).
- **TEMPORAL EVALUATION** is mandatory (`t=0..T`).
- **PHASE 1 scope** remains **satellite-to-satellite** connectivity only.
- **Determinism**: all outputs reproducible via `config_hash + seed + epoch (+ code version)`.
- Large jobs (dataset generation / training) run on a tunneled remote machine.

This plan focuses on **ML pipeline alignment** and **external dataset ingestion scaffolding**. Ground segment + GUI + traffic/routing integration are captured as **Phase 2** roadmap items.

---

## Alignment Target (Advisor Methodology → Repo Mapping)

| Advisor Step | What the deck implies | Current repo status | Gap to close |
|---|---|---|---|
| 1. Data Collection & EDA | Download “real” datasets + run EDA | Tier 1 produces canonical simulated truth datasets; no external loaders | Add external dataset manifests + loaders + EDA scripts |
| 2. Data Pre-processing | Cleaning, encoding, balancing, split | Tier 1 dataset already schema-stable; ad-hoc preprocessing for ML | Add shared preprocessing utilities + deterministic splits |
| 3. Feature Selection | Define features + risk values | Tier 1 features exist; needs formal feature catalog + no-leakage framing | Add feature catalogs + enforce non-leaky feature sets |
| 4. Space Segment ML | Physics-based temporal graphs | Implemented (HypatiaAdapter + rollout + baseline RF + temporal GNN) | Add evaluation harness; optionally add LightGBM baseline to match deck |
| 4.5 “Generate Simulated Data” | Monte Carlo simulation for training | Implemented (`tier1_rollout`, `monte_carlo`, export scripts) | Add provenance manifests + stable artifact layout |
| 5. Ground Segment ML | SAT–GS / gateways modeled | **Deferred (Phase 2)** | Add interfaces + contracts only |
| 6. GUI | Interactive parameter manipulation | **Deferred (Phase 2)** | Add minimal CLI “runner” first; GUI later |
| 7. Parameter Tuning | Search / Bayesian optimization | Partial (manual knobs) | Add standardized experiment runner + metrics export |
| 8. Evaluate / Validate | Accuracy + runtime + external baseline comparisons | Partial | Add evaluation harness + runtime measurement + external validation script |

---

## External Datasets to Ingest (Downloaded / Requested)

These datasets should **not** be committed to git. We will standardize **paths, manifests, loaders, and EDA**, and keep integration into the Tier 1 pipeline strictly controlled.

- **LEO Congestion Control Logs** (12.71 GB)  
  Source: figshare (INFOCOM 2026)  
  Link: https://figshare.com/s/32821d6f5940f72fd946

- **NASA SMAP/MSL Anomaly Dataset** (~50k records; many files)  
  Source: Kaggle  
  Link: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl

- **Synthetic Satellite Health Data** (~10k rows)  
  Source: Kaggle  
  Link: https://www.kaggle.com/datasets/jeffdjeffd/synthetic-data-satellite-health

- **SatFlow Planning Dataset** (author request; 2023-2024)  
  Source: ResearchGate “Request file” (Cen et al.)  
  Link: https://researchgate.net/publication/387540164_SatFlow_Scalable_Network_Planning_for_LEO_Mega-Constellations

- **LEO Routing Simulation Logs** (author request; 2020)  
  Source: ResearchGate “Request file” (Rabjerg et al.)  
  Link: https://researchgate.net/publication/342377712_Metrics_for_multirate_routing_in_dense_LEO_constellations

### Why these can be “real” in an advisor-safe way

- Congestion logs and telemetry anomaly datasets are **externally sourced**.
- They are **not** direct truth labels for “LEO ISL partition risk”. They are best used as:
  - **Traffic/QoS modeling inputs** (Phase 2), and/or
  - **Satellite health/anomaly priors** that could inform failure processes (future work).

---

## Proposed Data Layout (do not bloat repo)

Standardize on a single local data root (on the remote machine):

- `SATNET_DATA_ROOT=/path/to/big_disk/satnet_data`

Within that:

- `external/leo_congestion_control/` (raw downloads)
- `external/nasa_smap_msl/` (raw downloads)
- `external/synthetic_satellite_health/` (raw downloads)
- `external/satflow_planning/` (author-requested)
- `external/leo_routing_sim_logs/` (author-requested)
- `derived/` (EDA summaries, extracted features, train/test splits)

Repo `data/` remains for **small** Tier 1 artifacts only (CSV/Parquet outputs from our generator).

---

# Phase 1 Implementation Plan (Atomic Steps)

Each step is intended as a small PR with explicit acceptance checks.

## Step 0 — Freeze the “Two Tracks” story (advisor-safe)

**Goal:** Align language without breaking Tier 1 constraints.

- Document that we have two data tracks:
  - **Track A (primary):** Tier 1 simulated truth datasets (temporal connectivity under failures).
  - **Track B (supporting):** external datasets (traffic/telemetry/health) for EDA + future integration.

**Acceptance:** A short section exists in docs (or README_THESIS) stating Phase 1 vs Phase 2 and “truth data” definition.

---

## Step 1 — Add an external dataset manifest + path contract

**Goal:** Make external downloads usable without committing data.

- Add `docs/datasets/external_datasets_manifest.md`:
  - dataset name, source URL, license notes (if known), expected on-disk layout
  - acquisition status (downloaded vs author-requested) and expected file count (when known)
  - optional checksums if available
- Add a tiny `satnet.data.paths` module:
  - resolves `SATNET_DATA_ROOT`
  - raises a clear error if missing

**Acceptance:** Unit tests validate path resolution and error messaging without requiring real downloads.

---

## Step 2 — Implement external dataset loaders (read-only, lazy)

**Goal:** Provide authoritative ingestion points for EDA and feature extraction.

Create `src/satnet/data/external/` with loaders:

- `leo_congestion_control.py`
  - lazy iterator over archives/files
  - returns `pd.DataFrame` batches + metadata
- `nasa_smap_msl.py`
  - loads the canonical SMAP/MSL structures (`.npy` / `.csv`)
  - yields standardized per-signal frames with anomaly labels
- `synthetic_satellite_health.py`
  - loads CSV; validates required columns

- (once acquired) `satflow_planning.py`
  - loads planning records; validates schema
- (once acquired) `leo_routing_sim_logs.py`
  - loads routing simulations/logs; validates schema

**Acceptance:** Each loader has:
- deterministic behavior
- schema validation
- a small unit test using a minimal synthetic fixture (not the real dataset)

---

## Step 3 — Add EDA scripts that output reproducible summaries

**Goal:** Match advisor expectation of “we ran EDA” without mixing it into simulation code.

Add `scripts/eda_external_datasets.py`:
- outputs `derived/eda/<dataset_name>_summary.json`
- includes counts, missingness, label distribution, basic stats

**Acceptance:** Script runs on a small synthetic sample locally; full runs happen remotely.

---

## Step 4 — Standardize preprocessing utilities (shared)

**Goal:** One deterministic way to clean/split datasets.

Add `src/satnet/data/preprocessing.py`:
- `clean_dataframe(df) -> df`
- `split_train_test(df, seed, stratify_col=None) -> (train, test)`
- `split_train_test_by_group(df, group_col, seed, stratify_col=None) -> (train, test)`
- `validate_no_leakage(feature_cols, label_cols, forbidden_cols)`

**Acceptance:** Tests cover deterministic splitting and leakage checks. For Tier 1 temporal datasets, the default must be a **run-level split** (e.g., by `run_id`) to prevent cross-timestep leakage.

---

## Step 5 — Formalize feature catalogs (Tier 1 + external)

**Goal:** The advisor will ask “what are your features”. Make it explicit and non-leaky.

Add docs:
- `docs/datasets/tier1_feature_catalog_v1.md` (design features vs labels)
- `docs/datasets/external_feature_catalog_v0.md` (candidate features for telemetry/traffic)

**Acceptance:** A reviewer can point to a single doc that answers: features, labels, and leakage policy.

---

## Step 6 — Align “design-space baseline model” with the deck (LightGBM vs RF)

**Goal:** Remove mismatch between deck claims and repo.

Choose one:

- **Option A (implement):** add a LightGBM training script for Tier 1 design dataset
  - new script: `scripts/train_design_lightgbm.py`
  - outputs metrics JSON comparable to RF baseline
  - add `lightgbm` as an optional dependency

- **Option B (declare):** update advisor-facing docs to state RandomForest baseline (already implemented)

**Acceptance:** The repo’s “baseline model” claim is consistent across code + slides.

---

## Step 7 — Graph caching for temporal GNN (performance deliverable)

**Goal:** Match advisor expectation of scalable training runs.

- Add `scripts/precompute_temporal_graphs.py`:
  - reads `tier1_design_runs.csv`
  - reconstructs graphs deterministically
  - writes cached tensors to `SATNET_DATA_ROOT/derived/graphs_cache/`
- Update `SatNetTemporalDataset` to optionally load from cache.

**Acceptance:** A single run index can be regenerated and cached; reloading yields identical tensors.

---

## Step 8 — Unified evaluation harness (accuracy + runtime)

**Goal:** Provide one command that generates the metrics the advisor expects.

Add `scripts/evaluate_models.py`:
- evaluates RF baseline and temporal GNN on the same deterministic split
- measures:
  - accuracy, F1, ROC-AUC (where applicable)
  - confusion matrix
  - inference time / run time
- writes to `models/*_metrics.json`

**Acceptance:** Produces stable metrics artifacts given a fixed dataset + seed, and persists the exact train/test split (run IDs) so all models are evaluated on identical partitions.

---

## Step 9 — Decision binning integration (end-to-end “decision output”)

**Goal:** Close the loop: prediction → tier → recommended action.

- Add a script that takes model outputs and produces a decision table using:
  - `src/satnet/metrics/risk_binning.py`

**Acceptance:** Produces a CSV/JSON mapping each run to risk tier and action.

---

## Step 10 — External validation (small but defensible)

**Goal:** A minimal, advisor-safe validation beyond internal tests.

- Add a script comparing high-level invariants vs a reference (when possible):
  - ISL counts / GCC stats for a reference constellation vs an external simulator output (if available)
  - or sanity checks across SGP4 vs fallback mode (must be explicit)

**Acceptance:** Script runs on a small reference config and saves a validation report.

---

# Phase 2 Roadmap (Deferred, but aligns to the deck)

## P2.1 Ground segment modeling (time-varying visibility)

- Introduce SAT–GS visibility provider and time-varying service labels.
- Requires Earth-fixed frames and careful rotation/transform fidelity.

## P2.2 Traffic/routing integration

- Use congestion-control logs to inform traffic models and end-to-end QoS labels.

## P2.3 GUI

- Start with a CLI “scenario runner”; GUI later.

---

## Definition of Done (for this alignment plan)

Phase 1 alignment is “done” when:

- External datasets have **manifests + loaders + EDA summaries**, without being committed to git.
- Tier 1 connectivity pipeline remains canonical and unchanged in its guardrails.
- Baseline model claims match the advisor plan (either implemented LightGBM or explicitly documented RF).
- Training and evaluation produce reproducible metrics artifacts.
- Train/test splits are persisted (run IDs) and reused across baselines and temporal models.

---

## Non-goals (this plan)

- Implementing ground stations/gateways now.
- Implementing traffic/routing labels now.
- Treating external datasets as direct truth labels for ISL partitioning (they are not).
