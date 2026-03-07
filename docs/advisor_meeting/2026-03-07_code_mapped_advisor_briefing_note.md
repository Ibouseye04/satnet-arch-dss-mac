# Code-Mapped Advisor Briefing Note

**Date:** 2026-03-07  
**Audience:** Alex briefing for advisor technical walk-through  
**Purpose:** Give a plain-English, code-grounded explanation of how the Tier 1 pipeline is developed, executed remotely, and used to produce RF/GNN results.

---

## 1. Executive summary

This project is an **end-to-end Tier 1 workflow** for studying satellite network partition risk.

In plain English, the workflow is:

1. Connect to the remote execution machine.
2. Generate a **time-evolving satellite network** using the Tier 1 physics pipeline.
3. Inject controlled failures and measure how connectivity changes over time.
4. Export those runs as a dataset.
5. Train ML models on that dataset.
6. Evaluate how well the models detect partition risk.

The important advisor-safe point is that the **truth labels come from temporal graph behavior**, not from a toy random topology and not from hand-assigned labels.

---

## 2. Remote execution workflow

The heavy runs are performed on Alex's machine via remote access / VS Code tunnel, while development and review can happen elsewhere.

This matters because it shows a realistic engineering workflow:

- one machine acts as the **development/control station**,
- another machine acts as the **execution environment**,
- results are captured in reproducible output directories and documented in advisor notes.

Reference execution record:

- `docs/advisor_meeting/2026-02-17_phase1_restart_execution_update.md`

That file records the actual rerun commands and outputs, including:

- dataset regeneration,
- RF retraining,
- temporal GNN smoke/full training runs.

---

## 3. How the Tier 1 simulation works in code

### A) Physics/network layer

**File:** `src/satnet/network/hypatia_adapter.py`

This is the bridge between the repository and the underlying satellite propagation / link computation machinery.

What it does:

- builds Walker-style constellations,
- generates TLEs,
- propagates satellites over time,
- computes inter-satellite links,
- returns a graph for each time step.

Plain-English explanation:

> This is the code that turns orbital configuration into a changing satellite network we can actually analyze.

### B) Temporal rollout layer

**File:** `src/satnet/simulation/tier1_rollout.py`

This is the authoritative Tier 1 rollout contract.

What it does:

- defines the rollout configuration (`Tier1RolloutConfig`),
- constructs the `HypatiaAdapter`,
- generates temporal graphs from `t = 0..T`,
- samples persistent failures,
- applies those failures at each time step,
- computes temporal connectivity metrics,
- returns per-step results plus a run summary.

Plain-English explanation:

> This is the time-based experiment loop. Instead of looking at one frozen snapshot, it watches the network over time and measures how badly it breaks under failures.

### C) Pure labeling layer

**File:** `src/satnet/metrics/labels.py`

This file computes graph-based metrics such as:

- number of connected components,
- GCC size,
- GCC fraction,
- whether the network is partitioned,
- longest partition streak.

Plain-English explanation:

> These functions do not know why the graph changed. They only inspect the graph itself and compute the connectivity outcome.

This is important because it means the labels are **derived from graph state only**.

---

## 4. How the dataset is created

### A) Monte Carlo dataset generation

**File:** `src/satnet/simulation/monte_carlo.py`

This file repeats the Tier 1 rollout many times.

What it does:

- samples constellation parameters,
- samples failure probabilities,
- calls `run_tier1_rollout(...)`,
- stores one summary row per run,
- stores one detailed row per time step.

Plain-English explanation:

> This turns a single simulation into a large training dataset by running many different scenarios.

### B) Command-line entry point for dataset export

**File:** `scripts/export_design_dataset.py`

This is the script used to generate the canonical Tier 1 design dataset.

Documented full rerun command:

- `python scripts/export_design_dataset.py --num-runs 10000 --seed 42 --output-dir data/runs/2026-02-17_full_10k`

Recorded output from the execution note:

- `10000` run rows
- `110000` step rows
- `partition_probability = 0.672`
- `mean_gcc_fraction = 0.483`

Plain-English explanation:

> This script is the dataset factory entry point. It creates the examples that both the baseline model and the GNN learn from.

---

## 5. How the baseline RF model is trained

### A) Training script

**File:** `scripts/train_design_risk_model.py`

This script loads the generated runs table and trains a RandomForest classifier.

### B) Training logic

**File:** `src/satnet/models/risk_model.py`

What it does:

- loads `tier1_design_runs.csv`,
- selects design-time features,
- uses `partition_any` as the label,
- splits train/test,
- trains the classifier,
- computes accuracy, ROC AUC, confusion matrix, and feature importances.

The baseline features printed by the script are:

- `num_planes`
- `sats_per_plane`
- `inclination_deg`
- `altitude_km`

The label is:

- `partition_any`

Plain-English explanation:

> The RF baseline tries to predict whether a design is likely to encounter partitioning, using only the design parameters.

Recorded rerun metrics from the execution note:

- `accuracy = 0.913`
- `roc_auc = 0.971`
- confusion matrix = `[[594, 62], [112, 1232]]`

---

## 6. How the temporal GNN is trained

### A) GNN training script

**File:** `scripts/train_gnn_model.py`

This is the real training entry point for the temporal graph model.

What it does:

- loads the Tier 1 dataset directory,
- builds a `SatNetTemporalDataset`,
- splits samples into train/test,
- trains for multiple epochs,
- evaluates accuracy, precision, recall, and F1,
- saves the best checkpoint.

Plain-English explanation:

> This script trains the model that looks at the graph sequence over time, not just summary design features.

### B) GNN dataset reconstruction layer

**File:** `src/satnet/models/gnn_dataset.py`

This file is important because the GNN does **not** just consume a flat CSV row.
It reconstructs temporal graph sequences on the fly from the exported run metadata.

What it does:

- reads `tier1_design_runs.csv`,
- rebuilds the constellation using `HypatiaAdapter`,
- uses `epoch_iso`, `duration_minutes`, and `step_seconds`,
- reapplies `failed_nodes_json` and `failed_edges_json`,
- converts each time step into a PyTorch Geometric `Data` object,
- returns a list of graph snapshots for each run.

Plain-English explanation:

> The CSV stores the recipe, and this dataset class rebuilds the actual graph movie so the GNN can learn from how connectivity changes over time.

### C) GNN model definition

**File:** `src/satnet/models/gnn_model.py`

This file defines `SatelliteGNN`, a GCLSTM-based temporal graph model.

What it does:

- processes each graph snapshot in sequence,
- updates hidden state over time,
- pools node embeddings into a graph-level representation,
- outputs a binary classification.

Plain-English explanation:

> Instead of looking only at design parameters, the GNN learns from the evolving network structure itself.

### D) Documented full-run command and results

From the execution note:

- Command:
  - `python scripts/train_gnn_model.py --data-dir data/runs/2026-02-17_full_10k --epochs 20 --device cpu --output-model models/satellite_gnn_10k.pt`

- Best checkpoint summary:
  - `best_epoch = 3`
  - `best_test_acc = 0.7600`
  - `precision = 0.8168`
  - `recall = 0.8242`
  - `f1 = 0.8205`
  - confusion summary: `TP=1097, FP=246, FN=234, TN=423`

Plain-English explanation:

> The GNN learned meaningful patterns from the temporal graph sequences. It is not perfect, but it is clearly detecting real signal rather than guessing.

---

## 7. Recommended advisor walk-through order

If Alex is walking the advisor through the repository, the clean order is:

1. `docs/advisor_meeting/2026-02-17_phase1_restart_execution_update.md`
   - show the actual commands and reported results.
2. `src/satnet/network/hypatia_adapter.py`
   - show the physics/network foundation.
3. `src/satnet/simulation/tier1_rollout.py`
   - show the temporal evaluation loop.
4. `src/satnet/metrics/labels.py`
   - show that labels come from graph state only.
5. `src/satnet/simulation/monte_carlo.py`
   - show how many runs become a dataset.
6. `scripts/export_design_dataset.py`
   - show the script that generated the dataset.
7. `scripts/train_design_risk_model.py`
   - show the baseline RF training path.
8. `src/satnet/models/gnn_dataset.py`
   - show graph reconstruction for temporal ML.
9. `src/satnet/models/gnn_model.py`
   - show the GNN architecture.
10. `scripts/train_gnn_model.py`
   - show the actual GNN training/evaluation loop.

---

## 8. Short talk track Alex can say

> We remotely connected to the execution machine, regenerated a Tier 1 temporal satellite dataset, and used that dataset to train both a baseline RandomForest model and a temporal GNN. The physics layer builds the evolving satellite network, the rollout layer applies failures and evaluates connectivity over time, the labels are computed purely from graph state, and the exported runs are then used for ML training. The RF baseline learns from design parameters, while the GNN reconstructs and learns from the full graph sequence over time. The key research point is that the labels are grounded in temporal network behavior, not in a toy topology or manually assigned outcomes.

---

## 9. Advisor-safe bottom line

This repository demonstrates a complete technical pipeline:

- **open-source scientific simulation** connected to
- **custom temporal network analysis**, producing
- **reproducible datasets**, used for
- **baseline and temporal ML evaluation**.

That is the core technical story Alex can walk through with confidence.
