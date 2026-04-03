# Advisor Demo: Training Run Video Walkthrough

**Purpose:** Step-by-step script for Alex to record a screen-capture video walking his advisor through a full training run of the satnet-arch-dss pipeline.

**Estimated recording time:** 12-15 minutes (with narration)

**Prerequisites:** Python 3.11+ environment with dependencies installed (see Setup below).

---

## Before You Record

### 1. Environment Setup (do this off-camera)

```bash
cd /path/to/satnet-arch-dss
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Core + dev
python -m pip install -e ".[dev]"

# RF workflow
python -m pip install scikit-learn scipy joblib

# GNN stack (optional but recommended for the full demo)
python -m pip install -e ".[ml]"
```

### 2. Clean Slate (optional, makes the demo cleaner)

```bash
# Remove old artifacts so the video shows fresh outputs
rm -f data/tier1_design_runs.csv data/tier1_design_steps.csv
rm -f models/design_risk_model_tier1*
rm -f models/satellite_gnn*
rm -f experiments/rf_log.jsonl experiments/gnn_log.jsonl
```

### 3. Terminal Tips for Recording

- Use a **large font** (16-18pt) so the advisor can read it on video.
- Keep one terminal open for commands, one for showing output files if needed.
- Pause briefly after each command so the output is legible on playback.

---

## Part 1: Introduction (narrate over IDE / repo overview)

> **Talking point:** "This is the satnet-arch-dss repository, our satellite network architecture decision support system. The pipeline has three layers: a physics layer that simulates satellite orbits, a simulation layer that injects failures and measures connectivity, and an ML layer that learns to predict risk from design parameters."

Show the project tree briefly. Highlight these directories:

| Directory | Role |
|-----------|------|
| `src/satnet/network/` | Physics layer (SGP4 orbital propagation, ISL link budgets) |
| `src/satnet/simulation/` | Simulation layer (temporal rollout, Monte Carlo dataset generation) |
| `src/satnet/metrics/` | Pure metrics (GCC fraction, partition labels) |
| `src/satnet/models/` | ML models (RandomForest, temporal GNN) |
| `scripts/` | Runnable entry points for each pipeline stage |

**Time:** ~1 minute

---

## Part 2: Generate the Dataset (~3 minutes)

> **Talking point:** "First, we generate our training dataset. Each run samples a random constellation design -- number of orbital planes, satellites per plane, altitude, inclination -- then simulates it over time under random node and link failures, measuring connectivity at each timestep."

### Command

```bash
python scripts/export_design_dataset.py --num-runs 200 --seed 42
```

> **What to say while it runs:** "Each run creates a Walker Delta constellation, generates TLEs, propagates orbits with SGP4, computes inter-satellite links using a 1550nm optical link budget, injects persistent failures, and evaluates the Giant Connected Component fraction at every timestep."

### Expected Output (approximate)

```
Generating Tier 1 Temporal Design Dataset...
  Runs: 200
  Planes: (3, 8), Sats/plane: (4, 12)
  Altitude: 300-1200 km
  Inclination: 30-98 deg
  Duration: 10 min @ 60s steps
  Seed: 42
============================================================
[progress messages for each run]

Generated 200 runs, XXXX step records
Written 200 run rows to .../data/tier1_design_runs.csv
Written XXXX step rows to .../data/tier1_design_steps.csv

=== Dataset Summary ===
Partition probability: X.XXX
Mean GCC fraction: X.XXX
```

### After it finishes, show the output files

```bash
head -5 data/tier1_design_runs.csv
wc -l data/tier1_design_runs.csv data/tier1_design_steps.csv
```

> **Talking point:** "The runs table has one row per simulation. Each row records the design parameters, the seed, a config_hash for reproducibility, and the temporal resilience labels -- including whether the network partitioned at any point, the minimum and mean GCC fraction, and the longest consecutive partition streak."

### Key columns to highlight

| Column | Meaning |
|--------|---------|
| `config_hash` | Deterministic hash of all design + failure parameters |
| `num_planes`, `sats_per_plane`, `altitude_km`, `inclination_deg` | Design features (model inputs) |
| `node_failure_prob`, `edge_failure_prob` | Failure scenario parameters |
| `partition_any` | **Binary label**: 1 if network partitioned at any timestep |
| `gcc_frac_min` | **Continuous label**: worst-case connectivity fraction |
| `gcc_frac_mean` | **Continuous label**: average connectivity fraction |

---

## Part 3: Train the RandomForest Baseline (~2 minutes)

> **Talking point:** "Now we train a RandomForest classifier to predict whether a constellation design will partition under failure. This is our baseline model -- fast to train, interpretable feature importances, and gives us a benchmark for the GNN."

### Command

```bash
python scripts/train_design_risk_model.py --target-name partition_any --seed 42
```

### Expected Output (approximate)

```
Target: partition_any (classification)

=== Classification Metrics (test split) ===
Accuracy: 0.XXX
  Class 0: precision=X.XXX recall=X.XXX f1=X.XXX
  Class 1: precision=X.XXX recall=X.XXX f1=X.XXX

Feature importances:
  altitude_km: 0.XXX
  sats_per_plane: 0.XXX
  num_planes: 0.XXX
  inclination_deg: 0.XXX
  ...

Predictions saved to .../models/design_risk_model_tier1_predictions.csv
Metrics written to .../models/design_risk_model_tier1_metrics.json
Model saved to .../models/design_risk_model_tier1.joblib
```

> **Talking point:** "The feature importances tell us which design parameters matter most for network resilience. The predictions CSV contains per-sample predictions with stable config_hash join keys, so we can compare this model against the GNN later."

### Show the metrics file

```bash
cat models/design_risk_model_tier1_metrics.json
```

### (Optional) Also train a regression target

```bash
python scripts/train_design_risk_model.py --target-name gcc_frac_min --seed 42
```

> **Talking point:** "We can also train on continuous targets like gcc_frac_min -- the worst-case connectivity fraction. This lets us rank designs by severity, not just classify them."

---

## Part 4: Train the Temporal GNN (~4 minutes)

> **Talking point:** "The GNN is our thesis model. Unlike the RandomForest which only sees aggregate features, the GNN operates directly on the temporal graph sequences -- it sees the actual network topology evolving over time. We use a GCLSTM architecture: Graph Convolutional layers to capture spatial structure, and LSTM cells to capture temporal dynamics."

### Command

```bash
python scripts/train_gnn_model.py \
    --target-name partition_any \
    --epochs 20 \
    --lr 0.01 \
    --hidden-dim 64 \
    --data-dir data/ \
    --output-model models/satellite_gnn.pt \
    --device auto
```

### Expected Output (approximate)

```
2026-XX-XX HH:MM:SS | INFO | Using device: cpu
2026-XX-XX HH:MM:SS | INFO | Target: partition_any (classification)
2026-XX-XX HH:MM:SS | INFO | Loading dataset from .../data
2026-XX-XX HH:MM:SS | INFO | Dataset: 200 samples (neg: XX, pos: XX)
2026-XX-XX HH:MM:SS | INFO | Train/Test split: 160/40
2026-XX-XX HH:MM:SS | INFO | Model parameters: XX,XXX
2026-XX-XX HH:MM:SS | INFO | Starting training for 20 epochs...
------------------------------------------------------------
2026-XX-XX HH:MM:SS | INFO | Epoch   1/20 | Train Loss: X.XXXX Acc: X.XXXX | Test Loss: X.XXXX Acc: X.XXXX F1: X.XXXX
2026-XX-XX HH:MM:SS | INFO | Epoch   2/20 | Train Loss: X.XXXX Acc: X.XXXX | Test Loss: X.XXXX Acc: X.XXXX F1: X.XXXX
...
2026-XX-XX HH:MM:SS | INFO | Epoch  20/20 | Train Loss: X.XXXX Acc: X.XXXX | Test Loss: X.XXXX Acc: X.XXXX F1: X.XXXX
------------------------------------------------------------
2026-XX-XX HH:MM:SS | INFO | Training complete! Best test loss: X.XXXX at epoch X
2026-XX-XX HH:MM:SS | INFO | Model saved to: .../models/satellite_gnn.pt
2026-XX-XX HH:MM:SS | INFO | Predictions saved to: .../models/satellite_gnn_predictions.csv
```

> **What to narrate while training runs:**
> - "Each epoch iterates over the training samples. For each sample, the GNN reconstructs the temporal graph sequence from the dataset -- using the same Hypatia physics engine, same seed, same failure realization."
> - "Watch the test accuracy and F1 score improve over epochs. The model is learning to distinguish resilient designs from fragile ones, just by looking at the graph topology over time."
> - "The best checkpoint is saved automatically based on lowest test loss."

### After training, show the artifacts

```bash
ls -la models/satellite_gnn*
head -5 models/satellite_gnn_predictions.csv
```

> **Talking point:** "We now have the trained model checkpoint, and a predictions CSV with per-sample predictions keyed by config_hash. This lets us directly compare GNN predictions against the RF baseline on the same test samples."

---

## Part 5: Review Experiment Logs (~1 minute)

> **Talking point:** "Every experiment is logged to a structured JSONL file with timestamps, hyperparameters, metrics, and artifact paths. This ensures full reproducibility."

```bash
cat experiments/rf_log.jsonl | python -m json.tool | head -40
cat experiments/gnn_log.jsonl | python -m json.tool | head -40
```

> **Talking point:** "Each log entry records the model type, target, seed, data path, training duration, and final metrics. This is our experiment tracking system."

---

## Part 6: Run the Test Suite (~1 minute)

> **Talking point:** "Finally, we verify that the codebase is healthy with our test suite."

```bash
python -m pytest tests/ -v --ignore=tests/test_tier1_contract.py --ignore=tests/test_tier1_guardrails.py
```

Or use the Makefile shortcut:

```bash
make test
```

> **Talking point:** "The test suite validates our metrics are pure functions, our schema contracts hold, and our model training pipelines produce valid artifacts. The Tier 1 contract and guardrail tests are excluded here because they require the Hypatia orbital data -- they're run separately in CI."

---

## Part 7: Wrap-Up (narrate over artifacts)

> **Talking points for closing:**
> - "To recap: we generated 200 constellation simulations using physics-based orbital propagation with SGP4."
> - "We trained a RandomForest baseline that predicts partition risk from design parameters alone."
> - "We trained a temporal GNN that operates directly on the evolving graph topology."
> - "All experiments are logged, all artifacts are reproducible via config_hash and seed, and the pipeline is tested."
> - "Next steps: scale to larger datasets (2000+ runs), add graph caching for faster GNN re-training, and validate that cheap proxy metrics like gcc_frac_min preserve the ranking of expensive targets."

### Summary of generated artifacts

| Artifact | Path |
|----------|------|
| Design dataset (runs) | `data/tier1_design_runs.csv` |
| Design dataset (steps) | `data/tier1_design_steps.csv` |
| RF model | `models/design_risk_model_tier1.joblib` |
| RF metrics | `models/design_risk_model_tier1_metrics.json` |
| RF predictions | `models/design_risk_model_tier1_predictions.csv` |
| GNN model checkpoint | `models/satellite_gnn.pt` |
| GNN predictions | `models/satellite_gnn_predictions.csv` |
| RF experiment log | `experiments/rf_log.jsonl` |
| GNN experiment log | `experiments/gnn_log.jsonl` |

---

## Quick Reference: All Commands in Order

```bash
# 1. Generate dataset
python scripts/export_design_dataset.py --num-runs 200 --seed 42

# 2. Train RF (binary classification)
python scripts/train_design_risk_model.py --target-name partition_any --seed 42

# 3. Train RF (continuous regression, optional)
python scripts/train_design_risk_model.py --target-name gcc_frac_min --seed 42

# 4. Train GNN (binary classification)
python scripts/train_gnn_model.py \
    --target-name partition_any \
    --epochs 20 \
    --lr 0.01 \
    --hidden-dim 64 \
    --data-dir data/ \
    --output-model models/satellite_gnn.pt \
    --device auto

# 5. Run tests
python -m pytest tests/ -v --ignore=tests/test_tier1_contract.py --ignore=tests/test_tier1_guardrails.py
```

Or using Makefile shortcuts:

```bash
make dataset          # Step 1 (500 runs)
make train-rf-binary  # Step 2
make train-rf-gcc     # Step 3
make train-gnn-binary # Step 4
make test             # Step 5
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'satnet'` | Run `pip install -e .` from the repo root |
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -e ".[ml]"` |
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` |
| `FileNotFoundError: tier1_design_runs.csv` | Run Step 1 (dataset generation) first |
| `CUDA out of memory` | Add `--device cpu` to the GNN command |
| Dataset generation is slow | Reduce `--num-runs` to 50 for a quick demo |

---

## Part 8: Troubleshooting Walkthrough — From Experiments to Methodological Correction (~3 minutes)

> **Framing for Alex:** This section shows the advisor that the project follows a real engineering troubleshooting cycle. The story is **not** "we found one bug and fixed everything." The story is: "the experiments revealed a target/evaluation mismatch, and we tightened the pipeline so it better supports a dissertation-grade proxy/resilience workflow."

---

### A) What the Experiments Showed

> **Talking point:** "After running the temporal GNN experiments — including full 20-epoch training runs on 10,000 samples — the results showed the model peaking early (best checkpoint at epoch 3) and then overfitting rather than learning a robust resilience signal. That pushed us to step back and ask: is the original binary formulation actually aligned with the dissertation objective?"

Show the GNN training results from the execution note:

```bash
cat docs/advisor_meeting/2026-02-17_phase1_restart_execution_update.md
```

Key numbers to highlight:

| Metric | GNN Smoke Run | GNN Full 20-Epoch Run |
|--------|--------------|----------------------|
| Best epoch | — | **3** (out of 20) |
| Test accuracy | 0.616 | 0.760 |
| F1 | 0.750 | 0.821 |
| True negatives | 80 | 423 |

> **Talking point:** "The best checkpoint occurring at epoch 3 out of 20 is a clear signal — the model is not learning a richer representation with more training, it is memorizing. That is a red flag about the target formulation, not just about hyperparameters."

---

### B) The Diagnosis: Target and Evaluation Mismatch

> **Talking point:** "The core issue was that `partition_any` — our binary label — collapses too much information. A constellation that briefly dips below the connectivity threshold for one timestep gets the same label as one that is completely fragmented for the entire simulation. Binary classification cannot distinguish severity, and neither the model nor the evaluation framework was designed to capture that."

Show the resilience targets module that was created in response:

```bash
head -20 src/satnet/metrics/resilience_targets.py
```

> **Talking point:** "We defined a canonical set of resilience targets that capture different aspects of failure severity — not just whether partitioning happened, but how bad it was, how long it lasted, and what the worst-case connectivity looked like."

### Available Resilience Targets (after correction)

| Target | Type | What It Captures |
|--------|------|-----------------|
| `partition_any` | classification | Did it ever partition? (original binary) |
| `partition_fraction` | regression | What fraction of timesteps were partitioned? |
| `gcc_frac_min` | regression | Worst-case connectivity (severity) |
| `gcc_frac_mean` | regression | Average connectivity over time |
| `max_partition_streak` | regression | Longest consecutive outage window |

---

### C) The Pipeline Changes (show the commits)

> **Talking point:** "Based on that diagnosis, the codebase evolved in five specific directions. These are not just cleanup — they reflect a methodological correction."

```bash
git log --oneline 6e39341..ef12a07
```

Walk through these changes on screen:

#### 1. Continuous/regression targets added (Phases 2, 5–6, 9)

> "Both the RF baseline and the GNN now support any target from the canonical set. The training scripts auto-detect classification vs regression using `infer_task_type()`."

```bash
# Show that the same script handles both formulations:
python scripts/train_design_risk_model.py --target-name gcc_frac_min --seed 42
```

#### 2. Ranking-oriented evaluation for proxy usefulness (Phase 8)

> "We added a proxy ranking validation tool. The question is: if I use a cheap target like `gcc_frac_min` as a proxy for a more expensive one, does it preserve the ranking of designs? That is what matters for a decision-support system."

```bash
python tools/validate_proxy_rankings.py \
    models/rf_gcc_frac_min_predictions.csv \
    models/rf_partition_fraction_predictions.csv \
    --id-col config_hash \
    --score-col y_pred \
    --top-k 5,10,20
```

Reports: Spearman rho, Kendall tau, pairwise ordering accuracy, top-k overlap, NDCG@k.

#### 3. config_hash validation on prediction exports

> "Every prediction CSV now validates that `config_hash` values are non-null and non-empty before writing. This prevents broken join keys that would silently corrupt downstream model comparison."

```bash
# Show the validation in the code:
grep -n "config_hash" src/satnet/models/risk_model.py | head -5
```

#### 4. Graph-cache schema and provenance hardening (Phase 3 + v2 bump)

> "The GNN graph cache now enforces schema version, generator provenance, and payload mode. Stale caches from a previous pipeline version fail loudly instead of silently feeding mismatched data."

```bash
# Show the cache schema:
grep -n "CACHE_SCHEMA_VERSION" src/satnet/utils/graph_cache.py
```

#### 5. Standardized prediction exports across RF/GNN

> "RF and GNN prediction CSVs now use the same schema — `config_hash`, `target_name`, `task_type`, `seed`, `split`, `y_true`, `y_pred`. That means the proxy ranking validator can compare any two model outputs directly."

```bash
head -3 models/design_risk_model_tier1_predictions.csv
```

---

### D) The Advisor-Safe Takeaway

> **Talking point:** "The experiments showed that the original binary target was collapsing meaningful variation. That led us to expand the target set, add regression support to both models, build a ranking validation framework, and harden the reproducibility controls. The pipeline now supports a dissertation-grade proxy/resilience workflow — not just a single binary classification experiment."

> **If the advisor asks about a specific physical cause:** "The troubleshooting was primarily about the target/evaluation formulation, not a single physical bug. The strongest repo-backed story is the workflow shift toward better targets, ranking validation, and reproducibility controls."

---

### E) Summary of Troubleshooting Commits

| Commit | What Changed |
|--------|-------------|
| `6e39341` | Proxy metric repository map (Phase 1) |
| `3d84940` | Canonical resilience targets module with tests (Phase 2) |
| `c189be9` | Temporal graph sequence caching (Phase 3) |
| `8de2324` | Lightweight experiment logger (Phase 4) |
| `8cf15b0` | `--target-name` support + RF regression (Phases 5–6) |
| `7c7afbe` | Dataset audit script (Phase 7) |
| `8fbec44` | Proxy ranking validation script (Phase 8) |
| `f5f746d` | GNN regression compatibility (Phase 9) |
| `cb26090` | Proxy metric workflow guide + Makefile (Phase 10) |
| `ef12a07` | config_hash validation, cache schema v2, provenance checks |

---

## Scaling Notes for Advisor Q&A

If the advisor asks about scale:

- **200 runs** takes ~2-5 minutes to generate (depending on hardware).
- **500 runs** (the `make dataset` default) takes ~5-15 minutes.
- **2000+ runs** is the production target for thesis results; use `--num-runs 2000`.
- GNN training on 200 samples with 20 epochs takes ~5-10 minutes on CPU.
- Graph caching (`--write-cache` / `--use-cache`) eliminates graph reconstruction overhead on re-runs, cutting GNN training time by ~60-80%.
