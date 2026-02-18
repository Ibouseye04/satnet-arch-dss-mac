# Phase 1 Restart + Phase 2 Readiness Update (Execution Snapshot)

**Date:** 2026-02-17  
**Audience:** Advisor prep + multi-agent handoff  
**Status:** In progress (Phase 1 rerun executed; Phase 2 entry work staged)

---

## Why this update exists

The repository guardrails correctly state **Phase 1 = satellite-to-satellite only**.  
However, current execution work is now explicitly focused on:

1. Rebuilding the full Phase 1 dataset baseline after major codebase updates.
2. Verifying training outputs on remote hardware (Alex machine).
3. Staging external datasets as a supporting track to enable a controlled transition to Phase 2.

This file clarifies that "sat-to-sat only" is a **scope boundary**, not a dead-end.

---

## Tonight's completed execution work (2026-02-17)

### A) Tier 1 design dataset regeneration

- Command:
  - `python scripts/export_design_dataset.py --num-runs 10000 --seed 42 --output-dir data/runs/2026-02-17_full_10k`
- Output:
  - `10000` run rows
  - `110000` step rows
- Aggregate summary:
  - `partition_probability = 0.672`
  - `mean_gcc_fraction = 0.483`

### B) Baseline RF retraining on 10k design runs

- Input: `data/tier1_design_runs.csv` (copied from `data/runs/2026-02-17_full_10k/`)
- Command:
  - `python scripts/train_design_risk_model.py`
- Test metrics:
  - `accuracy = 0.913`
  - `roc_auc = 0.971`
  - confusion matrix = `[[594, 62], [112, 1232]]`

Interpretation: baseline remains strong and stable after rerun.

### C) Temporal GNN smoke test (completed)

- Run type: short validation run to confirm end-to-end GNN training loop executes.
- Reported test metrics:
  - `accuracy = 0.6155`
  - `precision = 0.6615`
  - `recall = 0.8648`
  - `f1 = 0.7496`
  - confusion summary: `TP=1151, FP=589, FN=180, TN=80`

Interpretation: smoke run is functioning and biased toward high recall (finds most
partitioned cases) with weak true-negative performance. This is acceptable for a
smoke test and motivates full training + threshold/calibration review.

---

## What this means for Phase 2

Phase 1 rerun was required to re-establish a clean, reproducible foundation after repo evolution.  
That is now happening successfully, and this directly enables Phase 2 by providing:

- trusted temporal graph labels,
- deterministic run metadata (`seed`, `config_hash`, `epoch_iso`),
- consistent artifacts for model comparison.

We are not blocked on the concept of Phase 2; we are sequencing it correctly.

---

## External datasets: how they feed Phase 2 (advisor-safe)

External datasets are **Track B (supporting/validation)**, not Phase 1 truth labels.

- Use in near term:
  - manifesting,
  - schema review,
  - EDA summaries,
  - validation targets for Phase 2 claims.
- Not allowed in Phase 1:
  - replacing physics-derived sat-to-sat truth labels.

Reference context:
- `docs/advisor_meeting/2026-02-07_advisor_meeting_notes.md`
- `docs/refactor_plans/2026-01-08_advisor_plan_alignment_ml_pipeline_and_external_datasets.md`

---

## Phase 2 entry criteria (gated)

Move to implementation of ground-segment logic when all are true:

- [ ] Phase 1 dataset rerun accepted (10k+ design runs archived with metadata).
- [ ] Baseline model metrics recorded and reproducible across reruns.
- [ ] External dataset manifest + path contract available.
- [ ] At least one external dataset EDA note completed (LENS priority).
- [ ] Slide deck language explicitly distinguishes Track A (truth) vs Track B (supporting).

After gates pass, Phase 2 starts with SAT-GS visibility contracts and service-availability labeling.

---

## TODO: Instrumentation features to implement (from Alex's advisor slides)

Alex's troubleshooting slides describe a debug/instrumentation design that is **not yet implemented**.
These should be built after Phase 1 rerun is accepted and before the next advisor review.

- [ ] `--single-seed` flag on `export_design_dataset.py` — run exactly one seed for minimal failing replay
- [ ] `--single-timeslice` flag — evaluate only one time step (t=N) for isolation
- [ ] `--dump-intermediates` flag — write per-stage parquet/json artifacts + SHA256 hash to output dir
- [ ] `--trace-routing` flag — verbose ISL routing log per time step
- [ ] `--dry-validate` flag — run pipeline through schema/invariant checks without writing output
- [ ] Artifact hash diffing utility — compare SHA256 hashes across two output dirs to find first mismatch stage
- [ ] Hard invariant checks as a callable (timestamps monotonic, no NaN/inf, link availability ∈ [0,1])

**Current workaround:** `--num-runs 1 --seed 42` gives a single-run replay. `--output-dir` writes intermediates but no hashing.

---

## Handoff note for agents

If an agent reads "Phase 1 is sat-to-sat only," it should interpret this as a current implementation boundary while preserving active workstreams for:

- external validation data onboarding,
- Phase 2 interface planning,
- advisor-facing transition narrative.
