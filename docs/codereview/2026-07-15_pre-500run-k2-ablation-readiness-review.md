# Code Review: Readiness for Full 500-Run K=2 TGNN Ablation + RF Ablation Methodology

**Date:** 2026-07-15
**Scope:** Read-only review. Verified the K=2 fix chain end-to-end (model → training script → runner → tests → artifacts) and audited the RF ablation path.
**Overall:** The K=2 implementation itself is sound; the risks are operational, not in the model code.

> **Historical status:** This pre-run review and the artifacts it references are immutable preliminary evidence. The historical `artifacts/ablation` output is K=1, and the later K=2 validation outputs are topology-sensitivity evidence only. Do not overwrite or use these directories for resumed or canonical experiments; all future runs require a new versioned output directory.

## Verdict

No hard blockers in the code. Two high-priority operational hazards and several medium methodology issues that could undermine the dissertation claims if unaddressed.

---

# High Priority

## H1. Stale K=1 artifacts + skip logic can silently contaminate the rerun

The existing full-scale ablation under `artifacts/ablation` is **pre-fix (K=1)**. Verified all 6 TGNN metrics JSONs there have `cheb_k: None`, and `run_manifest.json` has `cheb_k: None`, `smoke: false`.

The runner skips any condition whose metrics file already exists (`scripts/run_ablation_study.py:316-319`):

```python
def run_specs(specs: list[ExperimentSpec], *, overwrite: bool) -> None:
    for spec in specs:
        if spec.metrics_path.exists() and not overwrite:
            continue
```

If the rerun targets the default `--output-dir artifacts/ablation` without `--overwrite`, **every condition is skipped**, yet `collect_results()` still rebuilds `comparison/*.csv` from the stale K=1 metrics while the freshly written manifest claims `cheb_k=2`. The only tell is `cheb_k` being empty in the comparison rows.

**Mitigation:** Use a fresh output dir (e.g., `artifacts/validation/tgnn_k2_full_ablation`) exactly as the handover plan says — this is confirmed necessary, not optional. Alternatively `--overwrite`. Note the skip logic doubles as a resume mechanism after mid-run failure, which is useful for a long run.

## H2. Runner never enables graph caching → ~9,000 Hypatia regenerations per TGNN condition

`SatNetTemporalDataset` regenerates the full SGP4 + ISL sequence on **every** `dataset[idx]` access unless `use_cache=True` (default `False`, `src/satnet/models/gnn_dataset.py:158-159`). `train_gnn_model.py` exposes `--use-cache`/`--write-cache`, but the runner's TGNN command omits them (`scripts/run_ablation_study.py:232-265`).

Measured generation on the real dataset: 500 runs, **31 timesteps each**, ~0.06–0.15 s per sequence. Full run = 6 conditions × 20 epochs × ~450 train+val accesses + final passes ≈ **55k+ regenerations ≈ 1.5–2.5 h of pure redundant graph generation** on top of model compute. Not fatal — an overnight CPU run remains feasible — but wasteful and it slows any retry.

**Mitigation options:**

- Cheapest, no code change: accept the overhead.
- One-line-ish change: append `--use-cache --write-cache` to the TGNN command in `build_experiment_specs`. The cache layer is safe for this: cache identity includes `failed_nodes_json`/`failed_edges_json`/`failure_model`/`isl_policy` (`src/satnet/utils/graph_cache.py:49-68`) and validation is fail-closed. First condition warms the cache; the other 5 reuse it.

---

# Medium Priority (methodology / dissertation-defensibility)

## M1. RF vs TGNN class-imbalance treatment is inconsistent

Dataset is 329 positive / 171 negative for `partition_any`. RF uses `class_weight="balanced"` (`src/satnet/models/risk_model.py:806-813`), but the TGNN uses unweighted `nn.CrossEntropyLoss()` (`scripts/train_gnn_model.py:663-666`). The prior audit found the TGNN collapsing to the majority baseline — an unweighted loss on a 66/34 split makes that failure mode more likely, and it also weakens any RF-vs-TGNN comparison claim. Consider class weights (`CrossEntropyLoss(weight=...)`) or at least reporting balanced accuracy.

## M2. No per-epoch shuffling of training order

`train_epoch` iterates `train_indices` in the same fixed order every epoch (`scripts/train_gnn_model.py:389`). With per-sample (batch=1) Adam updates, fixed ordering biases late-epoch updates toward the same samples. A seeded per-epoch permutation would be more defensible and stays deterministic.

## M3. Comparison outputs omit the trivial baselines that gate the experiment

`collect_results`/`write_summary` report only within-family deltas (ablation vs `full`). The prior gate was "TGNN must beat majority-class / train-mean baselines" — those numbers aren't computed anywhere in the runner (`scripts/run_ablation_study.py:452-470`). Without them in the artifact set, the 500-run rerun can't answer its own pass/fail question. Recommend adding majority-class accuracy/F1 and train-mean MAE columns, or computing them post-hoc from the predictions CSVs (which do contain everything needed).

## M4. RF ablation naming + dead features (confirms and extends the handover finding)

- `architecture_only` = full minus failure probs; `no_geometry` = full minus altitude/inclination — internally consistent, not nested, and `architecture_only` is a misleading name (`src/satnet/models/risk_model.py:61-73`).
- New finding: in the actual 500-run dataset, `duration_minutes` and `step_seconds` are **zero-variance constants** (verified: nunique = 1 for both). They are dead features in all three sets, so "7 features" in `architecture_only` is effectively 5 informative ones. Harmless to RF math, but the dissertation should not imply these carry signal.

## M5. RF silent feature drop makes the missing-feature guard dead code

`src/satnet/models/risk_model.py:768-769`:

```python
    if feature_columns is None:
        feature_columns = [c for c in get_rf_feature_columns(feature_set_name) if c in df.columns]
```

The `if c in df.columns` filter silently drops any registry feature absent from the CSV, so the `missing_features` check at lines 771–773 can never fire on the default path. Today all 9 features exist, so no practical impact — but a future dataset missing `altitude_km` would silently train a mislabeled "full" condition. Fail-closed would be safer.

---

# Low Priority

- **`torch.load` future-compat:** the checkpoint reload at `scripts/train_gnn_model.py:732` pickles `args`/metrics; under PyTorch ≥2.6 `weights_only=True` defaults this would break. Works in the current env (smoke passed).
- **Empty-val edge case:** with `--val-split 0` and no manifest, `evaluate` returns loss 0.0, freezing checkpoint selection at epoch 1. Not hit by the planned run (val=0.1 via manifest).
- **`run_specs` captures full stdout in memory** — negligible at 20-epoch log volume.
- **Split-manifest reuse is fail-closed** — dataset SHA-256 + row-count validation (`src/satnet/utils/split_manifest.py:284-289`) means stale manifests error out rather than silently misalign. Also, same seed + same dataset ⇒ the new-dir rerun reproduces the exact splits of the K=1 run, which is good for before/after comparison.

# Tier 1 Compliance — Clean

- **No Tier 0 topology:** `satnet.network.topology` appears only in `src/satnet/legacy/engine.py` and in guardrail tests; the active pipeline uses `HypatiaAdapter` (`src/satnet/models/gnn_dataset.py:419-435`).
- **Temporal:** full `iter_graphs()` sequences (31 steps), not snapshots.
- **No leaky labels:** RF feature validation rejects outcome fields; TGNN node features are `plane_idx`, `sat_in_plane`, constant — no degree/GCC proxies.
- **Determinism:** seeds propagated everywhere; failure realizations replayed from stored JSON, never resampled.
- **K=2 fix chain verified:** default `cheb_k=2` with validation, persisted in checkpoint/metrics/config/predictions/manifest; K=1/K=2 guardrail tests present in `tests/models/test_gnn_model.py`.

# Recommended Pre-Run Checklist

1. **Must:** run into a fresh `--output-dir` (or `--overwrite`) — H1.
2. **Should:** decide on cache passthrough (H2) and per-epoch shuffling + class weighting (M1/M2) *before* the run, since changing them afterward invalidates comparability.
3. **Should:** plan baseline computation (M3) — post-hoc from predictions CSVs is fine, no code change required before launch.
4. **Defer:** RF renaming (M4/M5) is independent of the TGNN rerun and can be done with tests + dissertation wording together, per the handover plan.
