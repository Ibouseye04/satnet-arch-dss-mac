# Tier 1 Codebase Review: Simulation and ML Validity

**Review date:** 2026-07-15  
**Scope:** Current `satnet-arch-dss-mac` workspace  
**Review type:** Evidence-based architecture, simulation, dataset, ML, testing, and reproducibility audit  
**Change policy:** Review only; no source code, datasets, models, or existing artifacts were modified

> **Historical status:** The current K=2 results are preliminary topology-sensitivity evidence, not canonical dissertation evidence. The reviewed K=1 and K=2 artifact directories are immutable historical records and must not be overwritten or resumed; post-remediation work must use new versioned output directories.

## 1. Purpose of this document

This document explains the findings of the full codebase review and why they matter to the project's central research goal: producing defensible conclusions from a doctoral-grade, temporal, satellite-to-satellite resilience simulator.

The review evaluated whether the current implementation supports the following claims:

1. Orbital and link-state graphs are generated from reproducible Tier 1 physics.
2. Connectivity is evaluated over time rather than from one static snapshot.
3. Failures are applied consistently and can be reconstructed exactly.
4. Labels are calculated only from graph state.
5. Dataset rows contain enough information to reproduce and verify each rollout.
6. RF and TGNN experiments evaluate clearly defined prediction tasks.
7. Ablation results are reproducible and statistically strong enough to support research conclusions.

The main conclusion is that the architecture has advanced substantially beyond Tier 0, but the current dataset and K=2 ablation artifacts should be treated as **preliminary validation evidence**, not final dissertation evidence. Several physics-contract, reconstruction, provenance, and evaluation issues should be addressed before regenerating the final dataset and rerunning the experiments.

## 2. What is already working well

The review found several important strengths.

### 2.1 Temporal simulation is real

The rollout evaluates a sequence of graph states rather than using only `t=0`. This aligns with the Tier 1 requirement that connectivity be assessed over time.

### 2.2 The scope remains satellite-to-satellite

No gateway or ground-station behavior has been introduced into the Phase 1 simulation path. This preserves the intended space-segment research boundary.

### 2.3 Labels are graph-derived

The canonical resilience labels are calculated from graph properties such as giant connected component size, connected components, threshold crossings, and partition streaks. They do not directly inspect failure probabilities when assigning a label. This satisfies the non-leaky-label requirement.

### 2.4 Failure realization is stored

The dataset includes persistent failed-node and failed-edge realizations. This is an essential basis for regenerating the exact failed graph sequence used to produce each label.

### 2.5 The current dataset passed basic integrity checks

The reviewed 500-run dataset had:

- No null values.
- No duplicate `run_id` values.
- No duplicate `config_hash` values.
- 15,500 linked temporal step rows.
- Run summaries consistent with the corresponding step rows within floating-point tolerance.
- `partition_any` and `partition_fraction` values consistent with graph-derived threshold calculations.

### 2.6 Shared split manifests are implemented

RF and TGNN ablations can use the same deterministic train, validation, and test assignment. This is necessary for paired comparisons between ablation conditions.

### 2.7 The K=2 TGNN is topology-sensitive

The current model uses Chebyshev order `K=2`, and `node_state_only` removes inter-node edges. Unlike the historical `K=1` behavior, the current K=2 result changes materially when graph connectivity is removed. This is strong evidence that the corrected model can use topology.

## 3. Critical simulation-validity findings

### 3.1 `max_isl_distance_km` is not enforced

The ISL calculation accepts `max_isl_distance_km`, and that value can appear to be part of the simulation configuration. However, candidate links are not actually rejected when their distance exceeds the configured limit. A direct diagnostic produced the same edges with a one-kilometer limit and a 10,000-kilometer limit.

#### Why this matters

A configuration field should either affect the physics or not exist. If it is included in configuration hashes, logs, or experimental descriptions while having no effect, two apparently different experiments can be physically identical. That undermines sensitivity analysis and configuration provenance.

#### Required resolution

Choose and document one of these contracts:

1. Treat `max_isl_distance_km` as a hard geometric terminal-range limit and enforce it before link-budget evaluation.
2. Remove the parameter and state that feasibility is determined only by Earth line of sight and the link budget.

If retained, distance-limit rejections should be counted separately in ISL diagnostics and tested at values just below, at, and above the threshold.

### 3.2 SGP4 propagation errors silently create zero-position satellites

When SGP4 returns an error, the adapter currently inserts a satellite at `(0, 0, 0)` with zero altitude and continues. The adapter can also silently use a Keplerian fallback when SGP4 is unavailable.

#### Why this matters

A satellite at Earth's center is not a conservative approximation. It is invalid state that can alter Earth-obscuration tests, distances, link acceptance, and graph connectivity. Continuing the run allows a corrupted graph to receive normal labels and a normal `config_hash`.

A silent orbital-engine change also means two runs with the same configuration and seed can use different physics depending on the software environment.

#### Required resolution

- Canonical Tier 1 dataset generation should require SGP4.
- Any SGP4 propagation error should invalidate the rollout with a descriptive error.
- Keplerian propagation should be available only through an explicit noncanonical development option.
- Dataset metadata should record orbital engine name, implementation version, and propagation-error count.

### 3.3 Adaptive ISL selection does not enforce incident terminal capacity

The adaptive policy limits how many links each source satellite selects, but it does not limit how many neighboring satellites select that satellite. A targeted construction with `max_inter_plane_links_per_sat=1` produced an actual inter-plane degree of four for one satellite.

#### Why this matters

The field name implies a physical limit on incident inter-plane links or terminals. If the implementation instead limits only outgoing choices, high-degree hubs can form despite the stated capacity. This changes resilience, degree distributions, failure exposure, and GCC labels.

#### Required resolution

- Define whether the parameter means source choices or total incident terminal capacity.
- If it is a terminal-capacity limit, implement deterministic capacity-constrained matching.
- Add invariants over actual graph degree, not just per-source selection count.
- Rename the field if the current source-choice behavior is intentional.

### 3.4 Link-budget sensitivity needs explicit validation

In a representative optical diagnostic, all accepted candidates passed the optical budget and no candidate was rejected for insufficient budget. Geometry, rather than the budget, appeared to control the graph.

#### Why this matters

This is not automatically a defect. It may mean the chosen terminal assumptions provide ample margin over the evaluated design space. However, if the thesis claims that the 1550 nm optical budget materially determines connectivity, that claim requires a parameter sweep demonstrating active and physically plausible acceptance boundaries.

#### Required resolution

Add physics validation plots or tables for:

- Distance versus received power.
- Distance versus link margin.
- Link acceptance around the sensitivity threshold.
- Sensitivity to aperture, transmit power, pointing loss, and atmospheric assumptions.
- Counts of candidates rejected by LOS, range, and link budget.

## 4. Reconstruction and dataset-contract findings

### 4.1 CSV rows cannot independently reproduce their own `config_hash`

`Tier1RolloutConfig.config_hash()` hashes every rollout configuration field. The exported run rows omit at least:

- `phasing_factor`
- `max_isl_distance_km`
- `gcc_threshold`

The current dataset is reconstructable only because the omitted values happen to use implicit defaults. The CSV itself does not prove those defaults were used.

#### Why this matters

The project's reproducibility contract requires a row to identify the configuration that produced it. A stored hash is not independently verifiable if the fields used to calculate it are missing.

#### Required resolution

- Export every rollout-defining field.
- Implement one canonical `config_from_run_row()` function.
- Recompute and compare `config_hash` whenever a dataset is loaded.
- Introduce a new schema version for the expanded contract.

### 4.2 TGNN reconstruction fails open

If failed-node and failed-edge realization columns are missing, the TGNN dataset logs a warning and regenerates graphs without applying failures.

#### Why this matters

The regenerated graph can differ from the graph that generated the target label. This allows model inputs and labels to become silently inconsistent.

#### Required resolution

Canonical Tier 1 training should fail unless all reconstruction metadata is present and valid. Legacy compatibility should require an explicit flag and should clearly mark the resulting run as noncanonical.

### 4.3 Schema validation is too shallow

Current validation primarily checks required columns and validates only selected values. Targeted invalid data was accepted with impossible probabilities, negative counts, invalid temporal indices, empty hashes, wrong dataset versions, and broken run/step relationships.

#### Why this matters

CSV files are part of the scientific interface, not merely serialization details. Invalid rows can enter training without an obvious failure and affect labels, split manifests, or reported sample counts.

#### Required resolution

Validate:

- Every row, not only representative rows.
- Dataset and schema versions.
- Full configuration-hash format and recomputation.
- Finite numeric values.
- Probability and fraction ranges.
- Positive constellation and time dimensions.
- JSON failure realization structure.
- Run/step referential integrity.
- Exact expected step counts and step-index ranges.
- Agreement between step-level values and run-level aggregates.

## 5. Random Forest findings

### 5.1 Named feature sets can silently lose features

The RF training path filters canonical features to columns currently present before checking for missing values. Removing `altitude_km` still allowed a model to train under the name `full`.

#### Why this matters

An ablation name must have one stable scientific meaning. A `full` model with eight features is not directly comparable to a `full` model with nine features.

#### Required resolution

- Require all registered features for every named feature set.
- Raise an error listing missing columns.
- Permit different columns only through an explicitly named custom-feature mode.
- Store the exact ordered feature list in every model and metrics artifact.

### 5.2 `architecture_only` does not mean architecture only

The current condition removes node and edge failure probabilities, but retains geometry and temporal parameters. It is better described as `no_failure_probabilities`.

#### Why this matters

The current name can cause an ablation result to be interpreted as the predictive value of architecture alone, even though the condition includes altitude, inclination, timing, and other nonfailure information.

#### Required resolution

- Rename the current condition to `no_failure_probabilities`.
- If scientifically useful, define a separate true `architecture_only` condition.
- Update tests, artifact labels, plots, and dissertation wording together.

### 5.3 Feature importance requires careful interpretation

RF features include quantities that are mathematically dependent, such as plane count, satellites per plane, and total satellites. Correlated inputs distribute impurity-based importance unpredictably.

#### Required resolution

Report permutation importance on held-out data and, where possible, grouped importance for related architecture variables. Do not interpret raw impurity importance as a causal ranking.

## 6. TGNN findings

### 6.1 The `full` condition does not use physical edge attributes

The dataset constructs edge attributes such as distance, margin, link type, and mode. The current GCLSTM receives graph connectivity but not those attributes.

#### Why this matters

The `full` condition currently means binary graph adjacency plus node features. It is not a model of the full physical link state. This affects the interpretation of topology-versus-geometry conclusions.

#### Required resolution

Choose one of these paths:

1. Integrate physical edge information into a model that supports edge attributes.
2. Convert a justified scalar physical quantity into `edge_weight`.
3. Rename and document the current condition as adjacency plus node state.

### 6.2 RF and TGNN solve different prediction problems

The RF receives architecture and configured failure probabilities before a specific failure realization is known. The TGNN receives the realized failed graph sequence, while the labels are computed from that same sequence.

Therefore:

- The RF is approximately an **ex-ante design-risk predictor**.
- The TGNN is approximately an **ex-post graph-state outcome recognizer**.

#### Why this matters

Both tasks can be useful, but their headline metrics should not be compared as though they estimate the same quantity. The TGNN is given information much closer to the target outcome.

#### Required resolution

The dissertation should define the estimand for each model. If a direct RF/TGNN comparison is desired, both models must receive information available at the same decision point.

### 6.3 Classification evaluation needs stronger imbalance-aware metrics

The reviewed dataset contains 329 positive and 171 negative `partition_any` examples. A classifier that always predicts positive achieves approximately `0.658` accuracy. The node-state-only TGNN scored `0.66`, making it effectively a majority baseline despite an apparently high F1 score.

#### Required resolution

Add:

- Majority and dummy classifier baselines.
- Balanced accuracy.
- Matthews correlation coefficient.
- ROC-AUC and PR-AUC calculated from probabilities.
- Confusion matrices.
- Brier score or another calibration measure.
- Validation-only threshold selection where thresholding is used.

### 6.4 One seed is not enough for final ablation claims

The K=2 artifact uses one model seed and one train/validation/test split. Neural-network optimization variance is therefore unmeasured.

#### Required resolution

Run paired repeated-seed experiments using identical splits across ablation conditions. Report means, standard deviations, confidence intervals, and paired deltas relative to `full`.

## 7. Interpretation of the current K=2 results

The existing K=2 artifacts show:

| Task | Condition | Primary result |
|---|---|---:|
| Classification | Full | Accuracy 0.83, F1 0.8702 |
| Classification | Topology only | Accuracy 0.83, F1 0.8722 |
| Classification | Node state only | Accuracy 0.66, F1 0.7952 |
| Regression | Full | R² 0.8199, RMSE 0.1498 |
| Regression | Topology only | R² 0.8048, RMSE 0.1559 |
| Regression | Node state only | R² -0.0088, RMSE 0.3544 |

### What these results support

- The corrected K=2 model is sensitive to connectivity.
- Removing inter-node edges severely degrades both classification and regression.
- Topology alone contains most of the predictive information available to the current TGNN.

### What these results do not yet support

- They do not establish uncertainty across training seeds.
- They do not prove that physical edge attributes are unimportant because those attributes are not consumed by the model.
- They do not support a fair RF-versus-TGNN superiority claim because the models receive different information.
- They are not fully source-reproducible because the experiment log records only a commit SHA from a dirty worktree.

The appropriate description is **successful preliminary K=2 topology-sensitivity validation**.

## 8. Experiment provenance findings

### 8.1 Recorded Git SHA does not identify the executed code

The K=2 logs record commit `859141b`. The run used behavior that was committed later in `c353d25`, demonstrating that the experiment was run from a dirty worktree. The log does not record dirty state or a source diff.

#### Why this matters

A commit SHA identifies only committed source. It cannot reproduce uncommitted changes that were active during an experiment.

#### Required resolution

Every canonical experiment should record:

- Full Git commit SHA.
- Clean/dirty worktree state.
- A hash or patch of tracked modifications when dirty runs are allowed.
- Python version.
- Dependency versions or lockfile hash.
- CUDA, PyTorch, PyG, and temporal extension versions.
- Dataset content hash.
- Split-manifest hash.
- Exact command and configuration.

For final experiments, the simplest policy is to refuse execution from a dirty worktree.

### 8.2 Existing metrics can be silently reused

The ablation runner skips a condition if its metrics file already exists. It does not verify that the existing file matches the requested dataset, seed, split, epochs, K value, or source state.

#### Required resolution

Build a canonical experiment identity from all scientific inputs. Existing artifacts should be reused only when that identity matches exactly. Otherwise, the runner should fail and require a new output directory or explicit overwrite.

## 9. CI and verification findings

### 9.1 Local verification completed

- Full pytest run: 264 tests passed and one timing-based test failed.
- The failing test passed when rerun alone, indicating a flaky wall-clock assertion rather than a functional failure.
- Python compilation checks passed.
- The working tree was clean at the end of the review.

### 9.2 Static analysis was unavailable locally

Neither the global environment nor the project virtual environment contained `ruff` or `mypy`, so their configured checks could not be executed locally.

### 9.3 CI does not exercise all dependency modes

The test environment installs the core and development dependencies but not the full RF/scientific or TGNN stack. Some ML tests are therefore skipped, while other modules rely on packages not declared in the tested dependency group.

#### Required resolution

Create separate CI jobs for:

1. Core simulator and metrics.
2. RF/scientific dependencies.
3. TGNN CPU-compatible dependencies.
4. Lint and type checking.

The final thesis environment should be pinned or locked.

## 10. Documentation inconsistencies

The thesis README still describes edge failures as sampled only from edges present at `t=0`, although the current default uses the temporal union of candidate edges. It also describes rain margin as part of RF ISL evaluation even though the implementation correctly disables rain fade for satellite-to-satellite RF links.

### Why this matters

Documentation drift can lead to a technically incorrect dissertation description even when the code is correct.

### Required resolution

Update architecture diagrams, README material, experiment descriptions, and dissertation wording after the simulation contract is finalized.

## 11. Recommended remediation sequence

The following order minimizes wasted computation and keeps changes atomic.

### Step 1: Lock physics invariants

- Enforce or remove maximum ISL distance.
- Fail on SGP4 propagation errors.
- Make orbital-engine choice explicit.
- Enforce adaptive terminal capacity.
- Add focused invariant tests.

### Step 2: Introduce a complete versioned reconstruction schema

- Export every rollout field.
- Add canonical row-to-config reconstruction.
- Recompute configuration hashes during validation.
- Add engine and source provenance.
- Make TGNN loading fail closed.

### Step 3: Correct ML feature contracts

- Enforce exact RF named feature sets.
- Rename `architecture_only`.
- Decide whether a true architecture-only group is required.
- Clarify or extend the TGNN `full` input contract.

### Step 4: Harden experiment provenance

- Require a clean source tree for canonical runs.
- Record environment and dependency versions.
- Hash dataset, split manifest, and experiment specification.
- Reject stale artifact reuse.

### Step 5: Define the statistical protocol

- Define the RF and TGNN estimands.
- Add baselines and imbalance-aware metrics.
- Choose a paired multi-seed protocol.
- Define confidence intervals and ablation comparisons before running.

### Step 6: Regenerate and rerun

- Generate a new schema-versioned dataset.
- Run validation and truthfulness checks.
- Run smoke experiments.
- Run the full ablation into a new output directory.
- Preserve the current K=2 artifacts as preliminary evidence; do not overwrite them.

## 12. Final assessment

The project is no longer a toy topology prototype. Its current temporal rollout, graph-derived labeling, failure realization, shared splits, and topology-sensitive K=2 model provide a credible Tier 1 foundation.

The remaining blockers are primarily contract and evidence problems rather than a need to redesign the entire system:

- Some declared physics constraints are not enforced.
- Invalid propagation can fail silently.
- Dataset rows do not yet contain a complete reconstruction recipe.
- ML condition names and inputs can be misinterpreted.
- Existing artifacts do not fully identify their executed source environment.
- One-seed evaluation is not sufficient for final research claims.

After those issues are resolved and the dataset is regenerated, the codebase should be in a much stronger position to support defensible temporal satellite-network resilience conclusions.
