# Tier 1 Historical Artifact Blast-Radius Audit

**Audit date:** 2026-07-16

**Scope:** Repository-local historical datasets, split manifests, models, metrics, predictions, comparison outputs, and graph caches after V-01 through V-06 remediation

**Method:** Read-only filesystem inventory and metadata inspection; no historical artifact was altered, deleted, regenerated, or overwritten

## 1. Classification rules

- **Canonical regeneration required:** Known to use affected physics semantics, or cannot satisfy the current fail-closed canonical data contract.
- **Historical only:** Preserve as preliminary evidence, but do not use as final canonical evidence.
- **Unaffected by a specific defect:** Available metadata excludes that defect's trigger. This does not establish validity against other defects.
- **Unresolved:** Historical metadata cannot prove or exclude exposure. No favorable assumption is made.

Preservation and scientific validity are separate decisions. Every inventoried artifact remains immutable historical evidence.

## 2. Remediation baseline

| Defect | Remediation commit | Current contract |
|---|---|---|
| V-02 | `57b220a` | Canonical execution requires SGP4 and fails on every nonzero propagation code. |
| V-03 | `0525b72` | Adaptive capacity is deterministic total incident inter-plane degree. |
| V-01 | `8d2c2d8` | Maximum ISL distance is a hard inclusive limit. |
| V-04 | `2286d90` | Canonical reconstruction requires schema v2 graph-defining metadata. |
| V-05 | `df4ffce` | Graph cache schema v5 requires complete scientific identity and valid payload structure. |
| V-06 | `b875579` | Experiment reuse requires exact condition identity; comparison collection requires common provenance. |

The post-V-06 full suite passed: **318 tests**.

## 3. Primary dataset blast radius

`data/tier1_design_runs.csv` has SHA-256 `9e7cf202082c363b755404e48f315724e009a52d6d9e47df121339d3b32ab0cb` and contains 500 rows.

| Observed field | Historical value |
|---|---|
| Schema | `1` for 500/500 rows |
| Dataset version | `tier1_temporal_connectivity_v1` for 500/500 rows |
| ISL policy | `grid_adaptive` for 500/500 rows |
| Adjacent search K | `1` for 500/500 rows |
| Maximum inter-plane links per satellite | `1` for 500/500 rows |
| Failure model | `persistent_temporal_union_edges_v1` for 500/500 rows |
| Schema-v2 fields absent | `phasing_factor`, `max_isl_distance_km`, `orbital_engine` |

### Finding

**Canonical regeneration is required for the complete 500-row dataset and its 15,500-row steps table.**

- **V-03 — definitive exposure:** Every row uses the corrected adaptive policy and capacity setting. Total-incident endpoint enforcement can change temporal graphs, failure-edge universes, labels, and dataset content.
- **V-04 — current incompatibility:** The run table is schema v1 and lacks three mandatory reconstruction fields. The remediated canonical TGNN loader rejects it rather than synthesizing defaults.
- **V-02 — unresolved historical incidence:** No row records canonical engine identity or propagation-error telemetry. Nonzero SGP4 incidence cannot be proven or excluded. Canonical final evidence therefore requires a fail-closed rerun.
- **V-01 — not independently demonstrated for this dataset:** The 10,000 km historical limit is absent from run rows, and temporal edge distances are not exported. The range defect alone does not prove these 500 rows changed, but V-03 and V-04 already require regeneration.

The historical CSV files must remain preserved under their existing names. Regenerated data must use new versioned locations.

## 4. Dataset-bound experiment artifacts

Six split manifests under the three experiment roots below all record the exact primary dataset hash and 500-row identity:

| Experiment root | Split manifests | Dataset identity | Classification |
|---|---:|---|---|
| `artifacts/ablation` | 2 | Exact primary v1 dataset hash | Canonical regeneration required |
| `artifacts/validation/space_segment_k2_500` | 2 | Exact primary v1 dataset hash | Canonical regeneration required |
| `artifacts/validation/tgnn_k2_ablation_smoke` | 2 | Exact primary v1 dataset hash | Historical smoke evidence only |

Because each split is content-bound to the affected v1 dataset, all models, predictions, metrics, and comparison tables produced through those manifests are downstream of V-03 and are not canonical after remediation. This includes the fresh-directory K=2 results: V-06 does not show them to be stale-reused, but freshness cannot cure upstream physics exposure.

Repository inventory for these roots:

| Root | Files | Notable serialized artifacts |
|---|---:|---|
| `artifacts/ablation` | 110 | 6 RF `.joblib`, 6 TGNN `.pt`, 22 CSV, 39 JSON |
| `artifacts/validation/space_segment_k2_500` | 55 | 6 TGNN `.pt`, 9 CSV, 21 JSON |
| `artifacts/validation/tgnn_k2_ablation_smoke` | 55 | 6 TGNN `.pt`, 9 CSV, 21 JSON |

The split manifests themselves are also historical only because a regenerated dataset will have a different content hash and potentially different labels.

## 5. V-06 experiment identity audit

Observed across `artifacts/`:

- 28 metrics JSON files.
- 3 run manifests.
- 0 `experiment_identity.json` sidecars.
- All 3 run manifests lack condition identities.

### Finding

**No historical experiment condition is eligible for automatic reuse under the remediated V-06 contract.** Existing metrics are not thereby proven numerically wrong; they are unidentifiable for safe resume and mixed-provenance collection.

The three pre-remediation manifests are:

| Manifest | Recorded K | Recorded seed | Identity status |
|---|---:|---:|---|
| `artifacts/ablation/run_manifest.json` | absent | 42 | Legacy; no condition identities |
| `artifacts/validation/space_segment_k2_500/run_manifest.json` | 2 | 42 | Legacy; no condition identities |
| `artifacts/validation/tgnn_k2_ablation_smoke/run_manifest.json` | 2 | 42 | Legacy; no condition identities |

Do not add identity sidecars retrospectively: doing so would assert provenance that was not emitted atomically with the historical run. Future experiments must write identities in new output directories through the remediated runner.

## 6. Controlled-validation artifacts

| Artifact group | Evidence | Classification |
|---|---|---|
| `adaptive_k1_failure_pilot` | 250 schema-v1 rows; run metadata records `grid_adaptive`, K=1 | Historical only; adaptive results are in V-03 scope and canonical rows lack schema-v2 identity |
| `isl_policy_comparison` | 100 `grid_fixed` and 100 each of adaptive K=1, K=2, K=3 | Adaptive subsets are in V-03 scope; fixed subset is outside V-03 but remains pre-v2 with unresolved V-02 provenance |
| `baseline_no_failures` | 100 schema-v1 rows; policy/capacity not recorded in the CSV | Unresolved policy exposure; historical only because canonical engine/range/phasing identity is absent |
| `stage_diagnostic` | 19,689 accepted links; maximum accepted distance 7,922.130 km; 1,465 candidates over 10,000 km were all LOS-rejected | No observed accepted-edge violation of the historical 10,000 km range in this diagnostic only |
| `artifacts/smoke` | 20 legacy smoke files, including RF/TGNN metrics and serialized models | Historical smoke evidence only; not a canonical result set |
| `models` | 2 legacy metrics JSON files, no serialized models in the directory | Historical metrics only; provenance is insufficient for canonical reuse |

The stage diagnostic narrows V-01 impact only for its own sampled candidate set. It does not establish the 500-run dataset's edge-level range compliance.

## 7. Graph-cache blast radius

No cache-named directory and no `artifacts/graph_cache` entry were detected.

### Finding

**No repository-local historical artifact is invalidated by consuming a detected V-05 cache.** If external or deleted caches existed, this audit cannot classify them. Any cache created before schema v5 must not be reused by the canonical loader.

## 8. Defect-by-defect disposition

| Defect | Confirmed historical blast radius | Unresolved boundary |
|---|---|---|
| V-01 | No accepted over-10,000 km edge in the controlled stage diagnostic | Primary dataset lacks edge distances and stored range identity |
| V-02 | All canonical final evidence needs fail-closed regeneration because error telemetry is absent | Actual historical nonzero SGP4 incidence is unknowable from artifacts |
| V-03 | All 500 primary rows; six hash-bound splits; downstream RF/TGNN models, predictions, metrics, and comparisons; adaptive controlled validations | Legacy files that omit policy/capacity cannot be classified from file content alone |
| V-04 | Primary and controlled run tables are schema v1 and fail the new canonical reconstruction contract | Complete v1 nominal replay may remain historically interpretable, not canonical |
| V-05 | None detected locally | External or previously removed caches are outside audit scope |
| V-06 | All 28 historical metrics lack identity sidecars; all 3 manifests lack condition identities | Historical numerical correctness is not inferable solely from missing identity |

## 9. Required next actions

1. Preserve all historical artifacts unchanged.
2. Generate a new schema-v2, dataset-v2 canonical dataset in a new versioned directory using mandatory fail-closed SGP4, corrected adaptive capacity, and hard inclusive range enforcement.
3. Build new split manifests from the regenerated dataset content hash.
4. Retrain RF and TGNN models in new output roots.
5. Produce new predictions, metrics, and comparison tables only through the identity-enforcing experiment runner.
6. Do not treat any legacy run manifest or metrics-path existence as permission to resume.

These are future regeneration actions, not actions performed by this audit.

## 10. Conclusion

The historical blast radius is broad but bounded:

- **Broad:** V-03 definitively reaches the entire primary 500-run adaptive dataset and all six content-bound experiment splits and downstream evidence.
- **Mandatory for canonical evidence:** V-02's absent error telemetry and V-04's schema-v1 omissions require a clean versioned rerun even where no historical failure is proven.
- **Narrow or absent locally:** V-01 has no demonstrated 10,000 km violation in the controlled stage sample, and V-05 has no detected local cache consumer.
- **Experiment-level:** V-06 makes all historical outputs ineligible for automatic reuse because none carries the new atomic identity sidecar.

No historical artifact was modified during this audit.
