# Final Integrated Dataset API Audit

## Status

Audited for the contract-only final integrated dataset phase on 2026-07-19 and amended on 2026-07-20 to bind corrected numeric run identity semantics.

The audited repository HEAD is `e4475f9bc22a83b30cdc6862d3337a1e2b6dbc3f`, the validated integrated-pilot report commit. The protected G1-to-G5 science base remains `62beda9df1576958d9e33d33d2d9eb5489e24b20`.

The tracked working tree was clean at preflight. Unrelated untracked paths were present and excluded from this task. The complete pre-change baseline was 809 passing tests.

## Scope

This audit authorizes contract artifacts, deterministic design tooling, schemas, design and run manifests, and a pre-outcome grouped split. It does not authorize simulation execution, G1-to-G5 scientific changes, current RF or TGNN implementation changes, or a release tag.

## Authoritative pilot inputs

| Input | Path | Identity |
|---|---|---|
| Pilot design manifest | `artifacts/integrated_ground_pilot_25/inputs/pilot_designs.json` | `514795b9cc17deacd81ffb9308e0d944f8da64b80444047647e919b375ef27bf` |
| Pilot catalog | `artifacts/integrated_ground_pilot_25/inputs/pilot_catalog.csv` | production semantic catalog hash `810c64dfb030b042311c90f2f42f8dee866a48fc63a6a29e362ee328c52eaa6e` |
| Pilot catalog provenance | `artifacts/integrated_ground_pilot_25/inputs/pilot_catalog_provenance.json` | synthetic, pilot-only, not scientifically reviewed |
| Pilot report | `docs/experiments/integrated_ground_pilot_25_report.md` | validated report at audited HEAD |

The catalog hash is a semantic identity over parsed canonical station records. It is not a raw CSV-byte hash.

## Production execution interfaces

| Stage | Authoritative interface | Source |
|---|---|---|
| Satellite rollout | `Tier1RolloutConfig`, `run_tier1_rollout()` | `src/satnet/simulation/tier1_rollout.py` |
| G1 selection | `GroundSegmentEnabledConfig`, `select_ground_stations()` | `src/satnet/ground/selection.py` |
| G1 persistence | `make_enabled_ground_design_record()` | `src/satnet/ground/persistence.py` |
| G2 positions | `reconstruct_operational_satellite_position_sequence()` | `src/satnet/ground/position_adapter.py` |
| G2 visibility | `GroundVisibilityPolicy`, production visibility persistence | `src/satnet/ground/visibility.py`, `src/satnet/ground/visibility_persistence.py` |
| G3 satellite graph | `reconstruct_operational_satellite_graph_sequence()` | `src/satnet/ground/satellite_graph_adapter.py` |
| G3 integrated graph | `build_integrated_ground_graph()` and production persistence | `src/satnet/ground/integrated_builder.py`, `src/satnet/ground/integrated_persistence.py` |
| G4 service | `GroundServicePolicy`, production service persistence | `src/satnet/ground/service_policy.py`, `src/satnet/ground/service_persistence.py` |
| G5 ground failure | `GroundFailurePolicy`, production failure realization and adjusted service persistence | `src/satnet/ground/failure_policy.py`, `src/satnet/ground/failure_service_persistence.py` |

The final generation phase must orchestrate these APIs without recreating their equations.

## Production seed purposes

`Tier1RolloutConfig` exposes one `seed` field. Production samples persistent satellite-node and accepted-edge failures from the same seeded random stream. The authoritative repository-equivalent purpose is therefore:

```text
satellite_rollout_and_failure
```

G1 selection exposes one station-selection seed. The final contract intentionally derives it once per design, not once per realization.

G5 exposes one ground-failure seed per run realization.

The final seed purposes are therefore:

```text
design-level: ground_station_selection
realization-level: satellite_rollout_and_failure
realization-level: ground_failure_realization
```

No separate node-failure, edge-failure, orbit, visibility, G4, or G5 timestep seeds exist in production.

## Production run identity

The final run manifest uses exact integer `run_id` values 0 through 499 as the sole authoritative production execution, filename, replay, and dataset-join identity. Boolean values are rejected. With zero-based design and realization indices, `run_id = design_index * 5 + realization_index`.

The human-readable `run_key` is `design_id + "-" + realization_id`, with realization labels `R00` through `R04`. Run records bind `run_id`, `run_key`, both indices, all approved seeds, and the frozen design-level split assignment. The obsolete `run_index` alias is absent and has no production role.

## Fixed satellite and temporal profile

| Field | Locked value |
|---|---|
| `duration_minutes` | 10 |
| `step_seconds` | 60 |
| inclusive state count | 11 |
| `phasing_factor` | 1 |
| `max_isl_distance_km` | canonical binary64 `10000` |
| `isl_policy` | `grid_fixed` |
| `adjacent_search_k` | 1 |
| `max_inter_plane_links_per_sat` | 1 |
| `orbital_engine` | `sgp4` |
| `epoch_iso` | `2000-01-01T12:00:00+00:00` |
| `failure_model` | `persistent_temporal_union_edges_v1` |
| satellite rollout schema version | 2 |
| satellite dataset version | `tier1_temporal_connectivity_v2` |

The final DOE ranges are subsets of parameter points already exercised by the pilot: planes 4 through 6, satellites per plane 5 through 8, altitude 300 through 1200 km, inclination 30 through 98 degrees, node-failure probability 0 through 0.20, and accepted-edge failure probability 0 through 0.25.

## Link-budget and physics identity

The production physics identifier is:

```text
tier1_space_segment_physics_v2
```

The exact default `LinkBudgetEngine` configuration is:

```text
optical_tx_power_dbm = 37
optical_aperture_m = 0.10000000000000001
optical_sensitivity_dbm = -45
optical_wavelength_m = 1.55e-06
rf_tx_power_dbm = 30
rf_antenna_gain_dbi = 40
rf_sensitivity_dbm = -90
rf_frequency_hz = 28000000000
rf_rain_margin_db = 10
```

The machine specification obtains these values from `LinkBudgetEngine().to_config()` and canonicalizes each binary64 value with `canonical_float_string()`. There is no separate production link-budget-version field.

## G1-to-G5 versions and policy identities

| Identifier | Value |
|---|---|
| ground station selection version | `1` |
| G1 ground design schema version | `1` |
| ground visibility model version | `1` |
| ground visibility frame contract | `hypatia_ecef_gmst_v1` |
| WGS84 model | `wgs84_geodetic_ecef_v1` |
| G2 visibility schema version | `1` |
| G3 integrated graph model version | `1` |
| G3 integrated graph schema version | `1` |
| G4 service model version | `1` |
| G4 service policy version | `1` |
| G4 step schema version | `1` |
| G4 run schema version | `1` |
| G5 failure model version | `1` |
| G5 failure policy version | `1` |
| G5 failure sampling version | `1` |
| G5 realization schema version | `1` |
| G5 adjusted service model version | `1` |
| G5 adjusted step schema version | `1` |
| G5 adjusted run schema version | `1` |

The fixed visibility policy uses minimum elevation 10 degrees. The fixed G4 service policy uses space and ground thresholds equal to the binary64 value represented canonically as `0.80000000000000004`.

The ground-failure model and sampling versions are constant. `GroundFailurePolicy.ground_failure_policy_hash` varies by design because its probability varies.

## Ground identity behavior

Production G1 selection preserves class-concatenated ordering:

```text
civilian IDs, then government IDs, then military IDs
```

The selection hash binds the catalog hash, requested class counts, selected IDs, selection seed, and selection version. The ground-design hash binds the selection hash and enabled state; it does not bind the satellite configuration hash. A run-level G1 record separately carries its run ID and satellite configuration hash.

Therefore one final design can correctly hold a constant selected geography, selection hash, and ground-design hash while its five run-level G1 records carry different satellite configuration hashes.

## Authoritative target fields

The final targets are copied from verified G5 run summaries.

| Role | Field | Type |
|---|---|---|
| primary classification | `overall_threshold_breach_any` | Boolean |
| secondary classification | `ground_threshold_breach_any` | Boolean |
| secondary classification | `space_threshold_breach_any` | Boolean |
| primary regression | `failure_adjusted_overall_service_fraction_mean` | finite binary64 fraction |
| safety secondary regression | `failure_adjusted_overall_service_fraction_min` | finite binary64 fraction |
| secondary regression | `failure_adjusted_ground_service_fraction_min` | finite binary64 fraction |
| secondary regression | `space_gcc_fraction_original_min` | finite binary64 fraction |
| diagnostic only | `ground_service_loss_due_to_failures_max` | finite binary64 fraction |

The primary regression choice follows the pilot evidence: its mean target had better unique-value and within-design stochastic spread than its minimum target. The diagnostic-only loss field failed the pilot useful-spread gate.

## Current RF boundary

The current RF registry in `src/satnet/models/risk_model.py` is satellite-only and does not support the final integrated feature schema. The final `integrated_rf_export_schema_v1` is a future one-row-per-run export contract. It does not authorize loader, registry, trainer, or model changes.

RF predictors are pre-outcome design values only. Seeds, selected station IDs, hashes, realized failures, graph metrics, service metrics, targets, statuses, and runtime values are prohibited predictors.

## Current TGNN boundary

The current `SatNetTemporalDataset` in `src/satnet/models/gnn_dataset.py` reconstructs satellite-only graphs. The current `SatelliteGNN` in `src/satnet/models/gnn_model.py` is not an integrated satellite-ground model and does not consume general `edge_attr`.

The final `integrated_tgnn_adapter_schema_v1` is a future adapter specification over canonical G3 graphs plus the verified G5 persistent ground-failure overlay. It is not implemented or directly consumable by the current model. Satellite ECEF features are omitted. Satellite failures retain G3 absent-node semantics; failed ground stations remain as nodes with a G5 operational indicator.

## Canonical persistence rules

Scientific binary64 values use `canonical_float_string()`. Canonical hash payloads contain no raw JSON floats. Canonical JSON is UTF-8, sorted by key, compact, and non-ASCII preserving. Negative zero is rejected at persistence boundaries.

G1 class-specific selected tuples retain production order. G5 station tuples use their own globally sorted semantics and must not be substituted for G1 ordering.

## Protected implementation boundary

The contract task must leave unchanged every authoritative module under:

```text
src/satnet/ground
src/satnet/network
src/satnet/simulation/tier1_rollout.py
src/satnet/models/gnn_dataset.py
src/satnet/models/gnn_model.py
src/satnet/models/risk_model.py
src/satnet/utils/graph_cache.py
```

New final-contract tooling belongs under `src/satnet/experiments`. New tests belong under `tests/experiments`. Contract artifacts and documentation do not become G1-to-G5 scientific evidence.

## Audit conclusion

The production API supports the approved contract without scientific changes. Exact fixed-ground architecture is compatible with G1 identities, the satellite stochastic purpose is confirmed as `satellite_rollout_and_failure`, and all requested target fields exist in verified G5 run summaries.
