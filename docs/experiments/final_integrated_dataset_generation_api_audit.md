# Final Integrated Dataset Generation API Audit

## Status

Audited on 2026-07-20 at frozen starting commit `a1967185e80327e4b00c1831828dc975ab6819fc`. The existing production APIs support final-generation orchestration without protected-science changes. The final-generation layer must add only contract validation, configuration mapping, orchestration, satellite/target/result containers, inventories, operational state, replay coordination, and acceptance validation.

## Scientific boundary

The generation layer imports authoritative APIs and persists their native records. It does not recreate orbital, visibility, graph, service, or failure equations. `contract_bundle_hash` is operational contract-package provenance and is excluded from generated scientific payloads and hash preimages.

## Authoritative interfaces

| Stage | Import | Signature / return |
|---|---|---|
| Satellite | `satnet.simulation.tier1_rollout.Tier1RolloutConfig` | Frozen dataclass configuration; `config_hash()` is the authoritative satellite configuration identity. |
| Satellite | `satnet.simulation.tier1_rollout.run_tier1_rollout` | `(cfg) -> (list[Tier1RolloutStep], Tier1RolloutSummary, Tier1FailureRealization)` |
| G1 | `satnet.ground.selection.GroundSegmentEnabledConfig` | `(civilian_count, government_count, military_count, station_selection_seed)` |
| G1 | `satnet.ground.selection.select_ground_stations` | `(*, catalog, config) -> GroundStationSelection` |
| G1 | `satnet.ground.persistence.make_enabled_ground_design_record` | `(*, run_id, satellite_config_hash, selection) -> GroundRunDesignRecord` |
| G1 persistence | `satnet.ground.persistence.write_ground_design_manifest` | `(records, path, *, overwrite=False) -> None` |
| G2 positions | `satnet.ground.position_adapter.reconstruct_operational_satellite_position_sequence` | `(*, satellite_config, failure_realization) -> tuple[OperationalSatellitePositionSnapshot, ...]` |
| G2 visibility | `satnet.ground.visibility.GroundVisibilityPolicy` | `(minimum_elevation_deg)` |
| G2 visibility | `satnet.ground.visibility.evaluate_ground_design_visibility_sequence` | `(*, ground_design, catalog, satellite_sequence, policy) -> tuple[GroundVisibilitySnapshot, ...]` |
| G2 record | `satnet.ground.visibility_persistence.make_ground_visibility_record` | `(*, run_id, snapshot) -> GroundVisibilityRecord` |
| G2 persistence | `satnet.ground.visibility_persistence.write_ground_visibility_manifest` | `(records, path, *, overwrite=False) -> None` |
| G3 satellite graph | `satnet.ground.satellite_graph_adapter.reconstruct_operational_satellite_graph_sequence` | `(*, satellite_config, failure_realization) -> tuple[OperationalSatelliteGraphSnapshot, ...]` |
| G3 integrated graph | `satnet.ground.integrated_builder.build_integrated_ground_graph` | `(*, ground_design, catalog, satellite_graph_snapshot, verified_visibility_snapshot) -> IntegratedGroundGraphSnapshot` |
| G3 record | `satnet.ground.integrated_persistence.make_integrated_graph_record` | `(*, ground_design, snapshot) -> IntegratedGroundGraphRecord` |
| G3 persistence | `satnet.ground.integrated_persistence.write_integrated_graph_manifest` | `(records, path, *, overwrite=False) -> None` |
| G4 policy | `satnet.ground.service_policy.GroundServicePolicy` | `(space_gcc_threshold, ground_service_threshold)` |
| G4 calculation | `satnet.ground.service_persistence.generate_verified_ground_service_records` | `(*, satellite_config, failure_realization, ground_design, catalog, visibility_policy, visibility_records, integrated_records, service_policy) -> (step_records, run_record)` |
| G4 persistence | `satnet.ground.service_persistence.write_ground_service_step_manifest`, `write_ground_service_run_manifest` | Native canonical JSONL writers. |
| G5 policy | `satnet.ground.failure_policy.GroundFailurePolicy` | `(ground_station_failure_probability)` |
| G5 calculation | `satnet.ground.failure_service_persistence.generate_verified_ground_failure_service_records` | Accepts verified satellite through G4 evidence plus exactly one of `ground_failure_seed` or `persisted_realization_record`; returns realization, adjusted steps, and adjusted run. |
| G5 persistence | `satnet.ground.failure_service_persistence.write_ground_failure_realization_manifest`, `write_ground_failure_service_step_manifest`, `write_ground_failure_service_run_manifest` | Native canonical JSONL writers. |

## Replay entrypoints

| Stage | Authoritative replay |
|---|---|
| G2 | `satnet.ground.visibility_persistence.replay_ground_visibility_records` |
| G3 | `satnet.ground.integrated_persistence.replay_integrated_graph_records` |
| G4 | `satnet.ground.service_persistence.replay_ground_service_records` |
| G5 | `satnet.ground.failure_service_persistence.replay_ground_failure_service_records` |

Satellite replay is an exact rerun through `run_tier1_rollout()` using the frozen persisted configuration and seed. G1 replay reconstructs the selection through `select_ground_stations()` and compares the resulting design record. The final replay coordinator must treat the source run tree as read-only and must not call the pilot wrapper that writes into its input directory.

## Persistence boundaries and ordering

Native G1–G5 writers use canonical JSON/JSONL, deterministic record ordering, temporary siblings, flush, file `fsync`, and atomic replacement. Final generation must preserve their filenames within stage directories and publish `result.json` only after the satellite artifact, G1–G5 records, target artifact, and scientific inventory validate.

Required order is satellite, G1, G2, G3, G4, G5, target, inventory, result. G4 authoritatively replays G3 before metrics. G5 authoritatively replays G4 and supports replay from the persisted ground-failure realization.

## Frozen identities and versions

Runtime preflight must compare `PHYSICS_MODEL_VERSION`, canonical `LinkBudgetEngine().to_config()`, satellite schema/dataset versions, ground-selection/design versions, G2 model/frame/WGS84/schema versions, G3 model/schema versions, G4 model/policy/step/run versions, and G5 model/policy/sampling/realization/step/run versions to `contract_specification.json`.

G1 selection is design-level and preserves civilian, government, then military station order. Its selection and ground-design hashes remain fixed across a design's five realizations. The run-level G1 record varies by `run_id` and satellite configuration hash.

## Target source

All eight final target values are copied from `FailureAdjustedGroundServiceRunRecord.summary`. Classification values remain exact Booleans. Fraction values are finite binary64 values persisted through `canonical_float_string()` and validated in the closed unit interval.

## Audit conclusion

The audited API chain is compatible with the frozen 100-design, 500-run contract. No protected scientific API or equation change is required. The implementation may proceed in `satnet.experiments.final_generation` with final-generation-specific identity domains and strict reuse of the production persistence and replay boundaries above.
