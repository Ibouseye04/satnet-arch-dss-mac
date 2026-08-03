# Stage A Random Forest Model-Ready Dataset v1

This artifact is constructed only from the frozen accepted Stage A production evidence. It contains one row per complete simulation run, with five realization rows per design. No model was trained or tuned, no TGNN dataset was built, and no test target values were materialized.

## Targets

- `partition_any`: source field `overall_threshold_breach_any`.
- `gcc_frac_min`: source field `space_gcc_fraction_original_min`.

## Predictors

The 12 predictors are the pre-outcome design/configuration fields documented in `rf_feature_schema.json`: `num_planes`, `sats_per_plane`, `configured_satellite_count`, `altitude_km`, `inclination_deg`, `satellite_node_failure_probability`, `satellite_edge_failure_probability`, `civilian_count`, `government_count`, `military_count`, `total_ground_station_count`, and `ground_station_failure_probability`.

Identifiers and provenance are retained in every artifact but are explicitly non-feature. The complete included/excluded field audit is in `rf_leakage_exclusion_report.json`. Target fields, target derivatives, graph/service summaries, threshold outcomes, realized failures, seeds, hashes, paths, status, acceptance/replay fields, and split labels are not predictors.

## Splits

- `rf_train.csv`: 70 designs / 350 runs, with targets.
- `rf_validation.csv`: 15 designs / 75 runs, with targets.
- `rf_test_sealed_index.csv`: 15 designs / 75 runs, identifiers and provenance only; target values are sealed.

The frozen split manifest is verified by exact-file SHA-256 `047b7cc99d83a003add8bca4bb037bce9d642774443eeef1a3df377f855db90d`. The freeze manifest SHA-256 is `b2d1fbd9510d3d828fe051b4da04f6747088ae00d78d65e0a9851dc85559844e`.

## Verification

`rf_construction_report.json` records row/design cardinalities, duplicate and missing-value checks, source hash verification, and test-seal status. The independent RF dataset acceptance gate is intentionally not run by this construction task.
