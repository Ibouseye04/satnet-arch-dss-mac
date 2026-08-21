# SATNET Final DSS — Phase 1 Backend Contract

## Purpose

Phase 1 provides one Python entry point that transforms a validated user architecture through the existing SATNET Tier 1 temporal pipeline, evaluates the frozen operational TGNN regression model across five deterministic failure realizations, and returns an engineering decision-support result. This phase does not provide a GUI, HTTP server, persistence, authentication, or deployment packaging.

## Operational model decision

The DSS exposes exactly one ML model:

- family: `TGNN`
- task: `tgnn_space_regression`
- winning configuration: `tgnn_010`
- reporting seed: `42`
- target: `space_gcc_fraction_original_min`
- architecture: node features 3, hidden dimension 64, one regression output, Chebyshev order 2, one recurrent layer

The Random Forest and classification models remain dissertation research and evaluation benchmarks. They are not operational DSS choices. The DSS does not retrain, tune, select models, expose hyperparameters, or change the frozen TGNN architecture.

The operational checkpoint is loaded from `SATNET_DSS_TGNN_CHECKPOINT`. The service requires the file to exist and to have SHA-256:

`22cabef076428ba5c118b10fa230c5930af1b1c2da53bdd5c51b118dc0c9960a`

A missing file, hash mismatch, architecture mismatch, or load failure fails closed. The checkpoint is not committed to Git.

## Request and validated domain

`DSSArchitectureRequest` accepts:

- `num_planes`: integer 4–6
- `sats_per_plane`: integer 5–8
- `altitude_km`: 300–1200
- `inclination_deg`: 30–98
- `satellite_node_failure_probability`: 0.0–0.20
- `satellite_edge_failure_probability`: 0.0–0.25
- `civilian_count`, `government_count`, `military_count`: nonnegative catalog-backed counts
- `ground_station_failure_probability`: 0.0–0.40
- `required_minimum_connectivity`: [0, 1], default 0.80

Space inputs outside the validated TGNN training domain are hard-rejected. The ground counts are validated against the enabled ground-station catalog named by `SATNET_DSS_GROUND_CATALOG`; silent reduction or extrapolation is not allowed.

## Scientific scenario construction

The DSS uses the existing fixed final-dataset profile:

- SGP4 orbital propagation
- J2000 epoch
- 10-minute duration
- 60-second step
- inclusive sequence length of exactly 11 snapshots
- `grid_fixed` ISL policy
- 10,000 km maximum ISL distance
- existing persistent temporal-union edge-failure semantics

The DSS orchestrates existing `HypatiaAdapter`, Tier 1 rollout, G3 graph, ground visibility, integrated graph, G4 service, and G5 ground-failure implementations. It does not replace SGP4, LOS, link budgets, failure semantics, visibility, selection, or service definitions.

## Five-realization policy

Every request executes exactly five realizations indexed 0 through 4. Seeds are derived with SHA-256 over canonical architecture data, a frozen DSS master/domain identifier, a separate seed domain, and the realization index where applicable.

Separate domains are used for:

- satellite failure realization
- ground failure realization
- architecture-level ground-station selection

The canonical physical architecture payload excludes `required_minimum_connectivity`. Therefore changing only the requirement threshold preserves realization seeds, SATNET physical configuration, temporal graph identities, TGNN feature tensors, predictions, and calculated ground/system values.

The existing SATNET failure implementation remains authoritative: node and edge failures are sampled persistently for a temporal rollout, with edge eligibility determined by the configured SATNET failure model.

## TGNN feature contract

Each realization is transformed into exactly 11 PyG graph snapshots using the existing SATNET conversion contract:

- node feature dimension: 3
  - `plane_idx_normalized`
  - `sat_in_plane_normalized`
  - `node_exists_constant`
- edge feature dimension: 4
  - `distance_km_scaled_10000`
  - `margin_db_scaled_100`
  - `link_type_code_scaled_2`
  - `link_mode_binary`

For the frozen model API, `edge_weight` is exactly `edge_attr[:, 0]`. No additional edge feature is consumed by the model. Model inference runs in evaluation mode under `torch.no_grad()`.

Raw predictions are retained without clipping. The result records whether a prediction is outside the physical [0, 1] interval; an out-of-range value is not silently corrected.

## Expected Minimum Connectivity

**Expected Minimum Connectivity** means:

> the arithmetic mean of the five TGNN-predicted minimum GCC values.

The internal result field is `expected_minimum_gcc`. It is not a calibrated probability, likelihood, or probability of success.

The DSS also returns the lowest and highest modeled values, the configured requirement, expected and lowest margins, the count of modeled realizations meeting the requirement, and a boolean realization risk flag. A count such as “4 of 5 modeled failure realizations met the requirement” is descriptive only and must not be presented as “80% probability of success.”

The primary assessment is `MEETS_EXPECTED_REQUIREMENT` when the arithmetic mean is greater than or equal to the requirement; otherwise it is `BELOW_EXPECTED_REQUIREMENT`. The risk flag is true when any raw realization prediction is below the requirement.

## Threshold semantics

`required_minimum_connectivity` is a post-inference engineering requirement. It is not passed to:

- SATNET graph generation
- failure seed derivation
- TGNN feature construction
- TGNN inference

Changing only the threshold changes the requirement, margins, assessment, meeting count, and risk flag. It does not change modeled physical scenarios or raw TGNN predictions. No threshold tuning or clipping is performed.

## Ground and system context

The operational ML model is TGNN-only. Ground and integrated values are SATNET-calculated outputs, not ML predictions.

Using the same five realizations, the service invokes authoritative G4/G5 metrics and aggregates:

- `mean_ground_service_fraction`
- `minimum_ground_service_fraction`
- `mean_overall_service_fraction`
- `minimum_overall_service_fraction`
- `limiting_segment`

Ground values use the existing G4 ground-service and G5 failure-adjusted service definitions. Output provenance is explicitly labeled `SATNET_CALCULATED`; space values are labeled `TGNN_PREDICTION`. If the catalog or a qualified G1–G5 prerequisite is unavailable, system context is returned as `BLOCKED` with an explanation rather than an invented metric.

## Result and provenance

`analyze_architecture(request)` returns `DSSAnalysisResult`, with dictionary serialization containing architecture, model metadata, space resilience, system context, provenance, and an `analysis_details` field. Individual realization predictions, seeds, temporal graph hashes, feature dimensions, and out-of-range flags are retained in analysis details rather than the primary summary.

Provenance includes the checkpoint hash, scenario profile, physical architecture hash, seed policy, graph identity evidence, and `training_performed: false`.

## Limitations

- Five realizations are a deterministic engineering scenario policy, not a statistical probability estimate.
- The TGNN prediction is not clipped and may require engineering interpretation if outside [0, 1].
- Results are limited to the validated training domain and fixed temporal profile.
- Ground results depend on the validated catalog supplied through `SATNET_DSS_GROUND_CATALOG`.
- No gateway routing, traffic, throughput, authentication, persistence, web API, or frontend is included in Phase 1.

## No-retraining policy

Phase 1 performs inference only. It contains no optimizer path, fitting path, model-selection control, threshold tuning, or checkpoint-writing behavior. Frozen training, test, external-validation, checkpoint, and production dataset artifacts are not modified or committed.
