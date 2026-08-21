# Ground Segment Stage G4 Service Metric Contract

**Status:** Approved

**Decision date:** 2026-07-16

**Validated G3 implementation SHA:** `8f63531c9bedac97c1c28d76e44f0a8b24ca909a`

**G3 documentation/tagged HEAD and G4 base:** `a90c122b4f047bf0045fe10d7a899305a97ad5f4`

## Scope

Stage G4 consumes only snapshots returned by verified G3 replay and produces deterministic timestep ground-service metrics, temporal run summaries, standalone canonical JSONL evidence, and verified G1-to-G4 replay.

G4 adds no ground failures, datasets, Monte Carlo generation, labels, caches, ML features, RF features, model training, routing, scheduling, capacity, traffic, or communications link budget.

## Scientific identity and policy

The service model and policy versions are both `1`. Policy identity uses domain `satnet_ground_service_policy`, version `1`.

`GroundServicePolicy` contains inclusive `space_gcc_threshold` and `ground_service_threshold` values. Inputs must be exact integers or floats, excluding Booleans. Numeric strings, NaN, infinity, and values outside `[0.0, 1.0]` fail. Accepted values are normalized to finite floats, including normalization of negative zero. Policy identity hashes canonical binary64 float strings.

## Pure timestep calculation

The required pure API is:

```python
def compute_ground_service_step(
    *,
    integrated_snapshot: IntegratedGroundGraphSnapshot,
    ground_design: GroundRunDesignRecord,
    catalog: GroundStationCatalog,
    configured_satellite_count: int,
    policy: GroundServicePolicy,
) -> GroundServiceStepMetrics:
    ...
```

The API accepts an intrinsically valid typed G3 snapshot, reconstructs the exact G1 selection, computes G4 metrics without mutating inputs, and does not independently prove complete upstream provenance. Production persistence and replay invoke it only with snapshots returned by verified G3 replay.

The orchestration layer derives `configured_satellite_count` from `Tier1RolloutConfig.total_satellites`. The pure API requires an exact positive integer.

Every projected satellite ID must satisfy:

```text
0 <= satellite_id < configured_satellite_count
```

This contextual G4 bound is required because G3 intrinsic validity permits any nonnegative satellite ID.

## Satellite GCC

Connected components are calculated only on the G3 projected satellite subgraph. Ground nodes never participate in selecting the satellite GCC.

The selected component has maximum cardinality. If multiple components have the same maximum size, the lexicographically smallest ascending tuple of integer satellite IDs is authoritative. This tie-break can change serviced stations and every downstream G4 metric.

The recorded fractions are:

```text
space_gcc_fraction_surviving = satellite_gcc_size / operational_satellite_count
space_gcc_fraction_original = satellite_gcc_size / configured_satellite_count
```

Both are `0.0` when no satellite is operational. `space_gcc_fraction_original` is authoritative for threshold evaluation, the overall bottleneck fraction, run minima and means, and breach reporting. The surviving-node fraction is diagnostic only.

## Ground service

A selected G1 station is serviced exactly when at least one canonical `IntegratedEdgeKind.SATELLITE_GROUND` edge connects it to a satellite in the authoritative satellite GCC. A station with multiple qualifying edges is counted once. Stations with no edge or only edges to satellites outside the selected GCC remain unserviced.

Every selected G1 station remains in the denominator. G4 models no ground failures. The G3 ground-node identity must equal the exact reconstructed G1 selected set.

Class authority comes exclusively from reconstructed G1 selection and the validated catalog. Class is reporting metadata and does not affect visibility, graph topology, service, priority, weighting, or thresholds. An absent class has fraction `None`.

```text
ground_service_fraction = serviced_ground_station_count / total_ground_station_count
overall_service_fraction = min(space_gcc_fraction_original, ground_service_fraction)
```

## Threshold compliance

Threshold fields are named:

```text
space_threshold_met
ground_threshold_met
overall_threshold_met
```

Comparisons are exact and inclusive, without epsilon. `overall_threshold_met` is the conjunction of the space and ground states.

A zero threshold allows a zero-valued service state to meet that threshold. This represents exact policy compliance and must not be described as graph connectivity or meaningful delivered service.

## Timestep evidence

`GroundServiceStepMetrics` is scientifically run-independent. Identity domain is `satnet_ground_service_step`, version `1`, and its hash includes all upstream scientific hashes, canonical timestep and timestamp, counts, canonical GCC and station IDs, class totals and serviced counts, canonical fractions, threshold states, and service model and policy identity. It excludes run ID, paths, and mutable graph objects.

Intrinsic validation enforces exact types, ranges, ordering, count algebra, fraction algebra, threshold algebra where the policy is supplied, and hash recomputation. Contextual validation enforces exact G1 station identity and class membership, configured satellite ID bounds, selected GCC membership, typed-edge service, and policy comparisons.

`GroundServiceStepRecord` binds schema version `1`, exact nonnegative integer run ID, timestep, canonical timestamp, and scientific step hash. Record identity domain is `satnet_ground_service_step_record`, version `1`.

## Run aggregation

`summarize_ground_service_run()` requires at least one bound step record from one run, in strictly increasing input order with unique contiguous timestep indices and strictly increasing timestamps. Satellite, ground, visibility, policy, and model hashes and configured satellite, ground, and class counts remain constant.

Every mean is calculated as:

```python
math.fsum(values_in_timestep_order) / len(values)
```

The summary records original- and surviving-denominator satellite minima and means; overall and class-specific ground minima and means; overall bottleneck minima and means; and breach flags, counts, and first timesteps. Class aggregates are `None` exactly when that class is absent.

The step-sequence identity domain is `satnet_ground_service_step_sequence`, version `1`. It hashes the ordered timestep, canonical timestamp, and scientific step hash for every contributing step. Missing, extra, reordered, or reassigned identities change the sequence identity or fail validation.

The scientific run-summary identity domain is `satnet_ground_service_run_summary`, version `1`. It includes the sequence identity and every aggregate but excludes run ID. The run persistence record uses schema version `1` and identity domain `satnet_ground_service_run_record`, version `1`, to bind the exact run ID and scientific summary hash.

## Persistence and verified replay

G4 writes standalone `ground_service_steps.jsonl` and `ground_service_runs.jsonl` files. Existing G1, G2, and G3 persistence is unchanged. Files use compact sorted-key UTF-8 JSON, canonical timestamps and float strings, explicit null values, Unix newlines, strict schemas, duplicate-key rejection, atomic synchronized sibling replacement, and no overwrite by default.

Step records are ordered by run then timestep and require unique keys. Run records are ordered by run and require unique run IDs. G4 step keys must exactly equal verified G3 keys, with exactly one summary for each represented run and no orphan summary.

Canonical production G4 replay accepts satellite configuration and failure realization, G1 design and catalog, G2 policy and records, G3 records, G4 policy, and persisted G4 step and run records. It verifies exact G1-to-G4 run binding, invokes existing verified G3 replay, derives configured satellite count from the satellite configuration, recalculates every step and aggregate, and compares every scientific field and persistence hash. Persisted counts, IDs, fractions, states, summaries, and hashes are never trusted without recomputation.

A self-consistent arbitrary G3 graph can be accepted by the pure primitive when intrinsically valid but is not canonical production evidence unless verified G3 replay reconstructs it exactly.

## Validation layers

Permanent intrinsic and contextual validation are both required. Replacing a hash does not make relationally inconsistent fields valid. Production factories, record construction, persistence reconstruction where evidence is available, and replay invoke the applicable contextual validators.

## Scientific interpretation

Space resilience is measured using the satellite GCC divided by the original configured satellite count. The surviving-node GCC fraction is retained separately as a topology diagnostic.

A deterministic equal-size GCC tie-break selects the authoritative component and can therefore affect ground service and every downstream G4 metric.

Ground stations are endpoints and do not participate in determining the satellite GCC.

G4 threshold fields describe policy compliance, not graph-theoretic connectivity.

Ground-station class authority comes from verified G1 catalog and selection evidence.

Ground-station failures are not modeled in G4. Every selected station remains in the denominator.

The overall-service fraction is the minimum of the original-denominator satellite GCC fraction and the ground-service fraction.

Canonical production G4 evidence is calculated only from integrated graph snapshots returned by verified G3 replay.

Every run summary is cryptographically bound to the complete ordered sequence of contributing G4 step identities.

## Deferred work

Ground failure probabilities, seeds, failed station IDs, failure-adjusted service evaluation, datasets, labels, caches, model features, training, and Monte Carlo generation are deferred. G5 will define deterministic persistent ground-station failures while retaining failed selected stations in the resilience denominator.
