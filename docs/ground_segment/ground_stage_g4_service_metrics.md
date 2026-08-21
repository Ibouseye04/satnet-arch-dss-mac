# Ground Segment Stage G4 Service Metrics

## Scope

Stage G4 derives deterministic service and resilience evidence from verified G1 station selection, verified G2 visibility, and verified G3 integrated graph snapshots.

G4 adds policy identity, per-timestep service metrics, temporal run summaries, standalone JSONL persistence, exact replay, and diagnostics. It does not add ground failures, link budgets, traffic, routing, scheduling, datasets, labels, caches, model features, training, or Monte Carlo generation.

## Verified trust boundaries

The pure `compute_ground_service_step()` primitive accepts an intrinsically valid typed G3 snapshot and verifies its relationship to reconstructed G1 selection and catalog evidence. It does not independently establish the complete G1-to-G3 provenance chain.

Canonical production G4 evidence is calculated only from integrated graph snapshots returned by verified G3 replay.

Production generation and replay therefore receive:

- Satellite rollout configuration and persistent satellite failure realization.
- G1 ground-design record and catalog.
- G2 visibility policy and records.
- G3 integrated graph records.
- G4 service policy.
- Persisted G4 step and run records during replay.

The existing G3 replay path first reconstructs operational satellite positions and graphs, replays G2, rebuilds G3, and compares canonical G3 evidence. Only its returned snapshots enter G4 calculation.

## Satellite projection

G4 calls the validated G3 satellite projection and computes connected components only on that projected satellite graph.

Ground stations are endpoints and do not participate in determining the satellite GCC.

Computing a component over the mixed integrated graph is prohibited because ground endpoint leaves and station quantity would distort satellite topology and component selection.

Every projected satellite ID must lie in the exact configured range. The configured count is derived from `Tier1RolloutConfig.total_satellites`; it is not inferred from failures, operational nodes, or edge counts.

## Configured and operational satellite counts

`configured_satellite_count` is the original constellation size. `operational_satellite_count` is the exact number of nodes in the projected G3 satellite graph after persistent satellite failures.

The two recorded satellite fractions are:

```text
space_gcc_fraction_original
= satellite_gcc_size / configured_satellite_count

space_gcc_fraction_surviving
= satellite_gcc_size / operational_satellite_count
```

For zero operational satellites, both fractions, component count, and GCC size are zero and GCC IDs are empty.

Space resilience is measured using the satellite GCC divided by the original configured satellite count. The surviving-node GCC fraction is retained separately as a topology diagnostic.

The original-denominator value is authoritative for space threshold compliance, overall-service calculation, run minima and means, and threshold-breach aggregation.

## Deterministic authoritative GCC

Every component is canonicalized as an ascending tuple of integer satellite IDs. G4 selects maximum cardinality and then the lexicographically smallest tuple among equal-size components.

A deterministic equal-size GCC tie-break selects the authoritative component and can therefore affect ground service and every downstream G4 metric.

The selected tuple is used for all attachment decisions in that timestep.

## Station service

A selected station is serviced when a canonical typed `satellite_ground` edge connects it to at least one satellite in the authoritative satellite GCC. Multiple qualifying neighbors count once.

A station remains unserviced when it has no ground edge or when every attached satellite lies outside the authoritative GCC. G4 inspects typed canonical edges rather than unqualified mixed-graph adjacency.

G4 verifies that the integrated ground-node set exactly equals the reconstructed G1 selected-station set.

## Ground denominator and classes

Ground-station failures are not modeled in G4. Every selected station remains in the denominator.

The overall ground-service fraction is:

```text
ground_service_fraction
= serviced_ground_station_count / total_ground_station_count
```

Ground-station class authority comes from verified G1 catalog and selection evidence.

Civilian, government, and military fractions use their selected class totals. An absent class has `None` for both timestep and run-level class fractions. Classes affect reporting only and do not change physics, topology, service rules, priority, weighting, or thresholds.

## Overall bottleneck metric

The overall-service fraction is the minimum of the original-denominator satellite GCC fraction and the ground-service fraction.

```text
overall_service_fraction
= min(space_gcc_fraction_original, ground_service_fraction)
```

Space, surviving-topology, ground, and overall values remain separately available. The bottleneck value does not replace its components.

## Policy and threshold terminology

The immutable service policy stores space-GCC and ground-service thresholds in `[0.0, 1.0]`. It rejects Booleans, strings, NaN, infinity, and out-of-range values, normalizes accepted integers and floats to binary64, normalizes negative zero, and hashes canonical float strings.

Threshold fields are:

```text
space_threshold_met
ground_threshold_met
overall_threshold_met
```

G4 threshold fields describe policy compliance, not graph-theoretic connectivity.

Comparisons are exact and inclusive. Overall compliance is the conjunction of space and ground compliance. No epsilon is applied.

A zero threshold allows a zero-valued service state to meet that threshold. This is exact policy compliance, not graph connectivity or meaningful delivered service.

## Timestep evidence and validation

`GroundServiceStepMetrics` records upstream hashes, exact counts, component state, canonical GCC IDs, canonical serviced and unserviced station IDs, class counts and fractions, space fractions, ground and overall fractions, threshold states, and scientific identity.

Intrinsic validation covers exact types, finite ranges, canonical ordering, count algebra, fraction algebra, conjunction state, and hash recomputation. Graph-aware contextual validation reconstructs G1 evidence and recomputes every field from the G3 snapshot and G4 policy. A replacement hash cannot make relationally inconsistent evidence valid.

The scientific step identity excludes run ID. `GroundServiceStepRecord` separately binds run ID, schema, timestep, canonical timestamp, and scientific step hash.

## Run aggregation

`summarize_ground_service_run()` requires one run, nonempty records, unique contiguous increasing timesteps, strictly increasing timestamps, and constant scientific identities and denominators.

Each mean uses exactly:

```python
math.fsum(values_in_timestep_order) / len(values)
```

The summary contains original- and surviving-denominator space minima and means, ground and overall minima and means, optional class minima and means, and space, ground, and overall breach flags, counts, and first timesteps.

Absent classes retain `None` for both minimum and mean.

## Ordered sequence identity

The step-sequence hash binds each contributing timestep index, canonical UTC timestamp, and scientific step hash in strict timestep order.

Every run summary is cryptographically bound to the complete ordered sequence of contributing G4 step identities.

Reordered input is rejected. Missing, extra, duplicate, reassigned, or scientifically changed steps alter sequence identity or fail validation.

The run-summary scientific hash excludes run ID. A separate run-record identity binds schema, run ID, and scientific summary hash.

## Persistence

G4 uses standalone files:

```text
ground_service_steps.jsonl
ground_service_runs.jsonl
```

Existing G1, G2, and G3 files and hashes remain unchanged.

Writers use compact sorted-key UTF-8 JSON, canonical UTC timestamps, canonical float strings, explicit null values, Unix newlines, synchronized sibling temporary files, atomic replacement, and no overwrite by default. Readers reject duplicate JSON keys, empty content, empty lines, unknown or missing fields, noncanonical floats and timestamps, wrong exact types, duplicate record keys, and invalid relationships or hashes.

Step records are ordered by run and timestep. Run records are ordered by run.

## Replay

G4 replay requires exact G1, G2, G3, step-record, and run-record run equality. G4 step keys must exactly equal verified G3 keys, and one run summary must exist.

Replay regenerates verified G3 snapshots, recalculates every G4 step field and hash, recalculates the ordered sequence and every run aggregate, and compares complete immutable records. Persisted counts, IDs, fractions, states, summaries, and hashes are never trusted without recomputation.

An intrinsically valid arbitrary self-rehashed G3 snapshot can be evaluated by the pure primitive, but production replay rejects it when it does not reproduce from verified upstream evidence.

## Identity layers

G4 distinguishes five identities:

- Scientific step identity: run-independent scientific timestep state.
- Step persistence identity: schema and run binding for one step.
- Step-sequence identity: complete ordered contributing sequence.
- Scientific run-summary identity: run-independent temporal aggregates.
- Run persistence identity: schema and run binding for one summary.

## Dependency boundary

Pure G4 policy, metric, and aggregation modules import no network, simulation, dataset, cache, model, or training modules. The persistence orchestration imports only the verified G1-to-G3 contracts and the existing satellite configuration and failure-realization boundary.

The package initializer has an export-only change for new G4 public symbols. No G1, G2, or G3 scientific implementation module was modified.

## Scientific limitations

A G2/G3 satellite-ground edge still represents geometric visibility, not RF or optical viability, capacity, scheduling, routing, traffic delivery, or user service quality. G4 service means topological access to the authoritative satellite GCC under that visibility contract.

G4 uses one unweighted threshold for all stations and classes. It does not model ground-station priority or class-weighted service.

## Deferred work

G5 will define deterministic persistent ground-station failures and reapply G4 metrics while retaining every selected station in the resilience denominator.

Dataset migration, RF and TGNN features, labels, model training, cache changes, Monte Carlo regeneration, and final experiment generation remain deferred.
