# Ground Segment Stage G4 Validation Report

**Validation date:** 2026-07-16

**Branch:** `feature/ground-service-metrics`

**Validated G1 implementation SHA:** `753b28be767b7bcee213e239bbfd086b26b1944b`

**Validated G2 implementation SHA:** `5cc9e71399cc6587162634958e0173985c4ee228`

**Validated G3 implementation SHA:** `8f63531c9bedac97c1c28d76e44f0a8b24ca909a`

**G3 documentation/tagged HEAD and G4 branch base:** `a90c122b4f047bf0045fe10d7a899305a97ad5f4`

**Validated G4 implementation SHA:** `bcb56ff64730a31fc5b75591c44fc93abe1c7812`

**Documentation commit:** handled separately by this report commit

## Scope result

Stage G4 implemented deterministic satellite-GCC selection, original- and surviving-denominator space fractions, overall and class-specific ground service, conservative overall service, exact threshold compliance, temporal aggregation, five distinct identity layers, standalone JSONL persistence, verified G1-to-G4 replay, diagnostics, and upstream isolation.

No ground failures, dataset migration, labels, RF or TGNN features, caches, training, Monte Carlo generation, routing, scheduling, capacity, traffic, or link budgets were implemented.

## Validation summary

| Gate | Result |
|---|---:|
| G4 baseline complete suite | 642 passed |
| Final focused G4 suite | 91 passed |
| Final complete suite | 733 passed |
| Changed Python module compilation | Passed |
| Diagnostic harness | Passed |
| `git diff --check` | Passed |
| G3 base ancestry | Passed |
| Protected satellite diff | Empty |
| Protected G1-G3 ground diff | Empty |

A known unrelated Windows 10 ms timer-resolution test failed in two intermediate complete-suite runs. The unchanged suite passed on immediate rerun, including the final implementation gate. No timing code was modified.

## Original-denominator space resilience

The authoritative space quantity is:

```text
satellite_gcc_size / configured_satellite_count
```

The configured count comes from the satellite rollout configuration in verified orchestration. Tests reject Boolean, zero, inconsistent, and out-of-range configured counts and projected satellite IDs.

Space threshold decisions, overall service, authoritative run minima and means, and space breach reporting all use the original denominator.

## Surviving-node topology diagnostic

G4 separately reports:

```text
satellite_gcc_size / operational_satellite_count
```

This diagnostic describes connectivity among surviving satellites but is not treated as total constellation resilience.

## Catastrophic-attrition evidence

The deterministic diagnostic used 100 configured satellites, one operational satellite, and GCC size one:

| Metric | Value |
|---|---:|
| `space_gcc_fraction_surviving` | 1.0 |
| `space_gcc_fraction_original` | 0.01 |
| `ground_service_fraction` | 0.25 |
| `overall_service_fraction` | 0.01 |

This proves a lone surviving satellite does not produce perfect authoritative space resilience.

## Satellite-only GCC and mixed-graph prohibition

G4 calls the validated G3 satellite projection and applies connected-component analysis only to integer satellite nodes and ISLs. Ground nodes never enter component count, GCC size, or GCC membership.

Tests show a station attached to a nonauthoritative component remains unserviced even though it is connected in the mixed integrated graph. The diagnostic case selected `(0, 1)` from two satellite components and reported ground-service fraction `0.0` for a station attached only to satellite `2`.

## Deterministic equal-GCC tie-break

Two equal-size components `(0, 1)` and `(2, 3)` with different station attachments selected `(0, 1)` lexicographically. Only stations attached to that component were serviced.

Tests verify the resulting canonical GCC IDs, serviced and unserviced IDs, class fractions, ground fraction, overall fraction, and threshold states. The tie-break is operationally binding for all downstream G4 values.

## Ground denominator evidence

Every reconstructed G1 station remains in the denominator, including stations with no edge and stations attached only outside the GCC. A four-station diagnostic with one no-access station produced ground-service fraction `0.75`; the station remained explicitly unserviced.

The exact integrated ground-node set must equal the reconstructed G1 selected set. Missing or extra nodes fail context validation.

## Class-specific evidence

Mixed-class diagnostics produced numeric civilian, government, and military fractions from G1 selection and catalog evidence. A case with two serviced stations produced:

| Class | Fraction |
|---|---:|
| Civilian | 0.5 |
| Government | 1.0 |
| Military | 0.0 |

Absent government and military classes produced `None`, not zero, one, NaN, or an empty string.

Valid alternate G1, G2, and G3 evidence chains prove that changing catalog-authoritative class membership changes G4 class reporting and scientific identity. G4 never trusts a free-form G3 class attribute to classify stations.

## Threshold-boundary and zero-threshold evidence

Tests cover below, exact, and above space and ground boundaries plus all four truth combinations. Exact equality meets the policy without epsilon.

A zero-satellite, zero-ground-service case with both thresholds set to `0.0` produced all three threshold-met states as true while all fractions remained `0.0`. This is policy compliance only and is not described as connectivity or delivered service.

## Overall bottleneck evidence

Every step enforces:

```text
overall_service_fraction
= min(space_gcc_fraction_original, ground_service_fraction)
```

The catastrophic case yielded `min(0.01, 0.25) = 0.01`. Component values remain separately persisted.

## Zero-operational-satellite evidence

With no operational satellites, G4 produced zero components, zero GCC size, empty GCC IDs, both space fractions `0.0`, all stations unserviced, ground fraction `0.0`, and overall fraction `0.0`.

## Run aggregation and `math.fsum` evidence

Run aggregation requires nonempty, single-run, strictly increasing, unique, contiguous timestep records and strictly increasing timestamps. Scientific hashes and satellite, ground, and class denominators remain constant.

Every mean uses `math.fsum(values_in_timestep_order) / len(values)`. Tests compare exact results and reject replacement-hash corruption of minima, means, first breaches, counts, and sequence identity.

The disconnect-and-recovery diagnostic produced:

| Metric | Minimum | Mean |
|---|---:|---:|
| Original-denominator space | 0.5 | 0.8333333333333334 |
| Surviving-denominator space | 0.5 | 0.8333333333333334 |
| Ground service | 0.0 | 0.6666666666666666 |
| Overall service | 0.0 | 0.6666666666666666 |

Ground and overall breach counts were one, with first breach at timestep `1`. Space had no breach.

## Ordered step-sequence identity

The diagnostic sequence hash was:

```text
a2c6945372e70b13fdb50c9d8357ce0d482f7af00425a60a50914e9fc2695902
```

It binds ordered timestep, canonical timestamp, and scientific step identity. Reordered input fails. Reassigning scientifically different steps to new timestep/timestamp keys changes the sequence hash. Missing, extra, and duplicate steps fail.

## Identity separation

G4 distinguishes:

1. **Scientific step identity:** all scientific inputs and outputs for one timestep; excludes run ID.
2. **Step persistence identity:** binds schema, run ID, timestep, timestamp, and scientific step hash.
3. **Step-sequence identity:** binds the complete ordered sequence of contributing scientific steps.
4. **Scientific run-summary identity:** binds all run aggregates and sequence identity; excludes run ID.
5. **Run persistence identity:** binds schema, run ID, and scientific run-summary hash.

Tests prove equivalent science across different run IDs retains scientific hashes while persistence hashes change.

## Contextual validation and corruption resistance

Intrinsic dataclass validation enforces types, ranges, ordering, count and fraction algebra, supported versions, and hash recomputation. Contextual validators recompute graph-aware GCC and station service, policy comparisons, exact G1 station identity and class membership, and complete run aggregates.

Corruption tests cover configured and operational counts, component count, GCC size and IDs, station IDs and counts, class counts, original and surviving fractions, ground and overall fractions, threshold states, minima, means, first breaches, breach counts, sequence hash, scientific hashes, and persistence hashes. Replacement hashes do not make relational inconsistencies valid.

## Cross-stage run binding

Replay requires exact equality across G1 design, G2 records, G3 records, G4 step records, and the G4 run record. Tests reject independently valid alternate-run G2, G3, step, and summary records.

Step keys exactly equal verified G3 keys. Missing, extra, duplicate, and reordered step evidence fails. Missing and duplicate run summaries fail.

## Persistence and replay

`ground_service_steps.jsonl` and `ground_service_runs.jsonl` are standalone canonical artifacts. Round-trip tests prove compact sorted keys, canonical timestamps and float strings, explicit null values, UTF-8 Unix newlines, atomic synchronized replacement, path-independent bytes, no overwrite by default, duplicate-key rejection, empty-line rejection, exact fields, exact types, and record uniqueness.

Production generation invokes existing G3 verified replay and computes G4 only from returned snapshots. Replay regenerates and compares every step field, aggregate, scientific hash, and persistence hash.

A self-consistent arbitrary G3 graph was accepted by the pure G4 primitive as intrinsically valid but rejected by production replay because verified G3 reconstruction did not reproduce it.

Valid alternate upstream elevation/range evidence retained the same raw service fraction while changing the G2 snapshot, G3 graph, and G4 step scientific identities.

## Upstream identity invariance

Tests capture and compare catalog, selection, ground-design, G2 snapshot and record, G3 graph and record, satellite configuration, and failure-realization identities before and after G4 replay. All remain unchanged.

The G4 package-initializer change exports new public symbols only. No G4 state is written into G1, G2, or G3 records, nodes, edges, graph attributes, or persistence files.

## Dependency isolation

AST guards prove the pure policy, metric, and aggregation modules import no satellite network or simulation internals, model code, metrics package, graph-cache internals, datasets, or training code.

Production persistence imports only verified G1-to-G3 contracts and the existing rollout configuration and failure-realization boundary. Protected satellite modules do not import G4.

## Protected satellite diff

Release command compared `a90c122b4f047bf0045fe10d7a899305a97ad5f4..bcb56ff64730a31fc5b75591c44fc93abe1c7812` across:

```text
src/satnet/network
src/satnet/simulation/tier1_rollout.py
src/satnet/models/gnn_dataset.py
src/satnet/utils/graph_cache.py
```

Result: **empty**.

## Protected G1-G3 ground-module diff

Release validation compared the same range across all authoritative G1-G3 implementation modules:

```text
catalog.py
selection.py
scenario.py
persistence.py
canonical.py
coordinates.py
position_adapter.py
visibility.py
visibility_persistence.py
graph_attributes.py
integrated_graph.py
integrated_builder.py
satellite_graph_adapter.py
integrated_persistence.py
```

Result: **empty**.

## Known limitations

- Service is topological access through G2 geometric visibility, not RF or optical link viability.
- No capacity, traffic, routing, scheduling, user terminals, priority, or class weighting exists.
- One shared ground-service threshold applies to all classes.
- Ground-station failures are not modeled; every selected station remains in the denominator.
- No G4 fields enter existing satellite datasets, labels, caches, RF inputs, TGNN inputs, or training.

## Deferred G5 work

G5 may define deterministic persistent ground-station failures and reapply G4 metrics while retaining failed selected stations in the resilience denominator. Ground failure probabilities, seeds, failed IDs, dataset migration, ML features, training, and Monte Carlo generation remain outside G4.

READY FOR GROUND FAILURE MODEL DESIGN
