# Ground Segment Stage G5 Architecture Decision

## Status

Accepted and implemented on 2026-07-16.

Validated implementation SHA: `b350be7303d202038b775227c78d354f56054d16`.

The validated upstream G4 implementation remains `bcb56ff64730a31fc5b75591c44fc93abe1c7812`. The G5 branch base and G4 documentation HEAD remain `5da9dd818fc452a0938cfe97144ee38bbf0bf893`.

## Decision

Stage G5 is implemented as a deterministic persistent operational eligibility overlay on verified G4 ground-service evidence. One station-local Bernoulli trial is evaluated for every selected G1 station using a run-constant seed. Failed and operational station sets remain fixed for the complete run.

G5 does not remove failed stations or visibility edges from G3. At every timestep, failure-adjusted service is the verified G4 serviced set minus the persistent failed set. All selected G1 stations remain in the overall and class-specific denominators.

## Scientific boundary

The G5 boundary is limited to satellite-to-satellite network state already evaluated by G4 and persistent ground-station eligibility. It adds no Earth rotation, gateway logic, ground-link failure process, satellite failure process, communications subsystem model, traffic, routing, scheduling, capacity, dataset generation, label generation, machine-learning features, cache behavior, or integrated pilot execution.

Disabled and empty ground designs are invalid G5 inputs. Canonical execution requires an enabled nonempty G1 selection.

## Deterministic failure policy

`GroundFailurePolicy` accepts exact integer or float values in `[0.0, 1.0]`, rejects Booleans and non-finite values, stores the normalized binary64 value, and converts negative zero to positive zero. Policy identity includes model, policy, and sampling versions and the canonical probability string.

The ground-failure seed has exact type `int`, excluding Boolean, and lies in `[0, 2**63 - 1]`.

Every station trial uses canonical JSON over the locked domain, version, failure model version, sampling version, seed, and station ID. SHA-256 produces a 256-bit unsigned big-endian integer. Failure is decided by exact integer-ratio comparison:

```text
digest_int * probability_denominator < probability_numerator * 2**256
```

No digest-to-float conversion, epsilon, mutable random state, catalog order, selection iteration order, process state, or wall-clock state affects an outcome. Probability is intentionally excluded from the station-trial payload so paired designs share station outcomes for equal station IDs and seeds. Probability remains bound through policy and realization identities.

## Population binding and ordering

G1 retains its established class-concatenated station order. G5 reconstructs the authoritative G1 selection and binds by cardinality plus set equality. G5 stores selected, failed, operational, serviced, and unserviced station tuples in globally ascending station-ID order.

A realization contains the complete selected population, exact failed and operational partitions, exact counts, G1 ground-design hash, failure-policy hash, seed, model and sampling versions, and a scientific realization hash. The run-bound realization record adds a nonnegative run ID without contaminating the scientific identity.

## Failure-adjusted timestep science

A failure-adjusted step copies these values exactly from verified G4 evidence:

- Configured and operational satellite counts.
- Satellite component count.
- Satellite GCC size and IDs.
- Original-denominator and surviving-denominator space GCC fractions.
- Space-threshold state.
- Satellite, ground-design, visibility, integrated-graph, G4 policy, and baseline G4 step identities.

It derives:

```text
adjusted serviced = verified G4 serviced - failed
adjusted unserviced = all selected - adjusted serviced
adjusted ground fraction = adjusted serviced count / all selected count
adjusted overall fraction = min(G4 original-denominator space fraction, adjusted ground fraction)
```

Class counts and fractions use the authoritative G1 selection and catalog. An absent class has `None`; a nonempty fully failed class has `0.0`. Ground and overall losses are exact nonnegative differences from G4 baselines. Generated zero-valued evidence uses positive zero.

Ground-threshold compliance uses the unchanged inclusive G4 ground threshold. Overall compliance is unchanged space compliance conjoined with adjusted ground compliance. Ground failures cannot improve overall service, class service, ground service, or threshold compliance.

## Temporal aggregation

G5 run aggregation receives the actual verified G4 step records and the actual verified G4 run record. It first verifies that the G4 run record exactly summarizes the supplied G4 steps. It then requires exact G4/G5 run-and-timestep key equality and exact baseline G4 step-hash binding.

The G5 summary binds the G4 run-summary hash and G4 step-sequence hash. It uses contiguous increasing timesteps, strictly increasing timestamps, constant run-level identities and populations, minimum and ordered `math.fsum` mean fractions, maximum and ordered `math.fsum` mean losses, class minimum and mean fractions, breach flags, first breach timesteps, and breach counts.

G4-derived aggregate fields are checked against the authoritative G4 summary rather than inferred independently.

## Identity model

G5 separates the following identities:

- Failure-policy scientific identity.
- Failure-realization scientific identity.
- Failure-realization record identity.
- Failure-adjusted step scientific identity.
- Failure-adjusted step record identity.
- Failure-adjusted step-sequence scientific identity.
- Failure-adjusted run-summary scientific identity.
- Failure-adjusted run-record identity.

Scientific identities exclude run ID. Record identities include exact nonnegative run IDs. Canonical JSON, canonical UTC timestamps, canonical float strings, explicit nulls, ascending numeric satellite IDs, and ascending station IDs define the byte-level identity surface.

## Validation architecture

Intrinsic constructors validate only fields stored by the object. They enforce exact types, Boolean-versus-integer distinctions, supported versions, hash formats, canonical tuple order, uniqueness, partition and count algebra, fraction algebra, loss algebra, threshold conjunction, positive-zero evidence, and canonical hash recomputation.

Contextual validators receive authoritative external evidence. They enforce G1 population identity, catalog class authority, design and policy binding, deterministic realization resampling, G4 serviced-set subtraction, exact satellite evidence copying, G4 threshold reuse, run-ID equality, G4/G5 key equality, baseline step-hash equality, sequence binding, G4 summary binding, and exact temporal aggregation.

Production factories perform intrinsic construction followed by contextual validation. Persistence parsing establishes intrinsic validity only. Canonical production evidence requires successful contextual replay.

## Persistence and replay

G5 persists three standalone canonical JSONL artifacts:

- Ground-failure realization records.
- Failure-adjusted service step records.
- Failure-adjusted service run records.

Writers use UTF-8 without BOM, Unix newlines, compact sorted-key JSON, canonical timestamps, canonical float strings, explicit nulls, deterministic collection ordering, temporary sibling files, `fsync`, atomic replacement, and no overwrite by default. Readers reject duplicate JSON keys, missing or unknown fields, malformed values, noncanonical floats, duplicate run IDs, and duplicate run-and-timestep keys.

Generation mode accepts an authoritative policy and separate seed. Replay mode accepts an authoritative policy and persisted realization. Supplying both or neither fails. Replay never reconstructs probability from a hash or from observed failure counts; it resamples through the production station-trial implementation.

Canonical G5 generation and replay first invoke the existing authoritative G4 replay boundary. G5 does not reconstruct G2 or G3 independently. Replay then regenerates the realization, every adjusted step, and the adjusted run record and requires exact equality with persisted evidence.

## Module boundaries

The implementation adds:

- `failure_policy.py` for policy identity and seed validation.
- `failure_realization.py` for exact station trials, realizations, records, and contextual resampling.
- `failure_service_metrics.py` for adjusted timestep metrics and records.
- `failure_service_aggregation.py` for G4-bound temporal summaries and run records.
- `failure_service_persistence.py` for strict JSONL, verified generation, and replay.
- `validate_ground_failure_model.py` for deterministic zero, full, and partial failure diagnostics.

All authoritative G1, G2, G3, and G4 implementation modules remain unchanged. The ground package initializer changes exports only. Satellite network, rollout, TGNN reconstruction, and graph-cache modules remain unchanged. Static tests enforce the import and protected-module boundaries.

## Verification

The implementation is covered by 51 focused G5 tests spanning policy validation, golden station-trial vectors, endpoint probabilities, ordering, persistent realization partitions, class semantics, original denominators, satellite invariance, threshold transitions, negative-zero rejection, G4-bound aggregation, JSONL corruption, replay corruption, diagnostics, and upstream isolation.

The complete suite passes 784 tests on Python 3.11.9. Compilation, deterministic diagnostics, whitespace checks, protected satellite diffs, and protected G1-to-G4 diffs also pass.

## Consequences

G5 completes the major ground-model architecture with reproducible persistent station failures and auditable temporal service degradation. It preserves upstream scientific evidence and exposes failure effects without introducing a new graph topology or a second G4 reconstruction path.

The next phase is a separate 25–50 run integrated G1-to-G5 pilot. That pilot is intentionally not part of this decision or implementation.
