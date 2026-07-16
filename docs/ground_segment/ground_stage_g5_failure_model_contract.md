# Ground Segment Stage G5 Failure Model Contract

**Status:** Approved

**Decision date:** 2026-07-16

**Validated G4 implementation SHA:** `bcb56ff64730a31fc5b75591c44fc93abe1c7812`

**G4 documentation HEAD and G5 branch base:** `5da9dd818fc452a0938cfe97144ee38bbf0bf893`

## Scope

Stage G5 applies deterministic persistent ground-station failures as a separate operational eligibility overlay on verified G4 service evidence. It adds one class-neutral station-failure probability, one deterministic seed per run realization, persistent failed and operational station sets, failure-adjusted timestep service, temporal run aggregation, standalone identities, strict JSONL persistence, and verified G1-to-G5 replay.

G5 does not modify G1, G2, G3, or G4 scientific artifacts. It adds no time-varying failures, repair, correlated failures, class-specific probabilities, geographic probabilities, ground-edge failures, satellite failures, communications failures, traffic, routing, scheduling, capacity, datasets, labels, caches, RF features, TGNN features, training, Monte Carlo regeneration, or integrated-pilot execution.

## Persistent failure semantics

Ground-station failures are sampled once per run realization and remain persistent for the complete run. Every selected station receives exactly one deterministic station-local trial. Failed stations are ineligible for service at every timestep; operational stations remain eligible at every timestep. No repair, recovery, or repeated timestep sampling occurs.

One shared failure probability applies to civilian, government, and military stations. Station class affects reporting only.

## Enabled nonempty designs

Canonical G5 execution requires an enabled G1 ground design with at least one selected station. Disabled or empty designs fail before sampling, record creation, service calculation, aggregation, persistence generation, or replay.

## Station ordering and population binding

G1 retains its class-concatenated deterministic selection order. G5 uses globally ascending station-ID order for selected, failed, operational, failure-adjusted serviced, and failure-adjusted unserviced tuples.

G5 binds to G1 by equal cardinality and set equality. It never requires tuple equality between G5 globally sorted selected IDs and G1 class-concatenated selected IDs.

## Failure policy

The failure model, policy, and sampling versions are `1`. `GroundFailurePolicy` accepts exact integer or float probability inputs, rejects Booleans, strings, NaN, infinities, and values outside `[0.0, 1.0]`, converts accepted inputs to finite binary64 floats, and normalizes negative zero to positive zero.

Policy identity uses domain `satnet_ground_failure_policy`, version `1`, and includes the model, policy, and sampling versions and the canonical binary64 probability. It excludes run ID, seed, ground design, station IDs, paths, and timestamps.

## Seed contract

The ground-failure seed must have exact type `int`, excluding Boolean, and must be within `[0, 2**63 - 1]`. It is constant for the complete run realization.

## Station-local deterministic trial

For each selected station, the exact trial payload is:

```python
trial_payload = {
    "identity_domain": "satnet_ground_failure_trial",
    "identity_version": "1",
    "ground_failure_model_version": GROUND_FAILURE_MODEL_VERSION,
    "ground_failure_sampling_version": GROUND_FAILURE_SAMPLING_VERSION,
    "ground_failure_seed": ground_failure_seed,
    "station_id": station_id,
}
```

The payload bytes are exactly:

```python
canonical_payload_bytes = canonical_json(trial_payload).encode("utf-8")
```

The trial digest is exactly:

```python
digest = hashlib.sha256(canonical_payload_bytes).digest()
```

No local JSON options, `repr`, delimiter encoding, global random state, NumPy random state, process state, wall-clock state, catalog order, or selection iteration order may affect sampling.

The station trial excludes probability, catalog hash, ground-design hash, satellite configuration, and station class. Therefore, the same station ID and failure seed produce the same digest across paired designs. Probability remains part of policy identity and realization identity.

## Exact probability comparison

Let `digest_int` be the unsigned big-endian integer represented by the 256-bit digest. Let the normalized binary64 probability provide `probability_numerator` and `probability_denominator` through `as_integer_ratio()`.

A station fails exactly when:

```python
digest_int * probability_denominator < probability_numerator * (1 << 256)
```

No digest-to-float conversion and no epsilon are permitted. Probability zero fails no stations; probability one fails every selected station.

Locked golden vectors must include endpoint probabilities and interior binary64 probabilities producing both failed and nonfailed outcomes. Each vector binds the station ID, seed, canonical JSON, UTF-8 bytes, digest, ratio, comparison, and Boolean outcome.

## Failure realization

A realization binds model and sampling versions, ground-design hash, policy hash, seed, globally sorted selected, failed, and operational station IDs, exact counts, and its scientific hash. Failed and operational IDs are unique disjoint subsets whose union equals selected IDs. Counts must match the tuples exactly.

Realization identity uses domain `satnet_ground_failure_realization`, version `1`. It excludes run ID. The run-bound realization record uses schema version `1`, domain `satnet_ground_failure_realization_record`, version `1`, and binds run ID to the scientific realization hash.

Production sampling reconstructs the exact G1 selection from the authoritative catalog, verifies catalog and design identity, evaluates one station-local trial per selected station, and returns exact sorted complements.

## Original denominator

Failed selected stations remain in the original ground-service denominator and therefore count against resilience. The overall and class-specific denominators remain the complete selected G1 populations, never the operational, visible, connected, or serviced subsets.

## Failure-adjusted service

G5 does not remove failed stations or their visibility edges from G3. It applies an operational eligibility overlay to verified G4 service evidence.

At each timestep:

```text
failure-adjusted serviced IDs = verified G4 serviced IDs - failed IDs
failure-adjusted unserviced IDs = all selected IDs - adjusted serviced IDs
```

Class authority comes only from verified G1 selection and catalog evidence. An absent class has fraction `None`; a nonempty fully failed class has fraction `0.0`.

The adjusted ground fraction is adjusted serviced count divided by all selected stations. The adjusted overall fraction is the minimum of verified G4 original-denominator space GCC fraction and adjusted ground fraction.

Ground and overall losses are the exact nonnegative differences from their verified G4 baseline fractions. Ground failures cannot improve serviced sets, class service, ground fraction, overall fraction, or threshold compliance.

## Satellite and threshold invariance

Satellite GCC membership, satellite resilience fractions, configured and operational satellite counts, satellite component count, GCC size, GCC IDs, and space-threshold state remain exactly equal to verified G4 evidence.

G5 reuses the authoritative G4 `GroundServicePolicy`. Ground threshold compliance is the exact inclusive comparison of adjusted ground fraction with the G4 ground threshold. Overall compliance is the conjunction of unchanged space compliance and adjusted ground compliance. Zero thresholds retain G4 policy-compliance semantics.

## Validation layers

Intrinsic validation uses only fields contained in the object. It enforces exact types, Boolean-versus-integer distinctions, finite positive-zero floats, hash formats, versions, canonical tuple ordering, uniqueness, count and partition algebra, fraction and loss algebra, Boolean conjunction, and scientific or record hash recomputation.

Contextual validation uses authoritative external evidence. It enforces enabled nonempty G1 selection, selected-population set equality, class authority, realization-to-design and realization-to-policy binding, deterministic resampling, G4 serviced-set subtraction, exact copied satellite evidence, G4 threshold comparisons, monotonicity, cross-stage run identity, step-key equality, sequence binding, G4 summary binding, and aggregate equality.

Production factories and replay invoke intrinsic validation followed by contextual validation. Persistence parsing establishes intrinsic validity only. Parsed records are not canonical production evidence until contextual validation succeeds.

Policy input negative zero is normalized to positive zero. Generated scientific floats always use positive zero. Directly constructed or persisted negative-zero scientific evidence is rejected. Hashing and persistence use `canonical_float_string()`.

## Scientific and record identities

G5 distinguishes:

```text
failure policy identity
failure realization scientific identity
failure realization record identity
failure-adjusted step scientific identity
failure-adjusted step record identity
failure-adjusted step-sequence identity
failure-adjusted run-summary scientific identity
failure-adjusted run-record identity
```

Scientific identities exclude run ID. Record identities bind exact nonnegative run IDs. Hash payloads use canonical JSON, canonical UTC timestamps, canonical float strings, explicit `None`, ascending numeric satellite IDs, and ascending station IDs.

## Single-run execution scope

Scientific generation and replay operate on exactly one run per invocation, matching G4. All G1, G2, G3, G4, realization, G5 step, and G5 run record IDs must equal the G1 run ID. G5 step keys must exactly equal verified G4 step keys.

JSONL readers and writers may hold multiple runs and enforce canonical ordering, unique step keys, and unique realization and run IDs. Multi-run scientific orchestration is deferred to the integrated pilot.

## Generation and replay modes

Generation mode requires an authoritative `GroundFailurePolicy` and a separate seed and rejects a persisted realization input. Replay mode requires an authoritative policy and persisted realization and rejects a separate seed input. Both or neither realization inputs fail.

Replay validates the supplied policy hash against persisted evidence and resamples using the supplied policy, persisted seed, and production station-trial code. Probability is never reconstructed from a hash or inferred from counts.

## Authoritative upstream replay boundary

Canonical production G5 begins by invoking the existing verified G4 replay boundary. G5 does not independently reconstruct G2 or G3 evidence. The required flow is verified G1-to-G4 replay, followed by the G5 persistent failure overlay.

The G5 run summary receives actual verified G4 step and run records. It verifies that the G4 run record exactly summarizes the supplied G4 steps, requires exact G4/G5 key equality and baseline-step-hash equality, and binds the exact G4 run-summary and G4 step-sequence hashes.

## Persistence

G5 writes standalone canonical JSONL artifacts for failure realizations, failure-adjusted steps, and failure-adjusted runs. Upstream artifacts remain unchanged.

Files use UTF-8 without BOM, compact sorted-key JSON, canonical UTC timestamps, canonical float strings, explicit nulls, Unix newlines, duplicate-key rejection, exact schemas, atomic synchronized sibling replacement, and no overwrite by default. Realizations and runs are ordered by run ID; steps are ordered by run ID then timestep.

## Protected boundaries

All authoritative G1, G2, G3, and G4 implementation modules are protected from G5 changes, including the complete G4 policy, metrics, aggregation, and persistence surface. A package-initializer diff may only export new G5 symbols. Satellite network, rollout, TGNN reconstruction, and graph-cache modules remain unchanged.

Pure G5 modules may depend on Python standard-library modules, shared canonical utilities, authoritative G1 catalog and design types, authoritative G4 policy and metric types, and other pure G5 modules. G5 orchestration may depend on authoritative G1 and G4 persistence/replay. Pure G5 modules must not depend on G2/G3 reconstruction, satellite simulation, datasets, caches, models, training, or experiment runners.

## Stop condition

G5 stops after deterministic persistent failure realizations, failure-adjusted step metrics, G4-bound run aggregation, identities, strict JSONL persistence, verified single-run replay, diagnostics, tests, documentation, and validation reporting are complete.

G5 completes the major ground-model architecture. The next phase is a separate 25–50 run integrated G1-to-G5 pilot. That pilot is not part of this stage.
