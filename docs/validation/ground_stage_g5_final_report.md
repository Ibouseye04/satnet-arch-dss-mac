# Ground Segment Stage G5 Final Validation Report

## Validation status

**Result:** Passed

**Validation date:** 2026-07-16

**Validated implementation SHA:** `b350be7303d202038b775227c78d354f56054d16`

**Documentation HEAD:** The docs-only commit containing this report and the G5 architecture decision. It is intentionally later than, and is not, the validated implementation SHA.

**Validated upstream G4 implementation SHA:** `bcb56ff64730a31fc5b75591c44fc93abe1c7812`

**G5 branch base and G4 documentation HEAD:** `5da9dd818fc452a0938cfe97144ee38bbf0bf893`

**Python:** 3.11.9

## Scope validated

The validation covers the Stage G5 deterministic persistent ground-station failure model and its integration over verified G4 service evidence. It includes:

- Versioned failure policy identity.
- Exact station-local SHA-256 Bernoulli trials.
- Run-persistent failed and operational station sets.
- Scientific and run-bound realization identities.
- Original-denominator failure-adjusted timestep service.
- Exact satellite-state and space-threshold invariance.
- Class-aware adjusted service reporting.
- G4-bound temporal aggregation.
- Scientific, record, and sequence identities.
- Strict realization, step, and run JSONL persistence.
- Verified generation and replay through the authoritative G4 replay boundary.
- Deterministic diagnostics.
- Static import and protected-module isolation.

The validation does not include the separate 25–50 run integrated G1-to-G5 pilot. It does not validate time-varying failures, repairs, correlated failures, class-specific probabilities, ground-edge failures, satellite failures, traffic, routing, scheduling, capacity, datasets, labels, caches, machine-learning features, training, or Monte Carlo orchestration.

## Repository gates

The implementation began from branch `feature/ground-failure-model` at exact base SHA `5da9dd818fc452a0938cfe97144ee38bbf0bf893`. The required G4 implementation SHA `bcb56ff64730a31fc5b75591c44fc93abe1c7812` is an ancestor of the base.

Initial workspace inspection showed only these pre-existing untracked items, which were left untouched:

- `data/`
- `docs/refactor_plans/2026-07-15_tier1_validity_remediation_atomic_gameplan.md`
- `docs/validation/tier1_defect_verification.md`

The initial complete baseline passed 733 tests.

## Contract gate

The approved contract was recorded before production implementation in the atomic commit:

```text
836561a docs: lock persistent ground failure contract
```

The contract locks persistent failures, a shared class-neutral probability, exact seed validation, canonical station-local SHA-256 trials, integer-ratio probability comparison, original denominators, global station-ID ordering, G1 set binding, G4-bound aggregation, scientific-versus-record identities, strict JSONL behavior, verified replay, protected boundaries, and the stop condition.

## Atomic implementation history

The validated implementation SHA contains this approved sequence:

```text
836561a docs: lock persistent ground failure contract
17399d1 feat: add deterministic ground failure realizations
caf6eca feat: add failure adjusted service metrics
123dd3c feat: add failure adjusted run aggregation
2ab3175 feat: persist and replay ground failure evidence
b350be7 test: validate ground failures and upstream isolation
```

The documentation commit containing this report is deliberately excluded from the validated implementation SHA.

## Focused test evidence

The focused G5 command passed 51 tests:

```text
python -m pytest \
  tests/ground/test_g5_failure_realization.py \
  tests/ground/test_g5_failure_service_metrics.py \
  tests/ground/test_g5_failure_service_aggregation.py \
  tests/ground/test_g5_failure_service_persistence.py \
  tests/ground/test_g5_diagnostics_and_isolation.py -q
```

Result:

```text
51 passed in 1.32s
```

The focused tests cover:

- Probability type, range, finiteness, identity, and negative-zero normalization.
- Seed type, Boolean exclusion, range, and identity behavior.
- Locked canonical JSON, UTF-8 bytes, SHA-256 digest, binary64 ratio, and endpoint/interior golden outcomes.
- Probability-zero and probability-one station populations.
- G1 class-concatenated ordering versus G5 global ordering.
- Failure and operational set partitions and exact counts.
- Deterministic resampling against authoritative G1 design, catalog, policy, and seed.
- Run-bound realization record identities.
- No-failure, full-failure, partial-failure, and already-unserviced failure cases.
- Original selected-population denominators.
- Absent-class `None` and nonempty fully failed class `0.0` semantics.
- Exact satellite evidence and space-threshold invariance.
- Ground and overall threshold degradation and zero-threshold semantics.
- Positive-zero generation and negative-zero rejection.
- G4 run-summary verification, exact G4/G5 key equality, baseline step-hash binding, sequence identity, and aggregate equality.
- Missing, extra, duplicate, reordered, and mixed-run records.
- Strict JSONL schema, duplicate-key, corruption, newline, ordering, and no-overwrite behavior.
- Replay-mode exclusivity, authoritative policy input, production resampling, and exact evidence equality.
- Deterministic zero, full, and partial diagnostics.
- Pure-module import restrictions and protected upstream isolation.

## Complete suite evidence

The final serial complete-suite command passed:

```text
python -m pytest
```

Result:

```text
784 passed in 9.01s
```

Two earlier post-commit full-suite attempts encountered the unchanged Windows timing-sensitive `TestExperimentLogger.test_timer`: one measured zero elapsed time after a 10 ms sleep. The test passed in isolation at approximately 15 ms, and the final complete serial suite passed all 784 tests. No unrelated utility or test code was changed.

## Compilation evidence

All new production modules, the export surface, and the diagnostic script compiled successfully:

```text
python -m py_compile \
  src/satnet/ground/failure_policy.py \
  src/satnet/ground/failure_realization.py \
  src/satnet/ground/failure_service_metrics.py \
  src/satnet/ground/failure_service_aggregation.py \
  src/satnet/ground/failure_service_persistence.py \
  src/satnet/ground/__init__.py \
  scripts/validation/validate_ground_failure_model.py
```

## Diagnostic evidence

The deterministic diagnostic completed successfully:

```text
PYTHONPATH=src python scripts/validation/validate_ground_failure_model.py
```

It validated:

- Zero failures at probability zero.
- Full failure at probability one.
- Deterministic partial failure at an interior probability.
- Equal station outcomes for equal station IDs and seeds.
- Failed and operational station partitions.
- Baseline versus adjusted serviced sets.
- Ground and overall losses.
- Class-specific failure counts.
- Space, ground, and overall threshold states.
- Run minima and means.
- Loss maxima and means.
- First breach timesteps and breach counts.
- Realization, step-sequence, and run-summary hashes.

## Persistence evidence

The three G5 artifact families round-trip exactly:

- Ground-failure realization JSONL.
- Failure-adjusted step JSONL.
- Failure-adjusted run JSONL.

Persistence validation confirms UTF-8, no BOM, Unix newlines, compact sorted-key JSON, canonical UTC timestamps, canonical float strings, explicit nulls, deterministic ordering, strict schemas, duplicate-key rejection, duplicate run/key rejection, temporary sibling writes, `fsync`, atomic replacement, and no overwrite by default.

Parsing establishes intrinsic validity. Canonical production status is established only by contextual replay through authoritative G1 and G4 evidence.

## Replay evidence

Generation mode requires exactly one separate failure seed and rejects a persisted realization. Replay mode requires exactly one persisted realization and rejects a separate seed. Both or neither fail.

Verified generation and replay first invoke the existing G4 replay boundary. G5 does not independently reconstruct G2 or G3. Replay then:

1. Validates the supplied failure policy.
2. Intrinsically validates the persisted realization.
3. Reconstructs the authoritative G1 selection.
4. Resamples every station with the persisted seed and production trial implementation.
5. Requires exact realization equality.
6. Regenerates every adjusted G5 step from verified G4 steps.
7. Requires exact key completeness and record equality.
8. Regenerates the G4-bound G5 run summary and record.
9. Requires exact run-record equality.

Wrong policy, wrong realization, wrong step, wrong run, missing records, extra records, duplicate keys, and reordered steps are rejected.

## Protected-boundary evidence

The diff from base SHA `5da9dd818fc452a0938cfe97144ee38bbf0bf893` to validated implementation SHA `b350be7303d202038b775227c78d354f56054d16` is empty for:

- Satellite network modules.
- `tier1_rollout.py`.
- TGNN reconstruction.
- Graph cache.
- Every authoritative G1, G2, G3, and G4 implementation module, including the complete G4 policy, metrics, aggregation, and persistence surface.

The only existing ground package file changed is `src/satnet/ground/__init__.py`, and its diff contains imports and `__all__` exports for new G5 symbols only.

Static AST tests also verify that pure G5 modules do not directly import satellite simulation, network, model, metric-label, or graph-cache surfaces; G5 orchestration uses the authoritative G4 persistence/replay boundary; and protected satellite modules do not import G5.

## Scientific conclusions

The validated implementation satisfies the approved G5 semantics:

- Failures are sampled once and persist for the entire run.
- The probability is shared across station classes.
- Station outcomes are deterministic functions of station ID and seed.
- Probability remains bound through policy and realization identities.
- Failed stations remain in original selected-population denominators.
- G5 service is exact set subtraction over verified G4 serviced IDs.
- Satellite metrics and space-threshold state remain unchanged.
- Ground failures cannot improve service or threshold compliance.
- Class authority comes from verified G1 and catalog evidence.
- Run summaries bind actual G4 step and run evidence.
- Scientific and record identities remain distinct.
- Persistence is standalone and upstream artifacts remain unchanged.

## Handoff

Stage G5 is complete at validated implementation SHA `b350be7303d202038b775227c78d354f56054d16`.

The current documentation HEAD is the later docs-only commit containing this report and `ground_stage_g5_architecture_decision.md`. It must not be substituted for the validated implementation SHA in scientific records.

The next phase is a separate 25–50 run integrated G1-to-G5 pilot. That pilot should consume the validated G5 API and persist the configuration hash and seeds for every run. It remains outside this report and outside Stage G5.
