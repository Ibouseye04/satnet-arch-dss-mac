# Integrated Ground Pilot 25-Run Contract

## Status

Locked before full-pilot execution on 2026-07-16.

This is a controlled engineering experiment over the validated G1-to-G5 architecture. It is not a new architecture stage and is not a final scientific dataset.

## Validated foundations

The pilot branch is `experiment/integrated-ground-pilot` and its required base is `62beda9df1576958d9e33d33d2d9eb5489e24b20`.

Validated scientific implementation SHAs are:

| Stage | Implementation SHA |
|---|---|
| G1 | `753b28be767b7bcee213e239bbfd086b26b1944b` |
| G2 | `5cc9e71399cc6587162634958e0173985c4ee228` |
| G3 | `8f63531c9bedac97c1c28d76e44f0a8b24ca909a` |
| G4 | `bcb56ff64730a31fc5b75591c44fc93abe1c7812` |
| G5 | `b350be7303d202038b775227c78d354f56054d16` |

The complete baseline at the pilot base is 784 passing tests on Python 3.11.9.

## Objective

The pilot contains exactly five deterministic design specifications and five stochastic realizations per design, producing exactly 25 expected runs. It evaluates generation, exact replay, target variance, space-versus-ground information, persistent satellite and ground failure outcomes, temporal bottlenecks, class service, runtime, artifact volume, and readiness for the next experiment phase.

The pilot stops after 25-run generation, 25-run replay, deterministic repeat checks, analysis, validation, protected diffs, and the final report. It does not expand automatically to 50 runs.

## Fixed temporal and policy profile

All 25 runs use:

| Parameter | Locked value |
|---|---:|
| Duration | 10 minutes |
| Step | 60 seconds |
| Inclusive timestep count | 11 |
| Minimum elevation | 10 degrees |
| Space GCC threshold | 0.80 |
| Ground service threshold | 0.80 |
| Pilot master seed | 20260716 |
| Orbital engine | SGP4 |
| Satellite edge failure model | `persistent_temporal_union_edges_v1` |

Visibility-policy, service-policy, and ground-failure-policy hashes are persisted in the design and run manifests. Thresholds are not tuned after observing outcomes.

## Synthetic pilot catalog

The pilot uses a deterministic pilot-only catalog with exactly:

- 50 enabled civilian stations.
- 50 enabled government stations.
- 50 enabled military stations.
- 150 enabled stations total.

Every station uses country code `ZZ`, an explicitly synthetic name, a valid G1 station ID, valid WGS84 latitude and longitude, deterministic altitude, and one of multiple deterministic synthetic regions. It contains no real facility names or sensitive locations.

The catalog CSV is scientific content and is preserved under its G1 catalog hash. Provenance is stored separately and is excluded from that scientific hash.

The formal research-catalog status remains unchanged:

```text
Production research catalog availability: NOT AVAILABLE
Production research catalog scientific review: NOT PERFORMED
```

## Design matrix

The pilot uses these values without post-outcome tuning:

| ID | Name | Planes | Sats/plane | Altitude km | Inclination deg | Node failure | Edge failure | Civilian | Government | Military | Ground failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P01 | Large robust reference | 6 | 8 | 1200 | 98 | 0.00 | 0.00 | 8 | 6 | 6 | 0.00 |
| P02 | Large constellation, light failures | 6 | 8 | 800 | 60 | 0.05 | 0.05 | 12 | 4 | 4 | 0.05 |
| P03 | Balanced middle case | 5 | 6 | 600 | 55 | 0.10 | 0.10 | 4 | 3 | 3 | 0.15 |
| P04 | Sparse constellation, elevated failures | 4 | 5 | 500 | 45 | 0.15 | 0.20 | 2 | 4 | 4 | 0.25 |
| P05 | Stress case | 4 | 5 | 300 | 30 | 0.20 | 0.25 | 1 | 1 | 1 | 0.40 |

All values map directly to existing `Tier1RolloutConfig`, `GroundSegmentEnabledConfig`, `GroundVisibilityPolicy`, `GroundServicePolicy`, and `GroundFailurePolicy` fields. No scientific adjustment is required by current production constraints.

A design hash binds all design-level satellite parameters, temporal parameters, failure probabilities, class counts, policy hashes, fixed epoch, orbital engine, ISL policy, and failure-model identity. It excludes run ID and realization seeds.

## Realization matrix and run IDs

Each design has realizations `R01` through `R05`. Run IDs use design-major order:

```text
P01-R01 -> 0
P01-R02 -> 1
...
P05-R05 -> 24
```

The design group ID is the design ID. Future ML grouping must keep all five realizations of one design in one split. This pilot creates no ML splits.

The five realizations in one design share every design-level parameter and differ only by locked stochastic seeds.

## Seed derivation

The production satellite rollout exposes one stochastic seed, `Tier1RolloutConfig.seed`, which controls both persistent satellite-node and persistent accepted-edge failure sampling. Therefore the repository-equivalent satellite purpose is locked as `satellite_rollout_and_failure` rather than inventing a second unsupported production field.

Three independent seed purposes are derived for each design-realization pair:

```text
satellite_rollout_and_failure
ground_station_selection
ground_failure_realization
```

The canonical seed payload is:

```json
{
  "design_id": "P01",
  "identity_domain": "satnet_integrated_ground_pilot_seed",
  "identity_version": "1",
  "master_seed": 20260716,
  "realization_id": "R01",
  "seed_purpose": "satellite_rollout_and_failure"
}
```

Canonical JSON uses UTF-8, sorted keys, no insignificant whitespace, and no ASCII escaping requirement beyond the repository canonicalizer. SHA-256 is interpreted as one unsigned big-endian integer. The derived seed is:

```text
digest_integer modulo 2**63
```

Thus every seed lies in `[0, 2**63 - 1]`, is independent of iteration order and wall-clock state, and is reproducible from the manifest. Golden derived seeds are locked by focused tests.

## Required execution path

Each run coordinates existing production APIs only:

```text
Tier 1 rollout with persistent satellite failures
-> G1 deterministic selection and ground-design record
-> G2 operational positions, visibility snapshots, records, and JSONL
-> G3 operational graphs, integrated snapshots, records, and JSONL
-> G4 verified service persistence
-> G5 verified failure-adjusted persistence
```

The runner does not recreate orbital, failure, visibility, graph, or service equations.

The satellite configuration and `Tier1FailureRealization` have no standalone production JSONL writer. The pilot may therefore use a strict pilot container that serializes the exact `Tier1RolloutConfig` dataclass fields, exact sorted failed-node IDs, exact sorted failed-edge endpoint pairs, production config hash, rollout summary, and a pilot artifact hash. This container adds no scientific calculation and is validated before replay.

## Canonical run artifacts

Every successful run retains:

- Pilot satellite configuration and failure artifact.
- G1 ground-design JSONL.
- G2 visibility JSONL.
- G3 integrated-graph JSONL.
- G4 service-step JSONL.
- G4 service-run JSONL.
- G5 failure-realization JSONL.
- G5 adjusted-step JSONL.
- G5 adjusted-run JSONL.
- Stage runtime JSON.
- Artifact inventory JSON.
- Supplemental validated run-summary JSON.
- Deterministic run log.

The supplemental row never replaces canonical stage evidence.

## Output policy

The output root is `artifacts/integrated_ground_pilot_25`. Large canonical run artifacts under its `runs` and repeat directories remain local and are not committed. The runner, locked manifests, small aggregate summaries, hash inventory, feature inventory, replay summary, and final report may be committed.

No overwrite is permitted by default. Resume validates the complete existing artifact set through authoritative production replay and exact summary validation. File existence alone never marks a run complete. Invalid or partial existing evidence fails without replacement unless the user explicitly selects a new output root.

## Replay contract

Every expected run must replay through authoritative boundaries. Replay reconstructs the exact satellite configuration and satellite failure realization, validates the G1 design against the catalog and satellite hash, replays G2 and G3, replays G4, and replays G5 including deterministic ground-failure resampling.

Required final counts are:

```text
25 expected
25 attempted
25 generated
25 replayed
0 missing artifacts
0 extra artifacts
0 identity mismatches
0 source-key mismatches
0 step mismatches
0 graph mismatches
0 service mismatches
0 failure-realization mismatches
0 run-summary mismatches
```

Any replay failure prevents readiness for final dataset design.

## Summary contract

One deterministic row per run binds identity, satellite design, ground design, satellite outcomes, G4 baseline outcomes, G5 adjusted outcomes, class outcomes, generation status, replay status, monotonic runtime, and artifact bytes.

All scientific values are copied from authoritative G1, G4, and G5 records. Pilot-only derived analyses are explicitly identified. Rows reject nonfinite floats, negative counts, fractions outside `[0, 1]`, missing identities, run mismatches, duplicate timesteps, and incomplete timestep ranges.

## Diagnostic contract

Candidate classification labels are evaluated without selecting a final label:

- Space threshold breach anywhere.
- Ground threshold breach anywhere.
- Overall threshold breach anywhere.
- Their whole-run met inverses.

Candidate regression targets are evaluated without selecting a final target:

- Original-denominator space GCC minimum.
- Failure-adjusted ground-service minimum.
- Failure-adjusted overall-service minimum.
- Failure-adjusted overall-service mean.
- Maximum ground-service loss due to failures.

Regression means use `math.fsum`. Diagnostics report finite counts, range, standard deviation, unique values, quantiles, and values by design.

Every timestep is classified as space, ground, or tie by comparing original-denominator space GCC fraction with failure-adjusted ground-service fraction. Run and design summaries report counts and dominant bottlenecks.

Class diagnostics remain descriptive because class counts differ by design. They do not claim intrinsic class resilience.

## Runtime and artifact contract

Monotonic elapsed time is recorded for satellite rollout, G1, G2, G3, G4, G5, persistence, replay, total generation, and total run execution. Analysis reports minimum, median, ordered `math.fsum` mean, maximum, and total.

Artifact bytes are measured for each artifact, stage, run, design, and the complete pilot. A 500-run estimate is a labeled linear extrapolation from these pilot artifacts and excludes undefined future ML datasets.

## Determinism repeat subset

Runs 0, 12, and 24 are regenerated into a separate repeat root. The repeat must match original scientific hashes, record hashes, summaries, failed station IDs, satellite failures, visibility records, integrated graphs, G4 records, and G5 records. Runtime values and filesystem paths are excluded from scientific equality.

## Failure handling

The expected manifest remains fixed at 25 runs. A failed run is logged with its design, realization, stage, exception type, and message. Other runs continue where safe. A failed run is never silently omitted or replaced with another seed.

## Acceptance gates

Pipeline gate:

- 25 of 25 runs generate.
- 25 of 25 runs replay exactly.

Integrity gate:

- Complete manifests.
- Unique run IDs and design-realization pairs.
- All stage identities present.
- All summary relationships valid.

Isolation gate:

- No authoritative G1-to-G5 implementation changes.
- No satellite physics, TGNN reconstruction, or graph-cache changes.
- Focused and complete tests pass.

Variance gate:

- Report whether each classification candidate contains both classes.
- Report whether any candidate has minority count at least 5.
- Report whether regression candidates have at least five unique values and nonzero standard deviation.

Variance failure does not invalidate the architecture. It selects a revised 50-run pilot rather than final dataset design.

Runtime and artifact gates report operational implications without inventing an undocumented compute threshold.

## Protected files

The final diff from `62beda9df1576958d9e33d33d2d9eb5489e24b20` must be empty for all authoritative modules in:

- Shared ground canonicalization.
- G1 catalog, selection, scenario, and persistence.
- G2 coordinates, position adapter, visibility, and visibility persistence.
- G3 graph attributes, integrated graph, integrated builder, satellite graph adapter, and integrated persistence.
- G4 service policy, metrics, aggregation, and persistence.
- G5 failure policy, realization, adjusted metrics, adjusted aggregation, and persistence.
- Satellite network modules.
- Tier 1 rollout.
- TGNN reconstruction.
- Graph cache.

Pilot code remains outside those modules.

## Readiness statuses

The report selects exactly one:

```text
READY FOR FINAL INTEGRATED DATASET DESIGN
READY FOR REVISED 50-RUN PILOT
NOT READY FOR DATASET DESIGN
```

The first requires complete generation and replay, integrity and isolation, adequate candidate variance, and understood runtime and volume. The second requires successful pipeline, integrity, and isolation but inconclusive variance. The third applies to generation, replay, identity, isolation, or artifact failure.

## Stop condition

Stop after exactly 25 primary runs, exact replay, three deterministic repeat runs, analysis, focused tests, complete tests, protected diffs, and the final report.

Do not modify validated science, expand automatically to 50 runs, choose final labels, create ML splits, generate 500 runs, alter RF or TGNN inputs, or train models.
