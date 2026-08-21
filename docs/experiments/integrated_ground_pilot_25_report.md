# SATNET Integrated Ground Pilot: 25-Run Final Report

## 1. Decision record

- Execution date: 2026-07-16.
- Branch: `experiment/integrated-ground-pilot`.
- Locked base SHA: `62beda9df1576958d9e33d33d2d9eb5489e24b20`.
- Contract commit: `565610dd7400a951de9d2f838ab3488ca3d20714`.
- Validated pilot-tooling implementation SHA: `dedb26aec44e9e8054d4d5ab844fe837b52d178f`.
- The documentation-only report commit is intentionally recorded in Git history and the final handoff rather than embedded here, because a commit cannot contain its own stable SHA.
- Scientific implementation boundary: unchanged from the locked base.

This pilot is a deterministic engineering validation of the integrated Tier 1 satellite and G1-G5 ground pipeline. It is not a production-scale training dataset and is not evidence that the synthetic station locations are scientifically representative of a real ground network.

## 2. Objectives and disposition

| Objective | Result |
|---|---:|
| Generate exactly 5 designs × 5 realizations | Passed: 25/25 |
| Complete Tier 1 and G1-G5 generation | Passed: 25/25 |
| Replay every generated run | Passed: 25/25 |
| Preserve every fixed seed without substitution | Passed |
| Detect failed, missing, duplicate, reordered, or corrupt artifacts | Covered by strict readers, replay, and focused tests |
| Demonstrate distinct space and ground metrics | Passed |
| Obtain candidate classification spread | Passed |
| Obtain candidate regression spread | Passed for 4/5 audited candidates |
| Validate independent deterministic repeats | Passed: runs 0, 12, and 24 |
| Measure runtime and artifact volume | Passed |
| Keep protected scientific modules unchanged | Passed: empty protected diffs |

No run was silently skipped, replaced, reseeded, or dropped.

## 3. Locked experiment identity

### 3.1 Manifest and catalog

- Design manifest hash: `514795b9cc17deacd81ffb9308e0d944f8da64b80444047647e919b375ef27bf`.
- Run manifest hash: `1b1e128561a7f25233bc89677e5c40da7c4df8945e6070e2827f12d6c42535c7`.
- Synthetic catalog hash: `810c64dfb030b042311c90f2f42f8dee866a48fc63a6a29e362ee328c52eaa6e`.
- Catalog population: 150 enabled stations, with 50 civilian, 50 government, and 50 military stations.
- Country code: reserved synthetic code `ZZ`.
- Region labels: deterministic synthetic regions `region_01` through `region_10`.

The catalog is pilot-only synthetic geometry. It contains no sensitive locations, no operational ground-station claims, and no scientifically reviewed production inventory. Production catalog availability remains `NOT AVAILABLE`; production scientific review remains `NOT PERFORMED`.

### 3.2 Fixed policy identities

- Visibility policy hash for all designs: `5dda243a39d13746d4e9d16922318df1774c4cf29a95fbc70677b618a41c45b1`.
- Ground-service policy hash for all designs: `e314b9d4123ef84832965b1baadd04a118805f8cab368bd548a3f8f65ae02950`.
- Ground-failure policy hashes:
  - P01: `d29cac302069099deb9014e0ec6a0018d039cc42a075587cc604a671c779d6c7`.
  - P02: `6c70000af244c32df77b676f0c195ec319906e5503dfd35289d18be180e6eefe`.
  - P03: `b11b7de98c5eceb3aba56ea2035316dace68194ec4ca9de31c8ba446dc89d43f`.
  - P04: `2727500757257183510527c7abe32b31fed09df39b51da82a7c237513ec6cd74`.
  - P05: `6a80f131447fc6a4057809f3dbe83d361d0d29f2b23defe61876fc483746b505`.

The fixed temporal profile was 10 minutes at 60-second steps, yielding 11 persisted and replayed timesteps per run. Minimum elevation was 10 degrees. Space and ground thresholds were both 0.80. The ISL policy was `grid_fixed` with the locked adapter parameters.

### 3.3 Seed derivation

The master seed was `20260716`. Three purpose-separated 63-bit seeds were derived for every `(design_id, realization_id)` pair through canonical SHA-256 payloads:

- `satellite_rollout_and_failure`.
- `ground_station_selection`.
- `ground_failure_realization`.

The P01-R01 golden vectors were:

- Satellite rollout and failure: `7354080756714418247`.
- Ground-station selection: `8343673171039191876`.
- Ground-failure realization: `783383988117819731`.

Every row in the final one-row-per-run summary includes the catalog hash and all three exact manifest seeds.

## 4. Execution and replay evidence

### 4.1 Generation

- Attempted runs: 25.
- Successful runs: 25.
- Failed runs: 0.
- Resumed runs: 0.
- Generated run IDs: exactly 0 through 24.
- Timesteps per run: 11.
- Total generated G2, G3, G4, and G5 step records: 275 per stage.

Each run persisted nine canonical artifact groups:

1. Complete Tier 1 satellite config, rollout steps, summary, and persistent satellite failure realization.
2. G1 ground design.
3. G2 visibility sequence.
4. G3 integrated graphs.
5. G4 step records.
6. G4 run record.
7. G5 ground-failure realization.
8. G5 step records.
9. G5 run record.

### 4.2 Authoritative replay

- Attempted replays: 25.
- Successful replays: 25.
- Failed replays: 0.
- Missing artifact failures: 0.
- Identity mismatches: 0.
- Ordering or duplicate failures: 0.
- G4/G5 baseline binding mismatches: 0.

Replay regenerated the Tier 1 satellite rollout from the exact config and seed, then invoked the authoritative G1 validation, G2 replay, G3 replay, G4 replay, and G5 replay paths. Supplemental summaries were accepted only after their source scientific records passed replay.

### 4.3 Independent repeats

Runs 0, 12, and 24 were regenerated under independent repeat paths and replayed. They represent the robust, middle, and stress regions.

| Run | Design/realization | Canonical artifacts | Scientific summary |
|---:|---|---|---|
| 0 | P01-R01 | Exact byte match | Exact value match |
| 12 | P03-R03 | Exact byte match | Exact value match |
| 24 | P05-R05 | Exact byte match | Exact value match |

All nine canonical artifact groups matched for all three repeats. Runtime and other explicitly non-scientific supplemental timing fields were excluded from scientific-value comparison.

## 5. Outcome diagnostics

### 5.1 Classification candidates

| Candidate | Positive | Negative | Minority count | Pilot gate |
|---|---:|---:|---:|---|
| Space threshold breach at any timestep | 15 | 10 | 10 | Passed |
| Ground threshold breach at any timestep | 20 | 5 | 5 | Passed |
| Overall threshold breach at any timestep | 20 | 5 | 5 | Passed |

The corresponding entire-run threshold-met labels are exact Boolean inverses and have the same minority support. These labels are calculated from graph-derived service state only; they do not inspect the configured failure parameters.

Per-design behavior was strongly ordered:

| Design | Space breach runs | Ground breach runs | Overall breach runs | Adjusted overall minimum | Mean adjusted overall service |
|---|---:|---:|---:|---:|---:|
| P01 | 0/5 | 0/5 | 0/5 | 0.85 | 0.9409 |
| P02 | 0/5 | 5/5 | 5/5 | 0.65 | 0.8155 |
| P03 | 5/5 | 5/5 | 5/5 | 0.00 | 0.0255 |
| P04 | 5/5 | 5/5 | 5/5 | 0.00 | 0.0182 |
| P05 | 5/5 | 5/5 | 5/5 | 0.00 | 0.0000 |

The class balance is adequate for moving to final dataset design, but the 25 rows are not enough for production model estimation or stable confidence intervals.

### 5.2 Regression candidates

| Candidate | Range | Unique values | Population SD | Useful pilot spread |
|---|---:|---:|---:|---|
| Minimum original-denominator space GCC fraction | 0.05-1.00 | 6 | 0.4465 | Yes |
| Minimum failure-adjusted ground-service fraction | 0.00-0.95 | 7 | 0.3995 | Yes |
| Minimum failure-adjusted overall-service fraction | 0.00-0.95 | 7 | 0.3995 | Yes |
| Mean failure-adjusted overall-service fraction | 0.00-0.9727 | 15 | 0.4263 | Yes |
| Maximum ground-service loss due to failures | 0.00-0.10 | 4 | 0.0376 | No as a primary target |

The strongest candidate in this pilot is mean failure-adjusted overall-service fraction because it has 15 unique values and retains within-design stochastic variation. Minimum space GCC and minimum adjusted service are valid but show substantial design-level clustering. Maximum ground-service loss due to failures should remain a diagnostic or secondary target unless the final design broadens its support.

## 6. Bottleneck findings

Space and ground metrics were demonstrably distinct. Across 275 timesteps:

- Ground bottleneck: 231 timesteps, 84.0%.
- Space bottleneck: 21 timesteps, 7.6%.
- Exact tie: 23 timesteps, 8.4%.

| Design | Ground | Space | Tie | Dominant bottleneck |
|---|---:|---:|---:|---|
| P01 | 42 | 0 | 13 | Ground |
| P02 | 55 | 0 | 0 | Ground |
| P03 | 34 | 21 | 0 | Ground |
| P04 | 45 | 0 | 10 | Ground |
| P05 | 55 | 0 | 0 | Ground |

P03 is the only design where the satellite layer is the limiting factor for a substantial timestep subset. Ground access or ground failure is the dominant integrated constraint elsewhere. This is an engineering result for the locked synthetic catalog and design matrix, not a universal architectural conclusion.

## 7. Ground-class diagnostics

| Class | Selected across runs | Failed across runs | Operational across runs | Mean adjusted class service | Runs below 0.80 at minimum |
|---|---:|---:|---:|---:|---:|
| Civilian | 135 | 11 | 124 | 0.3511 | 19 |
| Government | 90 | 13 | 77 | 0.3591 | 20 |
| Military | 90 | 9 | 81 | 0.3809 | 20 |

All three class minima reached zero in at least one run. The selected class counts differ by design, so these descriptive values do not establish causal or intrinsic class resilience. No claim should be made that one class is physically or operationally more resilient than another from this pilot.

## 8. Runtime and scaling

### 8.1 Measured runtime

| Stage | Mean seconds/run | Maximum seconds/run | Total seconds |
|---|---:|---:|---:|
| Satellite rollout | 0.0180 | 0.1964 | 0.4495 |
| G1 | 0.0003 | 0.0006 | 0.0084 |
| G2 | 0.1862 | 0.4240 | 4.6541 |
| G3 | 0.1130 | 0.2665 | 2.8258 |
| G4 | 0.2687 | 0.5677 | 6.7179 |
| G5 | 0.2841 | 0.5679 | 7.1021 |
| Explicit persistence | 0.0729 | 0.1600 | 1.8227 |
| Complete generation | 0.9439 | 2.1476 | 23.5968 |
| Authoritative replay | 1.2193 | 2.7294 | 30.4820 |
| Generation plus replay | 2.1632 | 4.8770 | 54.0788 |

These are workstation observations, not portable performance guarantees. Linear extrapolation gives approximately 471.94 seconds for 500-run generation and 609.64 seconds for 500-run replay, or 1,081.58 seconds total. That projection ignores parallelism, scheduling contention, cache state, hardware differences, and future dataset-export cost.

## 9. Artifact volume

Canonical run artifacts totalled 50,406,344 bytes.

| Artifact group | Bytes | Share |
|---|---:|---:|
| G2 visibility | 32,601,223 | 64.7% |
| G3 integrated graphs | 16,132,493 | 32.0% |
| G5 steps | 869,817 | 1.7% |
| G4 steps | 517,940 | 1.0% |
| Satellite rollout | 92,395 | 0.2% |
| G5 run | 88,063 | 0.2% |
| G4 run | 48,833 | 0.1% |
| G5 realization | 28,481 | 0.1% |
| G1 design | 27,099 | 0.1% |

Per-run canonical volume ranged from 278,912 bytes to 4,201,155 bytes, with a mean of 2,016,254 bytes. A direct 20× extrapolation to 500 runs is 1,008,126,880 bytes, excluding future ML tables, indexes, models, logs, and filesystem overhead. G2 and G3 account for 96.7% of canonical volume and are the primary storage-optimization targets if scale becomes limiting. Their scientific content must not be reduced without a separately locked evidence-retention decision.

## 10. Model-development implications

### 10.1 Random-forest readiness

The final dataset design may use pre-run satellite design variables, ground class counts, explicit failure probabilities, visibility policy values, and service policy values as candidate features. Hashes may be retained as provenance but must not become predictive features. Realized failure counts are post-treatment information and must be excluded when the task is pre-run design prediction.

Recommended initial targets are:

1. Overall threshold breach at any timestep.
2. Space threshold breach at any timestep.
3. Mean failure-adjusted overall-service fraction.
4. Minimum failure-adjusted overall-service fraction.
5. Minimum original-denominator space GCC fraction.

The maximum ground-service-loss target requires broader support before promotion to a primary regression target.

### 10.2 TGNN framing

The current TGNN consumes complete temporal sequences. It is therefore an ex-post run-level assessment, not a forecast. A future forecasting experiment must define an explicit prefix cutoff and prevent all suffix graph states, labels, aggregate statistics, and replay-derived information from entering the inputs. Ex-post and future-prefix results must be reported separately.

### 10.3 Leakage prohibitions

The final schema must prohibit:

- Outcome labels as inputs.
- G4 or G5 summaries when predicting those same outcomes.
- Realized failures for pre-run prediction.
- Run, sequence, scientific, record, and replay hashes as features.
- Runtime, artifact size, generation status, and replay status as features.
- Any future timestep information in a forecasting task.

## 11. Validation gates

### 11.1 Focused pilot tests

- Result: 25 passed.
- Includes deterministic manifest and seed vectors, strict identity parsing, synthetic catalog contract, satellite artifact integrity, failure handling, resume validation, repeat comparison, diagnostics, artifact accounting, one real production G1-G5 smoke run, protected-file isolation, and final summary seed/catalog bindings.

### 11.2 Complete regression suite

- Result: 809 passed in 9.90 seconds.
- Windows timer resolution was set to 1 ms for the clean sample because `TestExperimentLogger.test_timer` uses a 10 ms sleep and is known to return zero on coarse Windows clock samples.
- Unadjusted samples repeatedly produced 808 passes and only that unchanged timer failure.
- No unrelated timing utility was modified.

### 11.3 Compilation and repository checks

- Pilot Python compilation: passed.
- `git diff --check`: passed.
- Protected G1-G5 diff against the locked base: empty.
- Protected satellite network, Tier 1 rollout, TGNN dataset, and graph-cache diff against the locked base: empty.
- No legacy static topology import was introduced.
- No production catalog replacement was introduced.
- No scientific thresholds were tuned after seeing pilot outcomes.

## 12. Limitations

1. Twenty-five rows provide engineering evidence, not definitive statistical power.
2. Each design has only five stochastic realizations.
3. The design matrix intentionally couples constellation size, orbit, failure probabilities, and ground composition; causal effects cannot be isolated from these 25 runs.
4. The catalog is deterministic and synthetic, not operationally representative.
5. The 10-minute window is appropriate for this pilot but does not establish long-horizon resilience.
6. Runtime and storage projections are linear approximations from one Windows workstation.
7. Threshold labels depend on the locked 0.80 policies and should be re-audited if policy values change.
8. P03-P05 exhibit floor concentration in several adjusted-service minima; final sampling must preserve intermediate regimes.
9. The stress-region service floor is informative for boundary coverage but should not dominate a training dataset.

## 13. Final integrated dataset design recommendations

1. Retain deterministic purpose-separated seed derivation and export all seeds, design hashes, policy hashes, catalog hash, and source artifact hashes.
2. Increase realizations per design and add intermediate designs between P02 and P03 to reduce the abrupt outcome transition.
3. Vary satellite and ground failure dimensions more independently to improve attribution and target support.
4. Preserve robust and stress controls while allocating more rows to the transition region.
5. Keep exact replay as a dataset release gate, not an optional diagnostic.
6. Preserve one-row-per-run tables alongside canonical temporal graph evidence.
7. Partition train, validation, and test data by design group when evaluating generalization to unseen architectures.
8. Treat catalog sensitivity as a separate experimental factor once a scientifically reviewed catalog is available.
9. Estimate storage primarily from G2 and G3 evidence and plan approximately 1.01 GB of canonical artifacts for 500 runs before ML derivatives.
10. Keep random-forest ex-ante prediction and TGNN ex-post assessment or future-prefix forecasting as explicitly separate tasks.

## 14. Final readiness verdict

READY FOR FINAL INTEGRATED DATASET DESIGN
