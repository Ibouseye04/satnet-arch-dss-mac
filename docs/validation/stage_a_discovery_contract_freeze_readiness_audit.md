# Stage A Discovery Contract Freeze-Readiness Audit

## Final verdict

`APPROVED FOR STAGE A CONTRACT FREEZE`

This verdict means only that corrected proposal commit `509f2449dbbaf4c1f5153ecfa4bc1652f24f75da` is ready for a separate contract-freeze task. The Stage A contract remains `NOT_FROZEN`; simulation authorization remains `false`; no Stage A simulation, replay, acceptance, freeze, Stage B execution, model training, merge, push, or tag was performed.

## Audit scope

This was an independent, read-only scientific and operational audit of the corrected SATNET Stage A Discovery Contract Proposal v1. The audit parsed tracked proposal bytes directly, independently recomputed scientific signatures, normalized distances, design/run hashes, all four seed purposes, partition arithmetic, class-support semantics, margin quantization, gate feasibility, and output-root isolation. Canonical proposal tooling was used only for the separate byte-for-byte artifact reproduction check.

Audit outputs were written outside all production and proposed Stage A roots at `C:\Users\johns\satnet-stage-a-freeze-audit-20260721` and mirrored byte-for-byte into `artifacts/stage_a_discovery_contract_freeze_audit`. The tracked audit bundle contains 17 files: 16 inventoried outputs plus the self-excluding inventory. Audit inventory SHA-256 is `16ce1a1b138567a144cbf9b1d715c74b30f05e1d8262841d9580240080339d10`.

## Input identities

| Input | Independently verified identity |
|---|---|
| Corrected proposal commit | `509f2449dbbaf4c1f5153ecfa4bc1652f24f75da` |
| Starting correction commit | `f61521e96edc1bb8b5c3f05e8ebbcd25bca1eb7d` |
| Production tooling | `9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab` |
| Frozen contract tag | `final-integrated-dataset-contract-v1` |
| Frozen contract commit | `a1967185e80327e4b00c1831828dc975ab6819fc` |
| Contract specification identity | `482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b` |
| Generation ledger | `a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1` |
| Replay ledger | `4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd` |
| Freeze archive | `375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc` |
| Class-support analysis inventory | `5ac295ba0e08797b63a2ce3f062a1995dc7aa850f11a2bf5d83a0580731afd85` |
| Prior independent-audit inventory | `20d2e7037940d708e133936d3907ad822caac3b46cd73c84329f3efd16c4037a` |
| Corrected proposal inventory | `69fd2a9fbcc1b461ae4230a282cfcaaadc26ba7b65563bc90a81e86e1ca98127` |
| Seed manifest | `ab37564cca81e926b6e43caf74f1ff2be641b8bc203e32c3ae6a219ea6f7bace` |

The contract specification identity is the embedded canonical `contract_spec_hash`; the JSON container's raw byte hash is not the contract identity.

## Prior binding findings

All five prior binding findings and the later near-neighbor blocker are resolved:

1. The proposal has exact 30-design, 150-run, and 150-seed manifests, exact region bounds, partitions, roots, criteria, policies, and inventory.
2. Stage A uses namespace `stage_a_discovery_v1`, design IDs `SA-D000` through `SA-D029`, and global run IDs `500` through `649`; no original identity collision exists.
3. Stage A is discovery evidence only. Development, validation, and sealed holdout are excluded from the primary final corpus. The primary final classification corpus is the original frozen corpus plus a future frozen Stage B corpus.
4. Preassigned boundary-region membership is explicitly separated from observed boundary status. Boundary and class-support criteria operate at design level.
5. Final classification gates define majority classes, mixed-design counting, observed boundary status, canonical margin quantization, run ratios, scope, and exact per-split thresholds. They are mathematically feasible and do not apply to Stage A.
6. The `SA-D013`/`SA-D020` development-validation blocker was resolved without an exception or pending review.

## Proposal artifact reproduction

All 11 proposal-bound files reproduced byte-for-byte from canonical tooling. Relative paths, SHA-256 values, byte lengths, CSV record counts, schema identifiers, UTF-8 encoding, LF newlines, deterministic ordering, and inventory self-reference avoidance passed.

- Corrected proposal inventory: `69fd2a9fbcc1b461ae4230a282cfcaaadc26ba7b65563bc90a81e86e1ca98127`
- Seed manifest: `ab37564cca81e926b6e43caf74f1ff2be641b8bc203e32c3ae6a219ea6f7bace`
- Artifact result: `11/11 byte-identical`

## Design-manifest audit

The manifest contains exactly 30 rows, 30 unique IDs, and indexes `0` through `29`. IDs exactly span `SA-D000` through `SA-D029`.

Region allocation is 12 resilient-core, 12 boundary, and 6 global-control designs. Partition allocation is 20 development, 5 validation, and 5 sealed holdout designs. Region-by-partition allocation is `8/2/2`, `8/2/2`, and `4/1/1` respectively.

All scientific values are finite, typed, inside frozen base DOE bounds, and inside assigned region bounds. Configured satellite counts and ground-station compositions are arithmetically exact. All fixed fields represented directly in the design rows match the frozen profile. Inclusive timestep count, physics model, link-budget configuration, and G1-G5 version identities are inherited through the exact immutable base-contract reference and specification hash.

The independently defined scientific parameter vector contains architecture, orbit, failure assumptions, station composition, and fixed-profile values while excluding IDs, hashes, partitions, and proposal metadata. Results were:

- Duplicate Stage A parameter vectors: 0
- Exact original-design parameter duplicates: 0
- Design ID/index collisions: 0
- Base-bound violations: 0
- Region-bound violations: 0
- Design-parameter hash mismatches: 0
- Design-record hash mismatches: 0

## Run-manifest audit

The run manifest contains exactly 150 rows. Global IDs exactly span `500` through `649`. Run keys exactly span five realizations for every design, from `SA-D000-R00` through `SA-D029-R04`.

All 150 run keys and all 150 design-realization pairs are unique. Every design has realization indexes `0` through `4`; all five realizations retain the design's region and partition; only holdout runs are sealed. No run ID collides with original IDs `0` through `499`. All 150 run-record hashes independently reproduced.

## Seed audit

The seed policy uses SHA-256 over UTF-8 compact canonical JSON, all 32 digest bytes, unsigned big-endian conversion, and reduction modulo `2**63`. The corpus namespace, proposal version, design identity, realization identity where applicable, and purpose are domain-separated.

All 150 seed records independently reproduced for design construction, ground selection, satellite rollout/failure, and ground failure. Design and ground-selection seeds are fixed by design; failure seeds vary by realization. Retry identity is stable and seed substitution or outcome-driven changes are prohibited.

The corrected SA-D020 parameter did not alter seeds because the declared seed policy is identity-only. The corrected seed manifest is byte-identical to the starting proposal seed manifest.

## Partition audit

The only partitions are `development`, `validation`, and `sealed_holdout`:

- Development: 20 designs / 100 runs
- Validation: 5 designs / 25 runs
- Sealed holdout: 5 designs / 25 runs

Assignments are pre-simulation and immutable with respect to outcomes. Designs do not cross partitions and all five realizations are colocated. Stage A partitions are not mapped to final train/validation/test semantics.

All original design IDs `D000` through `D099`, run IDs `0` through `499`, 500 original run split assignments, original seed records, and the frozen target-schema identity remain preserved. Original split counts remain 70/15/15 designs and 350/75/75 runs for train/validation/test.

## Holdout audit

The holdout is sealed at 5 designs and 25 runs. Its prohibited-use list covers Stage B parameter bounds, regions, design density, sample size, split allocation, seeds, acceptance gates, and frozen-contract modification.

Unsealing requires the Stage B contract, design manifest, run manifest, seed manifest, split, independent audit, and recorded contract hash. The machine-readable policy combines run- and seed-manifest generation into one condition but requires both actions. Independent negative tests reject every individually absent prerequisite and a missing or malformed contract hash.

After lawful unsealing, holdout evidence may only confirm or reject the already-frozen Stage B contract. A required change rejects that version and requires a new proposal, freeze, and audit.

## Final-corpus and Stage B adaptation audit

Stage A development, validation, and sealed holdout are all excluded from the primary final classification corpus. Stage A development and validation may inform a separately proposed Stage B contract; the holdout may not. Stage B requires new design/run/seed manifests, a frozen split, an independent audit, and a recorded contract hash. Stage B remains proposal-only and simulation-unauthorized.

## Boundary and class-support definitions

The authoritative margin is:

`failure_adjusted_overall_service_fraction_min - 0.80`

Margin zero is non-breach. A design is observed boundary if its five margins straddle zero (`minimum < 0` and `maximum >= 0`) or at least two realizations have absolute margin at most `0.05`. Exact `0.05` qualifies.

Independent cases for non-breach counts `0` through `5` verified:

- Counts 0, 1, and 2 are breach-majority.
- Counts 3, 4, and 5 are non-breach-majority.
- Counts 1 through 4 are mixed.
- Every design counts toward exactly one majority-class minimum.
- Mixed designs may separately count as observed boundary but never toward both class minima.

## Margin quantization audit

Canonical margin quantization is `0.000001` using `ROUND_HALF_EVEN`. Positive and negative ties, values immediately above and below half increments, and negative zero all reproduced exactly. Distinct-margin count is the number of unique canonical six-decimal margin strings.

## Discovery-criteria audit

The proposal contains 22 uniquely identified criteria in four required groups: development, validation, pre-holdout Stage B proposal, and sealed-holdout confirmation. Every criterion contains its ID, description, input partition, metric, operator, threshold, rationale, failure consequence, and Stage B influence flag.

Criteria cover completeness, resilient-core reproduction, independent non-breach support, mixed boundary designs, near-boundary localization, realization stability, validation reproduction, global-control consistency, no holdout consultation, and sealed-holdout confirmation. Every threshold is mathematically feasible for its input partition. Final classification gates explicitly do not apply to Stage A.

## Decision-state audit

All six required states are exact. The compact nine-line artifact workflow semantically preserves all 11 required actions in order by combining validation execution/evaluation and Stage B freeze/audit. It requires the Stage B contract hash before holdout unsealing and permits only execution or rejection of the unchanged frozen Stage B contract afterward. No transition permits silent post-holdout modification.

## Near-neighbor resolution audit

Independent comparison to the starting proposal proved that only `ground_station_failure_probability` changed for SA-D020, from `0.075` to `0.100`. Architecture, orbital geometry, satellite failures, station composition, fixed profile, identity, region, validation partition, sealed state, and seeds are unchanged. The corrected value remains within the boundary range `[0.03, 0.15]`.

| Measurement | Independent result |
|---|---:|
| Original SA-D013/SA-D020 | `0.07681919236933395` |
| Corrected SA-D013/SA-D020 | `0.12039492645571381` |
| Minimum development-validation | `0.12039492645571381`, SA-D013/SA-D020 |
| Minimum development-holdout | `0.1205683310788027`, SA-D013/SA-D011 |
| Minimum validation-holdout | `0.1500946005884104`, SA-D020/SA-D022 |
| Minimum within-partition | `0.1249155271916213`, SA-D011/SA-D022 |
| Minimum Stage A-to-original | `0.06382978723404255`, SA-D024/D004 |

The complete matrix contains 435 Stage A pairs and 3,000 Stage A-to-original pairs. There are zero cross-partition pairs below `0.10`, zero holdout pairs below `0.10`, zero exact Stage A pairs, zero exact Stage A-to-original pairs, no exceptions, and no pending scientific reviews.

The SA-D024/D004 proximity is not a leakage or independence failure: the designs have different purposes and namespaces, are not exact scientific duplicates, and Stage A is excluded from the final corpus.

The ten-candidate SA-D020 review is deterministic, uses admissible boundary-region values, preserves identity, excludes duplicates, and contains no simulation-outcome input. The selected ground-failure-axis change preserves the intended validation-boundary sampling role while avoiding architecture and orbital changes.

## Region-bounds audit

Every region declares base bounds, region-specific allowed values or intervals, fixed values, exclusions, construction rationale, and source evidence. All 30 designs satisfy their assigned region. Corrected SA-D020 satisfies every boundary-region constraint.

## Output-root audit

The four proposed generation, replay, acceptance, and freeze roots are exact, mutually non-overlapping, external to all Git worktrees, outside frozen evidence and audit roots, and absent. Their creation remains unauthorized. No Stage A generation ledger, replay ledger, acceptance report, scientific run output, or production root exists.

## Final-gate feasibility

Final classification gates apply only to the original frozen corpus plus a future frozen Stage B corpus. They do not apply to Stage A.

The expected combined sizes are 130 designs / 650 runs for train and 30 designs / 150 runs each for validation and test. Independent integer-interval analysis found all class-design minima, class-run minima, run-ratio maxima, observed-boundary minima, and distinct-margin minima simultaneously feasible in every split. No gate exceeds design or run capacity and no class minima conflict.

## Authorization and fail-closed audit

All authoritative proposal artifacts are `NOT_FROZEN` and simulation-unauthorized. No contradictory authorization state was found. No explicit `contract_frozen` field is present; equivalent false state is established by `NOT_FROZEN`, false simulation authorization, absence of a frozen Stage A contract hash, and the explicit requirement for a future separate freeze.

Independent negative tests reject duplicate designs/runs, identity collisions, duplicate scientific vectors, original-design duplicates, incorrect counts, missing realizations, out-of-bounds and region-invalid values, hash mismatches, seed substitutions, incomplete holdout prerequisites, missing Stage B hash, output overlap, sub-threshold cross-partition/holdout neighbors, infeasible gates, frozen status, simulation authorization, and contract-frozen state. Any proposal byte mutation also fails canonical artifact and inventory reproduction.

## Protected science and frozen evidence

Tracked and untracked protected-science changes are empty. Production generation, replay, acceptance, frozen contract, and original manifests were not modified.

The audit verified all 8,502 generation files and 502 replay files before and after proposal analysis: 9,004 files, 1,337,549,193 bytes, and 9,004 exact SHA-256 values each time. All generation, replay, freeze metadata, archive, and archive-hash files remain read-only. Before and after results are identical.

## Validation results

- Stage A focused, fail-closed, and near-neighbor tests: `25 passed`
- Class-support analysis tests: `8 passed`
- Prior independent-audit tests: `8 passed`
- Logger tests: `7 passed`
- New freeze-readiness audit tests: `20 passed`
- Isolation tests: `4 passed`
- Initial monolithic suite before the final added preservation test: `968 passed, 1 failed` due the known Windows timer-resolution test returning `0.0` after a 10 ms sleep
- Intermediate monolithic suite before the final added preservation test: `969 passed`
- Final monolithic attempts after all 20 audit tests were present: each `969 passed, 1 failed`, with only the same Windows timer test failing
- Exact timer-test reruns: `1 passed` independently
- Deterministic split verification: `969 passed, 1 deselected` for all non-timer tests, followed by the exact timer test `1 passed`
- In-memory compilation: `192` Python files passed
- Whitespace: passed
- Proposal artifact reproduction: `11/11` byte-identical
- Protected science: passed
- Frozen evidence before/after: passed and unchanged

The increase from the previous `950` tests to `970` collected outcomes is explained by 20 new independent audit tests. The complete 970-test set passed when the known timer test was isolated from the monolithic Windows timing context. No scientific code was changed to address the timer flake.

## Findings

### Binding findings

None.

### Nonbinding findings

None.

### Observations

1. SA-D024 is close to original D004 (`0.06382978723404255`) but is not an exact duplicate and does not enter the final corpus.
2. Inclusive timestep count, physics model, link budget, and G1-G5 version identities are inherited through the exact base-contract reference rather than duplicated in each Stage A row.
3. Stage A seeds intentionally bind identity and purpose rather than scientific parameters, preserving all seed bytes after the SA-D020 correction.

## Required corrections

None.

## Governance state

- Proposal status: `NOT_FROZEN`
- Simulation authorized: `false`
- Contract frozen: no
- Stage A simulation/replay/acceptance: not performed
- Stage B work: not performed
- RF/TGNN training: not performed
- Merge/push/tag: not performed

## Final verdict

`APPROVED FOR STAGE A CONTRACT FREEZE`

Approval is limited to readiness for a separate Stage A contract-freeze task. It does not freeze the contract or authorize any execution.
