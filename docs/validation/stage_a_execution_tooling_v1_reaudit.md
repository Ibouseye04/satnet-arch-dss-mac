# SATNET Stage A Execution Tooling v1 Independent Re-Audit

## Scope

This re-audit independently evaluated remediation commit `21eea8bc8d713b8799ff4e2fc3c0365139e61de3` and stable executable commit `5a4bb751bc379596abd938e68db377570222ed73`. It did not modify execution tooling, authorize execution, create a reserved Stage A root, expose sealed-holdout identities, or execute a Stage A simulation.

Approval was evaluated only for a later, separate development-partition execution-authorization preparation task. Simulation remains unauthorized.

## Input identities

- Re-audit branch: `audit/stage-a-execution-tooling-v1-reaudit`
- Remediation HEAD: `21eea8bc8d713b8799ff4e2fc3c0365139e61de3`
- Stable executable commit: `5a4bb751bc379596abd938e68db377570222ed73`
- Executable inventory SHA-256: `8404e0cda2864a111598bd8b1209a341a1f1ecfc1c12f7c17c611c0156a2f94a`
- Corrected tooling proposal inventory SHA-256: `845f4c72548b584af371ae1d3b868b7805673e05832ff4dbfff09703a3b2a7f9`
- Scientific artifact-contract file SHA-256: `ea03ce40d849815c6a5d4c0275e68e0fa1cba165a44e351e6a6e8e6db99e8fe3`
- Frozen Stage A contract hash: `e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a`
- Frozen tag: `stage-a-discovery-contract-v1`
- Frozen tag type: annotated
- Frozen tag target: `301d8a224daa070b15ecc6447f503d42d5d1e70a`
- Frozen-contract audit commit: `c350d693bf47d007a0cb2a8c3b5cfb2259d2da48`
- Frozen-contract audit inventory: `2fe074c018f69da95f29ffaca06ae45c50e7098988d7b181fe9756452db41df9`

## Original audit findings and remediation provenance

The original tooling HEAD was `94e0a9b2ee2d5699eba172eef84cbb857f5bd4cf`, with stable executable commit `c1605fbe62c57c1968856b592185e3f08bb9541d`, proposal hash `75c7832f6adf65a3b584acdb313153dc0ebaba251d0fc3bd569efa51f86bcc17`, independent audit HEAD `4d9ad633329bd3e0710ab566a1b25c7e01af889f`, and audit inventory `5c40277ffb85576eea9bf8c5f6772f99f0375a4828e4fcbba73a15c78a5368c8`.

The original verdict was `NOT APPROVED FOR STAGE A DEVELOPMENT EXECUTION-AUTHORIZATION PREPARATION`. This re-audit preserved the authoritative original mapping of `BINDING-001` through `BINDING-009`. The remediation report incorrectly reassigned the topics of `BINDING-006` through `BINDING-009`; that closure matrix was not treated as proof.

## Stable executable identity

The 70-file executable inventory was independently regenerated from current bytes. Relative paths, byte lengths, SHA-256 values, lexical ordering, uniqueness, source membership, and self-exclusion matched exactly. Real executable preflight passed from a clean repository and resolved imports to the expected worktree files.

The stable commit is an ancestor of the remediation HEAD. No inventoried executable path changed between `5a4bb751bc379596abd938e68db377570222ed73` and `21eea8bc8d713b8799ff4e2fc3c0365139e61de3`.

## Repository clean-state enforcement

Executable preflight rejects all tracked and untracked repository dirt, including documentation-only dirt, executable dirt, staged changes, untracked executable modules, and unrelated untracked files. This is stricter than the minimum requirement and matches the implemented specification.

## Mandatory preflight

Generation, resume, replay, and acceptance require a sealed `PreflightCertificate` before root validation, campaign-lock acquisition, root creation, ledger creation, persistent scientific writes, or adapter invocation. Synthetic monkeypatch tests supplied an invalid certificate and observed zero adapter calls, zero roots, zero ledgers, and zero locks.

The public CLI has no debug, force, environment-variable, or alternate execution alias that bypasses authorization and preflight.

## Root isolation

Temporary-directory tests confirmed rejection of existing roots, overlapping roots, containment in either direction, repository and worktree containment, frozen-evidence containment, traversal paths, and resolved link or junction escape. Windows case normalization uses `os.path.normcase`. Root creation follows successful preflight.

No reserved Stage A root was created.

## Artifact success contract

The corrected artifact-contract file SHA-256 is `ea03ce40d849815c6a5d4c0275e68e0fa1cba165a44e351e6a6e8e6db99e8fe3`; its canonical payload identity is `bdda72a8e3d6cd11fb79a42c1384aae042e93ebb7c912bda2803e08061f3eda1`.

Exact required paths, canonical parsing, selected schemas, scientific identities, three operational seeds, no unexpected artifacts, and adapter/filesystem manifest equality are enforced. Arbitrary text, missing files, malformed synthetic output, wrong identities, wrong seeds, wrong hashes, and unexpected files are rejected.

A binding gap remains. The public `execute_generation` API accepts an injected adapter. Its orchestration validator `_validate_production` does not call `validate_run_authoritatively`; only the standard production adapter does. A custom adapter can therefore bypass authoritative satellite and G1-G5 replay while still reaching the orchestration validator. Complete scientific validation is not non-bypassable before `SUCCEEDED`.

## Adapter-result validation

`SimulationAdapterResult` binds run, design, realization, three operational seeds, artifact manifest, inventory hash, simulator return state, validation state, and a self-identity hash. Orchestration independently compares the claimed manifest and hash with filesystem reality.

The structured result is adequate for manifest binding, but it does not close the complete-science bypass because the public orchestration surface permits a nonstandard production adapter and does not independently perform authoritative G1-G5 replay.

## Replay source-ledger binding

Replay validates generation-ledger semantic identity, campaign identity, tooling identities, roots, run membership, three operational seeds, states, and artifact inventories before replay.

Replay does not persist or bind the source generation-ledger path, byte length, or SHA-256. A canonical JSON formatting-only change alters source ledger bytes without altering its semantic payload hash and is accepted. The replay ledger therefore is not bound to the exact source generation ledger required by the audit contract.

## Acceptance cross-ledger binding

Acceptance validates generation and replay ledger semantics, run ordering, success states, stored artifacts, replay reports, and selected identity fields.

Acceptance cannot cross-bind the replay ledger to an exact source-generation-ledger SHA-256 because replay stores no such identity. A synthetic end-to-end test changed generation-ledger bytes after successful replay without changing JSON semantics; acceptance still returned `PASSED`.

## Seed binding

The frozen seed manifest contains:

- `design_construction_seed`
- `ground_selection_seed`
- `satellite_failure_seed`
- `ground_failure_seed`

Plans and ledgers bind only the latter three. `design_construction_seed` is omitted. The complete frozen seed set is therefore not carried through plan, generation, replay, and acceptance.

## Frozen-evidence preflight

Mandatory preflight invokes full frozen production-evidence verification before campaign writes. Synthetic mutation tests rejected changed ledger identities, archive identity, file count, byte count, verified-hash count, and read-only state.

The real frozen corpus was verified before and after the re-audit:

- 9,004 files
- 1,337,549,193 bytes
- 9,004 SHA-256 checks
- generation ledger `a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1`
- replay ledger `4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd`
- archive `375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc`
- all generation and replay files read-only
- archive and freeze metadata read-only
- before/after results identical

## Campaign locking

Campaign locks use atomic exclusive creation and reject a second owner. The payload contains a generic identity, scope, PID, process-start identity, nonce-like lock token, and creation time.

The required campaign fields are not explicit: campaign ID, operation, contract hash, plan hash, authorization hash, stable executable commit, and host identity are collapsed or absent. The campaign lock contract is incomplete.

## Per-run locking

Generation and replay acquire per-run locks with exclusive creation. Same-path concurrency is rejected.

The per-run payload does not explicitly carry campaign identity, global run ID, or run key as independent validated fields. The run key appears only in the scope string and the caller supplies only `run_record_hash` as generic identity. The required per-run lock binding is incomplete.

## Stale-lock recovery

Recovery is explicit, identity checked, guarded by an exclusive recovery lock, and records process-start identity. PID reuse on the local host is partially addressed by comparing process-start identity.

The following required controls are absent:

- host identity and foreign-host denial
- minimum lock age
- explicit campaign, contract, plan, authorization, and run identity fields
- active campaign-owner proof
- completed-output validation
- append-only immutable recovery evidence

A newly created inactive lock with no age delay, host proof, campaign-owner proof, or output validation was recovered successfully. Stale-lock recovery is unsafe under the authoritative audit contract.

## Windows byte portability

A fresh external checkout at remediation HEAD was created with `core.autocrlf=true`. All 95 unique executable/proposal inventory-bound files matched expected byte lengths and SHA-256 values. The checkout remained clean, executable verification passed, and the 40 corrected tooling tests passed.

The byte result is successful, but the policy result is not. `.gitattributes` contains repository-wide `*.py text eol=lf`, explicitly forbidden by this audit handover. Narrow inventory-bound coverage was required. `BINDING-008` remains open even though the fresh-checkout bytes matched.

## Proposal generation and inventory

The checked-in deterministic generator reproduced all 16 proposal artifacts twice in external roots. Both reproductions were byte-identical to the tracked proposal. The corrected authoritative proposal inventory remained `845f4c72548b584af371ae1d3b868b7805673e05832ff4dbfff09703a3b2a7f9`.

The proposal inventory contains 44 records excluding itself, has deterministic lexical ordering, no duplicates, no self-reference, and no record mismatches. Binding proposal artifacts contain no hostnames, temporary paths, wall-clock values, test timestamps, timer results, or environment-dependent test counts.

Schema SHA-256 values:

- tooling specification: `6a5acb5f0b4f04495274e04288098f643ba95f07ed701419b61669f1017a74cd`
- authorization schema: `c3451ba984a8f17c2306c970d6ae17ed8202d4236813e360b0f94da07d1906a4`
- plan schema: `fcf62acaf1bcb332225327e608d7c2318d55ac79775f797c1c59c06d901a315b`
- execution-ledger schema: `3ee6766baef4457ce710b95e738e0e4ad85acac95415d4116d924b383d4c0e2e`
- replay-ledger schema: `63ffa7402ad7a14f68611d791d34b12a1507250595ce54c8df5dad73158e888e`
- acceptance schema: `d91ae5afcc9442ef8c8848e2669a4d852218ff98ea606b90605114dee88affe0`

## Development plan

The independently derived nonexecuting development plan has:

- 20 development designs
- 100 development runs
- 5 validation designs
- 25 validation runs
- 5 sealed-holdout designs
- 25 sealed-holdout runs
- first development run `SA-D000-R00`
- last development run `SA-D027-R04`
- ordered-run-ID SHA-256 `94a3fa9098c01b348ced330b0d12f69fcf3d36003d47aa19294fe4637a27c732`
- corrected development plan hash `e517526db56f8cbce170980a02fdc6991787f2fbb435e68def7747db48029b97`

Scientific run membership, ordering, design values, run values, three operational seeds, and partition membership were unchanged. The complete seed-binding defect concerns the omitted frozen `design_construction_seed` field.

## Authorization default deny

The proposal state is exactly:

- status `TOOLING_PROPOSAL`
- `execution_authorized = false`
- `simulation_authorized = false`
- `production_authorized = false`

Contract verification and the nonexecuting development preview are allowed. Generate, resume, replay, and accept without authorization return exit code 2. No hidden bypass was found.

## Partition isolation and holdout redaction

The development plan contains zero validation runs, zero sealed-holdout runs, and zero duplicate runs. Ordinary output exposes only aggregate sealed-holdout counts and `identities = REDACTED`. It exposes no sealed-holdout design IDs, run IDs, seeds, or paths. There is no general-purpose sealed-holdout execution option.

## Synthetic end-to-end result

Synthetic-only tests used synthetic contracts, IDs, seeds, temporary roots, mocked adapters, and synthetic evidence. They covered authorization validation, mandatory preflight, root initialization, campaign and run locks, valid artifacts, `SUCCEEDED`, replay, mismatch rejection, acceptance, seed mutation, and exact-ledger-byte mutation.

No real Stage A run identity, real execution authorization, reserved root, or Tier 1 simulator was used. The exact-ledger-byte mutation test is the reproducible failure for replay/acceptance binding.

## Nonexecution evidence

- Real authorization artifacts: 0
- Reserved roots: 0
- Generation ledgers: 0
- Replay ledgers: 0
- Acceptance reports: 0
- Campaign locks: 0
- Per-run locks: 0
- Stage A scientific output roots: 0
- Stage A simulations: 0

## Protected science and frozen contract

No remediation change was found beneath the protected ground, network, rollout, GNN, RF, or graph-cache paths relative to the frozen-contract audit commit. Frozen contract, specification, declaration, README, approved proposal inventory, seed manifest, readiness audit inventory, frozen-contract audit inventory, designs, runs, partitions, targets, thresholds, failure models, and G1-G5 definitions retained their expected identities.

Historical audit and remediation artifacts were not modified.

## Validation

- New independent re-audit tests: `16 passed`
- Remediation regression tests: `7 passed`
- Corrected Stage A execution-tooling tests: `40 passed`
- Historical original audit harness: `37 skipped`
- Frozen-contract, proposal, fail-closed, near-neighbor, class-support, logger, and isolation group: `208 passed, 1 failed`
- Exact unchanged repository-isolation file at remediation HEAD in the fresh Windows checkout: `4 passed`
- Complete audit-branch suite excluding the unrelated timer assertion: `1161 passed, 37 skipped, 1 deselected, 1 failed`
- Complete suite excluding both the unrelated timer assertion and the audit-branch allowlist assertion: `1161 passed, 37 skipped, 2 deselected`
- Exact timer assertion rerun: `1 passed`
- Fresh Windows corrected tooling suite: `40 passed`
- Tracked Python compilation: `228` files compiled
- Whitespace check: passed
- Proposal reproduction: 16 artifacts, zero mismatches
- Executable inventory reproduction: 70 files, zero mismatches
- Fresh Windows inventory qualification: 95 unique bound files, zero byte mismatches
- Protected-science diff: empty
- Frozen-contract diff: empty
- Historical audit/remediation diff: empty
- Reserved-root check: all four roots absent

The one audit-branch failure is the unchanged historical `test_contract_changes_are_confined_to_approved_surfaces` assertion. Its allowlist ends at remediation outputs and therefore rejects the newly added re-audit script, test, report, and artifacts. The exact unchanged test passes `4/4` at remediation HEAD in the fresh checkout. The assertion and authoritative proposal-bound test bytes were not modified or weakened.

The 37 skipped tests are the immutable original audit harness scoped to `audit/stage-a-execution-tooling-v1`. Active corrected and independent re-audit tests cover every original binding topic; the independent tests intentionally assert the observed open-control behavior rather than weakening historical assertions.

## Binding findings

| Finding | Independent status | Result |
|---|---|---|
| BINDING-001 | CLOSED | Exact current executable identity and clean-state enforcement passed. |
| BINDING-002 | CLOSED | Mandatory preflight and root-before-write ordering passed. |
| BINDING-003 | OPEN | Complete authoritative production-science validation remains bypassable through the public injected-adapter surface. |
| BINDING-004 | OPEN | Replay does not bind exact source generation-ledger path, length, and SHA-256. |
| BINDING-005 | OPEN | Acceptance lacks exact source-ledger cross-binding and complete frozen seed binding. |
| BINDING-006 | CLOSED | Real and synthetic frozen-evidence preflight verification passed. |
| BINDING-007 | OPEN | Lock payloads and stale recovery do not satisfy required campaign/run/host/age/owner/output controls. |
| BINDING-008 | OPEN | Fresh bytes match, but the line-ending policy is the explicitly forbidden broad Python rule. |
| BINDING-009 | CLOSED | All 16 proposal artifacts regenerate deterministically and byte-identically. |

## Nonbinding findings

None.

## Observations

1. The remediation report reassigns the original meanings of `BINDING-006` through `BINDING-009`; this re-audit used the authoritative handover mapping.
2. Fresh Windows bytes are portable despite the noncompliant broad policy.
3. The unchanged historical repository-isolation allowlist rejects newly added re-audit paths on the audit branch but passes at the exact remediation HEAD; it was preserved rather than weakened.

## Required corrections

1. Make complete authoritative production-science validation non-bypassable in every public execution path before `SUCCEEDED`.
2. Bind replay and acceptance to exact source ledger paths, byte lengths, and SHA-256 identities.
3. Carry and compare `design_construction_seed` with every frozen seed across plans and all ledgers.
4. Redesign lock payloads and stale recovery to include all campaign/run/host identities, minimum age, active-owner proof, completed-output validation, and immutable recovery evidence.
5. Replace the broad repository-wide Python line-ending rule with narrow inventory-bound rules and regenerate stable/proposal identities.

## Final verdict

```text
NOT APPROVED FOR STAGE A DEVELOPMENT EXECUTION-AUTHORIZATION PREPARATION
```

No authorization-preparation task is approved until all open binding findings are remediated and independently re-audited. Simulation remains unauthorized.
