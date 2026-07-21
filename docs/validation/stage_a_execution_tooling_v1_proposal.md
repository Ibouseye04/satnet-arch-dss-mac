# SATNET Stage A Execution Tooling v1 Proposal

## Purpose

This proposal provides deterministic, contract-bound, fail-closed orchestration for a future separately authorized SATNET Stage A campaign. It does not authorize simulation, production, replay, acceptance, validation execution, or sealed-holdout access.

**No Stage A execution has been authorized or performed.**

## Frozen contract identity

The loader binds annotated tag `stage-a-discovery-contract-v1` to commit `301d8a224daa070b15ecc6447f503d42d5d1e70a` and verifies contract inventory SHA-256 `e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a`. It also verifies the frozen specification, declaration, README, all 12 contract-bound artifacts, source copies, approved proposal, seed manifest, audit identities, counts, record hashes, seeds, partitions, reserved roots, and false authorization state.

## Tooling architecture

The package separates contract loading, external authorization, deterministic planning, path isolation, preflight, ledger state, locking, generation orchestration, resume, replay, acceptance, integrity, and CLI concerns. The generation adapter maps exact frozen design/run/seed records into the validated final integrated G1-G5 pipeline. Its base-pipeline local ID is explicitly recorded while the Stage A wrapper preserves global run ID, run key, design hash, run hash, seeds, contract hash, and plan identity.

## Authorization model

Authorization is external, operation-specific, partition-specific, run-set-specific, output-root-specific, and bound to the exact contract and tooling identities. `PLAN`, `GENERATE`, `REPLAY`, and `ACCEPT` permissions are non-transitive. Missing, malformed, mismatched, non-approved, or non-`AUTHORIZED` documents fail closed. No real authorization artifact exists in this proposal.

## Partition isolation and development-first execution

Ordinary tooling supports only `development` and `validation`; there is no sealed-holdout execution option. Development derives 20 designs and 100 runs directly from frozen manifests. Validation remains a separate future authorization domain. Sealed holdout remains inaccessible and ordinary status/plan output reports only frozen counts with identities redacted.

## Run planning

Plans sort by global run ID and bind the frozen contract, tooling identity, partition, operation, ordered records, relative output paths, all output roots, and optional authorization identity. Timestamps are excluded. The development preview contains 100 runs, 20 designs, deterministic ordering, no authorization, and plan hash `b666d8c7700a10418de85180624617fae744cb80a64b675ca95d1c89c1fc61c3`.

## Output-root controls and preflight

Root validation rejects existing target roots, overlap, containment, repository/worktree placement, frozen-evidence placement, traversal, and link/junction escape. Preflight verifies contract/tag/tooling/authorization identities, exact partition and run set, root states by operation, disk space, parent write access, dependencies, simulation entrypoint availability, protected evidence, and campaign-lock absence. Parent probes are temporary and removed.

## Ledger state, atomicity, locking, and resume

The deterministic ledger supports `PLANNED`, `STARTING`, `RUNNING`, `SUCCEEDED`, `FAILED`, and `INTERRUPTED`. Every write uses temporary-file, flush, fsync, and atomic replace behavior. Campaign and run identity locks deny duplicate work. Success requires nonempty verified artifact inventories; process return alone is insufficient. Resume requires exact contract, plan, authorization, tooling, roots, run set, seeds, and completed artifact hashes. Changed output is never overwritten.

## Generation adapter, replay, and acceptance

The adapter receives all scientific values from frozen records and does not sample parameters or seeds. Replay requires distinct authorization, writes only to a distinct root, regenerates deterministic artifacts, and compares canonical inventories. Acceptance requires distinct authorization and validates generation/replay completeness, equality, run identity, seeds, manifests, ledgers, partition integrity, required artifacts, and absence of unexpected artifacts. It does not relabel targets.

## Dry-run and failure behavior

Contract verification and unauthorized dry-run planning are permitted. Unauthorized execution commands report `EXECUTION NOT AUTHORIZED` and create no reserved root. Crashes leave failed or interrupted evidence, never success. Retry requires explicit policy. Mutation, concurrency, replay mismatch, unexpected artifacts, and partial output fail closed.

## Security controls

The implementation rejects traversal, absolute relative-artifact paths, symlink/junction escape, duplicate identities, duplicate output paths, unknown runs, cross-partition authorization, sealed-holdout runs, count mismatch, root mismatch, contract/tooling mismatch, ledger mutation, completed-artifact mutation, hash mismatch, and seed mismatch.

## Tooling proposal identity

The inventory binds 28 source, schema, specification, byte-policy, isolation, and test files. Its SHA-256 and authoritative tooling proposal hash are both `75c7832f6adf65a3b584acdb313153dc0ebaba251d0fc3bd569efa51f86bcc17`.

## Validation

The new suite passed 32 tests. The focused Stage A, frozen-contract, audit, byte-policy, class-support, and isolation set passed 226 tests. Logger tests passed 7 tests; the independently rerun Windows timer test passed. The final monolithic suite recorded 1,138 passed and one known Windows timer-resolution failure; the exact timer test passed independently. Compilation covered 217 Python files. Whitespace, protected science, frozen contract, annotated tag, 9,004-file frozen evidence, tooling inventory, reserved-root absence, and simulation nonexecution checks passed.

## Prohibited actions and next task

No Stage A simulation, production generation, production replay, production acceptance, evidence freeze, holdout unsealing, Stage B work, RF training, TGNN training, merge, push, tag creation, or execution authorization was performed.

The required next task is **Independent audit of SATNET Stage A Execution Tooling v1**.
