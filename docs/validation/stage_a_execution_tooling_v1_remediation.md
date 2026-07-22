# Stage A Execution Tooling v1 Remediation

## Status

**READY FOR INDEPENDENT REAUDIT**

This correction does not authorize execution, simulation, production use, or the sealed holdout. No production output root was created and no Stage A scientific simulation was run.

## Immutable inputs

- Audit commit: `4d9ad633329bd3e0710ab566a1b25c7e01af889f`
- Frozen Stage A tag: `stage-a-discovery-contract-v1`
- Frozen Stage A commit: `301d8a224daa070b15ecc6447f503d42d5d1e70a`
- Frozen Stage A contract hash: `e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a`
- Frozen production tooling commit: `9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab`
- Frozen contract specification SHA-256: `482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b`
- Frozen generation ledger SHA-256: `a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1`
- Frozen replay ledger SHA-256: `4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd`
- Frozen archive SHA-256: `375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc`

## New stable executable identity

The corrected executable source was finalized in commit `5a4bb751bc379596abd938e68db377570222ed73`.

The executable inventory contains 70 source files and has SHA-256 `8404e0cda2864a111598bd8b1209a341a1f1ecfc1c12f7c17c611c0156a2f94a`. Preflight verifies the exact inventory bytes, source membership, each file hash and byte length, a clean repository, stable-commit ancestry, no post-stable executable diff, and import resolution to the expected repository files.

The scientific artifact contract has SHA-256 `ea03ce40d849815c6a5d4c0275e68e0fa1cba165a44e351e6a6e8e6db99e8fe3`. The corrected proposal inventory has SHA-256 `845f4c72548b584af371ae1d3b868b7805673e05832ff4dbfff09703a3b2a7f9`.

## Finding closure

- **BINDING-001:** Closed by current executable byte verification and stable ancestry checks.
- **BINDING-002:** Closed by mandatory sealed preflight certificates on every public execution API before locks, directories, or writes.
- **BINDING-003:** Closed by structured adapter results and exact scientific artifact-contract validation.
- **BINDING-004:** Closed by replay validation of generation ledger, campaign identity, run records, roots, and all seeds.
- **BINDING-005:** Closed by acceptance validation of both source ledgers, source authorization and plan identities, artifacts, and all seeds.
- **BINDING-006:** Closed by explicit authoritative production-run validation before adapter success.
- **BINDING-007:** Closed by the portable deterministic generator and fresh Windows byte-for-byte regeneration test.
- **BINDING-008:** Closed by campaign and per-run locks with PID plus process-start identity and explicit recorded stale-lock recovery.
- **BINDING-009:** Closed by end-to-end contract, executable, proposal, artifact-contract, authorization, run, design, realization, and seed bindings.

## Validation evidence

- Focused Stage A suite: `40 passed`.
- Fresh Windows checkout with `core.autocrlf=true`: executable verification passed and `40 passed`.
- Fresh-checkout proposal regeneration: 16 artifacts, zero byte mismatches.
- Frozen production evidence: 9,004 files and 1,337,549,193 bytes verified; 9,004 SHA-256 checks passed; read-only checks passed.
- Complete suite excluding one unrelated timer assertion: `1146 passed, 37 skipped, 1 deselected`.
- The unrelated timer assertion passed separately: `1 passed`.
- Protected science diff: empty.
- Reserved production, replay, acceptance, and freeze roots: absent.

The 37 skipped tests are the immutable historical audit harness retained for the original audit branch. The correction branch adds independent replacement regressions covering the resolved findings.

## Remaining gate

An independent reaudit must review this branch and the remediation evidence. Any later execution authorization must be a separate explicit decision bound to the corrected stable executable commit, executable inventory, proposal inventory, artifact contract, exact operation, exact partition, exact run set, and exact output roots.
