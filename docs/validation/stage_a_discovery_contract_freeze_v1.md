# SATNET Stage A Discovery Contract v1 Freeze

## Verdict

**STAGE A CONTRACT FROZEN — SIMULATION NOT AUTHORIZED**

The frozen contract state is `FROZEN_PENDING_INDEPENDENT_AUDIT`. The contract is frozen, but simulation, production, and execution authorization remain false. The required next task is an independent audit of the frozen SATNET Stage A Discovery Contract v1.

## Restart identity

- Restart branch: `freeze/stage-a-discovery-contract-v1-restart`
- Byte-policy correction commit: `9c55e5999f308a84b857fdf6828cf48a9ca5b360`
- Feature and contract-artifact commit: `93019b2`
- Focused test commit: `df81c1b38ac73e65b1d2dfdb3ec6a7dd6e517f39`
- New generating worktree: `C:\Users\johns\satnet-stage-a-contract-freeze-restart-tooling-20260721`
- Stale worktree left untouched: `C:\Users\johns\satnet-stage-a-contract-freeze-tooling-20260721`

The new worktree inherited `core.autocrlf=true`. Its first materialization reproduced the approved audit inventory as 3,691 bytes, SHA-256 `16ce1a1b138567a144cbf9b1d715c74b30f05e1d8262841d9580240080339d10`, 107 LF sequences, and zero CRLF sequences. All 17 tracked audit files and all 16 inventory-bound files matched the approved audit inventory.

## Approved inputs

- Proposal commit: `509f2449dbbaf4c1f5153ecfa4bc1652f24f75da`
- Proposal inventory SHA-256: `69fd2a9fbcc1b461ae4230a282cfcaaadc26ba7b65563bc90a81e86e1ca98127`
- Seed-manifest SHA-256: `ab37564cca81e926b6e43caf74f1ff2be641b8bc203e32c3ae6a219ea6f7bace`
- Freeze-readiness audit commit: `a1514a654fe76518db16001b98e899f773eb9d1e`
- Freeze-readiness audit inventory SHA-256: `16ce1a1b138567a144cbf9b1d715c74b30f05e1d8262841d9580240080339d10`
- Freeze-readiness verdict: `APPROVED FOR STAGE A CONTRACT FREEZE`

Proposal reproduction passed for all 11 artifacts. No proposal or audit bytes changed.

## Frozen contract identity

- Frozen root: `artifacts/stage_a_discovery_contract_v1`
- Source artifacts: 11 byte-exact proposal copies
- Contract-bound artifacts: 12
- Frozen specification SHA-256: `c1822db61182e6ff6436c767ac39a35065ff84741119bd7306b261dc2a1f7373`
- Frozen inventory SHA-256: `e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a`
- Authoritative frozen contract hash: `e88b2f3f3fd545a2876e79262be73901eb77a656e08835ff01aeb52d6a9ba51a`
- Freeze declaration SHA-256: `e0987858e1eca4d04e7008a468de232ef26eaa867f9d2a750c83fc8c89717bd4`
- Freeze README SHA-256: `d4d9d8b796b194ef6bebe7ad1a4520cc38cee4f3fb40889fc922cf3705928b0c`

The authoritative contract hash is the SHA-256 of the exact frozen inventory bytes, not a Git commit identity.

## Frozen scope

- Designs: 30, `SA-D000` through `SA-D029`
- Runs: 150, global IDs 500 through 649
- Seed records: 150
- Realizations: 5 per design, `SA-D000-R00` through `SA-D029-R04`
- Regions: 12 resilient core, 12 boundary, 6 global control
- Partitions: 20 development designs and 100 runs; 5 validation designs and 25 runs; 5 sealed-holdout designs and 25 runs

The threshold is 0.80. The canonical margin remains `failure_adjusted_overall_service_fraction_min - 0.80`; a margin greater than or equal to zero is non-breach and a negative margin is breach. Quantization remains 0.000001 with `ROUND_HALF_EVEN`.

The corrected `SA-D013`/`SA-D020` distance is `0.12039492645571381`. The minimum development-holdout distance is `0.1205683310788027`, and the minimum validation-holdout distance is `0.1500946005884104`. No near-neighbor exceptions or pending scientific reviews remain.

All Stage A partitions remain excluded from the primary final corpus. The sealed holdout may only confirm or reject an already-frozen and independently audited Stage B contract and may not alter that contract.

## Validation

- Byte-policy tests: 17 passed; dedicated fresh-checkout test: 1 passed
- Frozen-contract focused tests: 53 passed
- Stage A proposal, fail-closed, and near-neighbor tests: 8, 8, and 9 passed
- Class-support analysis, prior audit, freeze-readiness audit, and logger tests: 8, 8, 20, and 7 passed
- Repository isolation tests: 4 passed
- Complete repository suite: 1,041 passed
- Independent timer test: 1 passed
- Compilation: 196 tracked Python files passed
- Whitespace and protected-science checks: passed
- Frozen production evidence: 9,004 files and 1,337,549,193 bytes reverified; all content hashes and read-only states passed

A final fresh clone at `C:\Users\johns\satnet-stage-a-contract-freeze-restart-qualification-final-20260721` was configured with `core.autocrlf=true` and checked out at `df81c1b38ac73e65b1d2dfdb3ec6a7dd6e517f39`. It reproduced all 17 audit files, all 11 source-bundle artifacts, all five derived artifacts, the authoritative contract hash, declaration, authorization state, and reserved-root absence. Its 71 focused qualification tests passed and the checkout remained clean.

## Authorization and next step

- `contract_frozen`: `true`
- `simulation_authorized`: `false`
- `production_authorized`: `false`
- `execution_authorized`: `false`
- Reserved Stage A production, replay, acceptance, and freeze roots: absent
- Push, merge, simulation, replay, acceptance, Stage B, RF, and TGNN actions: not performed

Required next task: **Independent audit of the frozen SATNET Stage A Discovery Contract v1**.
