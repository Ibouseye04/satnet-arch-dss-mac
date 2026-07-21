# Stage A Discovery Contract Correction v1

## Decision

This correction resolves the binding and non-binding findings in the independent class-support audit at `5add33fcd8205f59e8694b104c9b34f82bd66cd1` without authorizing simulation.

- Proposal status: `NOT_FROZEN`
- Simulation authorized: `false`
- Stage A namespace: `stage_a_discovery_v1`
- Stage A designs: `SA-D000` through `SA-D029`
- Stage A runs: global IDs `500` through `649`
- Stage B: proposal-only and not frozen

## Scientific scope

Stage A is a discovery corpus. It tests whether the D000/D001 resilience evidence and D001-to-D022 transition evidence reproduce across independent designs and realizations. It does not satisfy or inherit final classification acceptance gates.

The exact sampling regions are preassigned DOE regions. The word `boundary` in a design row identifies a sampling region only. An observed boundary design is outcome-defined after simulation: five margins must straddle zero, or at least two margins must have absolute value at most `0.05`.

A non-breach realization has margin greater than or equal to zero. A breach realization has negative margin. A design belongs to exactly one majority class: at least three of five non-breach realizations or at least three of five breach realizations. A mixed design can separately qualify as an observed boundary design, but it never counts toward both majority-class minima.

## Frozen evidence preserved

The proposal binds the following existing evidence identities:

- Production tooling SHA: `9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab`
- Frozen contract commit: `a1967185e80327e4b00c1831828dc975ab6819fc`
- Contract specification hash: `482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b`
- Generation ledger SHA-256: `a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1`
- Replay ledger SHA-256: `4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd`
- Freeze archive SHA-256: `375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc`

No frozen production, replay, freeze, archive, original contract, original design, original run, original target, original split, or protected science file is modified.

## Exact Stage A design and partition contract

The authoritative table is `artifacts/stage_a_discovery_contract_proposal/stage_a_design_manifest.csv`. It contains 30 exact rows, fixed simulation-profile values, deterministic ground-selection parameters, a base-contract reference, a design-parameter hash, and a design-record hash.

The allocation is:

| Region | Development | Validation | Sealed holdout | Total |
|---|---:|---:|---:|---:|
| Resilient core | 8 | 2 | 2 | 12 |
| Boundary | 8 | 2 | 2 | 12 |
| Global control | 4 | 1 | 1 | 6 |
| Total | 20 | 5 | 5 | 30 |

Each design has five colocated realizations. The run formula is `500 + design_index * 5 + realization_index`. Design/run identities, partitions, regions, and seeds are fixed before simulation and cannot be changed in response to outcomes.

## Seed policy

All seeds use SHA-256 of UTF-8 canonical JSON, all 32 digest bytes, unsigned big-endian conversion, and reduction modulo `2**63`. The Stage A domain is `satnet_stage_a_discovery_v1_seed`.

- Design-construction seed: fixed by design
- Ground-selection seed: fixed by design
- Satellite rollout/failure seed: varies by realization
- Ground-failure seed: varies by realization

Retries retain the same identity and seeds. Seed substitution is prohibited.

## Holdout policy

The five-design, 25-run holdout is sealed. It may be opened only after the Stage B contract is completely specified, its design/run/seed manifests and split are frozen, it passes independent audit, and its contract hash is recorded.

Holdout outcomes cannot choose Stage B parameter bounds, regions, density, sample size, split allocation, seeds, gates, or any other contract field. The holdout is not the final model test set. If it indicates a change is needed, Stage B v1 is rejected and a separately frozen and audited Stage B v2 is required.

## Stage B adaptation boundary

Stage A development and validation may inform a new Stage B proposal. Stage B remains separate, proposal-only, and not simulation-authorized. A future Stage B contract must specify its exact DOE, design/run/seed manifests, split, output roots, target references, gates, and inventory, then pass independent audit before Stage A holdout unsealing.

The planning baseline remains 90 Stage B designs and 450 runs, allocated 36 resilient-core, 36 boundary, and 18 global-control designs. It is not frozen by this correction.

## Final corpus membership and gates

The primary final classification corpus is the frozen original corpus plus a future frozen Stage B corpus. Stage A development, validation, and holdout are excluded. If Stage B remains 90 designs, the combined corpus has 190 designs and 950 runs.

The corrected gates use majority-class design counts, outcome-defined observed boundary counts, and canonical signed margins quantized to `0.000001` with `ROUND_HALF_EVEN`. The proposal contains exact feasible Stage B non-breach design/run intervals and exact minimum/maximum evidence for every split. Original designs remain in their frozen splits; Stage B can only append new grouped designs.

## Fail-closed controls

The implementation rejects:

- Duplicate Stage A parameter vectors
- Exact duplicates of original designs
- Design or run identity collisions
- Incorrect region or partition counts
- Seed substitutions or out-of-range seeds
- Holdout leakage or early unsealing
- Overlapping or pre-existing output roots
- Frozen-evidence output descendants
- Infeasible final gates
- Any proposal state other than `NOT_FROZEN`
- Any simulation authorization

## Proposal artifacts

The tracked proposal directory contains:

- Exact design manifest
- Exact run manifest
- Partition manifest
- Seed policy and seed manifest
- Region bounds
- Output-root manifest
- Discovery criteria
- Near-neighbor policy and reports
- Contract proposal
- Deterministic proposal inventory

The inventory excludes its own bytes to avoid a self-referential hash cycle. It records every other proposal artifact by relative path, byte count, schema identifier, record count where applicable, and SHA-256.

## Remaining work before simulation

This correction is not a freeze-readiness approval. Before any Stage A run:

1. Independently audit every tracked artifact and inventory hash.
2. Recompute identities, duplicates, seeds, record hashes, and distance reports.
3. Resolve any cross-partition near-neighbor review.
4. Verify proposed external roots remain absent and isolated.
5. Freeze exact contract bytes and record the contract hash.
6. Obtain explicit execution authorization in a later task.
