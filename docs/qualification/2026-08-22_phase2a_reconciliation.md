# Phase-2A Reconciliation — Adaptive-v2

**Result: PASS**

The accepted Phase-2 inventory is stale only for `metadata/validation_gates.json`. The exporter writes compact gates before freezing the inventory, then appends expanded historical immutability evidence to the Phase-2 manifest and rewrites `validation_gates.json` without regenerating the inventory. The expanded file is valid JSON and remains consistent with the accepted Phase-2 PASS.

## Complete inventory comparison

- Accepted inventory: `29ef628bbec3ad6a003ede89749b1ba64925098b1edc55cec7e3cb98946af6b8`
- Accepted bundle/content hash: `1399f12e6ed7e159076028bc01971cf2274924fc55b43cca0a44219db8c978d1`
- Inventory entries: 10,022; unique paths: 10,022
- Exact matches: 10,021
- Missing files: 0
- Unreadable files: 0
- Duplicate paths: 0
- Unexplained extras: 0 (the three exporter-excluded root files are not inventory artifacts)
- Size mismatches: 1
- SHA-256 mismatches: 1

The complete mismatch is:

| Path | Frozen | Current |
| --- | --- | --- |
| `metadata/validation_gates.json` | 7,310 bytes; `5544ca16c6e4d79021812ed705fea3b4312a65e65fdd9f683ab1a3d21110fa47` | 3,515,889 bytes; `64d2a7700b4d04136df0721bba6f197a705cd5ac01f2a424c4fa9c0453818661` |

## Scientific payload proof

Scientific/training payload mismatch count: **0**.

Metadata-only mismatch count: **1** (`metadata/validation_gates.json`). All four RF CSVs, all RF schemas and manifests, all TGNN schemas/manifests/target manifests, and all 10,000 TGNN sequence files match their accepted identities. The accepted RF CSV hashes and TGNN manifest identities are preserved in the external Phase-2A report.

Current identities remain: 10,000 runs; 2,000 designs; five realizations/design; run splits 7,000/1,500/1,500; design splits 1,400/300/300; design leakage 0; missing IDs 0; duplicate IDs 0; NaN 0; Inf 0; illegal/out-of-range 0; target mismatch 0; TGNN 10,000 × 11 with node dimension 3 and edge dimension 4; adaptive topology `('grid_adaptive', 1, 1, 'persistent_temporal_union_edges_v1')`; fixed-policy contamination 0.

## Phase-2A freeze

Sidecar: `C:\Users\johns\external\satnet-10k-final-ml-datasets-v2-adaptive-reconciliation`

- Phase-2 evidence SHA: `f47800bd5121203b6ba8ec918aaf0b6adc6d0aa1`
- Current-tree inventory SHA-256: `53a8ad2fbc6203d4901d15a9ef87d38be124002462b4f81d9ab956d161cefc3f`
- Scientific/training-payload inventory SHA-256: `a9d199a674c20ca31b2a91c6172f371b504603a2e7e09da29c51cd06654c9473`
- Deterministic training-payload bundle/content hash: `af91fd86702403c232757ab4b5e123b2fabc314b1c4ce26f7bd9a6a0d58d5167`

No dataset correction or regeneration occurred. No Phase-1 rerun, 10K simulation, authoritative replay, RF/TGNN regeneration, historical artifact mutation, or training occurred before reconciliation.
