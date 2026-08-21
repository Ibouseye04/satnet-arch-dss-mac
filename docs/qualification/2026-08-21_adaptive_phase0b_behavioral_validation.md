# SATNET Adaptive Remediation Phase 0B Behavioral Validation

## Authorization boundary

This qualification uses controlled fixtures, one small real SGP4 architecture, one
prepared adaptive frozen run, and small deterministic failure overlays. It does
not generate the 10K dataset, export RF/TGNN production data, train a model, run
held-out evaluation, or perform external evaluation.

## Exact executed call chain

1. `load_adaptive_contract` validates the prepared adaptive contract and its
   `fixed_profile`, then returns design/run records. The loader is explicitly
   read-only and does not launch simulation.
2. `map_adaptive_run` calls `map_frozen_run(...,
   production_profile=FINAL_ADAPTIVE_PRODUCTION_PROFILE)`. The mapping creates a
   `Tier1RolloutConfig` carrying `isl_policy`, `adjacent_search_k`, endpoint
   capacity, epoch, SGP4 engine, and failure model.
3. `run_tier1_rollout` constructs `HypatiaAdapter`, generates TLEs, copies all
   topology fields into `calculate_isls`, iterates every temporal graph, forms
   the temporal edge union, samples persistent failures, and applies failures
   at every timestep.
4. `HypatiaAdapter.calculate_isls` propagates SGP4 positions for every
   timestep and calls `_compute_grid_plus_isls` with the policy, search radius,
   and endpoint capacity. Returned `ISLLink` records become the cached Tier-1
   graph edges.
5. G3 reconstruction repeats the same Tier-1 adapter call, applies the frozen
   failure realization, and canonicalizes every timestep. TGNN reconstruction
   repeats the same graph-defining fields and converts each effective graph to
   the exported edge index.

Source references:

- Contract loader: `src/satnet/experiments/final_generation/adaptive_contract.py:35-53`
- Adaptive mapping: `src/satnet/experiments/final_generation/mapping.py:39-53,67-94,133-146`
- Production rollout mapping and temporal failure overlay: `src/satnet/simulation/tier1_rollout.py:269-373`
- SGP4 temporal ISL entrypoint: `src/satnet/network/hypatia_adapter.py:1279-1385`
- Graph reconstruction: `src/satnet/ground/satellite_graph_adapter.py:14-77`
- TGNN reconstruction: `src/satnet/models/gnn_dataset.py:438-481`

## White-box algorithm proof

`_compute_grid_plus_isls` executes the following behavior:

- Intra-plane links are evaluated first using the same `next_in_plane` logic for
  both policies (`hypatia_adapter.py:962-975`). The adaptive branch does not
  alter this construction.
- Fixed policy evaluates only same-index partner `sat_index(next_plane,
  sat_in_plane)` (`hypatia_adapter.py:984-999`).
- Adaptive policy starts with offset `0` and extends candidates in deterministic
  `[-1,+1]` order for `k=1` (`hypatia_adapter.py:979-982`). Candidate IDs are
  wrapped modulo satellites per plane and every candidate is passed to the same
  evaluator (`hypatia_adapter.py:1010-1029`).
- The evaluator computes maximum-distance feasibility, then Earth LOS, then link
  budget viability. Rejection reasons are recorded as
  `maximum_distance`, `earth_obscuration`, or `link_budget`
  (`hypatia_adapter.py:899-930`).
- Viable adaptive candidates are sorted by descending margin, ascending distance,
  then canonical endpoint IDs (`hypatia_adapter.py:1038-1046`).
- The selector checks the incident inter-plane degree of both endpoints before
  accepting an edge and increments both endpoint counters after acceptance
  (`hypatia_adapter.py:1047-1059`). Thus capacity `1` is enforced at both
  endpoints, not only at the source.
- Candidate offsets and timestep are included in adaptive selection diagnostics
  (`hypatia_adapter.py:1028`, `1071-1081`, `1383-1385`) so an accepted edge can
  be traced to a nonzero candidate offset without inferring from labels.

## Controlled behavioral sentinel

The deterministic synthetic geometry uses a selective budget that rejects the
90-km candidate and accepts the 100-km alternate candidates. With identical
positions and `max_inter_plane_links_per_sat=1`:

```text
FIXED_INTERPLANE_EDGES = []
ADAPTIVE_INTERPLANE_EDGES = [(0, 4), (1, 5)]
ADAPTIVE_ONLY_EDGES = [(0, 4), (1, 5)]
FIXED_ONLY_EDGES = []
```

The first adaptive source evaluates candidates at offsets `[0,-1,+1]`; offset 0
is rejected by Earth obscuration, the alternate candidates are evaluated, and
one alternate is selected by deterministic ranking. The controlled budget was
actually called for budget-eligible candidates and both fixed/adaptive runs
recorded distance, LOS, and budget rejections. Adaptive inter-plane endpoint
incident degrees are at most one. The intra-plane edge set is identical under
both policies.

## Real SGP4 differential: P03

Architecture: 5 planes x 6 satellites, 600 km, 55 degrees, phasing factor 1,
J2000 epoch, SGP4/WGS72, 10 minutes at 60-second cadence, 10,000 km distance
limit, zero node and edge failures. Only `isl_policy` changes.

| timestep | fixed inter-plane | adaptive inter-plane | adaptive-only | fixed components | adaptive components | fixed GCC/original | adaptive GCC/original |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 8 | 6 | 28 | 22 | 0.066667 | 0.066667 |
| 1 | 2 | 8 | 6 | 28 | 22 | 0.066667 | 0.066667 |
| 2 | 2 | 8 | 6 | 28 | 22 | 0.066667 | 0.066667 |
| 3 | 2 | 8 | 6 | 28 | 22 | 0.066667 | 0.066667 |
| 4 | 4 | 8 | 6 | 26 | 22 | 0.066667 | 0.066667 |
| 5 | 4 | 8 | 6 | 26 | 22 | 0.066667 | 0.066667 |
| 6 | 4 | 8 | 6 | 26 | 22 | 0.066667 | 0.066667 |
| 7 | 4 | 8 | 6 | 26 | 22 | 0.066667 | 0.066667 |
| 8 | 4 | 8 | 6 | 26 | 22 | 0.066667 | 0.066667 |
| 9 | 4 | 8 | 8 | 26 | 22 | 0.066667 | 0.066667 |
| 10 | 4 | 8 | 8 | 26 | 22 | 0.066667 | 0.066667 |

At timestep 0:

- Fixed graph hash: `d8c887479fc3cc999bc2e88989562ad626d86a9a291d22b466fb2508764b8862`
- Adaptive graph hash: `a93ff5adedb94e815cb6c2da363449e17fbda821dbcc66c6fd525eab04dfd1e8`
- Adaptive-only edges: `[(2,7), (5,10), (8,13), (11,16), (19,24), (22,27)]`

This is a graph-level differential across all 11 timesteps, not a metadata-only
comparison.

## Prepared adaptive production entrypoint

The first prepared frozen run (`D0000-R00`, P01: 6 planes x 8 satellites) was
loaded through the adaptive contract loader and `map_adaptive_run`, then
executed by `run_tier1_rollout`. Runtime fields were:

```text
isl_policy = grid_adaptive
adjacent_search_k = 1
max_inter_plane_links_per_sat = 1
failure_model = persistent_temporal_union_edges_v1
```

The rollout returned 11 steps with zero failures. The same mapped configuration
was run through `HypatiaAdapter.calculate_isls` with adaptive examples enabled.
The actual selected alternate trace was:

```text
 timestep             = 0
 source satellite     = 40
 same-index candidate = 0
 same-index reason    = earth_obscuration
 selected satellite   = 1
 alternate offset     = +1
 LOS                  = True
 distance             = 5643.28253471866 km
 margin               = 15.871781572634859 dB
 final edge identity  = (1, 40)
```

## Tier-1 -> G3 -> TGNN parity

The same prepared adaptive run and timestep 0 were canonicalized at all three
stages with zero failures:

```text
Tier1 count/hash = 71 / e1e2aa311bf79802d63fe1c2d3eacebad3041fc0b4c6fc45c9b6056ee96fbc8e
G3    count/hash = 71 / e1e2aa311bf79802d63fe1c2d3eacebad3041fc0b4c6fc45c9b6056ee96fbc8e
TGNN  count/hash = 71 / e1e2aa311bf79802d63fe1c2d3eacebad3041fc0b4c6fc45c9b6056ee96fbc8e
```

The adaptive-only edge `(1,40)` was present at all stages:

```text
Tier1: PRESENT
G3:    PRESENT
TGNN:  PRESENT
```

## Temporal-union failure overlay

A deterministic P03 adaptive qualification used seed 1, node failure
probability 0.15, edge failure probability 0.08, and the same 11-step physics.
The accepted adaptive temporal-union was:

```text
[(1,6), (1,25), (2,7), (4,9), (4,28), (5,10), (7,12), (8,13),
 (10,15), (11,16), (13,18), (16,21), (19,24), (22,27)]
```

The fixed-policy reconstruction union was only
`[(0,24), (1,25), (3,27), (4,28)]`; the sampled failed edge was
`(5,10)`, which exists only in the adaptive accepted universe. It was active at
timesteps `[0,1,2,3,4]` in the zero-failure baseline and was absent from the
effective graph at each of those timesteps after overlay. Failed nodes were
`[0,8,9,13,19,20,26]` and were absent from every effective snapshot.
Two identical executions with seed 1 produced identical step metrics, summary,
and failure realization.

The failure universe is constructed from `edge_union` over every accepted graph
step (`tier1_rollout.py:338-363`), so this evidence is explicitly adaptive and
temporal rather than a fixed t=0 reconstruction.

## Anti-fraud and drift guardrails

The mandatory string-only anti-fraud test monkeypatches the runtime builder to
force `grid_fixed` while the contract/configuration continues to say
`grid_adaptive`. It compares actual graph edge sets and detects the mismatch:

```text
STRING_ONLY_ANTIFRAUD = forced_fixed_edge_set_mismatch_detected=True
```

Configuration guardrails and behavioral guardrails are distinct:

- Configuration: adaptive contract loader validates the profile fields; adaptive
  run mapping binds `FINAL_ADAPTIVE_PRODUCTION_PROFILE`; DSS scenario construction
  validates the resulting runtime fields.
- Behavior: controlled actual edge sentinel, real SGP4 differential, endpoint
  degree assertions, selected-offset trace, and forced-fixed graph mismatch.

Deliberate fixed-policy substitutions failed at the adaptive contract loader,
final design/run mapping, replay mapping path (`map_adaptive_run`), and DSS
scenario construction.

## Verification and scope

Focused verification completed:

- 8 Phase 0B behavioral authorization tests passed.
- 7 existing ISL-policy and rollout-config tests passed.
- 6 existing adaptive production-profile tests passed; the existing 10K manifest
  mapping test was deselected from this focused invocation.
- 1 existing adaptive capacity validation test passed.
- Total focused result: **22 passed, 1 deselected**.
- `ruff` was not installed in the environment; `git diff --check` passed.

Files changed in Phase 0B:

- `src/satnet/network/hypatia_adapter.py`
- `src/satnet/dss/scenario_builder.py`
- `src/satnet/experiments/final_generation/adaptive_contract.py`
- `tests/network/test_isl_policy.py`
- `tests/validation/test_v07_adaptive_behavioral_authorization.py`
- `docs/qualification/2026-08-21_adaptive_phase0b_behavioral_validation.md`

No scientific parameters were changed. No 10K dataset generation occurred. No
model training occurred.
