# Final Integrated Dataset Scientific Contract

## Status

Locked on 2026-07-19 for external contract audit. This document authorizes a later 500-run final integrated dataset generation phase only after external approval. It does not generate simulations and does not alter G1-to-G5 science or current RF/TGNN implementations.

## Validated foundation

| Identity | Value |
|---|---|
| Integrated-pilot report SHA | `e4475f9bc22a83b30cdc6862d3337a1e2b6dbc3f` |
| Protected science base SHA | `62beda9df1576958d9e33d33d2d9eb5489e24b20` |
| Pilot design manifest hash | `514795b9cc17deacd81ffb9308e0d944f8da64b80444047647e919b375ef27bf` |
| Pilot semantic catalog hash | `810c64dfb030b042311c90f2f42f8dee866a48fc63a6a29e362ee328c52eaa6e` |

Production research catalog availability remains **NOT AVAILABLE**. Production research catalog scientific review remains **NOT PERFORMED**. The locked catalog is deterministic, synthetic, pilot-only engineering evidence.

## Machine authority

The machine-authoritative artifacts are under `artifacts/final_integrated_dataset_contract`.

| Artifact identity | Hash |
|---|---|
| Contract specification | `487b72217ef75e3edd5b460f0c8fa43e6c643f342f05b6c783a43ef9a125f286` |
| Contract bundle | `930ff61414f9806464fad8e27de4eaf1523cebaeadde71df89a5e2ec409ffcd9` |
| Target schema | `9088948d6b03db59877a129179ac3091c3690aeab9d2849a921bfa90f05da528` |
| Integrated RF export schema | `f60519fe5440645646697fa697207bed3411fece89720e0048663037ed54e9b4` |
| Integrated TGNN adapter schema | `bb1d5454904266b83a3c7457098df00d46733994e9f673e4a40b94f31c893cfa` |
| Design manifest | `9f7ded00d12c6dcff0155d8d145503582c3da0cddd534d41b775c43d4982654e` |
| Run manifest | `2a21031ef14a36462692731f839e46333fa3016a47e5c499ad07f170db216f51` |
| Split manifest | `3256e07451654cf68c2cf0654ac4c8cc38bec3230a7988f9899834150eed8969` |

The contract specification hash binds the scientific specification, seeds, DOE, production policy profile, schema hashes, split algorithm, and acceptance gates. Design records reference the specification hash. Run records reference the specification and design-record hashes. The split references the specification and design manifest. The bundle binds every resulting identity. Generated records never reference the bundle hash, so the graph is acyclic.

## Scope

The contract contains exactly:

```text
100 unique design groups
5 realizations per design
500 expected runs
```

The contract phase creates only specifications, deterministic manifests, split assignments, tests, and documentation. It does not create satellite rollout, G1, G2, G3, G4, or G5 run artifacts.

## DOE strata

| Design IDs | Stratum | Count |
|---|---|---:|
| `D000`–`D004` | Exact pilot scientific-parameter anchors | 5 |
| `D005`–`D039` | P02-to-P03 transition enrichment | 35 |
| `D040`–`D099` | Global coverage | 60 |

### Pilot anchors

The anchors copy every scientific parameter represented by P01 through P05. They retain `pilot_design_id`, `pilot_design_hash`, and the pilot design-manifest hash as provenance. They receive new final design identities and newly derived final design-level ground selections.

An anchor is an exact pilot scientific parameter point, not a copied pilot record, pilot realization, or pilot geography. P05 retains its three-station stress architecture. P01 retains its exact 8/6/6 class counts.

### Transition stratum

The transition stratum uses a repository-local five-dimensional 35-row maximin Latin hypercube over:

| Dimension | Range |
|---|---:|
| Altitude | 600–800 km |
| Inclination | 55–60 degrees |
| Satellite node failure | 0.05–0.10 |
| Accepted-edge failure | 0.05–0.10 |
| Ground-station failure | 0.05–0.15 |

Its exact ground schedule is the Cartesian product of totals 10, 12, 15, 18, and 20 with rational weights 1:1:1, 3:1:1, 1:3:1, 1:1:3, 2:2:1, 2:1:2, and 1:2:2. All 35 cells occur once.

Its satellite-pair frequencies are:

```text
(5,6): 6
(5,7): 6
(5,8): 6
(6,6): 6
(6,7): 6
(6,8): 5
```

### Global stratum

The global stratum uses a separate repository-local 60-row maximin Latin hypercube over:

| Dimension | Range |
|---|---:|
| Altitude | 300–1200 km |
| Inclination | 30–98 degrees |
| Satellite node failure | 0.00–0.20 |
| Accepted-edge failure | 0.00–0.25 |
| Ground-station failure | 0.00–0.40 |

Every combination of planes 4, 5, or 6 and satellites per plane 5, 6, 7, or 8 occurs exactly five times. Each total in 6, 10, 15, 20, 25, 30, 35, 40, 45, and 50 occurs six times. Each of the ten locked rational composition regimes occurs six times.

### Integer allocation

Every non-anchor class allocation starts with one station per class. Remaining stations use exact rational largest-remainder allocation. Equal remainders resolve civilian, then government, then military. No floating-point remainder comparison is permitted.

### LHS algorithm

Each LHS evaluates exactly 256 deterministic candidates. SHA-256 canonical payloads independently determine every dimension permutation and every within-stratum jitter. Jitter uses the 53 most-significant digest bits interpreted as an unsigned big-endian integer. Candidate selection maximizes minimum pairwise distance, then mean pairwise distance, then minimizes candidate ID.

Continuous rows, ground schedules, and satellite-pair schedules receive independent deterministic permutations before positional pairing. Golden payload, canonical JSON, digest, and seed vectors are persisted.

## Fixed production profile

Every design uses:

```text
10 minutes
60-second steps
11 inclusive sampled states
phasing factor 1
maximum ISL distance 10000 km
grid_fixed ISL policy
adjacent search k 1
one maximum inter-plane link per satellite
SGP4 orbital engine
2000-01-01T12:00:00+00:00 epoch
persistent_temporal_union_edges_v1 satellite failure model
10-degree minimum elevation
0.80 space GCC threshold
0.80 ground-service threshold
```

The contract specification binds actual production G1-to-G5 versions, `tier1_space_segment_physics_v2`, and the complete canonicalized default `LinkBudgetEngine` configuration. It does not invent absent version fields.

The visibility and G4 service policy hashes are constant. G5 model and sampling versions are constant. The G5 ground-failure policy hash varies with each design probability.

## Design-level ground architecture

One deterministic selection seed is derived per design. The selected class tuples preserve authoritative G1 class-concatenated ordering. Each design record materializes the seed, class-specific IDs, combined IDs, selection hash, and ground-design hash.

All five runs of a design use the same geographic architecture and ground-design identity. Satellite rollout/failure and ground-failure seeds vary by realization. A later run-level G1 record may carry a distinct satellite configuration hash while preserving the design-level selection and ground-design hash.

## Seed contract

```text
contract master seed = 20260719
split master seed = 20260720
seed modulus = 2^63
```

Every seed is the unsigned big-endian SHA-256 integer of its exact canonical payload modulo `2^63`. The one production satellite purpose is `satellite_rollout_and_failure`. The ground failure purpose is `ground_failure_realization`. No global random state, timestamp, host state, or process state is permitted.

## Target hierarchy

| Role | Authoritative G5 field |
|---|---|
| Primary classification | `overall_threshold_breach_any` |
| Secondary classification | `ground_threshold_breach_any` |
| Secondary classification | `space_threshold_breach_any` |
| Primary regression | `failure_adjusted_overall_service_fraction_mean` |
| Safety secondary regression | `failure_adjusted_overall_service_fraction_min` |
| Secondary regression | `failure_adjusted_ground_service_fraction_min` |
| Secondary regression | `space_gcc_fraction_original_min` |
| Diagnostic only | `ground_service_loss_due_to_failures_max` |

Classification value 1 means at least one threshold breach occurred. The mean overall-service target is primary because the integrated pilot found greater unique-value and within-design stochastic spread than for the minimum. The minimum remains the safety-oriented secondary target. The maximum ground-service-loss field remains diagnostic because it failed the pilot useful-spread gate.

Labels must be copied from verified G5 summaries. They must never be inferred from design probabilities or seed values.

## RF schema

`integrated_rf_export_schema_v1` is a future ex-ante one-row-per-run export schema. It is not the current satellite-only RF registry. Predictors are variable design values only. Fixed policy fields are provenance. Seeds, IDs, hashes, selected stations, failures, graph metrics, service metrics, outcomes, statuses, runtimes, and artifact sizes are prohibited predictors.

No RF loader, registry, trainer, or model change is authorized.

## TGNN adapter schema

`integrated_tgnn_adapter_schema_v1` is a future ex-post adapter over canonical G3 graph sequences plus the verified persistent G5 ground-failure overlay. It is not implemented, is not the current `SatNetTemporalDataset`, and is not directly consumable by the current `SatelliteGNN`.

Failed satellites retain G3 absent-node semantics. Selected ground stations remain as nodes and receive a G5 operational indicator. Satellite ECEF features are omitted. Ground ECEF may be deterministically derived through the existing WGS84 transformation. General edge attributes may be exported, but the current model does not consume them.

Heterogeneous message passing, edge-feature consumption, graph-static encoding, pooling changes, and prediction-head changes are deferred to a later model contract.

## Canonical numbers and persistence

Machine scientific binary64 values are canonical `.17g` strings. Raw JSON floats are prohibited in canonical hash payloads. Readers parse finite Python floats, reject negative zero, reserialize with `canonical_float_string()`, and require exact equality.

Examples:

```text
600.0 -> "600"
55.0 -> "55"
0.05 -> "0.050000000000000003"
0.8 -> "0.80000000000000004"
```

Design JSONL is ordered by design index. Run JSONL is ordered by run index. Canonical JSON uses sorted keys, compact separators, UTF-8, Unix newlines, and duplicate-key rejection.

## Pre-outcome grouped split

The split assigns 70 designs to training, 15 to validation, and 15 to test. Every realization remains with its design. Split candidate `164` was selected from exactly 4096 SHA-256 orderings.

The score balances marginally across planes, satellites per plane, DOE stratum, station total, reduced actual class ratio, and locked four-bin partitions of altitude, inclination, node failure, edge failure, and ground failure. Exact `Fraction` arithmetic determines expected counts and candidate scores.

Hard requirements ensure all plane categories, satellite-per-plane categories, and DOE strata occur in every split. The five anchors split 3/1/1 across train/validation/test. No target or outcome field participates. The assignment is frozen before simulation generation and cannot be silently reshuffled.

## Later generation acceptance gates

External contract approval authorizes a separate generation branch. That phase must require:

```text
500 expected runs
500 attempted runs
500 successful generation runs
500 exact authoritative G1-to-G5 replays
no missing or duplicate identities
no design-group split leakage
all target values finite and in-domain
both classes in every classification split
primary regression unique-value count at least 20
complete protected-science diff isolation
focused and complete tests passing
```

A failed run is not omitted, replaced, or assigned a new seed. A failed balance or variance gate does not trigger post-outcome reshuffling or threshold tuning.

## Protected boundary

This contract adds tooling only under `src/satnet/experiments`, tests under `tests/experiments`, documentation, and small contract artifacts. Authoritative modules under ground, network, simulation, current RF/TGNN models, and graph cache remain unchanged.

## Stop condition

Stop after contract artifacts, schemas, deterministic manifests, split, documentation, tests, complete-suite validation, and protected diffs. Do not run simulations, generate G1-to-G5 run evidence, train models, tag a release, or start final dataset generation.
