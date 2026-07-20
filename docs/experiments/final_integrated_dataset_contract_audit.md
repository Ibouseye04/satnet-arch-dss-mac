# Final Integrated Dataset Contract Audit

## Verdict

**READY FOR EXTERNAL CONTRACT AUDIT**

The contract-only implementation is complete at validated contract-tooling SHA `461d556f750491be1e521b81169b6a2462fd28a1`.

This verdict authorizes external audit only. It does not authorize simulation generation, G1-to-G5 scientific changes, RF/TGNN implementation changes, model training, release tagging, or final dataset generation.

## Audited repository identities

| Role | Full SHA |
|---|---|
| Protected G1-to-G5 science base | `62beda9df1576958d9e33d33d2d9eb5489e24b20` |
| Validated 25-run pilot report | `e4475f9bc22a83b30cdc6862d3337a1e2b6dbc3f` |
| API audit | `56734ee96ff36f2c7516553a5ee2dc2e592d7750` |
| Machine schemas and specification | `af89cc6d13de4508bd4399717fcdc61861019449` |
| Deterministic DOE tooling | `ad80765b5f0dc52f322c427b449330234f22b8f8` |
| Canonical manifests and split | `23c2ccff110f2fdad1debbb79a6dd88ae6680c52` |
| Human-readable scientific contract | `6e4a94543a99105759932fd6824df78a0f19c5fe` |
| Validated contract tooling and tests | `461d556f750491be1e521b81169b6a2462fd28a1` |

The pilot SHA originally supplied in abbreviated form as `e447e5c095f57e70c5fc08f92ea5155bf9f24da3` was not a valid repository object. Preflight established and used the actual validated pilot report SHA `e4475f9bc22a83b30cdc6862d3337a1e2b6dbc3f`.

## Artifact inventory

All contract artifacts are under `artifacts/final_integrated_dataset_contract`.

| Artifact | Purpose |
|---|---|
| `contract_specification.json` | Machine-authoritative scientific contract |
| `target_schema.json` | Exact G5 target hierarchy |
| `integrated_rf_export_schema.json` | Future ex-ante integrated RF export schema |
| `integrated_tgnn_adapter_schema.json` | Future ex-post integrated TGNN adapter schema |
| `designs.jsonl` | 100 ordered canonical design records |
| `runs.jsonl` | 500 ordered canonical run records |
| `split_manifest.json` | Frozen pre-outcome grouped split |
| `contract_bundle.json` | Acyclic binding identity over specification, schemas, catalog, manifests, and split |
| `golden_vectors.json` | Exact canonical payload, digest, and seed vectors |
| `doe_evidence.json` | DOE frequency and selected-LHS evidence |
| `manifest_inventory.json` | Counts and hash inventory |

No G1, G2, G3, G4, G5, satellite rollout, model, or training artifacts were created.

## Exact scientific identities

| Identity | Hash |
|---|---|
| Contract specification | `487b72217ef75e3edd5b460f0c8fa43e6c643f342f05b6c783a43ef9a125f286` |
| Contract bundle | `930ff61414f9806464fad8e27de4eaf1523cebaeadde71df89a5e2ec409ffcd9` |
| Target schema | `9088948d6b03db59877a129179ac3091c3690aeab9d2849a921bfa90f05da528` |
| RF export schema | `f60519fe5440645646697fa697207bed3411fece89720e0048663037ed54e9b4` |
| TGNN adapter schema | `bb1d5454904266b83a3c7457098df00d46733994e9f673e4a40b94f31c893cfa` |
| Pilot semantic catalog | `810c64dfb030b042311c90f2f42f8dee866a48fc63a6a29e362ee328c52eaa6e` |
| Pilot catalog file bytes | `e8855d4ded4c242f3e5b35b90610d5f9c4f218bdf717d2d08b2464cac39f1598` |
| Pilot design manifest | `514795b9cc17deacd81ffb9308e0d944f8da64b80444047647e919b375ef27bf` |
| Final design manifest | `9f7ded00d12c6dcff0155d8d145503582c3da0cddd534d41b775c43d4982654e` |
| Final run manifest | `2a21031ef14a36462692731f839e46333fa3016a47e5c499ad07f170db216f51` |
| Final split manifest | `3256e07451654cf68c2cf0654ac4c8cc38bec3230a7988f9899834150eed8969` |

The semantic catalog identity and raw file-byte identity are intentionally separate.

## Cardinality and split evidence

| Check | Result |
|---|---|
| Unique designs | 100 |
| Realizations per design | 5 |
| Run records | 500 |
| Training designs/runs | 70 / 350 |
| Validation designs/runs | 15 / 75 |
| Test designs/runs | 15 / 75 |
| Selected split candidate | 164 of 4096 |
| Pilot-anchor allocation | 3 train / 1 validation / 1 test |
| Outcome fields used for split | No |
| Design groups crossing splits | None |

The split score uses exact `Fraction` arithmetic. Its maximum normalized deviation is `17/3`; this is driven by rare actual reduced-composition categories, including anchor-only categories. No hidden tolerance, outcome balancing, candidate expansion, or fallback relaxation was used.

## Determinism evidence

The implementation locks:

- Contract master seed `20260719`.
- Split master seed `20260720`.
- Unsigned big-endian SHA-256 seed derivation modulo `2^63`.
- Production satellite seed purpose `satellite_rollout_and_failure`.
- One design-level ground-selection seed and identity.
- Five realization-level satellite and ground-failure seed pairs per design.
- Canonical dimension order.
- 256 maximin LHS candidates per sampled stratum.
- 53 most-significant digest-bit jitter extraction.
- Exact pair enumeration and deterministic tie-breaking.
- Independent schedule pairing digests.
- 4096 deterministic split candidates.
- Canonical UTF-8 sorted compact JSON and final newline persistence.
- Strict canonical binary64 parsing, finite-value enforcement, and negative-zero rejection.

Byte-for-byte rematerialization of all eleven contract artifacts passed in a temporary directory.

## DOE evidence

The final designs contain exactly:

```text
5 pilot-anchor designs
35 transition designs
60 global designs
```

Transition LHS candidate `72` and global LHS candidate `158` are frozen. Transition ground coverage contains all 35 requested total/composition cells exactly once. Transition satellite-pair counts are exactly 6, 6, 6, 6, 6, and 5 in the locked order. Every global satellite pair occurs five times. Every global station total occurs six times. Every global source composition regime occurs six times.

Pilot anchors preserve all pilot scientific parameters while receiving new final identities, seeds, ground selections, and ground identities. P01 retains 8/6/6 class counts. P05 retains one station per class.

## Ground-architecture evidence

Each design materializes:

- One design-level ground-selection seed.
- Class-specific selected station IDs.
- G1 class-concatenated selected-ID order.
- One ground-selection hash.
- One ground-design hash.

Every one of the five run records for a design references the same selection seed, selection hash, and ground-design hash. Each run has a distinct deterministic satellite seed and ground-failure seed. No run-level geography resampling exists.

## Target and leakage evidence

The target schema binds exact verified G5 summary fields. Classification polarity is explicit: value 1 means at least one threshold breach occurred.

The integrated RF schema contains only ex-ante design-time predictors. Tests confirm that target names, seeds, hashes, selected station identities, realized failures, graph metrics, service metrics, statuses, runtimes, and artifact sizes are absent from predictors.

The TGNN schema is explicitly future-only and incompatible with the current satellite-only dataset/model interface. Satellite ECEF is omitted. Failed satellites retain G3 absent-node semantics. Failed selected ground stations remain represented with a G5 operational indicator.

No current RF registry, trainer, loader, TGNN dataset, TGNN model, pooling, edge-feature, or prediction-head implementation was changed.

## Protected-diff evidence

The following protected paths have an empty diff from the validated pilot report SHA through validated tooling SHA:

```text
src/satnet/ground
src/satnet/network
src/satnet/simulation/tier1_rollout.py
src/satnet/models/gnn_dataset.py
src/satnet/models/gnn_model.py
src/satnet/models/risk_model.py
src/satnet/utils/graph_cache.py
```

The same protected paths also have an empty diff between the protected G1-to-G5 base and the validated pilot report SHA.

All contract changes are confined to:

```text
artifacts/final_integrated_dataset_contract
docs/experiments/final_integrated_dataset_*
src/satnet/experiments/final_dataset
tests/experiments/test_final_dataset_*
```

## Validation results

| Validation | Result |
|---|---|
| Pre-change complete baseline | 809 passed |
| Focused final-contract suite | 24 passed |
| Post-change complete suite | 833 passed |
| Python compilation | Passed |
| `git diff --check` | Passed |
| Protected pilot-to-tooling diff | Empty |
| Protected science-base-to-pilot diff | Empty |
| Ruff | Not executed; module not installed in the active Python environment |

One pre-existing Windows timing test, `TestExperimentLogger.test_timer`, intermittently measured a requested 10 ms sleep as zero during two complete-suite attempts. The same isolated test passed, and the subsequent unmodified complete-suite run passed all 833 tests. No timer, utility, or protected implementation code was changed. This environmental timing flake is recorded for audit transparency and is not a contract-tooling failure.

## Repository state warning

Three unrelated untracked paths existed before this task and remain untouched:

```text
data/
docs/refactor_plans/2026-07-15_tier1_validity_remediation_atomic_gameplan.md
docs/validation/tier1_defect_verification.md
```

They are not part of the contract commits or audit evidence.

## Catalog scientific limitation

Production research catalog availability remains **NOT AVAILABLE**. Production research catalog scientific review remains **NOT PERFORMED**. The locked catalog is synthetic pilot evidence only. The final contract is reproducible and externally auditable, but it is not evidence that the ground-station catalog is scientifically representative of an operational network.

## Stop condition confirmation

The task stopped after contract artifacts, schemas, deterministic manifests, split assignments, documentation, tests, protected diffs, and audit reporting.

No simulations were executed. No final dataset run was generated. No release tag was created. No generation branch was started.
