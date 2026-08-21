# Final Integrated Dataset 10k Contract and Qualification

## Status

`BLOCKED: FULL SUITE ENVIRONMENT GATE`

This document prepares and qualifies the frozen dissertation-scale 10,000-run
contract. It does not authorize execution of the full production campaign.

**FULL 10,000-RUN PRODUCTION HAS NOT BEEN EXECUTED.**

## Purpose and provenance

The active branch is `experiment/final-integrated-dataset-10k`, created from
`experiment/final-integrated-dataset-generation` at starting SHA
`a8fbfed18b1673f5fc9c6a291ccc02905f1392d6`. The validated foundation remains
protected; the candidate tooling SHA established by the prior qualification is
`9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab`.

This is a contract/DOE/control-plane expansion only. The validated SGP4/WGS72,
Walker-Delta, TEME/GMST/ECEF, LOS, ISL, 1550 nm link-budget, grid-fixed ISL,
temporal failure, G1-G5 integrated-ground, target, RF, and TGNN semantics are
unchanged.

## Frozen contract

| Item | Value |
|---|---|
| Contract version | `3` |
| Contract tag | `final-integrated-dataset-10k-contract-v1` |
| Contract root | `artifacts/final_integrated_dataset_10k_contract/` |
| Contract specification hash | `6c7dd365f9e7fb67f5f5e70879a19535ede55468aabfac53d82c2ab35b8307eb` |
| Design manifest hash | `d39830c861ae3e7c5222dd05c44ecef6fb66b365a6d7a42e26f335c8820600e0` |
| Run manifest hash | `e965d5daea19a958fe6ce20d5c4a75240a42497b6fde33a508d4df0df64888ea` |
| Split manifest hash | `07a84c324b255f21f2697b88150c2cc3171156405f0750e2db1312ff8204c4aa` |
| Contract bundle hash | `059dff74930d1125a46947a06d213dd07c3a93dce226a894558a01805c3ed94c` |
| Qualification tooling SHA | `83798a8561ec4aa2a6dac8cc0fb8bb7d295d79d6` |
| Frozen tag target | `a4a576607dd0d17ac4b6f350daff76a352c04f57` |

The historical `artifacts/final_integrated_dataset_contract/` contract remains
unchanged and retains its 500-run provenance.

## Cardinality and DOE

- 2,000 unique designs: `D0000` through `D1999`.
- Five realizations per design: `R00` through `R04`.
- 10,000 exact integer run IDs: `0` through `9999`.
- Run identity is `design_index * 5 + realization_index`.
- Transition has 395 designs (`D0005`-`D0399`) and global has 1,600 designs
  (`D0400`-`D1999`), in addition to the five unchanged pilot anchors.
- Transition ground cells retain the exact 35-cell Cartesian coverage and use
  deterministic 11-repeat plus 10-cell remainder balancing. Satellite pairs
  use counts 66,66,66,66,66,65.
- Global ground totals and composition categories are each represented exactly
  160 times. The 12 satellite pairs use deterministic balanced counts 134 for
  the first four product-order pairs and 133 for the remaining eight.
- The five-dimensional LHS retains the 256-candidate digest permutation,
  jitter, Euclidean score, and tie ordering. Pairwise distance evaluation is
  NumPy-vectorized; no candidate approximation or scientific simulation change
  was introduced.
- On the qualification Windows environment, direct LHS scoring measured 6.888 s
  for the 395-row transition search and 51.760 s for the 1,600-row global
  search. The 4,096-candidate split search was retained and its score counting
  was changed from repeated record scans to exact `Counter` aggregation.

The frozen pre-outcome grouped split candidate is **3958** with score recorded
in `split_manifest.json`:

- Train: 1,400 designs / 7,000 runs.
- Validation: 300 designs / 1,500 runs.
- Test: 300 designs / 1,500 runs.

All five realizations of every design are colocated. No outcomes or labels are
used by DOE construction or split selection.

## Qualification set

The deterministic, pre-outcome qualification IDs are:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
20, 21, 22, 23, 24, 25, 35, 60,
3039, 3539, 3735, 4115, 5564, 6099, 6225, 6874, 7195, 7365,
7369, 8624, 8645, 8980, 9174, 9514, 9999
```

The set contains 45 runs across train, validation, and test; pilot-anchor,
transition, and global strata; low/high architecture, continuous parameter,
failure-probability, station-count, and composition cases; and beginning,
middle, and end run identities. These IDs are frozen in the machine
specification and generation constants before any simulation outcome is read.

## Gate record

| Gate | Result |
|---|---|
| Contract materialization | Passed; reconstruction hashes match the frozen bundle |
| 10k no-simulation dry run | Passed: 2,000 designs, 10,000 mappings, IDs 0-9999, zero simulations |
| Focused tests | Passed: 103 passed |
| Full pytest suite | Blocked: Windows `0xc0000139` while importing optional `torch_scatter`/`torch_sparse` in `tests/models/test_gnn_model.py` |
| Qualification generation | Passed: 45/45 successful; generation ledger SHA-256 `cd2d493d61beb08d8d22e2c00c6bd6cc85ca71d06c5e5866c17c9a26222634d1` |
| Authoritative qualification replay | Passed: 45/45, all required stages matched; replay ledger SHA-256 `daab59e401ad60f3efabdedc75717d5a43927b10f97fddf623257569a39a06dc` |
| Deterministic repeat | Passed for run 4115; all scientific artifact bytes and result hash matched |
| Verified resume | Passed for run 0 with unchanged input-tree certificate |
| Protected-science audit | Passed: no protected science paths changed from the foundation SHA |
| `git diff --check` | Passed |

## Production stop condition

The full 10,000-run production generation, full authoritative production replay,
production acceptance, RF training, and TGNN training are explicitly outside
this qualification and have not been executed.

The contract and qualification-only execution gates passed except for the
complete-suite environment gate above. Therefore the pipeline is **not**
currently declared `READY FOR 10,000-RUN PRODUCTION EXECUTION`. No production
commands are authorized by this report. After the environment gate is resolved,
the complete suite and protected-science audit must be rerun before any
production command is considered.
