# SATNET Final Integrated Dataset Class-Support Audit

## Final verdict

**NOT APPROVED FOR STAGE A CONTRACT FREEZE**

This verdict does not reject the independently reproduced class-support science. It means the current proposal is not operationally complete enough to become a frozen Stage A discovery contract. No simulation, contract freeze, model training, merge, push, or tag is authorized by this audit.

## Audit scope

This audit independently evaluated the frozen SATNET Final Integrated Production Corpus v1 and the class-support analysis at commit `340af3f22f2bc7facc4f0e749b10490f06e80cde`.

The audit covered:

- Frozen evidence identity, cardinality, byte count, manifest membership, SHA-256, and read-only state.
- Corpus identities, frozen split colocation, classification counts, and minority-class identities.
- Boundary polarity, endpoint semantics, temporal breach metrics, DOE strata, parameter associations, nearest neighbors, and regression spread.
- Stage A and Stage B arithmetic, discovery semantics, leakage controls, identities, seeds, output roots, duplicates, normalized near neighbors, and gate feasibility.
- Analysis-focused tests, independent audit tests, the complete repository suite, and protected-science isolation.

The audit did not execute a simulation or replay, alter production evidence, freeze an augmentation contract, train RF/TGNN models, merge, push, or tag.

## Audit status taxonomy

| Status | Meaning |
|---|---|
| Independently reproduced | Recomputed directly from authoritative frozen scientific evidence or proposal rows without using an analysis-produced scientific summary as input. |
| Matched by inspection only | Verified against source code, proposal language, or artifact metadata without independently recomputing a scientific result. |
| Not independently verified | Evidence was unavailable or no independent calculation was performed. |

All binding scientific findings in this report are independently reproduced. Proposal semantics and readiness findings combine independent arithmetic/identity calculations with direct inspection of the proposal contract.

## Input provenance

| Input | Verified value |
|---|---|
| Analysis branch base | `analysis/final-dataset-class-support-v1` |
| Analysis commit audited | `340af3f22f2bc7facc4f0e749b10490f06e80cde` |
| Production tooling SHA | `9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab` |
| Frozen contract tag | `final-integrated-dataset-contract-v1` |
| Frozen contract commit | `a1967185e80327e4b00c1831828dc975ab6819fc` |
| Contract specification hash | `482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b` |
| Generation ledger SHA-256 | `a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1` |
| Replay ledger SHA-256 | `4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd` |
| Freeze archive SHA-256 | `375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc` |
| Canonical analysis inventory SHA-256 | `5ac295ba0e08797b63a2ce3f062a1995dc7aa850f11a2bf5d83a0580731afd85` |
| Audit inventory SHA-256 | `20d2e7037940d708e133936d3907ad822caac3b46cd73c84329f3efd16c4037a` |

The production tooling worktree was clean at the approved SHA. The frozen tag resolved to the required contract commit. The analysis inventory and every output it binds were verified from the canonical external analysis root.

Three high-volume canonical analysis outputs are external-only: `run_level_class_support.csv`, `design_level_class_support.csv`, and `temporal_breach_summary.csv`. All tracked summaries and plots were semantically equal to their canonical external counterparts. Tracked Windows checkout bytes differ because of line-ending conversion; this is not scientific drift.

## Frozen evidence verification

The audit independently parsed both frozen manifests, enumerated actual files, compared exact path sets, checked every recorded length, and recomputed every SHA-256.

| Evidence | Files | Bytes | SHA-256 verified | Read-only |
|---|---:|---:|---:|---:|
| Generation | 8,502 | 1,336,139,056 | 8,502 | 8,502 |
| Replay | 502 | 1,410,137 | 502 | 502 |
| Combined | 9,004 | 1,337,549,193 | 9,004 | 9,004 |

The archive hash record exactly matched the independently calculated archive hash. The eight freeze-metadata files, archive, and archive hash record were read-only. Freeze-internal SHA-256 records passed. No evidence mutation was observed.

**Result: independently reproduced — passed.**

## Independent corpus reproduction

The audit loaded each authoritative `design_record.json`, `run_record.json`, `target.json`, G5 step sequence, and G5 run summary directly from the generation root.

| Identity property | Result |
|---|---:|
| Runs | 500 |
| Designs | 100 |
| Realizations per design | 5 |
| Run IDs | Exactly 0 through 499 |
| Unique run keys | 500 |
| Unique design-realization pairs | 500 |
| Train runs/designs | 350 / 70 |
| Validation runs/designs | 75 / 15 |
| Test runs/designs | 75 / 15 |
| Colocated realizations | All 100 designs |

No missing, extra, duplicate, or non-colocated identity was found.

**Result: independently reproduced — passed.**

## Classification and non-breach reproduction

| Split | Non-breach | Breach | Total |
|---|---:|---:|---:|
| Train | 2 | 348 | 350 |
| Validation | 0 | 75 | 75 |
| Test | 5 | 70 | 75 |
| Total | 7 | 493 | 500 |

The only non-breach run IDs are `0, 1, 2, 3, 4, 5, 7`.

- D000 contains runs 0–4, is assigned to test, and is 5/5 non-breach.
- D001 contains non-breach runs 5 and 7, is assigned to train, and is 2/5 non-breach.
- Validation contains no non-breach run or design.
- The other 98 designs are 0/5 non-breach.

**Result: independently reproduced — passed.**

## Boundary audit

The production target implementation uses binary64 comparisons. A step meets its threshold when its service value is greater than or equal to `0.80`; a run breaches when at least one step is strictly below `0.80`.

The independently computed signed margin is:

```text
failure_adjusted_overall_service_fraction_min - 0.80
```

All 500 labels satisfy:

```text
margin >= 0  iff overall_threshold_breach_any is false
margin < 0   iff overall_threshold_breach_any is true
```

Runs 5 and 7 have binary64 margin exactly `0.0` and are non-breach. Equality is therefore inclusive threshold compliance.

| Finding | Independently reproduced value |
|---|---:|
| D000 margins | Five values of `0.1499999999999999` |
| D001 margins | `0.0`, `-0.050000000000000044`, `0.0`, `-0.10000000000000009`, `-0.15000000000000002` |
| Nearest breached run | Run 6, D001 |
| Nearest breached margin | `-0.050000000000000044` |
| Runs below `-0.20` | 480 |
| Design means below `-0.20` | 96 |
| Runs within inclusive ±0.01 | 2 |
| Runs within inclusive ±0.025 | 2 |
| Runs within inclusive ±0.05 | 2 |
| Runs within inclusive ±0.10 | 3 |

“Below `-0.20`” is exclusive. Symmetric windows are inclusive at both endpoints.

**Result: independently reproduced — passed.**

## Temporal audit

Every run contains 11 persisted authoritative G5 states in exact timestep order 0 through 10 with strictly increasing timestamps. No temporal metric was reconstructed from aggregate targets alone.

For every run, the audit independently computed:

- Threshold-breach sequence.
- Breach count and fraction.
- First and last breach timestep.
- Longest breach streak.
- Recovery count.
- Earliest minimum-service timestep.

The 5,500 step flags agree with the binary64 threshold comparison and the persisted run/target summaries.

| DOE stratum | Exact mean temporal breach fraction |
|---|---:|
| Pilot anchor | `0.6472727272727273` |
| Transition | `0.9677922077922079` |
| Global | `0.9893939393939394` |

The report’s rounded values of approximately `0.968` and `0.989` are correct.

D001 reproduces breach counts `0, 2, 0, 3, 8`; longest streaks `0, 2, 0, 2, 4`; and recovery counts `0, 0, 0, 1, 3` for runs 5–9.

**Result: independently reproduced — passed.**

## DOE stratum audit

The frozen design IDs and observed stratum labels agree exactly:

- D000–D004: 5 pilot anchors.
- D005–D039: 35 transition designs.
- D040–D099: 60 global designs.

Transition and global contain zero non-breach runs.

**Result: independently reproduced — passed.**

## Parameter-support audit

The audit independently computed run-level Spearman correlations using average ranks for ties and Pearson correlation of ranks. No normalization was applied before correlation. All 500 runs were used, and missing values would fail closed; none were observed.

| Parameter | Exact Spearman correlation with signed margin |
|---|---:|
| `altitude_km` | `0.6627247027078719` |
| `sats_per_plane` | `0.5338963974657238` |
| `configured_satellite_count` | `0.39215337541491824` |
| `inclination_deg` | `0.26015529260699544` |

The reported rounded values are correct.

Observed support must be stated narrowly:

- The seven non-breach runs have 6 planes, 8 satellites per plane, and 48 configured satellites.
- D000 has 1200 km, 98°, and zero node/edge/ground failure probabilities.
- D001 has 800 km, 60°, and `0.05` node/edge/ground failure probabilities.
- “800–1200 km,” “60–98°,” and “0–0.05” are intervals containing two designs, not densely observed resilient ranges.

The analysis appropriately labels the correlations exploratory and confounded. One phrase, “well inside the observed resilient region,” should be narrowed to “observed resilient anchor” or “candidate region.”

**Result: independently reproduced — passed with one nonbinding wording observation.**

## Nearest-neighbor audit

The audit independently used Euclidean distance over 11 full-DOE-range min-max-normalized variables:

```text
num_planes
sats_per_plane
altitude_km
inclination_deg
satellite_node_failure_probability
satellite_edge_failure_probability
total_ground_station_count
ground_station_failure_probability
civilian_fraction
government_fraction
military_fraction
```

No categorical variables were included. Fixed parameters were excluded. Failure probabilities were included. Every declared range is nonzero. Distance ties are ordered by ascending design index.

| Finding | Exact value |
|---|---:|
| D000 nearest design | D001 |
| D000→D001 distance | `0.8294212447374086` |
| D001 nearest design | D022 |
| D001→D022 distance | `0.3932311802307627` |
| Non-breach support among D001’s eight nearest neighbors | 0 designs |

The conclusion “no observed resilient local cluster” is justified under this declared metric. It remains descriptive and metric-dependent.

**Result: independently reproduced — passed.**

## Regression-only audit

The primary regression target is `failure_adjusted_overall_service_fraction_mean`.

| Split | Mean | Population standard deviation | Unique values |
|---|---:|---:|---:|
| Train | `0.2339977272727273` | `0.2484489820182131` | 237 |
| Validation | `0.1881074555074555` | `0.2447719355573115` | 51 |
| Test | `0.23218085618085618` | `0.3110362546438848` | 43 |

All four regression targets have nonzero population standard deviation in every split. The evidence supports reviewing a separately predeclared regression-only contract. It does not approve the corpus for regression and does not change the production verdict.

**Result: independently reproduced — passed.**

## Stage A review

### Arithmetic

| Region | Development/train | Validation | Internal holdout | Total |
|---|---:|---:|---:|---:|
| Resilient core | 8 | 2 | 2 | 12 |
| Boundary | 8 | 2 | 2 | 12 |
| Global control | 4 | 1 | 1 | 6 |
| Total designs | 20 | 5 | 5 | 30 |
| Total runs | 100 | 25 | 25 | 150 |

All arithmetic is correct at five realizations per design.

### Discovery semantics

The selected recommendation is **Approach B: discovery plus internal holdout**.

This is scientifically acceptable because the proposal explicitly prohibits using Stage A holdout outcomes to design Stage B. Development and validation can localize candidate regions; the five-design internal holdout provides an outcome-independent check at the cost of reduced discovery coverage.

The holdout should not be called an untouched final evaluation set until a future final combined-corpus contract declares its role. If any holdout outcome influences Stage B ranges, density, sample size, or gates, the holdout claim is invalid and a new holdout is required.

### Proposed Stage A stop/go criteria

A future Stage A contract should predeclare these feasible criteria for the 25 development-plus-validation designs:

1. **Operational:** exactly 30 designs and 150 fixed-identity runs generate and replay with no substitutions or retries under new identities.
2. **Core reproduction:** at least two distinct development/validation core designs each produce at least 3/5 non-breach realizations.
3. **Boundary mixing:** at least three distinct development/validation boundary designs are individually mixed, each with at least one breach and one non-breach realization.
4. **Transition localization:** at least four distinct boundary designs have at least one run within inclusive ±0.10; both signs occur across at least two distinct designs per sign.
5. **Independent support:** development/validation contain at least four non-breach-supporting designs and four breach-supporting designs. Mixed designs may support both only if that counting rule is explicitly declared.
6. **Realization stability:** report every design’s 0–5 non-breach count and margin spread; pooled run counts cannot replace design support.
7. **Control consistency:** at least four of five development/validation controls remain 0/5 non-breach; otherwise investigate population shift before Stage B.
8. **Go:** all operational, core, boundary, transition, independent-support, and control criteria pass without consulting the holdout.
9. **Revise:** core support exists but transition localization or control consistency fails; create a new Stage B proposal and contract without consulting the holdout.
10. **Stop:** core reproduction or independent support fails; do not freeze Stage B.

### Stage A readiness

The current proposal has no exact 30-design Stage A table, Stage A parameter bounds, Stage A design/run namespace, seed manifest, or explicit Stage A output-root manifest. Its existing boundary criterion can pass without one individually mixed design.

**Result: arithmetic passed; discovery concept is sound; current proposal is not freeze-ready.**

## Stage B review

| Region | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Resilient core | 24 | 6 | 6 | 36 |
| Boundary | 24 | 6 | 6 | 36 |
| Global control | 12 | 3 | 3 | 18 |
| Total designs | 60 | 15 | 15 | 90 |
| Total runs | 300 | 75 | 75 | 450 |

All arithmetic and five-realization grouping are correct. All 90 proposal rows are labeled `PROPOSAL ONLY — NOT FROZEN — DO NOT SIMULATE`; `simulation_authorized` is false and `seed_status` is `NOT_ISSUED`.

Stage B must remain `NOT_FROZEN` until Stage A development/validation results are reviewed and a new Stage B contract, manifests, seeds, and audit are completed. The current deterministic Stage B table is planning metadata, not a contract.

**Result: independently reproduced arithmetic and identities passed; Stage B is not freeze-ready.**

## Stage A and Stage B separation

No silent adaptation is permitted within one contract. Stage A findings may influence future Stage B region definitions, parameter bounds, design density, sample size, and gates only through:

1. A new Stage B proposal.
2. New frozen design/run manifests.
3. New domain-separated seeds.
4. A new independent audit.
5. Explicit authorization after that audit.

The proposal does not state whether Stage A evidence later joins the final corpus alongside Stage B or remains discovery-only. This affects final split cardinalities and gate interpretation and is a binding ambiguity.

## Leakage audit

The proposal correctly prohibits outcome-driven partition assignment, renaming, movement, seed choice, run replacement, and substitution. It preserves original design grouping and split assignments.

### Stage B duplicate and proximity results

| Check | Result |
|---|---:|
| Unique Stage B design identities | 90 |
| Duplicate proposed scientific vectors | 0 |
| Exact duplicates of original designs | 0 |
| New-new minimum cross-split distance | `0.13460738729509497` |
| New-original minimum distance, any split | `0.19308050168339316` |
| New-original minimum cross-split distance | `0.22042999245042424` |
| Cross-split pairs within inclusive radius `0.10` | 0 |

Definitions used:

- **Exact duplicate:** the same design identity occurs more than once.
- **Parameter-vector duplicate:** all comparable fixed and variable scientific fields are equal after numeric normalization.
- **Normalized near neighbor:** Euclidean distance in the declared 11-variable normalized space is at most `0.10`.
- **Cross-split near neighbor:** a normalized near-neighbor pair assigned to different splits.

The current Stage B table passes the `0.10` rule. A pre-freeze audit should repeat the calculation and report sensitivity to the chosen radius. Stage A cannot be audited for duplicates or proximity because no Stage A table exists.

## Original split preservation

The proposal does not move, relabel, delete, or replace any original design or run. Original train remains train, validation remains validation, and test remains test. D000 remains test and D001 remains train. The original seeds, targets, and evidence roots remain unchanged.

**Result: matched by inspection and identity comparison — passed.**

## Identity audit

Stage B uses `AUGV1-D000` through `AUGV1-D089` and `AUGV1-Dxxx-R00..R04`, which are disjoint from original `D000`–`D099` and integer run IDs 0–499.

Stage A has no identity namespace. Because Stage B already occupies the `AUGV1` range, a future Stage A contract must declare a distinct namespace, such as:

```text
Stage A designs: STAGEA-D000 through STAGEA-D029
Stage A runs:    STAGEA-D000-R00 through STAGEA-D029-R04
```

The exact prefix is not binding; unambiguous corpus separation is.

**Result: Stage B passed; Stage A identity isolation failed readiness review.**

## Seed-policy audit

No production seeds are issued by the proposal. This is correct for a non-frozen planning proposal.

A future Stage A contract must freeze deterministic, domain-separated derivation for:

- Design generation.
- Satellite rollout and failure realization.
- Ground selection.
- Ground failure realization.

Stable design and realization identities must determine seeds. A same-run retry retains the same identity and seed. Outcome-driven retries, seed replacement, and failed-run substitution are prohibited.

**Result: current non-issuance verified; future derivation policy remains to be frozen.**

## Output-root policy

Stage A must use new roots outside production, replay, freeze, analysis, and audit roots. Recommended future names are:

```text
C:\Users\johns\satnet-stage-a-discovery-production-<date>
C:\Users\johns\satnet-stage-a-discovery-replay-<date>
C:\Users\johns\satnet-stage-a-discovery-acceptance-<date>
```

These roots were not created by this audit.

## Classification-gate feasibility

The proposal says the gates apply to each “combined fixed-original-plus-augmentation split,” but it does not state whether augmentation means Stage B only or Stage A plus Stage B.

### Candidate final sizes

| Scope | Train designs/runs | Validation designs/runs | Test designs/runs |
|---|---:|---:|---:|
| Original only | 70 / 350 | 15 / 75 | 15 / 75 |
| Stage A only | 20 / 100 | 5 / 25 | 5 / 25 |
| Stage B only | 60 / 300 | 15 / 75 | 15 / 75 |
| Original + Stage A | 90 / 450 | 20 / 100 | 20 / 100 |
| Original + Stage B | 130 / 650 | 30 / 150 | 30 / 150 |
| Original + Stage A + Stage B | 150 / 750 | 35 / 175 | 35 / 175 |

The proposed final gates are mathematically feasible for Stage B augmentation alone, original plus Stage B, and original plus Stage A plus Stage B. They are intentionally infeasible for Stage A augmentation alone and therefore must not be used as Stage A discovery gates.

The tightest Stage B-only holdout check is:

```text
14 minimum non-breach runs + 60 minimum breach runs = 74 required
75 runs physically available
```

The corresponding design check is:

```text
4 minimum non-breach designs + 10 minimum breach designs = 14 required
15 designs physically available
```

No proposed final run count, design count, boundary-design count, distinct-margin count, or class-ratio limit is mathematically contradictory under either candidate combined final scope.

Before freeze, the contract must define:

- Whether a “boundary design” means preassigned DOE-region membership or observed margin behavior.
- Whether distinct margins use exact canonical binary64 equality or a predeclared tolerance/quantization rule.
- Whether one mixed design may satisfy both breach-design and non-breach-design minima.
- Whether Stage A is retained in the final corpus.

**Result: arithmetic feasible; operational definitions and final scope are bindingly incomplete.**

## Scientific interpretation audit

The analysis generally uses appropriate exploratory language and explicitly rejects causal interpretation. The following statements are supported:

- Altitude and constellation size are associated with margin in this DOE.
- Non-breach was observed only in D000 and D001.
- D000 is an observed resilient anchor.
- D001 is an observed exact-boundary, realization-sensitive anchor.
- The proposed regions require confirmation.

The evidence does not support claims that altitude causes resilience, larger constellations guarantee non-breach, D000 defines a stable region, the boundary region has been identified, or augmentation will achieve class balance.

## Binding findings

### B-001 — Stage A scientific design contract is incomplete

The proposal has allocation arithmetic but no exact Stage A design table, admissible bounds, identity manifest, or output-root manifest.

### B-002 — Stage A identity namespace is undefined

Stage B already occupies `AUGV1-D000`–`AUGV1-D089`; Stage A has no disjoint design/run namespace.

### B-003 — Final combined-corpus membership is ambiguous

The proposal does not declare whether Stage A joins the final corpus or remains discovery-only.

### B-004 — Stage A boundary discovery criterion is not design-level

The collective-sign criterion can pass with no individually mixed design and does not localize a realization-sensitive transition.

### B-005 — Final classification gate definitions are incomplete

Boundary-design membership, distinct-margin equality, and mixed-design class counting are undefined.

## Nonbinding findings

### N-001 — Stage A internal holdout terminology

Use “internal holdout” until a final combined-corpus contract defines its role.

### N-002 — Single-design regional wording

Replace “well inside the observed resilient region” with “observed resilient anchor” or “candidate region.”

### N-003 — Near-neighbor radius sensitivity

Repeat the `0.10` calculation and report sensitivity before freeze.

## Required corrections

Before a Stage A freeze can be approved:

1. Create an exact 30-design Stage A table with explicit admissible ranges and preassigned development, validation, and internal-holdout blocks.
2. Define disjoint Stage A design/run namespaces.
3. Replace collective boundary-sign criteria with design-level mixed-realization and near-boundary criteria.
4. Declare whether Stage A evidence enters the final combined corpus.
5. Define all final gate counting and numeric-equality semantics.
6. Freeze deterministic domain-separated seed derivation without issuing outcome-driven replacements.
7. Declare new Stage A production, replay, and acceptance roots.
8. Recompute exact duplicates and normalized cross-split proximity on the final Stage A table.
9. Conduct a separate pre-freeze audit.

## Validation results

| Validation | Result |
|---|---|
| Independent audit tests | 8 passed |
| Original focused class-support tests | 8 passed |
| Isolation/timer discrepancy rerun | 16 passed after audit placement correction and stale analysis-surface allowlist correction |
| Complete repository suite | 925 passed |
| Compilation | Passed |
| Protected tracked science diff | Empty |
| Protected untracked science status | Empty |
| Whitespace check | Passed |

The initial complete-suite run produced 922 passes and three failures. Two failures came from audit module placement and a historical isolation allowlist that omitted the already-audited class-support analysis surfaces. The module was moved into an isolation-safe subpackage and the allowlist was extended only for analysis/audit paths. The third failure was a timer test that passed immediately in isolation and did not involve audit code. The corrected complete-suite rerun passed all 925 tests.

## Audit outputs

Canonical external audit outputs are under `C:\Users\johns\satnet-class-support-audit-20260721`. Tracked copies are under `artifacts/final_integrated_dataset_class_support_audit`.

The audit inventory binds:

- `audit_findings.json`
- `audit_reproduction_summary.json`
- `audit_boundary_reproduction.csv`
- `audit_non_breach_reproduction.csv`
- `audit_stage_a_review.json`
- `audit_stage_b_review.json`
- `audit_gate_feasibility.csv`
- `audit_identity_collision_report.csv`
- `audit_near_neighbor_leakage.csv`

## Final authorization boundary

This audit authorizes none of the following:

- Stage A simulation.
- Stage B simulation.
- Augmentation contract freeze.
- RF training.
- TGNN training.
- Merge.
- Push.
- Tag.

The next permissible task is correction and review of an exact Stage A discovery contract proposal, followed by a separate freeze-readiness audit.
