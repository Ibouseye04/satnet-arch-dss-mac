# SATNET Final Integrated Dataset Class-Support Analysis

## Executive summary

The frozen SATNET Final Integrated Production Corpus v1 remains scientifically reproducible but is not production accepted. Read-only verification matched all 9,004 evidence files, 1,337,549,193 bytes, both authoritative ledgers, the frozen contract identity, the approved production tooling identity, and the freeze archive identity.

The classification failure is not merely a split accident. Only seven of 500 runs are non-breach, and those seven arise from only two of 100 designs. The validation split contains no non-breach run or design. One design, D000, is consistently resilient at 5/5 non-breach realizations with margin +0.15. D001 is realization-sensitive at 2/5 non-breach; its two non-breach realizations lie exactly on the boundary with margin 0.00. The other 98 designs are consistently breached.

The corpus is overwhelmingly distant from the decision boundary: 480 runs have margin below -0.20, while only two runs lie within ±0.05 and only three lie within ±0.10. The existing DOE therefore does not provide adequate independent design support for classification or boundary calibration.

A staged augmentation is recommended. Stage A should be a separately frozen 30-design, 150-run discovery contract. If its predeclared discovery criteria pass, Stage B should use a separately reviewed and frozen 90-design, 450-run augmentation with 36 resilient-core designs, 36 boundary designs, and 18 global controls. No simulation is authorized by this analysis.

## Frozen evidence provenance

| Identity | Frozen value |
|---|---|
| Approved production tooling SHA | `9ba5ea65ed718a9c50c9af776b6bcf978f9ba5ab` |
| Frozen contract tag | `final-integrated-dataset-contract-v1` |
| Frozen contract commit | `a1967185e80327e4b00c1831828dc975ab6819fc` |
| Contract specification hash | `482935e13017dc55cfbfcf2ba79ae50c09dfcffe69762806cc5448273406498b` |
| Generation ledger SHA-256 | `a887a9bad660945a3585369b2652511d4c9030cfdcbfbb09decb4721def15cb1` |
| Replay ledger SHA-256 | `4e15b33545a1bee63298a1597b8effab2f4ba8d26730b75b53a90ee132e501dd` |
| Freeze archive SHA-256 | `375e181e1a21a71386a3bad3c0f51de450203eb53d2a01853b517b5a86fa50cc` |

The preflight independently verified 8,502 generation files totaling 1,336,139,056 bytes and 502 replay files totaling 1,410,137 bytes against the frozen SHA-256 manifests. No evidence file was written, normalized, reserialized, moved, or made writable.

## Production acceptance failure

The frozen outcome was reproduced exactly:

| Split | Non-breach | Breach | Runs | Designs |
|---|---:|---:|---:|---:|
| Train | 2 | 348 | 350 | 70 |
| Validation | 0 | 75 | 75 | 15 |
| Test | 5 | 70 | 75 | 15 |
| Total | 7 | 493 | 500 | 100 |

The technical split failure is that validation contains zero non-breach outcomes. The underlying scientific support failure is more fundamental: only seven runs and two designs have any non-breach support globally.

Moving an observed non-breach design into validation after observing outcomes would be outcome-driven reshuffling. That is not an acceptable repair. The original design assignments remain fixed.

## Seven non-breach runs

| Run IDs | Design | Split | DOE stratum | Realizations | Architecture | Altitude / inclination | Node / edge / ground failure probabilities | Stations | Margin | Failure realization |
|---|---|---|---|---|---|---|---|---|---:|---|
| 0–4 | D000 | Test | Pilot anchor | R00–R04 | 6 planes × 8 satellites | 1200 km / 98° | 0 / 0 / 0 | 20 (8/6/6) | +0.15 | No failed ground stations, satellite nodes, or satellite edges |
| 5 | D001 | Train | Pilot anchor | R00 | 6 planes × 8 satellites | 800 km / 60° | 0.05 / 0.05 / 0.05 | 20 (12/4/4) | 0.00 | 0 ground stations, 3 satellite nodes, and 2 satellite edges failed |
| 7 | D001 | Train | Pilot anchor | R02 | 6 planes × 8 satellites | 800 km / 60° | 0.05 / 0.05 / 0.05 | 20 (12/4/4) | 0.00 | 1 ground station, 1 satellite node, and 5 satellite edges failed |

D000 is the only consistently resilient design. D001 is mixed and realization-sensitive. No design produced three or four non-breach realizations. No transition or global design produced a non-breach realization.

The seven-run sample is too small and too design-concentrated for a causal claim, validated predictive claim, or production feature-importance interpretation.

## Non-breach design analysis

The 100 design labels are:

| Descriptive class | Definition | Count |
|---|---|---:|
| Consistently resilient | 5/5 non-breach | 1 |
| Mostly resilient | 3–4/5 non-breach | 0 |
| Mixed | 1–2/5 non-breach | 1 |
| Consistently breached | 0/5 non-breach | 98 |

D000 has minimum, mean, and maximum margin +0.15. D001 has realization margins 0.00, -0.05, 0.00, -0.10, and -0.15, with mean margin -0.06. The two designs therefore represent different regimes: an isolated robust anchor and a stochastic transition anchor.

## Threshold-boundary analysis

The signed margin is the authoritative overall minimum minus 0.80. All 500 target labels agree with the margin polarity and all 5,500 G5 sampled states agree with the persisted threshold flags.

| Margin band | Runs | Designs by mean margin |
|---|---:|---:|
| Less than -0.20 | 480 | 96 |
| [-0.20, -0.10) | 12 | 2 |
| [-0.10, -0.05) | 1 | 1 |
| [-0.05, -0.025) | 0 | 0 |
| [-0.025, -0.01) | 0 | 0 |
| [-0.01, 0) | 0 | 0 |
| [0, 0.01] | 2 | 0 |
| (0.01, 0.025] | 0 | 0 |
| (0.025, 0.05] | 0 | 0 |
| (0.05, 0.10] | 0 | 0 |
| Greater than 0.10 | 5 | 1 |

Symmetric boundary windows contain two runs within ±0.01, ±0.025, and ±0.05, and three runs within ±0.10. Only one design has a best realization within each window. No design mean is within ±0.05; D001 is the only design mean within ±0.10.

The nearest breached run is run 6 from D001 at margin -0.05. The five D000 runs are not boundary cases; they are well inside the observed resilient region at +0.15. The two D001 non-breach runs are exactly on the inclusive boundary and are isolated stochastic outcomes within a mixed design.

## Temporal breach analysis

Every run contains 11 ordered authoritative G5 sampled states.

D000 has zero breach timesteps in every realization. D001 exhibits increasing degradation across realizations:

| Run | Margin | Breach timesteps | Breach fraction | Longest streak | Recoveries |
|---|---:|---:|---:|---:|---:|
| 5 | 0.00 | 0 | 0.000 | 0 | 0 |
| 6 | -0.05 | 2 | 0.182 | 2 | 0 |
| 7 | 0.00 | 0 | 0.000 | 0 | 0 |
| 8 | -0.10 | 3 | 0.273 | 2 | 1 |
| 9 | -0.15 | 8 | 0.727 | 4 | 3 |

The transition and global strata are not merely affected by isolated one-step crossings. Their mean temporal breach fractions are 0.968 and 0.989 respectively, compared with 0.647 across the five pilot anchors. This is consistent with broad strong-breach support, not a threshold-label artifact.

## Parameter-support findings

Descriptive run-level Spearman associations with boundary margin are largest for altitude (`+0.663`), satellites per plane (`+0.534`), configured satellite count (`+0.392`), and inclination (`+0.260`). Non-breach runs occur only with six planes, eight satellites per plane, 48 configured satellites, altitude 800–1200 km, inclination 60–98°, and node, edge, and ground failure probabilities between 0 and 0.05.

These associations are exploratory and confounded by the DOE. In particular, the weak marginal run-level correlations for node and edge failure probabilities do not establish weak physical effects; architecture, geometry, fixed grid ISL behavior, ground composition, and DOE pairing vary jointly. Direct support shows only that all observed non-breach outcomes occupy the high-capacity, low-failure corner represented by two pilot anchors.

The DOE produced only seven non-breach runs because:

1. Only five designs were pilot anchors, and only D000 and D001 reached the non-breach side.
2. None of the 35 transition or 60 global designs produced a non-breach realization.
3. Transition and global runs have median margins -0.744 and -0.778 and breach in almost all sampled states.
4. The resilient core is represented by one zero-failure 48-satellite design, while the only boundary support is one mixed 48-satellite design.
5. The 100-design DOE emphasized broad admissible coverage, not replicated independent support around the empirically narrow resilient and boundary regions.

This diagnosis is descriptive. It does not establish causality or justify changing protected physics, targets, or thresholds.

## Nearest-neighbor findings

The distance metric is Euclidean distance over 11 DOE-range-normalized variables: planes, satellites per plane, altitude, inclination, satellite node and edge failure probabilities, total ground stations, ground failure probability, and the three ground-composition fractions.

D000's nearest design is D001 at distance 0.829. Its nearest consistently breached design is D008 at distance 0.982. D001's nearest design is D022 at distance 0.393; D001's eight nearest neighbors are all 0/5 non-breach, with mean margins ranging from -0.29 to -0.80.

There is no observed resilient local cluster. D000 is an isolated resilient point. D001 is an isolated boundary transition point surrounded by breached designs. The large normalized gap between D000 and D001 and the lack of intermediate resilient designs make a discovery stage scientifically preferable to immediately freezing a large targeted augmentation.

## Split diagnosis

The seven non-breach runs are assigned as follows:

- Train: runs 5 and 7, both from D001.
- Validation: none.
- Test: runs 0–4, all from D000.

Only one non-breach-supporting design appears in train and one in test. Validation has none. This is both a technical split failure and an underlying global support failure. The appropriate repair is additive, preassigned, design-grouped augmentation, not movement or relabeling of existing evidence.

## Regression-only assessment

The frozen corpus passed the original regression gates. The primary regression mean, population standard deviation, and unique-value counts are:

| Split | Mean | Population standard deviation | Unique values |
|---|---:|---:|---:|
| Train | 0.233998 | 0.248449 | 237 |
| Validation | 0.188107 | 0.244772 | 51 |
| Test | 0.232181 | 0.311036 | 43 |

For the overall-minimum target, unique-value counts are 54, 24, and 14 in train, validation, and test. Ground-minimum unique counts are 53, 24, and 14. Space-minimum unique counts are 95, 28, and 26. All four targets have nonzero spread in every split.

This evidence supports review of a separately specified regression-only acceptance contract. It does not retroactively approve the corpus for regression and does not alter the failed production verdict.

## Augmentation-size comparison

All options retain the original 100 designs and frozen split assignments. Each new design has five grouped realizations and a new augmentation identity namespace.

| New designs | New runs | Core / boundary / control | Train / validation / test designs | Scenario non-breach run range | Estimated generation storage | Estimated generation / replay time |
|---:|---:|---|---|---|---:|---|
| 60 | 300 | 24 / 24 / 12 | 40 / 10 / 10 | 30–135 | 0.80 GB | 14.4 / 8.8 min |
| 90 | 450 | 36 / 36 / 18 | 60 / 15 / 15 | 45–202 | 1.20 GB | 21.6 / 13.1 min |
| 120 | 600 | 48 / 48 / 24 | 80 / 20 / 20 | 60–270 | 1.60 GB | 28.8 / 17.5 min |

The scenario ranges are planning scenarios, not confidence intervals, predictions, or guarantees. Pre-simulation design allocation can be guaranteed; post-simulation class support cannot.

The 60-design option may still provide weak independent holdout support. The 120-design option has the greatest coverage but risks overinvestment before the resilient region is localized. The 90-design option is the recommended Stage B scale because it provides six core and six boundary designs in each holdout split plus three global controls, while retaining manageable cost.

## Recommended staged augmentation strategy

### Stage A: discovery proposal

Stage A requires a separately reviewed and frozen contract before any simulation.

| Region | Train | Validation | Test | Total designs |
|---|---:|---:|---:|---:|
| Resilient core | 8 | 2 | 2 | 12 |
| Boundary | 8 | 2 | 2 | 12 |
| Global control | 4 | 1 | 1 | 6 |
| Total | 20 | 5 | 5 | 30 |

Stage A uses 30 designs, five realizations each, and 150 runs. Its proposed acceptance criteria are complete generation and replay, at least two resilient-core designs with at least three non-breach realizations, and at least four boundary designs collectively sampling both margin signs. The Stage A test partition remains sealed. Only Stage A train and validation findings may inform Stage B DOE refinement.

No silent adaptive change is permitted within Stage A. Any Stage B design, seed, split, or gate change requires a new reviewed and frozen contract.

### Stage B: recommended 90-design proposal

| Region | Train | Validation | Test | Total designs |
|---|---:|---:|---:|---:|
| Resilient core | 24 | 6 | 6 | 36 |
| Boundary | 24 | 6 | 6 | 36 |
| Global control | 12 | 3 | 3 | 18 |
| Total | 60 | 15 | 15 | 90 |

The resilient-core proposal spans high-capacity, low-failure combinations supported by D000 and bounded toward D001. The boundary proposal spans D001-like transition combinations and nearby architecture/failure gradients. The global control proposal covers the original admissible ranges to quantify targeted-population shift.

The proposal uses `AUGV1-D000` through `AUGV1-D089` and separate `AUGV1-Dxxx-R00..R04` run identities. No seeds or final run manifests are issued. All rows are labeled `PROPOSAL ONLY — NOT FROZEN — DO NOT SIMULATE`.

## Proposed stronger classification gates

The following gates are proposed for the combined fixed-original-plus-augmentation corpus. They require future review before freezing.

| Gate | Train | Validation | Test | Rationale |
|---|---:|---:|---:|---|
| Minimum non-breach designs | 12 | 4 | 4 | Prevent one design's repeated realizations from satisfying class support |
| Minimum non-breach runs | 50 | 14 | 14 | Provide repeated support while remaining subordinate to design-level support |
| Minimum breach designs | 40 | 10 | 10 | Preserve broad breach support after targeted augmentation |
| Minimum breach runs | 200 | 60 | 60 | Retain the majority scientific regime in every split |
| Maximum majority/minority run ratio | 12:1 | 10:1 | 10:1 | Reject technically present but scientifically unusable minority support |
| Minimum boundary-region designs | 10 | 3 | 3 | Support calibration around the 0.80 transition |
| Minimum distinct run-margin values | 30 | 12 | 12 | Reject margin collapse or discretization saturation |

Independent design support is binding. Run counts from a single design cannot compensate for a failed design-level gate. The holdout minimum of four non-breach designs is intentionally greater than the entire current corpus support of two designs and prevents a repeat of the present one-design split dependence.

## Leakage controls

The proposal requires grouped splitting by design, no design overlap, no movement of original designs, no outcome-driven reshuffling, no seed substitution, no replacement runs, no tuning against final test, and exact scientific-parameter deduplication.

Geometric proximity can create design-level leakage even without exact duplication. The proposed control is to normalize the same 11 DOE variables used in the neighborhood analysis, build a graph joining designs within a predeclared Euclidean radius of 0.10, and place each connected component wholly in one split. The current Stage B table has preliminary minimum cross-split distance 0.1346 and zero cross-split pairs within 0.10. This is a preliminary pass only and must be independently repeated before contract freeze.

## Limitations

- Seven non-breach runs from two designs do not support causal inference.
- D000 has zero configured failures, so its robustness does not establish robustness under nonzero failure rates.
- D001's two non-breach outcomes are exactly on the inclusive boundary and are realization-sensitive.
- The fixed DOE couples architecture, geometry, failure probabilities, and ground composition; marginal correlations are confounded.
- The proposed design table is deterministic planning metadata, not a frozen scientific contract.
- Scenario class-support ranges are not probabilistic guarantees.
- No production Random Forest, TGNN, or exploratory predictive model was trained.

## Reproducibility

The analysis reads authoritative run inputs, target artifacts, G5 realization/run/step evidence, and satellite failure evidence. It does not invoke a simulation, replay, RF, or TGNN entrypoint. Output roots inside generation, replay, or freeze evidence are rejected.

Machine-readable outputs use deterministic row and column ordering, UTF-8, explicit numeric formatting, and root-relative inventory paths. `analysis_inventory.json` excludes itself to avoid a self-referential hash cycle and binds all other final outputs to byte lengths, SHA-256 values, record counts where applicable, and schema identifiers.

## Next contract-freeze steps

1. Review the Stage A scientific ranges, exact design table, class-support gates, and normalized-distance leakage radius.
2. Create a separate Stage A contract with domain-separated frozen seeds, run identities, split manifest, and new output roots.
3. Perform an independent contract and protected-science audit before any Stage A simulation.
4. Execute Stage A generation and authoritative replay only after explicit authorization.
5. Use only Stage A train and validation evidence to formulate a separate Stage B proposal; keep Stage A test sealed.
6. Review and freeze Stage B independently. Do not silently adapt Stage A or reuse production evidence roots.

## Review verdict

The read-only analysis and proposal are complete and reproducible. The evidence supports review of a staged augmentation contract, but does not authorize simulation or claim production acceptance.

**READY FOR AUGMENTATION-CONTRACT REVIEW**
