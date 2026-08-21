# SATNET DSS v1 Release Audit — Phase 4

**Audit scope:** final product, scientific sanity, traceability, and release readiness. No product capability, model, checkpoint, or qualified science artifact was changed.

## Operational identity

- Operational model: frozen TGNN space regression (`tgnn_space_regression`, seed 42).
- Target: `space_gcc_fraction_original_min`.
- Checkpoint: `C:\Users\johns\external\satnet-10k-model-training-v1\final_robustness\tgnn_space_regression\seed_42\best_validation_checkpoint.pt`.
- Checkpoint SHA-256: `22cabef076428ba5c118b10fa230c5930af1b1c2da53bdd5c51b118dc0c9960a`.
- Inference remains CPU, evaluation-only, raw, and unclipped. No training, tuning, threshold fitting, or model selection was performed.
- Accepted export contract: space-segment-only TGNN; 3 node features and 4 edge features. The committed 14-node/10-edge integrated schema is explicitly future/not authorized and is not used by this DSS.

## Five-realization definition

Each analysis derives five deterministic satellite-failure realization seeds from the canonical physical architecture. The threshold is excluded from seed derivation. Each realization is a complete temporal SATNET rollout with 11 inclusive snapshots (`t=0..10`), followed by one frozen TGNN prediction. The primary result is:

> **Expected Minimum Connectivity = arithmetic mean of the five TGNN-predicted minimum original-denominator GCC values.**

The five-realization meeting count is descriptive engineering scenario context, not a probability, likelihood, confidence, or calibrated success rate.

## Scientific deployment sanity check

Configuration: 5 planes, 7 satellites per plane, 550 km, 53 degrees, 0.10 satellite-node failure, 0.12 satellite-edge failure, 10-minute duration, 60-second steps, and the frozen J2000 epoch/profile. The direct values below were read from the authoritative SATNET rollout summaries/steps for the same generated scenarios; they were not supplied to the TGNN.

| Realization | TGNN predicted min GCC | SATNET actual min GCC | Absolute error |
|---:|---:|---:|---:|
| 1 | 0.058312 | 0.057143 | 0.001169 |
| 2 | 0.068401 | 0.057143 | 0.011258 |
| 3 | 0.070415 | 0.057143 | 0.013272 |
| 4 | 0.075604 | 0.057143 | 0.018461 |
| 5 | 0.067640 | 0.057143 | 0.010497 |
| **Mean** | **0.068074** | **0.057143** | — |

- Five-case deployment-sanity MAE: **0.010931**.
- The direct SATNET minimum was 0.057143 in all five cases; the TGNN mean was 0.068074. Both are very low, so the low GUI result is scientifically consistent with this generated topology/failure scenario. No adapter discrepancy requiring investigation was found.
- The audit was not added to held-out performance statistics and is not a new performance experiment.

## Scenario versus training/export traceability

| Contract item | DSS implementation/evidence | Result |
|---|---|---|
| Timesteps | `DSS_SEQUENCE_LENGTH=11`; inclusive `10 min * 60 / 60 + 1` | PASS |
| Duration and step | `DSS_DURATION_MINUTES=10`, `DSS_STEP_SECONDS=60` | PASS |
| Epoch/profile | `2000-01-01T12:00:00+00:00`, `sgp4`, `tier1_space_segment_physics_v2` | PASS |
| Walker configuration | Walker Delta, phasing factor `F=1`; user plane/satellite counts are passed through | PASS |
| ISL policy | `grid_fixed` | PASS |
| Maximum ISL distance | Inclusive `10000 km` | PASS |
| Adjacent search | `adjacent_search_k=1` | PASS |
| Inter-plane cap | Maximum `1` link per satellite | PASS |
| Failure model | Persistent node failures; persistent temporal-union edge failures, `persistent_temporal_union_edges_v1` | PASS |
| Node semantics | Failed satellites are absent from each effective graph; node features are plane index, within-plane index, and constant existence value | PASS |
| Edge semantics | Accepted same-timestep satellite edges only; undirected physical edges are represented in both PyG directions | PASS |
| Link-mode encoding | Optical `1.0`; non-optical/RF `0.0` | PASS |
| Edge-weight semantics | `edge_weight = edge_attr[:, 0]` | PASS |
| Node normalization | `plane/(num_planes-1)`, `satellite/(sats_per_plane-1)`, constant `1.0` | PASS |
| Edge normalization | Distance `/10000`, margin `/100`, link type `intra=0`, `inter=0.5`, `seam=1`, link mode binary | PASS |
| Target | `space_gcc_fraction_original_min`, minimum of GCC/original nominal nodes over 11 snapshots | PASS |

Evidence was traced to `src/satnet/dss/scenario_builder.py`, `src/satnet/simulation/tier1_rollout.py`, `src/satnet/dss/tgnn_inference.py`, `src/satnet/experiments/final_training/tgnn_loader.py`, the accepted final ML export manifest/schema, and the frozen production contract. No semantic mismatch was found between the deployment DSS and the qualified space-only TGNN export.

## UI and product audit

### Display precision

- Primary percentages, failure rates, ground/system percentages, and comparison percentages now render to one decimal place (for example, `0.068074` → `6.8%`, `0.03697` → `3.7%`).
- Margins render to one decimal percentage point (for example, `-0.731926` → `-73.2 percentage points`).
- Integer counts remain whole numbers.
- Analysis Details retains the realization list but applies the same readable percentage presentation.

### Wording and provenance

- Primary result title: **Expected Minimum Connectivity**.
- Definition shown on the primary screen: **Mean of five TGNN-predicted minimum GCC values**.
- The five-realization count is described as deterministic modeled failure realizations/scenario context; no main UI or methodology text calls it probability, likelihood, confidence, or success probability.
- Threshold is a user-defined post-inference engineering requirement.
- Space provenance: **TGNN Prediction**.
- Ground/system provenance: **SATNET Calculated**.

### Engineering-manager UX

The primary screen exposes, in the main workspace: the evaluated architecture, Expected Minimum Connectivity, required connectivity, expected margin, meet/below assessment, and System Context limiting segment. The architecture summary and provenance are adjacent to the result; model/checkpoint detail remains secondary in Analysis Details. This supports the six requested questions without exposing technical realization detail as the primary decision surface.

### Default example decision

The existing 5×7, 550 km, 53-degree example remains the default. It is valid within the API domain but intentionally remains an illustrative, non-optimal architecture. A small bounded comparison against frozen pilot anchors found P02 produced approximately 96.0% mean TGNN prediction while P03 produced approximately 6.2%; selecting the favorable anchor would risk cherry-picking and would obscure the requested default sanity result. The UI therefore retains the current example and documents its low resilience honestly.

## API/UI acceptance status

- UI source audit: PASS.
- API contract and inference path: PASS by targeted DSS/API tests and the deployment sanity check.
- Ground/system service acceptance: BLOCKED for this workstation because `SATNET_DSS_GROUND_CATALOG` was not configured with the requested qualified authoritative catalog. The repository's committed pilot catalog is explicitly synthetic and not scientifically reviewed; it was not represented as authoritative and was not modified.
- A real API end-to-end analysis is therefore required with the already accepted authoritative catalog before merge acceptance can be declared.

## Known limitations

1. The current default architecture yields very low resilience under the accepted fixed topology/failure profile; it is not an optimal recommendation.
2. The DSS uses a space-only frozen TGNN; integrated TGNN support is not authorized.
3. Ground/system metrics require an externally supplied qualified catalog and remain unavailable when that deployment input is missing.
4. Five deterministic realizations are engineering scenarios, not a probability estimate or Monte Carlo uncertainty interval.
5. TGNN predictions are deliberately not clipped; out-of-range raw predictions remain visible in API/detail contracts for fail-open scientific inspection.
6. No automatic architecture optimization, recommendation engine, authentication, persistence, cloud deployment, RF UI, or retraining is part of this release.

## Verification results

- Starting branch: `feature/final-dss-ui-v1`.
- Starting SHA: `4283cef36b1b9032472dfe61dc93e81b2f90a9b5`.
- Starting remote SHA: matched starting SHA.
- Starting target worktree: clean.
- Frontend tests: `npm test` — **PASS**, 11 tests passed.
- Frontend build: `npm run build` — **PASS**; Vite emitted only the existing chunk-size advisory.
- Python DSS/API tests: `pytest tests/dss -q` — **PASS**, 41 passed with 6 dependency warnings. The real acceptance-path test used the committed synthetic pilot catalog because no authoritative catalog was configured.
- Real API end-to-end with the frozen checkpoint and synthetic pilot catalog: **PASS**, HTTP 200, expected minimum GCC `0.06807432174682618`, mean ground service `0.03696969696969697`, limiting segment `GROUND`, five realizations.
- Qualified-authoritative-catalog end-to-end: **BLOCKED** because `SATNET_DSS_GROUND_CATALOG` is not configured with the requested accepted catalog.
- Final branch/SHA: `feature/final-dss-ui-v1` / recorded in the final handoff after the audit commit; no merge to main.

## Artifact and science protection

- No checkpoint or ground catalog was created, modified, or committed.
- No frozen TGNN weights, architecture, training data, export data, or scientific model code was changed.
- No training, retraining, tuning, held-out experiment, Monte Carlo expansion, or threshold tuning occurred.
