# SATNET Final Simulator Verification — Adaptive-v2

## Validation Identity

- Validation date: 2026-09-08
- Validation branch: `validation/final-simulator-verification-adaptive-v2`
- Authoritative Adaptive-v2 source SHA: `9274ba721007c66f0b98aac7e50ba1e9b6b154b5`
- Production profile: `final_integrated_dataset_10k_adaptive_v2`
- Adaptive contract specification hash: `23c5fffc10849c3bc3ea027251ac3e5ad4c96f0eea85edf1e8deab079cb0871e`
- Adaptive contract bundle hash: `da3c73711b1d60635afcceee8bda0a60d1379e492e1ad0d588d5a0c25e10abe3`

This validation branch was created from the exact authoritative Adaptive-v2 source SHA. The frozen scientific contract, design manifest, run manifest, production outputs, and Adaptive-v2 scientific implementation were not modified during validation.

## Frozen Adaptive-v2 Scientific Configuration

- ISL policy: `grid_adaptive`
- Adjacent search K: `1`
- Maximum inter-plane links per satellite: `1`
- Orbital engine: `sgp4`
- Physics model: `tier1_space_segment_physics_v2`
- Simulation duration: `10 minutes`
- Time step: `60 seconds`
- Inclusive timestep count: `11`
- Satellite failure model: `persistent_temporal_union_edges_v1`
- Ground minimum elevation: `10 degrees`
- Space GCC threshold: `0.80`
- Ground service threshold: `0.80`

## DOE and Dataset Provenance Verification

| Verification | Result |
|---|---|
| Adaptive production profile tests | PASS |
| 2,000 unique designs | PASS |
| 10,000 total runs | PASS |
| Five realizations per design | PASS |
| Run IDs 0 through 9,999 | PASS |
| Design indices 0 through 1,999 | PASS |
| 1,400 / 300 / 300 design split | PASS |
| 7,000 / 1,500 / 1,500 run split | PASS |
| Zero cross-split design leakage | PASS |
| Design ID derivation | PASS |
| Run ID and run-key derivation | PASS |
| Satellite seed derivation | PASS |
| Ground selection seed derivation | PASS |
| Ground failure seed derivation | PASS |
| Deterministic primitive tests | PASS |
| Transition LHS candidate reproduction — candidate 51 | PASS |
| Global LHS candidate reproduction — candidate 49 | PASS |
| Exact 2,000-design record-for-record rebuild | PASS |
| Exact 10,000-run record-for-record rebuild | PASS |

### Frozen Manifest Identities

- Design manifest hash: `f29d519e551986732c8eb87f44de29ff3c16aa7956428df424757d7dbcb485f1`
- Run manifest hash: `3e24f4a58be51e1ccb9a8d2381cbc016dd36693bdb43b939c1d63b8671981a89`

### Frozen Metadata Observation

The Adaptive-v2 contract specification contains:

`run_identity.design_index_last = 99`

The actual frozen design and run records independently establish:

- 2,000 designs
- design indices 0 through 1,999
- 10,000 runs
- run IDs 0 through 9,999
- exactly five realizations per design

The field is therefore documented as a frozen metadata defect and was not altered during validation.

---

# Space-Segment Verification

## Walker-Delta Constellation Generation

Controlled 4-plane, 5-satellite-per-plane, phasing-factor-1 case:

- Total satellites: 20
- RAAN spacing: 90 degrees
- In-plane spacing: 72 degrees
- Inter-plane Walker phase step: 18 degrees
- Independently calculated mean motion: 15.078199602381 rev/day
- TLE field values: PASS
- TLE checksums: PASS
- RAAN structure: PASS
- Walker phasing: PASS

**Overall: PASS**

## SGP4 External Verification

Vallado/CelesTrak satellite 00005 verification case was used.

Installed SGP4 version: `2.27`

### Epoch

Position-vector error relative to Vallado reference:

`0.000006819 m`

**PASS**

### +360 minutes

Position-vector error relative to Vallado reference:

`0.000005926 m`

**PASS**

## SATNET Datetime-to-Julian-Date Wrapper

SATNET datetime conversion was passed directly into the SGP4 implementation and compared against the same Vallado reference states.

- Epoch: PASS
- +360 minutes: PASS

**Overall: PASS**

## TEME to Earth-Fixed Approximation

SATNET uses a GMST-only Z-axis rotation.

Independent Vallado GMST82 comparison:

- Epoch GMST error: approximately `-7.61e-11 rad`
- Epoch rotated-vector error: approximately `0.000545 m`
- +360-minute rotated-vector error: approximately `0.000616 m`
- Vector norm preserved

**Overall: PASS**

### Frame Terminology

The validated transformation is most precisely described as:

**GMST-rotated pseudo-Earth-fixed (PEF) / Earth-fixed approximation with polar motion neglected.**

It should not be described as a full high-precision ITRF transformation.

## Earth-Obscuration / LOS Geometry

SATNET effective obscuration radius:

`6371 km + 80 km = 6451 km`

Analytic 550-km-altitude equal-radius critical central angle:

`42.473791694097 degrees`

Known-answer cases:

- 30 degrees: clear — PASS
- critical minus 0.001 degrees: clear — PASS
- exact tangent: clear — PASS
- critical plus 0.001 degrees: obscured — PASS
- 60 degrees: obscured — PASS
- identical-position degenerate case: PASS

**Overall: PASS**

## RF Link Budget

Parameters:

- Frequency: 28 GHz
- Transmit power: 30 dBm
- Transmit gain: 40 dBi
- Receive gain: 40 dBi
- Receiver sensitivity: -90 dBm
- Rain margin when enabled: 10 dB

Known-answer cases:

| Distance | Received Power | Margin | Result |
|---:|---:|---:|---|
| 100 km | -51.39094 dBm | +38.60906 dB | PASS |
| 1,000 km | -71.39094 dBm | +18.60906 dB | PASS |
| 5,000 km | -85.37034 dBm | +4.62966 dB | PASS |
| 10,000 km | -91.39094 dBm | -1.39094 dB | PASS |

Zero-margin RF distance:

`8520.259212923111 km`

Boundary behavior: PASS

Explicit rain-margin subtraction at 1,000 km: exactly 10 dB — PASS

**Overall: PASS**

## Optical Link Budget

Parameters:

- Wavelength: 1550 nm
- Transmit power: 37 dBm
- Aperture diameter: 0.10 m
- Aperture efficiency: 0.55
- Receiver sensitivity: -45 dBm

Independent aperture gain:

`103.539990385419 dBi`

Known-answer cases:

| Distance | Received Power | Margin | Result |
|---:|---:|---:|---|
| 100 km | +5.902417 dBm | +50.902417 dB | PASS |
| 1,000 km | -14.097583 dBm | +30.902417 dB | PASS |
| 5,000 km | -28.076983 dBm | +16.923017 dB | PASS |
| 10,000 km | -34.097583 dBm | +10.902417 dB | PASS |

Zero-margin optical distance:

`35084.950867911808 km`

Boundary behavior: PASS

**Overall: PASS**

### Optical-First Modeling Observation

Because the Adaptive-v2 ISL distance cap is 10,000 km and the optical zero-margin range is approximately 35,085 km, candidates that pass distance and LOS generally also pass the optical link budget. Under the current optical-first link-selection policy, accepted Adaptive-v2 ISLs therefore normally use the optical mode within the modeled geometric range.

## Adaptive +Grid Topology Selection

Controlled two-plane, four-satellite-per-plane case:

### Fixed policy

Expected inter-plane edges:

`[(0,4), (1,5), (2,6), (3,7)]`

Observed:

`[(0,4), (1,5), (2,6), (3,7)]`

**PASS**

### Adaptive policy

Expected shifted best-partner edges:

`[(0,5), (1,6), (2,7), (3,4)]`

Observed:

`[(0,5), (1,6), (2,7), (3,4)]`

Additional checks:

- Adaptive topology differs from fixed topology: PASS
- Every satellite inter-plane degree = 1: PASS
- Endpoint link cap enforced: PASS
- All selected best-partner distances = 0 km in synthetic case: PASS
- Accepted inter-plane-link counts correct: PASS

**Overall: PASS**

This provides behavioral evidence that Adaptive-v2 is genuinely executing adaptive adjacent-plane partner selection rather than merely carrying an adaptive metadata label.

---

# Satellite Failure and Graph Verification

## Persistent Temporal-Union Edge Failures

Controlled temporal graph:

- t0: `0--1--2`, node 3 isolated
- t1: `0--1--2--3`
- edge `(2,3)` appears only at t1

With edge-failure probability 1.0:

Expected failed temporal-union edges:

`[(0,1), (1,2), (2,3)]`

Observed:

`[(0,1), (1,2), (2,3)]`

Late-appearing edge `(2,3)` was included in the persistent failure universe.

**Overall: PASS**

## Persistent Node Failure and GCC Denominators

Controlled graph:

`0--1--2--3`

Only node 1 failed.

Post-failure:

- configured satellites: 4
- surviving satellites: 3
- GCC size: 2
- original-denominator GCC fraction: `2/4 = 0.5`
- surviving-denominator GCC fraction: `2/3 = 0.666666...`

At threshold 0.60, partition classification used the original denominator and correctly indicated threshold breach.

**Overall: PASS**

## Canonical Graph Metrics

### K4

- nodes: 4
- edges: 6
- average degree: 3
- components: 1
- GCC fraction: 1
- global efficiency: 1

PASS

### P4

- nodes: 4
- edges: 3
- average degree: 1.5
- components: 1
- GCC fraction: 1
- global efficiency: `13/18 = 0.722222...`

PASS

### P3 + isolated node

- nodes: 4
- edges: 2
- average degree: 1
- components: 2
- GCC fraction: 0.75
- global efficiency: `5/12 = 0.416666...`

PASS

### Threshold Semantics

At threshold 0.80:

- GCC 0.79 -> partitioned = 1
- GCC 0.80 -> partitioned = 0
- GCC 0.81 -> partitioned = 0

**Overall: PASS**

SATNET `partitioned` therefore represents a threshold breach, not simply graph-theoretic disconnectedness.

---

# Ground-Segment Verification

## WGS-84 Geodetic-to-ECEF Conversion

Canonical cases:

### Equator / Greenwich

Expected:

`(6378.137, 0, 0) km`

Observed exactly.

PASS

### Equator / 90E

Expected:

`(0, 6378.137, 0) km`

Vector error approximately:

`3.91e-10 m`

PASS

### North Pole

Expected:

`(0, 0, 6356.752314245179) km`

Vector error approximately:

`3.92e-10 m`

PASS

**Overall: PASS**

## Topocentric ENU Geometry

Known-answer cases:

| Case | Elevation | Slant Range | Result |
|---|---:|---:|---|
| Directly overhead | 90° | 550 km | PASS |
| Controlled geometry | 30° | 1,000 km | PASS |
| Horizon | 0° | 1,000 km | PASS |
| Below horizon | -30° | 1,000 km | PASS |

**Overall: PASS**

## Frozen Adaptive-v2 Visibility Threshold

Frozen minimum elevation:

`10 degrees`

Controlled cases:

- 9.999° -> not visible — PASS
- 10.000° -> visible — PASS
- 10.001° -> visible — PASS

Visible satellite set:

`(1,2)`

**Overall: PASS**

### Floating-Point Boundary Observation

A separate artificial 30-degree equality construction produced:

`29.999999999999996 degrees`

This is exactly one IEEE-754 ULP below 30 degrees, so a direct `>= 30.0` comparison classified it below threshold.

The geometry itself was accurate and `math.isclose(..., abs_tol=1e-12)` evaluated true.

The actual frozen Adaptive-v2 10-degree policy passed the below/equality/above test. No production scientific code was modified.

This observation is documented as a generic floating-point boundary limitation and was not shown to materially affect the frozen Adaptive-v2 experiment.

## Ground Failure Sampling

Independent SHA-256 trial calculations were compared against SATNET for:

- p = 0.0
- p = 0.1
- p = 0.4
- p = 1.0

All station-level outcomes matched.

Endpoint behavior:

- p = 0.0 -> no stations fail — PASS
- p = 1.0 -> all stations fail — PASS

Repeated execution with the same station IDs, probability, and seed produced the exact same realization.

**Overall: PASS**

## Ground Failure Realization

Controlled six-station architecture:

Selected:

- CIV_TEST_001
- CIV_TEST_002
- GOV_TEST_001
- GOV_TEST_002
- MIL_TEST_001
- MIL_TEST_002

At p = 0.40 and seed 123456789:

Expected failed:

- CIV_TEST_002
- GOV_TEST_001
- GOV_TEST_002

Observed failed set matched exactly.

Expected operational:

- CIV_TEST_001
- MIL_TEST_001
- MIL_TEST_002

Observed operational set matched exactly.

Additional checks:

- failed and operational sets disjoint: PASS
- failed union operational equals selected: PASS
- counts consistent: PASS
- authoritative context replay: PASS
- repeated realization hash identical: PASS

Realization hash:

`99314d3ec63b3cb3e64bc52983f8315d31e07040087bfe0f317eccc32ef44828`

**Overall: PASS**

---

# Integrated Space + Ground Verification

## G3 Integrated Graph

Controlled satellite graph:

`0 -- 1 -- 2`

Three ground stations were used:

- CIV_TEST_001
- GOV_TEST_001
- MIL_TEST_001

Each station was analytically visible to satellites 0 and 2 and not satellite 1.

Expected visible pairs:

- CIV_TEST_001 -> 0, 2
- GOV_TEST_001 -> 0, 2
- MIL_TEST_001 -> 0, 2

Observed visible pairs matched exactly.

Integrated counts:

- satellite nodes: 3
- ground nodes: 3
- ISL edges: 2
- satellite-ground edges: 6

All counts: PASS

Satellite projection from the integrated graph reproduced the original satellite graph exactly:

`[(0,1), (1,2)]`

Visibility-edge equality validator: PASS

Integrated graph hash:

`a4abb55c301db0b3e2ea02073595997153c9f03999e915505da3cdc54eb7720b`

**Overall: PASS**

## G4 Baseline Service

Controlled system:

- space graph GCC: all 3 satellites
- all three ground stations attached to satellites in the GCC

Results:

- space GCC original fraction: 1.0
- space GCC surviving fraction: 1.0
- ground service fraction: 1.0
- overall service fraction: 1.0
- space threshold met: true
- ground threshold met: true
- overall threshold met: true

**Overall: PASS**

## G5 Failure-Adjusted Service

Ground failure realization:

- GOV_TEST_001 failed
- CIV_TEST_001 operational
- MIL_TEST_001 operational

Failure-adjusted results:

- space GCC original: 1.0
- space GCC surviving: 1.0
- adjusted ground service: `2/3 = 0.6666666666666666`
- adjusted overall service: `2/3 = 0.6666666666666666`
- ground service loss: approximately `1/3`
- overall service loss: approximately `1/3`
- space threshold met: true
- ground threshold met: false
- overall threshold met: false

Critical decomposition checks:

- space metric unchanged by ground failure: PASS
- overall service = minimum(space, ground): PASS

**Overall: PASS**

This verifies the intended distinction between the space-only resilience metric and the integrated space-plus-ground resilience metric.

---

# Overall Validation Conclusion

All controlled scientific verification cases executed against the authoritative Adaptive-v2 implementation passed, with one documented non-material floating-point boundary observation at an artificial 30-degree visibility threshold.

The verification campaign independently covered:

1. deterministic DOE construction and identity;
2. exact frozen design and run reproduction;
3. Walker-Delta architecture generation;
4. SGP4 propagation against published Vallado reference states;
5. SATNET time conversion;
6. GMST-only TEME-to-PEF transformation;
7. LOS geometry;
8. RF link-budget calculations;
9. optical link-budget calculations;
10. adaptive inter-plane topology selection;
11. persistent satellite node and temporal-union edge failures;
12. graph metrics and GCC threshold semantics;
13. WGS-84 ground coordinates;
14. ground elevation and visibility;
15. deterministic ground-station failures;
16. integrated satellite-ground graph construction;
17. baseline ground service;
18. failure-adjusted integrated service; and
19. preservation of the space-only metric under ground-only failures.

**Component-level simulator verification status: PASS**

## Independent Grouped Split Verification

An independent exhaustive reconstruction evaluated all 4,096 deterministic grouped-split candidates without calling SATNET's production candidate-assignment, scoring, or split-selection functions.

Results:

- total candidates evaluated: `4,096`;
- candidates satisfying all hard requirements: `636`;
- independently selected candidate: `3958`;
- frozen selected candidate: `3958`;
- exact frozen three-part rational score reproduced;
- frozen design assignments reproduced exactly;
- candidate `3958` was the unique best-scoring candidate;
- exact best-score tie count: `1`;
- runner-up candidate: `1814`.

**Independent grouped split verification status: PASS**

## Remaining Validation Work

The following are outside the completed component-level simulator known-answer campaign and remain separate validation tasks:

- formal upstream Hypatia provenance / implementation-lineage comparison;
- permanent executable consolidation of the controlled known-answer tests;
- final ML surrogate runtime, memory, scalability, and predictive-performance benchmarking;
- dissertation-facing presentation of the validation evidence.


