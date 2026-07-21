# Stage A Near-Neighbor Resolution v1

## Decision boundary

This proposal-only correction resolves the `SA-D013` development / `SA-D020` validation near-neighbor review before simulation.

- Proposal status: `NOT_FROZEN`
- Simulation authorized: `false`
- Contract frozen: no
- Simulation, replay, production, acceptance, model training, push, merge, and tag performed: no
- Starting correction commit: `f61521e96edc1bb8b5c3f05e8ebbcd25bca1eb7d`
- Resolution implementation commit: `fe35767eb3fc1f554d2068ece0e891d7c3727ca6`
- Resolution test commits: `7f9aa2e4d20b7a5f08216830de373f5de484abd2`, `54d34f15e2e615be151be7d76793478f67a09d41`

The result is ready for a separate independent Stage A freeze-readiness audit. This conclusion does not freeze the proposal or authorize execution.

## Original blocker reproduction

The distance is Euclidean distance over eleven dimensions normalized by frozen full-DOE min-max ranges. Ground-station class dimensions use class count divided by total ground-station count.

| Feature | Frozen range | SA-D013 | Original SA-D020 | Absolute normalized difference | Squared contribution |
|---|---:|---:|---:|---:|---:|
| `num_planes` | `[4, 6]` | 6 | 6 | 0 | 0 |
| `sats_per_plane` | `[5, 8]` | 8 | 8 | 0 | 0 |
| `altitude_km` | `[300, 1200]` | 750 | 725 | 0.027777777777777776 | 0.0007716049382716049 |
| `inclination_deg` | `[30, 98]` | 58 | 59 | 0.014705882352941176 | 0.00021626297577854672 |
| `satellite_node_failure_probability` | `[0, 0.2]` | 0.055 | 0.055 | 0 | 0 |
| `satellite_edge_failure_probability` | `[0, 0.25]` | 0.060 | 0.065 | 0.020000000000000018 | 0.0004000000000000007 |
| `total_ground_station_count` | `[3, 50]` | 18 | 20 | 0.0425531914893617 | 0.0018107741059302852 |
| `ground_station_failure_probability` | `[0, 0.4]` | 0.060 | 0.075 | 0.0375 | 0.00140625 |
| `civilian_fraction` | `[0, 1]` | 0.4444444444444444 | 0.45 | 0.005555555555555591 | 0.0000308641975308646 |
| `government_fraction` | `[0, 1]` | 0.2777777777777778 | 0.30 | 0.0222222222222222 | 0.0004938271604938261 |
| `military_fraction` | `[0, 1]` | 0.2777777777777778 | 0.25 | 0.02777777777777779 | 0.0007716049382716057 |

The squared contributions sum to `0.005901188316276734`; its square root is exactly reproduced as `0.07681919236933395`. The largest individual contributions are total ground-station count and ground-station failure probability. The designs are also unusually close because architecture and node-failure probability are identical and all remaining differences are small.

## Scientific roles

Both designs are preassigned boundary-region probes. The region is a sampling hypothesis, not an outcome label. Its construction rationale is controlled transition coverage spanning the independently observed D001-to-D022 neighborhood and nearby breached evidence.

| Field | SA-D013 | Original SA-D020 | Corrected SA-D020 |
|---|---:|---:|---:|
| Region | boundary | boundary | boundary |
| Partition | development | validation | validation |
| Sealed | false | false | false |
| `num_planes` | 6 | 6 | 6 |
| `sats_per_plane` | 8 | 8 | 8 |
| `configured_satellite_count` | 48 | 48 | 48 |
| `altitude_km` | 750 | 725 | 725 |
| `inclination_deg` | 58 | 59 | 59 |
| `phasing_factor` | 1 | 1 | 1 |
| `satellite_node_failure_probability` | 0.055 | 0.055 | 0.055 |
| `satellite_edge_failure_probability` | 0.060 | 0.065 | 0.065 |
| `civilian_count` | 8 | 9 | 9 |
| `government_count` | 5 | 6 | 6 |
| `military_count` | 5 | 5 | 5 |
| `total_ground_station_count` | 18 | 20 | 20 |
| `ground_station_failure_probability` | 0.060 | 0.075 | 0.100 |

Both designs retain the shared fixed profile: 10-minute duration, 60-second steps, epoch `2000-01-01T12:00:00+00:00`, SGP4, 10,000 km maximum ISL distance, `grid_fixed`, adjacent search `1`, one inter-plane link per satellite, persistent temporal-union edge failures, 10-degree minimum elevation, 0.8 space and ground thresholds, and the unchanged ground-service and visibility policy hashes.

### Role and nearest-neighbor evidence

| Evidence | SA-D013 | Original SA-D020 | Corrected SA-D020 |
|---|---:|---:|---:|
| Nearest original | D001: 0.21191635627866762 | D001: 0.22425868415787822 | D022: 0.24538558318207365 |
| Nearest Stage A | SA-D020: 0.07681919236933395 | SA-D013: 0.07681919236933395 | SA-D013: 0.12039492645571381 |
| Nearest cross-partition | SA-D020: 0.07681919236933395 | SA-D013: 0.07681919236933395 | SA-D013: 0.12039492645571381 |
| Distance to D000 | 0.8697812933804497 | 0.8893068193122207 | 0.9045498155851996 |
| Distance to D001 | 0.21191635627866762 | 0.22425868415787822 | 0.2490194920487611 |
| Distance to D022 | 0.25780389586583974 | 0.2644378204491048 | 0.24538558318207365 |

Neither design is an exact or parameter-vector duplicate. SA-D013 remains a development transition probe nearest D001. Corrected SA-D020 remains an independent validation transition probe and shifts modestly toward D022. Modifying SA-D020 is preferable to perturbing the established development scaffold.

## Deterministic candidate search

A deterministic one-parameter local search was performed around original SA-D020. Architecture, partition, region, identity, fixed profile, station composition unless explicitly varied, and positive station-class counts were fixed. Admissible grids respected integer/discrete fields and boundary-region limits: altitude in 25 km increments, integer inclination, failure probabilities in 0.005 increments, and integer station counts with total between 10 and 30. Candidates were rejected for Stage A or original-design duplication, region-bound failure, development distance below 0.11, or holdout distance below 0.10.

Ranking considered scientific-role preservation, a non-fragile cross-partition margin, perturbation from original SA-D020, regional coverage, D000/D001/D022 transition coverage, and sampling-gap avoidance.

| Rank | Changed field | Original | Candidate | Distance from original | Nearest development | Nearest validation | Nearest holdout | Nearest original | Bounds / duplicate | Rationale |
|---:|---|---:|---:|---:|---|---|---|---|---|---|
| 1 | `ground_station_failure_probability` | 0.075 | 0.100 | 0.06250000000000003 | SA-D013: 0.12039492645571381 | SA-D009: 0.31272960980120906 | SA-D022: 0.1500946005884104 | D022: 0.24538558318207365 | PASS / PASS | Selected minimal stress-axis correction with non-fragile separation. |
| 2 | `ground_station_failure_probability` | 0.075 | 0.095 | 0.05000000000000002 | SA-D013: 0.11023242860554572 | SA-D009: 0.30845349543556233 | SA-D022: 0.14852487712768675 | D001: 0.2429855086630126 | PASS / PASS | Smaller perturbation, but comparatively fragile margin. |
| 3 | `ground_station_failure_probability` | 0.075 | 0.105 | 0.07499999999999998 | SA-D013: 0.13096254547112593 | SA-D009: 0.31744063830331565 | SA-D022: 0.15267494596624046 | D022: 0.2433288497970454 | PASS / PASS | Same stress axis with a larger perturbation. |
| 4 | `satellite_edge_failure_probability` | 0.065 | 0.085 | 0.08000000000000002 | SA-D013: 0.1245037682814329 | SA-D009: 0.31692831815162315 | SA-D022: 0.14180845928855743 | D001: 0.25747224592220225 | PASS / PASS | Changes the space-failure axis rather than ground stress. |
| 5 | `inclination_deg` | 59 | 64 | 0.0735294117647059 | SA-D013: 0.11606201992265114 | SA-D009: 0.2548253252517674 | SA-D011: 0.13552483505048024 | D001: 0.23137826617230317 | PASS / PASS | Changes orbital geometry. |
| 6 | `altitude_km` | 725 | 650 | 0.08333333333333331 | SA-D013: 0.13219403311175132 | SA-D009: 0.3498082061196373 | SA-D022: 0.1987796338398332 | D022: 0.238129699044864 | PASS / PASS | Shifts the orbital shell toward the region floor. |
| 7 | `ground_station_failure_probability` | 0.075 | 0.110 | 0.0875 | SA-D013: 0.14184476837824062 | SA-D009: 0.3225675260258175 | SA-D022: 0.15621584146876538 | D022: 0.24190137216125862 | PASS / PASS | Same stress axis but farther from the original point. |
| 8 | `inclination_deg` | 59 | 65 | 0.08823529411764702 | SA-D013: 0.12760020044516768 | SA-D009: 0.24837874658068243 | SA-D011: 0.13945714680270999 | D001: 0.23554674448802745 | PASS / PASS | Larger orbital-geometry change. |
| 9 | `satellite_node_failure_probability` | 0.055 | 0.035 | 0.09999999999999995 | SA-D013: 0.1260999140216865 | SA-D009: 0.29604654844536943 | SA-D011: 0.17160505760132072 | D001: 0.2351424194402681 | PASS / PASS | Reduces satellite node-failure stress. |
| 10 | `satellite_edge_failure_probability` | 0.065 | 0.090 | 0.09999999999999998 | SA-D013: 0.14107157160915423 | SA-D009: 0.3250285508173343 | SA-D022: 0.14597821455886642 | D001: 0.26887163744103426 | PASS / PASS | Larger space-edge-failure change. |

## Selected correction

SA-D020 changes only `ground_station_failure_probability` from `0.075` to `0.100`. The corrected value is within the boundary-region range `[0.03, 0.15]`. It preserves the validation role while moving along a directly relevant failure-stress axis rather than modifying architecture, orbital geometry, station composition, or the sealed holdout.

The corrected SA-D013/SA-D020 distance is `0.12039492645571381`, providing `0.02039492645571380` above the required 0.10 threshold. No exception is retained.

## Coverage impact

| Metric | Before | After |
|---|---:|---:|
| Global minimum Stage A pairwise distance | 0.07681919236933395, SA-D013/SA-D020 | 0.12039492645571381, SA-D013/SA-D020 |
| Development-validation minimum | 0.07681919236933395, SA-D013/SA-D020 | 0.12039492645571381, SA-D013/SA-D020 |
| Development-holdout minimum | 0.1205683310788027, SA-D011/SA-D013 | unchanged |
| Validation-holdout minimum | 0.13945714680270999, SA-D011/SA-D020 | 0.1500946005884104, SA-D020/SA-D022 |
| Within-partition minimum | 0.1249155271916213, SA-D011/SA-D022 | unchanged |
| Minimum Stage A-to-original distance | 0.06382978723404255, SA-D024/D004 | unchanged |
| All-pair median | 0.7538462556117791 | 0.7544565022428936 |
| Cross-partition median | 0.75721630001553108 | unchanged |
| Within-partition median | 0.7534478205428954 | 0.7541513789273364 |
| Per-design nearest-neighbor median | 0.22175145107221444 | unchanged |
| Per-design nearest-neighbor mean | 0.3564218648389522 | 0.3593269137780442 |

All parameter ranges remain unchanged at the full-corpus, region, and partition levels. Boundary-region ground-failure coverage remains `[0.04, 0.14]`; validation coverage remains `[0.015, 0.35]`. Boundary-validation coverage changes from `[0.075, 0.11]` to `[0.10, 0.11]`, while neighboring boundary points at 0.07, 0.08, and 0.09 preserve transition coverage. The full Stage A ground-failure level `0.075` is removed and an already represented `0.10` level is reused, reducing distinct single-parameter levels by one without creating a multivariate duplicate or changing regional extrema.

The pairwise distribution changes are negligible outside the intended minimum-separation correction. D001 and D022 coverage remains represented by distinct development, validation, and holdout designs. Region and partition allocations remain unchanged.

## Identity, seed, and artifact effects

The corpus namespace, design IDs and indexes, partitions, regions, sealed flags, run IDs, run keys, realization IDs and indexes are unchanged. SA-D020 remains validation design `SA-D020` with global run IDs 600 through 604.

Design-parameter and design-record hashes for SA-D020 change. Its five run-record hashes change because they bind the corrected design-record hash. Seed derivation is identity-only; design, ground-selection, satellite-failure, and ground-failure seeds are unchanged. The seed manifest remains byte-identical with SHA-256 `ab37564cca81e926b6e43caf74f1ff2be641b8bc203e32c3ae6a219ea6f7bace`.

The original proposal inventory SHA-256 was `1ad81a30097a99da87ee6e51139f4bec0791e6d9d99032854930bd5245757b39`. The corrected proposal inventory SHA-256 is `69fd2a9fbcc1b461ae4230a282cfcaaadc26ba7b65563bc90a81e86e1ca98127`.

Changed canonical artifact hashes are:

| Artifact | Corrected SHA-256 |
|---|---|
| `stage_a_design_manifest.csv` | `073e48dbfb8fb6309bbbb9305fcbf4d0769a5ad85db9f1f3e04320f2dc413d75` |
| `stage_a_run_manifest.csv` | `f71556529f47ef642d8fe56d041312320cba22dcaed9575812d9693511b0b79a` |
| `stage_a_near_neighbor_policy.json` | `b5ebbd7ce681ed0fd57ab037bf2249b23c801787c7294691225d7e9919fdbc8f` |
| `stage_a_contract_proposal.json` | `35637c41199f8d7d824d18484cf00496c6d97aa71129b3a4d263bd155b5de0e3` |
| `stage_a_proposal_inventory.json` | `69fd2a9fbcc1b461ae4230a282cfcaaadc26ba7b65563bc90a81e86e1ca98127` |

All other proposal artifacts remain byte-identical. All 11 corrected proposal artifacts reproduce byte-for-byte from canonical tooling.

## Acceptance results

- Stage A designs: 30
- Stage A runs: 150
- Region allocation: 12 resilient core, 12 boundary, 6 global control
- Partition allocation: 20 development, 5 validation, 5 sealed holdout
- Region/partition allocation: 8/2/2 resilient core, 8/2/2 boundary, 4/1/1 global control
- Exact Stage A duplicates: 0
- Stage A parameter-vector duplicates: 0
- Exact original-design duplicates: 0
- Identity collisions: 0
- Minimum development-validation distance: `0.12039492645571381`, SA-D013/SA-D020
- Minimum development-holdout distance: `0.1205683310788027`, SA-D013/SA-D011
- Minimum validation-holdout distance: `0.1500946005884104`, SA-D020/SA-D022
- Minimum Stage A-to-original distance: `0.06382978723404255`, SA-D024/D004
- Remaining exceptions: none
- Pending scientific reviews: none
- External Stage A execution roots: absent
- Protected-science diff: empty
- Frozen production evidence verification: passed by the independent-audit tests
- Artifact reproduction: 11 of 11 byte-identical
- Focused Stage A tests: 25 passed
- Class-support analysis tests: 8 passed
- Independent-audit tests: 8 passed
- Isolation regression after test-path correction: 13 passed
- Complete repository suite: 950 passed in 47.65 seconds
- Logger tests independently: 7 passed
- Known Windows timer test: passed in the complete-suite run
- Compilation: 190 Python files compiled in memory
- Whitespace check: passed

## Fail-closed behavior

Every development-validation pair below normalized distance 0.10 now fails proposal generation unless an exact approved pre-freeze exception record binds the pair, exact distance, scientific justification, and approval reference. Any sealed-holdout cross-partition pair below 0.10 fails regardless of exception. The current corrected proposal has no exceptions and an empty `pending_scientific_reviews` list.

## Remaining limitations and next gate

This work is a pre-simulation proposal correction. A separate agent must independently audit the exact corrected bytes, hashes, identities, duplicate checks, distance calculations, seed invariants, absent execution roots, and protected evidence before any freeze decision. The proposal must remain `NOT_FROZEN`, and simulation authorization must remain `false`, until separately approved tasks perform those actions.

## Verdict

`READY FOR INDEPENDENT STAGE A FREEZE AUDIT`

This verdict does not freeze the Stage A contract and does not authorize simulation.
