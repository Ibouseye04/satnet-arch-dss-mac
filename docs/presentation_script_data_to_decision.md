# Presentation Script: Data-to-Decision Pipeline
## Satellite Network Reliability Analysis

---

## Opening (30 seconds)

"Today I'll walk you through our **Data-to-Decision Pipeline** for satellite network reliability analysis. This system takes constellation design parameters as input and produces actionable risk tiers as output—answering the question: *Will this satellite network stay connected under failure conditions?*"

---

## Section 1: The Big Picture (1 minute)

"The pipeline has **four core modules**:

1. **Data Ingestion** — Convert constellation specs into orbital elements
2. **Physics Simulation** — Propagate orbits and compute inter-satellite links over time
3. **Metrics Calculation** — Inject failures and measure connectivity degradation
4. **Decision Support** — Train ML models and bin predictions into risk tiers

Let me trace through each module."

---

## Section 2: Data Ingestion — Constellation to TLEs (2 minutes)

"We start with a **Walker Delta constellation specification**—the industry standard for LEO mega-constellations like Starlink and OneWeb.

The key parameters are:
- **Number of orbital planes** (e.g., 72 planes)
- **Satellites per plane** (e.g., 22 satellites)
- **Inclination** (e.g., 53°)
- **Altitude** (e.g., 550 km)

These parameters flow into our `HypatiaAdapter` class, which generates **Two-Line Elements (TLEs)** for each satellite.

```
HypatiaAdapter.__init__()
    └── WalkerDeltaConfig creation
    
HypatiaAdapter.generate_tles()
    └── Loop over planes → Loop over satellites
        └── Calculate RAAN & mean anomaly
        └── _generate_tle_lines() → (name, line1, line2)
```

The TLEs encode the orbital mechanics—epoch, inclination, RAAN, eccentricity, mean anomaly—in a standardized format that SGP4 propagators understand.

This is **Tier 1 physics**—no toy models, no random graphs. Real orbital mechanics."

---

## Section 3: Physics Simulation — SGP4 and ISL Computation (3 minutes)

"With TLEs in hand, we propagate satellite positions over time using **SGP4**—the same algorithm used by NORAD to track objects in orbit.

```
adapter.calculate_isls()
    └── Loop over time steps
        └── SGP4 position calculation
            └── satellite.sgp4(jd, fr) → TEME coordinates
            └── TEME → ECEF transformation
        └── ISL topology generation
            └── _compute_grid_plus_isls()
                └── Link budget evaluation
```

At each time step, we:
1. **Propagate positions** — SGP4 gives us satellite coordinates in the TEME frame
2. **Transform to ECEF** — Earth-Centered Earth-Fixed frame for geometry calculations
3. **Compute ISLs** — Which satellites can see each other?

The ISL computation uses a **+Grid pattern**—each satellite connects to its neighbors in the same plane and adjacent planes. But we don't just assume links exist; we run a **1550nm optical link budget analysis**:
- Free-space path loss
- Pointing loss
- Atmospheric absorption
- Earth obscuration (can't link through the planet!)

Only links that pass the budget check become edges in our graph.

The output is a **temporal sequence of NetworkX graphs**—one per time step—representing the evolving network topology."

---

## Section 4: Metrics Calculation — Failure Injection and GCC Analysis (3 minutes)

"Now we stress-test the network. The key question: *What happens when satellites fail?*

```
run_tier1_rollout()
    └── Sample persistent failures at t=0
        └── Node failures: if rng.random() < prob
        └── Edge failures: sample failed_edges
    └── Temporal loop over graphs
        └── Apply failures: G_eff.remove_nodes_from()
        └── Compute connectivity metrics
            └── compute_gcc_size(G_eff)
            └── compute_gcc_frac(G_eff)
            └── compute_partitioned()
```

**Failure model**: At t=0, we sample which nodes and edges fail. These failures are **persistent**—they stay failed for the entire simulation. This models hardware failures, not transient outages.

**Connectivity metric**: We use the **Giant Connected Component (GCC)**—the largest set of satellites that can still route traffic to each other.

- `compute_gcc_size()` — How many satellites are in the GCC?
- `compute_gcc_frac()` — What fraction of the network is connected?
- `compute_partitioned()` — Is GCC fraction below 80%? If yes, the network is **partitioned**.

**Temporal aggregation**: We track partition status across all time steps and compute:
- `partition_any` — Did the network partition at any point?
- `max_partition_streak` — Longest consecutive partition duration

These are our **labels** for machine learning."

---

## Section 5: ML Pipeline — From Simulations to Predictions (2 minutes)

"We run **Monte Carlo simulations**—hundreds or thousands of rollouts with different:
- Constellation designs (planes, satellites, altitude, inclination)
- Failure scenarios (different random seeds)

```
generate_tier1_temporal_dataset()
    └── Sample design parameters
    └── Execute rollout per config
        └── run_tier1_rollout() → (steps, summary)
    └── Extract temporal labels
    └── write_tier1_dataset_csv()
```

This produces a **labeled dataset**: design features → partition outcomes.

Then we train a **RandomForest classifier**:

```
train_tier1_v1_design_model()
    └── Load dataset from CSV
    └── Extract design features: [num_planes, sats_per_plane, inclination, altitude]
    └── Load partition labels
    └── clf.fit(X_train, y_train)
    └── clf.predict_proba(X_test) → reliability scores
```

The model learns: *Given only the design parameters, what's the probability this constellation will partition under failure?*

The output is a **continuous reliability score** between 0 and 1."

---

## Section 6: Decision Support — Risk Tiers (1.5 minutes)

"Finally, we convert continuous scores into **actionable risk tiers**:

```
bin_satellite_risk()
    └── scores.apply(compute_tier)
        └── score > 0.8 → Tier 1 (Healthy)
        └── score < 0.5 → Tier 3 (Critical)
        └── else → Tier 2 (Watchlist)
```

Each tier maps to a **recommended action**:

| Tier | Label | Score Range | Action |
|------|-------|-------------|--------|
| 1 | Healthy | > 0.8 | No action required |
| 2 | Watchlist | 0.5 - 0.8 | Schedule diagnostics |
| 3 | Critical | < 0.5 | Immediate maneuver |

This is the **decision support output**—a constellation architect can now evaluate design tradeoffs with quantified risk."

---

## Section 7: Key Architectural Principles (1 minute)

"A few design principles worth highlighting:

1. **Pure Functions for Metrics** — `labels.py` contains stateless functions. No side effects, easy to test, no circular dependencies.

2. **Determinism** — Every simulation is reproducible. We track seeds and config hashes.

3. **Separation of Concerns**:
   - Physics Layer (`hypatia_adapter`) — Deterministic orbital mechanics
   - Simulation Layer (`tier1_rollout`) — Stochastic failure injection
   - Metrics Layer (`labels.py`) — Pure math on graphs
   - ML Layer (`risk_model.py`) — Statistical learning

4. **Temporal, Not Static** — We don't just evaluate t=0. The network evolves, and so does our analysis."

---

## Closing (30 seconds)

"To summarize: We take a constellation design, simulate its behavior under failures using physics-based orbital propagation, measure connectivity degradation over time, train ML models on the results, and output actionable risk tiers.

**Design parameters in, risk decisions out.**

Questions?"

---

## Appendix: Key File Locations

| Component | File |
|-----------|------|
| Dataset Generation | `scripts/export_design_dataset.py` |
| Monte Carlo Engine | `src/satnet/simulation/monte_carlo.py` |
| Temporal Rollout | `src/satnet/simulation/tier1_rollout.py` |
| Orbital Adapter | `src/satnet/network/hypatia_adapter.py` |
| Connectivity Metrics | `src/satnet/metrics/labels.py` |
| ML Training | `src/satnet/models/risk_model.py` |
| Risk Binning | `src/satnet/metrics/risk_binning.py` |
