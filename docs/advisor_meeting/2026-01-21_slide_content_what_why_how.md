# Slide Content: What / Why / How + Design Rationale

**Prepared for:** Alex's Advisor Meeting (Saturday, Jan 25, 2026)  
**Date:** 2026-01-21  
**Purpose:** Provide ready-to-use content for filling out slide boxes and explaining design decisions

---

# Part A: What / Why / How for Slide Sections

## Section 1: Data Collection & EDA

### What
Obtain satellite network routing, congestion, and architecture datasets and perform exploratory data analysis (EDA) to understand data structure, variability, and flow of network data.

**Two Data Tracks:**
- **Track A (Primary):** Tier 1 simulated truth datasets — physics-based temporal ISL connectivity graphs generated via Monte Carlo simulation under various failure scenarios.
- **Track B (Supporting/Validation):** External datasets (LEO congestion logs, NASA anomaly data, synthetic satellite health) for EDA, validation, and future integration.

### Why
To acquire and understand network flow and possible failures in Satellite Network Architecture.

**Deeper reasoning:**
- Engineering managers cannot evaluate design risk without data that captures the *dynamics* of constellation behavior over time.
- Static "snapshot" analysis misses transient partitions and sustained degradations that emerge under real orbital mechanics.
- EDA validates that our simulated data exhibits realistic patterns (e.g., GCC fraction distributions, failure cascades) before feeding it to ML models.

### How
1. **Tier 1 Simulation Pipeline:**
   - Define Walker Delta constellation parameters (planes, sats/plane, altitude, inclination)
   - Generate TLEs → propagate orbits via SGP4
   - Compute ISLs using geometry checks (Earth obscuration) + link budgets (1550nm optical / Ka-band RF)
   - Inject persistent node/edge failures at t=0
   - Evaluate connectivity metrics at each timestep t=0..T
   - Export schema-validated CSVs with `config_hash` + `seed` for reproducibility

2. **External Data Ingestion:**
   - Download datasets from Figshare, Kaggle, ResearchGate (do not commit to git)
   - Standardize paths via `SATNET_DATA_ROOT` environment variable
   - Run EDA scripts to produce summary statistics and visualizations
   - Use for validation and Phase 2 traffic modeling

---

## Section 4: Ground Segment

### What
Earth-based infrastructure: ground stations (gateways), user terminals, and network operations centers that interface with the space segment. This includes the physical antennas, backhaul connections to terrestrial internet, and control systems.

### Why
- Connects the satellite mesh to end users and terrestrial networks
- Gateway placement directly affects:
  - **Coverage:** How much of Earth has access at any time
  - **Latency:** Fewer hops to ground = lower end-to-end delay
  - **Resilience:** Redundant gateways prevent single-point failures
- Without ground segment modeling, we can only assess "space segment connectivity" — not "service availability" to users

### How (Phase 2 Implementation Plan)
1. **Gateway Definition:**
   - Define candidate gateway locations (lat/lon coordinates)
   - Assign connectivity constraints (backhaul bandwidth, weather zone)

2. **Visibility Modeling:**
   - Model Earth rotation (ECEF frame)
   - Compute time-varying satellite-ground visibility windows
   - Apply elevation masks (e.g., >10° above horizon)
   - Apply atmospheric attenuation + weather margin in link budgets

3. **Integration:**
   - Extend temporal graph to include SAT-GS edges
   - Add "service availability" labels based on end-to-end connectivity
   - Retrain ML models with expanded feature set

---

### Sub-boxes for Ground Segment Row

#### Build Ground Architecture
| Aspect | Content |
|--------|---------|
| **What** | Design gateway placement: number of gateways, geographic distribution, redundancy, and backhaul connectivity to terrestrial networks. |
| **Why** | Optimal placement minimizes latency, maximizes coverage, and ensures resilience. Poor placement = coverage gaps or single-point failures. Trade-off between cost and performance. |
| **How** | (1) Enumerate candidate sites (existing teleports, favorable weather zones). (2) Model Earth rotation + satellite passes. (3) Compute visibility windows for each site. (4) Optimize placement for coverage/latency objectives using heuristics or ML. |

#### Develop and Train ML Model (Ground)
| Aspect | Content |
|--------|---------|
| **What** | Extend ML models to predict end-to-end service risk, including SAT-GS link availability and ground segment failures. |
| **Why** | Ground failures (weather, equipment) and visibility gaps affect mission success. A complete risk model must include both space and ground. |
| **How** | (1) Add SAT-GS link features to dataset (visibility duration, link margin, weather zone). (2) Add ground failure scenarios to Monte Carlo. (3) Define "service availability" label. (4) Retrain RF baseline and temporal GNN with expanded features. |

#### Simulations and Testing (Ground)
| Aspect | Content |
|--------|---------|
| **What** | Monte Carlo simulations over combined space + ground failure scenarios. |
| **Why** | Real-world failures are correlated (e.g., weather affecting multiple gateways in a region). Must stress-test the full system, not just space segment in isolation. |
| **How** | (1) Extend `tier1_rollout` with ground segment visibility provider. (2) Add correlated failure models (e.g., regional weather events). (3) Run N simulations with varying failure combinations. (4) Collect end-to-end connectivity metrics. |

#### Analyze Results (Ground)
| Aspect | Content |
|--------|---------|
| **What** | Compute service availability metrics, gateway handoff statistics, and end-to-end latency distributions. |
| **Why** | Translate raw connectivity data into user-facing SLA metrics that engineering managers care about. |
| **How** | (1) New pure metric functions: `compute_service_availability(G)`, `compute_handoff_frequency(G_series)`. (2) Aggregate across Monte Carlo runs. (3) Feed to risk binning for actionable tiers. |

---

## Section 5: GUI Segment

### What
A graphical user interface allowing engineering managers to interactively explore constellation designs, visualize risk predictions, and receive actionable recommendations — without writing code or running scripts.

### Why
- The thesis deliverable is a **decision-support tool** — the GUI is the human-facing layer that makes ML outputs accessible.
- Engineering managers are domain experts in satellite systems, not ML or Python. They need intuitive controls and clear visualizations.
- Enables "what-if" scenario exploration: "What happens if I add 2 more orbital planes?" → instant risk re-evaluation.

### How (Phase 2 Implementation Plan)
1. **CLI First:**
   - Build a "scenario runner" CLI that accepts parameters and outputs risk scores
   - Validates the inference pipeline end-to-end

2. **Web Dashboard:**
   - Framework: Streamlit or Dash (rapid prototyping, Python-native)
   - Input widgets for constellation parameters (sliders, dropdowns)
   - Visualizations: constellation 3D view, GCC fraction over time, risk tier gauge
   - Output: risk score, confidence, recommended action

3. **Integration:**
   - REST/gRPC API wrapping `HypatiaAdapter` + trained models
   - Async job runner for longer simulations with progress tracking

---

### Sub-boxes for GUI Segment Row

#### Build User Interface
| Aspect | Content |
|--------|---------|
| **What** | Design screens for: (1) Constellation parameter input, (2) Risk dashboard with visualizations, (3) Action recommendations display. |
| **Why** | Make ML outputs accessible to non-technical stakeholders. Clear UI = faster decisions. |
| **How** | Web framework (Streamlit/Dash) with interactive widgets. Responsive design for desktop use. |

#### Connect to Space and Ground Segments
| Aspect | Content |
|--------|---------|
| **What** | API layer that invokes the physics engine (HypatiaAdapter) and ML inference (trained models). |
| **Why** | Single source of truth for simulations. GUI should not duplicate physics logic. |
| **How** | REST endpoints: `/simulate` (run rollout), `/predict` (ML inference), `/recommend` (risk binning). |

#### Simulations and Testing (GUI)
| Aspect | Content |
|--------|---------|
| **What** | User-triggered simulation runs from the GUI, with real-time progress feedback. |
| **Why** | Enable interactive scenario exploration without command-line access. |
| **How** | Async job queue (Celery or simple threading). Progress bar in UI. Results cached for quick re-display. |

#### Analyze Results (GUI)
| Aspect | Content |
|--------|---------|
| **What** | Visualize risk tiers, GCC fraction time series, partition events, and recommended actions. |
| **Why** | Actionable decision support requires clear presentation, not just numbers. |
| **How** | Charts (Plotly/Altair), color-coded risk indicators (green/yellow/red), exportable reports. |

---

# Part B: Machine Learning Model Development — Design Rationale

## The High-Level Question: How Are We Choosing the Best ML Approach?

### Problem Characteristics That Drive Model Selection

| Characteristic | Implication for Model Choice |
|----------------|------------------------------|
| **Temporal data** — connectivity changes over time (t=0..T) | Need models that can capture sequences, not just static snapshots |
| **Graph-structured data** — satellites are nodes, ISLs are edges | Graph Neural Networks (GNNs) can exploit topology directly |
| **Interpretability matters** — engineering managers need to trust and explain | Classical models (RF, LightGBM) provide feature importance |
| **Limited labeled data** — simulated, but finite compute budget | Start with simpler models; avoid overfitting |
| **Binary classification** — Robust vs. Partitioned (primary task) | Standard classification metrics apply (accuracy, F1, ROC-AUC) |

---

### Why Two Models? (RF Baseline + Temporal GNN)

#### Model 1: Random Forest Baseline (Design-Time Risk)

**What it does:**
- Predicts partition risk from **design parameters only**: `num_planes`, `sats_per_plane`, `inclination_deg`, `altitude_km`
- Does NOT see the actual graph or failure realization

**Why this approach:**
1. **Fast and interpretable:** RF trains in seconds, provides feature importance
2. **Answers a design-time question:** "Given these constellation parameters, how risky is this architecture *in general*?"
3. **Baseline for comparison:** If a simple model achieves 90%+ accuracy on design features alone, that tells us density/geometry is highly predictive
4. **Matches industry practice:** Engineering teams often use rule-of-thumb heuristics; RF learns those patterns from data

**When to use:**
- Early design phase, before detailed simulation
- Quick screening of many candidate architectures
- Explainability to stakeholders ("altitude below 500km increases risk because...")

#### Model 2: Temporal GNN (GCLSTM) — Runtime Risk

**What it does:**
- Consumes a **sequence of graph snapshots** (one per timestep)
- Uses Graph Convolutional LSTM to learn:
  - **Spatial patterns:** which nodes/edges are structurally vulnerable
  - **Temporal dynamics:** how connectivity degrades over time
- Outputs: Robust (0) vs. Partitioned (1) classification

**Why this approach:**
1. **Captures dynamics:** Static models miss transient partitions and "near-miss" scenarios where GCC oscillates near threshold
2. **Exploits topology:** GNN sees the actual graph structure, not just scalar summaries
3. **Correct inductive bias:** Resilience is fundamentally a *time-domain property* — a constellation that briefly partitions then recovers is different from one that stays partitioned
4. **State-of-the-art for graph sequences:** GCLSTM (from PyTorch Geometric Temporal) is specifically designed for spatio-temporal graphs

**When to use:**
- After detailed simulation with failure injection
- Runtime monitoring (if deployed with live telemetry)
- When you need higher-fidelity predictions than design-only features allow

---

### Why Not Other Approaches?

| Alternative | Why Not (for Phase 1) |
|-------------|----------------------|
| **Static GNN (no LSTM)** | Misses temporal dynamics; treats each timestep independently |
| **Transformer on time series** | Overkill for current dataset size; harder to interpret |
| **Unsupervised (clustering)** | We have clear labels (partitioned/robust); supervised is more direct |
| **Reinforcement Learning** | Appropriate for *control* (routing decisions), not *prediction* (risk assessment) |
| **Regression (predict GCC fraction)** | Could do this, but binary classification aligns with decision tiers |

---

### How We Validate Model Choice

1. **Compare RF vs. GNN on same test set:**
   - If RF achieves ~90% and GNN achieves ~85% F1, RF is sufficient for design-time
   - If GNN significantly outperforms RF, temporal dynamics matter

2. **Feature importance analysis (RF):**
   - Confirms which design parameters drive risk
   - Sanity check: does it match physics intuition? (e.g., low altitude = more Earth obscuration = higher risk)

3. **Ablation studies (GNN):**
   - Remove temporal component → performance drop confirms temporal value
   - Remove edge features → confirms ISL properties matter

4. **Confusion matrix analysis:**
   - Are false negatives (missed partitions) or false positives (false alarms) more costly?
   - Tune threshold accordingly

---

# Part C: Satellite Constellation Development — Design Rationale

## The High-Level Question: Why This Simulation Approach?

### Why Physics-Based Simulation (Not Random Graphs)?

| Aspect | Physics-Based (Our Approach) | Random/Toy Graphs |
|--------|------------------------------|-------------------|
| **ISL connectivity** | Determined by orbital geometry + link budgets | Arbitrary edges based on probability |
| **Temporal evolution** | Satellites move → ISLs appear/disappear over time | Static or random rewiring |
| **Failure realism** | Persistent failures on actual t=0 edges | Random edge deletion |
| **Reproducibility** | SGP4 + epoch + seed = exact same graph sequence | Depends on random seed only |
| **Defense-ready** | Can explain every edge decision (physics) | "We used Erdős–Rényi with p=0.3" |

**Bottom line:** Toy graphs cannot answer "will *this specific constellation design* experience partitions under *realistic* failure scenarios?" Physics-based simulation can.

---

### Why Walker Delta Constellation Pattern?

1. **Industry standard:** Starlink, OneWeb, Kuiper all use variants of Walker Delta
2. **Parameterizable:** 4 numbers (planes, sats/plane, inclination, altitude) define the entire constellation
3. **Known trade-offs:**
   - More planes → better cross-track coverage, more seam complexity
   - Higher inclination → better polar coverage, more seam stress
   - Higher altitude → longer ISL visibility, higher latency, more radiation
4. **Design-space exploration:** By varying these 4 parameters, we can sample thousands of realistic architectures

---

### Why SGP4 Propagation?

1. **Accuracy:** Includes J2 perturbation (Earth oblateness), atmospheric drag approximation
2. **Standard:** Used by NORAD, space agencies, industry tools
3. **TLE compatibility:** Can validate against real satellite TLEs
4. **Determinism:** Same TLE + epoch = same position (no randomness)

**Fallback:** If SGP4 is unavailable, we use simplified Keplerian propagation (less accurate but still physics-based).

---

### Why +Grid ISL Topology?

The "+Grid" pattern connects each satellite to:
- **Intra-plane neighbors:** Satellite ahead and behind in same orbital plane
- **Inter-plane partners:** Corresponding satellite in adjacent planes

**Why this pattern:**
1. **Hardware constraint:** Real satellites have limited antenna pointing; +Grid requires only 4 ISL terminals
2. **Proven design:** Starlink, Iridium use variants of this pattern
3. **Seam handling:** Cross-seam links (between first and last plane) are explicitly tagged — these are the most stressed links
4. **Scalable:** Works for any (planes × sats/plane) configuration

---

### Why Persistent Failures (Not Dynamic)?

In Phase 1, failures are sampled at t=0 and persist for the entire rollout.

**Rationale:**
1. **Simplicity:** Easier to attribute outcomes to specific failure sets
2. **Worst-case analysis:** Persistent failure is more stressful than transient
3. **Reproducibility:** Given failure set + seed, exact same degraded graph sequence
4. **Phase 2 extension:** Dynamic failures (satellite reboots, link flapping) can be added later

---

### Why Temporal Evaluation (t=0..T)?

**The core Tier 1 principle:** Connectivity is evaluated at every timestep, not just t=0.

**Why this matters:**
1. **Orbital dynamics:** Satellites move; ISL visibility changes as Earth rotates beneath the constellation
2. **Transient vs. sustained:** A 5-minute partition (1 timestep) is less severe than a 60-minute partition (12 timesteps)
3. **Streak detection:** Consecutive partition timesteps indicate sustained degradation — captured by `max_partition_streak` metric
4. **ML training signal:** Temporal GNN needs the full sequence to learn degradation patterns

---

# Part D: Quick Reference — Advisor-Safe Talking Points

## "What is the core contribution?"

> An ML-powered decision-support tool that predicts satellite network partition risk from constellation design parameters and temporal connectivity simulations, enabling engineering managers to identify vulnerabilities before deployment.

## "Why ML instead of traditional simulation?"

> Traditional simulation tells you *what happens* for one scenario. ML learns patterns across thousands of scenarios, enabling:
> - Fast screening of new designs without re-running full simulations
> - Identification of which design parameters most strongly predict risk
> - Runtime risk estimation when combined with live telemetry (future work)

## "Why two models?"

> **RF baseline** answers design-time questions quickly and interpretably.  
> **Temporal GNN** captures dynamics that scalar features miss, for higher-fidelity runtime predictions.  
> Together they cover the full decision lifecycle: early design → detailed simulation → (future) operational monitoring.

## "What makes this 'Tier 1' / defense-ready?"

> 1. **Physics-based:** ISLs derived from orbital mechanics and link budgets, not random graphs
> 2. **Temporal:** Connectivity evaluated over time, not static snapshots
> 3. **Deterministic:** Explicit seeds, config hashes, and epoch for exact reproducibility
> 4. **Non-leaky labels:** Risk labels computed from graph state only, not from failure parameters

## "What's Phase 1 vs Phase 2?"

> **Phase 1 (complete):** Satellite-to-satellite connectivity, ISL partitions, RF + GNN models  
> **Phase 2 (planned):** Ground segment (gateways), end-to-end service availability, traffic modeling, GUI

---

# Appendix: Copy-Paste Content for Slide Boxes

## For "1. Data Collection & EDA" box:

**What?** Obtain Satellite Network routing, congestion, and architecture datasets and perform exploratory data analysis (EDA) to understand data structure, variability, and flow of network data.

**Why?** To acquire and understand network flow and possible failures in Satellite Network Architecture. EDA validates that training data captures realistic connectivity patterns before feeding it to ML models.

**How?** (1) Generate Tier 1 simulated truth datasets via physics-based temporal simulation (SGP4 + Monte Carlo). (2) Download external datasets (LEO congestion logs, NASA anomaly data) for validation. (3) Run EDA scripts to produce summary statistics. (4) Schema validation ensures data quality.

---

## For "4. Ground Segment" box:

**What?** Earth-based infrastructure: ground stations (gateways), user terminals, and backhaul to terrestrial networks.

**Why?** Connects the satellite mesh to end users. Gateway placement affects coverage, latency, and resilience. Required for "service availability" metrics.

**How?** (Phase 2) Define gateway locations → model Earth rotation and satellite passes → compute visibility windows with elevation masks → integrate SAT-GS links into temporal graph → add service availability labels.

---

## For "5. GUI Segment" box:

**What?** Graphical interface for engineering managers to explore constellation designs, visualize risk, and receive actionable recommendations.

**Why?** The thesis deliverable is a decision-support *tool*. The GUI makes ML outputs accessible to non-technical stakeholders and enables "what-if" scenario exploration.

**How?** (Phase 2) CLI scenario runner first → web dashboard (Streamlit/Dash) with parameter input, risk visualization, and action recommendations → API layer connecting to physics engine and trained models.

---

# Part E: Concrete Parameter Reference (With Reasoning)

This section provides the **exact values** used in the codebase, along with justifications for each choice. Alex can include these in his slides or use them to answer advisor questions about "what specific parameters are you using?"

---

## E.1 Constellation Design Space (Dataset Generation)

**Source:** `scripts/export_design_dataset.py` → `Tier1MonteCarloConfig`

| Parameter | Default Range | Reasoning |
|-----------|---------------|-----------|
| **Altitude** | 300 – 1,200 km | Covers LEO spectrum: 300km = very low (high drag, short life), 1200km = upper LEO (near Van Allen belts). Most operational constellations (Starlink, OneWeb) operate in 500-600km range. |
| **Inclination** | 30° – 98° | 30° = low-inclination (equatorial coverage bias), 98° = sun-synchronous (polar coverage). Starlink uses ~53°; polar constellations use ~87-98°. |
| **Orbital Planes** | 3 – 8 | Minimum viable mesh (3 planes) to moderately dense (8 planes). More planes = better cross-track coverage but more seam complexity. |
| **Satellites per Plane** | 4 – 12 | Sparse (4) to dense (12) per-plane. More sats/plane = better along-track coverage, more intra-plane redundancy. |
| **Phasing Factor** | 1 (default) | Walker Delta "F" parameter. F=1 is standard uniform phasing. |

**Temporal Parameters:**

| Parameter | Default | Reasoning |
|-----------|---------|-----------|
| **Duration** | 10 minutes | Captures ~1/10 of a 90-minute LEO orbit. Sufficient to observe ISL dynamics without excessive compute. |
| **Step Interval** | 60 seconds | 1-minute resolution balances fidelity vs. dataset size. Finer (10s) possible but 10× more data. |
| **Epoch** | J2000.0 (2000-01-01T12:00:00) | Standard astronomical reference epoch ensures determinism. Any fixed epoch works; J2000 is conventional. |

**Failure Injection:**

| Parameter | Default Range | Reasoning |
|-----------|---------------|-----------|
| **Node Failure Prob** | 0.0 – 0.2 (20%) | At most 20% of satellites fail. Higher would be catastrophic and unrealistic for typical missions. |
| **Edge Failure Prob** | 0.0 – 0.3 (30%) | ISL failures can be more common than node failures (laser pointing errors, thermal issues). 30% upper bound stresses the network. |
| **Failure Persistence** | Entire rollout | Failures sampled at t=0 persist for all timesteps. Models "permanent" failures (hardware loss, not transient glitches). |

**Why these ranges?**
- Cover realistic operational scenarios (normal ops to stressed conditions)
- Produce balanced dataset (~50% partitioned, ~50% robust) for ML training
- Enable design-space exploration across small-to-medium constellations

---

## E.2 Physics Engine Constants

**Source:** `src/satnet/network/hypatia_adapter.py`

### Physical Constants

| Constant | Value | Reasoning |
|----------|-------|-----------|
| **Earth Radius** | 6,371 km | Standard WGS84 mean radius |
| **Earth μ (GM)** | 398,600.4418 km³/s² | Standard gravitational parameter |
| **J2 Perturbation** | 1.08263 × 10⁻³ | Earth oblateness coefficient (used by SGP4) |
| **Atmosphere Buffer** | 80 km | Grazing height for Earth obscuration checks. Link must clear Earth + 80km buffer. |

### Optical ISL Link Budget (1550nm Laser)

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **Wavelength** | 1550 nm | Standard telecom wavelength; eye-safe, mature technology |
| **TX Power** | 37 dBm (5 W) | Typical space-qualified laser power |
| **Aperture** | 10 cm | Moderate telescope size; fits on small-to-medium satellites |
| **Receiver Sensitivity** | -45 dBm | Achievable with APD receivers |

**Why optical for ISLs?**
- Lower power than RF for same data rate
- No spectrum licensing needed
- Higher bandwidth potential (100+ Gbps)
- Industry trend (Starlink uses optical ISLs)

### RF ISL Link Budget (Ka-Band 28 GHz)

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **Frequency** | 28 GHz | Ka-band; common for satellite communications |
| **TX Power** | 30 dBm (1 W) | Modest power for ISL |
| **Antenna Gain** | 40 dBi | High-gain parabolic or phased array |
| **Receiver Sensitivity** | -90 dBm | Standard RF receiver |
| **Rain Margin** | 10 dB | Accounts for atmospheric attenuation (ground links) |

**Why RF as fallback?**
- More mature technology
- Works through some obstructions
- Useful for SAT-GS links (Phase 2) where atmosphere matters

---

## E.3 Labeling / Metrics Parameters

**Source:** `src/satnet/metrics/labels.py`, `src/satnet/simulation/monte_carlo.py`

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **GCC Threshold** | 0.8 (80%) | If Giant Connected Component contains <80% of nodes, network is "partitioned". 80% allows minor isolation while flagging significant fragmentation. |
| **Partition Detection** | `gcc_frac < threshold` | Pure function: looks only at graph topology, not failure parameters (non-leaky). |
| **Partition_any Label** | True if *any* timestep is partitioned | Conservative: flags designs that partition even briefly. |
| **Max Partition Streak** | Longest consecutive partitioned timesteps | Captures sustained degradation vs. transient glitches. |

**Why 80% threshold?**
- 100% would flag any single-node isolation (too sensitive)
- 50% would miss serious fragmentation (too permissive)
- 80% is a balanced operational threshold used in network reliability literature

---

## E.4 Random Forest Baseline Parameters

**Source:** `scripts/train_design_risk_model.py` → `RiskModelConfig`

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **Algorithm** | RandomForestClassifier | Robust, interpretable, handles mixed feature types, provides feature importance |
| **n_estimators** | 300 | Enough trees for stable predictions; diminishing returns beyond ~500 |
| **max_depth** | None (unlimited) | Let trees grow fully; RF handles overfitting via bagging |
| **class_weight** | balanced | Handles class imbalance by weighting minority class |
| **test_size** | 0.2 (20%) | Standard 80/20 train/test split |
| **random_state** | 42 | Fixed seed for reproducibility |

**Features (design-time only):**

| Feature | Type | Reasoning |
|---------|------|-----------|
| `num_planes` | int | More planes = more cross-plane redundancy |
| `sats_per_plane` | int | More sats = denser coverage |
| `inclination_deg` | float | Affects coverage pattern and seam stress |
| `altitude_km` | float | Affects ISL range, orbital period, Earth obscuration |

**Why RF as baseline?**
1. Fast to train (seconds on 2000 samples)
2. Provides feature importance → explains *why* a design is risky
3. No hyperparameter tuning needed for decent results
4. Establishes performance floor for temporal GNN to beat

---

## E.5 Temporal GNN (GCLSTM) Parameters

**Source:** `scripts/train_gnn_model.py`, `src/satnet/models/gnn_model.py`

### Architecture

| Component | Value | Reasoning |
|-----------|-------|-----------|
| **Model** | GCLSTM (Graph Convolutional LSTM) | Combines spatial (GCN) and temporal (LSTM) learning in one layer |
| **K (Chebyshev order)** | 1 | 1-hop neighborhood; keeps computation tractable |
| **Hidden Dimension** | 64 | Moderate capacity; larger (128, 256) possible but risk overfitting on 2000 samples |
| **Output Classes** | 2 | Binary: Robust (0) vs. Partitioned (1) |
| **Pooling** | Global Mean Pool | Aggregates node embeddings to single graph embedding |
| **Classifier** | Linear (64 → 2) | Simple linear head on graph embedding |

### Training

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **Epochs** | 20 | Sufficient for convergence on this dataset size |
| **Learning Rate** | 0.01 | Standard Adam LR; not too aggressive |
| **Optimizer** | Adam | Adaptive learning rates; works well out-of-box |
| **Loss** | CrossEntropyLoss | Standard for classification |
| **Train/Test Split** | 80/20 | Matches RF baseline for fair comparison |
| **Seed** | 42 | Reproducibility |

### Node Features (per satellite, per timestep)

| Feature | Dimension | Value Range | Reasoning |
|---------|-----------|-------------|-----------|
| `plane_idx_normalized` | 1 | [0, 1] | Which orbital plane (normalized) |
| `sat_in_plane_normalized` | 1 | [0, 1] | Position within plane (normalized) |
| `exists` | 1 | 1.0 | Constant "node exists" flag |
| **Total** | 3 | — | Minimal but sufficient for structure learning |

### Edge Features (per ISL, per timestep)

| Feature | Dimension | Normalization | Reasoning |
|---------|-----------|---------------|-----------|
| `distance_km` | 1 | ÷ 10,000 | ISL distance (longer = weaker) |
| `margin_db` | 1 | ÷ 100 | Link budget margin (higher = more reliable) |
| `link_type` | 1 | ÷ 2 | 0=intra-plane, 1=inter-plane, 2=seam |
| `link_mode` | 1 | 0 or 1 | 0=optical, 1=RF |
| **Total** | 4 | — | Captures link quality and topology role |

**Why GCLSTM?**
1. **Temporal:** LSTM captures degradation patterns over time
2. **Spatial:** GCN respects graph structure (not just node features)
3. **Proven:** Used in traffic prediction, social network analysis
4. **Available:** PyTorch Geometric Temporal provides tested implementation

---

## E.6 Risk Binning (Decision Support Output)

**Source:** `src/satnet/metrics/risk_binning.py`

| Tier | Score Range | Label | Recommended Action | Reasoning |
|------|-------------|-------|-------------------|-----------|
| **1** | > 0.8 | Healthy | No Action | High confidence in connectivity; nominal operations |
| **2** | 0.5 – 0.8 | Watchlist | Schedule Diagnostics | Moderate risk; monitor closely, plan contingencies |
| **3** | < 0.5 | Critical | Immediate Maneuver | High partition probability; engineering intervention needed |

**Why these thresholds?**
- **0.8:** Industry convention for "high reliability" (similar to "four nines" thinking)
- **0.5:** Coin-flip confidence; below this, the model is predicting likely failure
- Thresholds are configurable for sensitivity analysis

---

## E.7 Summary Table (Copy-Paste for Slides)

| Category | Parameter | Value | Why |
|----------|-----------|-------|-----|
| **Constellation** | Altitude | 300–1200 km | Full LEO range |
| | Inclination | 30–98° | Equatorial to polar |
| | Planes | 3–8 | Sparse to moderate density |
| | Sats/plane | 4–12 | Coverage variation |
| **Temporal** | Duration | 10 min | ~1/9 orbit |
| | Step | 60 sec | 1-min resolution |
| **Failures** | Node prob | 0–20% | Realistic stress |
| | Edge prob | 0–30% | ISL-heavy stress |
| **Labels** | GCC threshold | 80% | Partition = <80% connected |
| **RF Model** | Trees | 300 | Stable ensemble |
| | Features | 4 (design params) | Interpretable baseline |
| **GNN Model** | Hidden dim | 64 | Balanced capacity |
| | Epochs | 20 | Convergence |
| | Node features | 3 | Position + existence |
| | Edge features | 4 | Distance, margin, type, mode |
| **Risk Tiers** | Healthy | >0.8 | No action |
| | Watchlist | 0.5–0.8 | Monitor |
| | Critical | <0.5 | Intervene |

---

*End of document*
