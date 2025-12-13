# Tier 1 — Correlated Failure Models for Satellite Constellations
**Implementation Plan (Code-Ready)**  
Version: 1.0  
Scope: constellation reliability simulation **beyond iid Bernoulli**, including **plane-level outages**, **radiation storms**, **correlated ISL failures**, and **Markov‑modulated failure rates**.

---

## 0) Goal and Deliverables

### Goal
Provide a composable set of **generative failure models** you can drop into:
- a constellation network simulator (graph + routing), or
- a Monte‑Carlo availability tool (capacity + connectivity metrics)

…while capturing **correlation** that iid per-satellite Bernoulli models miss.

### Deliverables
1. **Interfaces** for failure processes: node failures, link failures, and “shocks”.
2. **Sampling algorithms** for each correlated failure type.
3. **Calibration** playbook for low-data scenarios (CCS/MCCV scoring + Bayesian priors).
4. **Validation tests** (invariants + statistical sanity checks).
5. **Runtime controls** (importance sampling + parallelization hooks).

---

## 1) Core Modeling Abstractions

### 1.1 Entities
- **Satellite node**: `sat_id`, `plane_id`, `slot_id`, `epoch`, `health_state`
- **ISL edge**: `(u, v)`, `edge_type ∈ {intra, inter}`, `terminal_id_u`, `terminal_id_v`
- **Constellation graph**: `G(t) = (V(t), E(t))` time-varying adjacency

### 1.2 Failure Events
Represent everything as events to keep your sim deterministic + debuggable:
```text
Event = {
  t: float,                        # simulation time (seconds)
  kind: "node_fail"|"node_recover"|"edge_fail"|"edge_recover"|"shock",
  targets: {sat_ids?:[], edge_ids?:[]},
  meta: {...}                      # cause tags + sampled latent variables
}
```

### 1.3 Failure Process Interface
A minimal “pluggable” model interface:
```python
class FailureProcess:
    def step(self, t: float, dt: float, state: "SimState") -> list[Event]:
        ...
```

Where `SimState` gives access to:
- topology `G(t)`
- satellite properties (plane, age, subsystem flags)
- current link metrics (distance, pointing residuals, SNR, etc.)
- environment regime (quiet/storm) if using HMM/MMPP

---

## 2) Plane-Level Outages

### 2.1 When to Use
Use when a constellation is vulnerable to **batch defects**, **shared deployment**, or **localized orbit-plane hazards**, causing multiple satellites within a plane to fail together.

### 2.2 Model A: Beta‑Factor Common Cause Failure (CCF)
You model plane failure as:
- independent per-node failures (base rate), plus
- a **plane-wide** common-cause event with probability proportional to `β`

**Parameters**
- `λ_T` total (effective) node failure rate (or time-step failure probability)
- `β ∈ [0,1]` fraction of failures that are common-cause (plane-wide)
- `plane_id → sat_ids[]` mapping

**Sampling (discrete-time Monte Carlo)**
At each time step for each plane:
1. Compute `p_plane = β * p_base_plane` (define base consistently)
2. Sample `u ~ U(0,1)`. If `u < p_plane`, emit `node_fail` for **all sats in plane**.
3. Else sample independent failures for sats in plane using `(1-β)` portion.

**Implementation snippet**
```python
def beta_factor_plane_step(t, dt, plane_sats, p_node, beta):
    events = []
    # plane-level common cause
    p_ccf = beta * (1 - (1 - p_node)**len(plane_sats))  # base: "any fails" in plane
    if rand() < p_ccf:
        events.append(Event(t, "node_fail", {"sat_ids": plane_sats}, {"cause":"plane_ccf"}))
        return events

    # independent residual failures
    p_ind = (1 - beta) * p_node
    failed = [s for s in plane_sats if rand() < p_ind]
    if failed:
        events.append(Event(t, "node_fail", {"sat_ids": failed}, {"cause":"independent"}))
    return events
```

### 2.3 Calibration: CCS/MCCV Scoring → β (Low Data)
Score 7 factors (1/5/10): separation, diversity, maturity, analysis/feedback, procedures, training, environmental control.

Compute:
- `CCS = Σ CCS_i`
- `β = (CCS / CCS_max) * MCCV`
  - `CCS_max = 70` (7 categories × 10)
  - `MCCV ≈ 0.10` (conservative space-industry ceiling)

**Operational tip**: treat β as a *distribution*, not a constant, in early phases:
- `β ~ Beta(a,b)` prior centered at your CCS estimate.

### 2.4 Model B: Partial Plane Degradation (k‑out‑of‑n)
If you need “some subset dies” rather than “all die”, swap the plane-wide all-or-nothing event with:
- **Alpha‑factor** or **Multiple Greek Letter** style sampling
- Practical simplification: sample `K ~ BetaBinomial(n, α, β_bb)` and fail `K` random sats in plane.

---

## 3) Radiation Storm Events (Correlated Node Failures)

### 3.1 When to Use
When space weather produces bursts of failures/anomalies across many planes simultaneously.

### 3.2 Model A: Shock Arrival Process (Poisson)
Storm arrivals: inter-arrival `ΔT ~ Exponential(λ_storm)`.

On arrival, emit `Event(kind="shock", meta={severity,...})`.

**Implementation**
```python
class PoissonShockProcess:
    def __init__(self, lam_per_sec):
        self.lam = lam_per_sec
        self.next_t = sample_exponential(self.lam)
    def step(self, t, dt, state):
        events=[]
        while t + dt >= self.next_t:
            events.append(Event(self.next_t, "shock", {}, {"cause":"radiation_storm"}))
            self.next_t += sample_exponential(self.lam)
        return events
```

### 3.3 Model B: Beta‑Binomial Fail Count (Given Shock)
Given a shock at time `t`, number of failed satellites `K` among `N_exposed` is:
- `K ~ BetaBinomial(N_exposed, α, β_bb)`

Interpretation:
- A shock induces a random per-sat failure probability `p ~ Beta(α, β_bb)`,
- Then `K ~ Binomial(N_exposed, p)`.

**Sampling**
1. Choose `N_exposed`: all sats, or only those in certain altitude/inclination “shell”.
2. Sample `p ~ Beta(α, β_bb)`
3. Sample `K ~ Binomial(N_exposed, p)`
4. Fail a random subset of size `K` (or weighted by “hardness” / shielding score).

**Parameterization from mean + correlation**
Often you’ll have:
- mean failure probability `E[p] = μ`
- intraclass correlation `ρ` (how correlated are outcomes)

Convert:
- `α = μ * (1/ρ - 1)`
- `β_bb = (1-μ) * (1/ρ - 1)`

(Valid for `0 < ρ < 1`.)

### 3.4 Model C: Hidden Markov “Quiet/Storm”
Instead of discrete shocks, model environment regime:
- hidden state `S_t ∈ {Quiet, Storm}`
- transition matrix `A`
- failure emission rates `p_fail(Quiet)`, `p_fail(Storm)` or per-node hazard rates.

**Sampling**
- Step regime by `A`
- Fail nodes with state-dependent probability

This is best once you have **time-series anomalies**.

### 3.5 “Starlink-style” Drag Catastrophes
Optional extension: a storm can increase atmospheric density → drag → increased deorbit risk in low orbits.
Model as shock that increases *deorbit hazard* for satellites below altitude threshold.

---

## 4) Correlated Link Failures (ISLs)

### 4.1 When to Use
When multiple ISLs fail together due to shared pointing jitter, thermal distortion, or ADCS issues, especially for optical ISLs.

### 4.2 Model A: Beta‑Factor on Pointing Subsystem
Treat “ADCS/beam-steering fault” as common-mode event for all links on a satellite:
- with probability `p_common`, mark all incident edges down.

Cheap and useful when you have no multivariate link telemetry.

### 4.3 Model B: Copula for Joint Link Quality
Represent per-link degradation metric (e.g., pointing error magnitude, SNR margin) as random variables with given marginals and **dependent joint** via copula.

#### Two-stage estimation (recommended)
1. Fit marginals for each link metric: `F_i`.
2. Transform samples `u_i = F_i(x_i)` (uniform in [0,1]).
3. Fit copula `C(u_1,...,u_d)` via MLE or Kendall’s τ.

#### Sampling to determine failures
At each step:
1. Sample correlated `u_1..u_d ~ C`.
2. Convert back: `x_i = F_i^{-1}(u_i)`.
3. Determine edge up/down via threshold (e.g., `margin_i - loss(x_i) > 0`).

#### Copula choice
- **Gumbel**: upper tail dependence (big joint outages during extremes)
- **Clayton**: lower tail dependence
- **Frank**: symmetric dependence

### 4.4 Model C: Vine Copula (High-dimensional)
For many links at once (constellation-scale), use vine copula decomposition.
This is heavier; only worth it with sufficient telemetry.

### 4.5 Practical “good enough” correlated link model (low data)
If you want correlation but not the copula machinery:

- Sample a **satellite-level jitter multiplier** `J_s(t)` common to all its terminals.
- Sample a **plane-level thermal term** `T_p(t)` common to a plane.
- Per-link loss: `L_point(u,v) = base(u,v) + a*J_u + a*J_v + b*T_plane(u) + ε_uv`
- Edge down if `margin - L_point < 0`.

This gives correlation via shared latent factors.

---

## 5) Markov‑Modulated Failure Rates (Time-Varying Hazards)

### 5.1 When to Use
When satellite failure rates are non-stationary (infant mortality → useful life → wear-out).

### 5.2 Model A: HMM on Health Regime
Hidden states: e.g. `Healthy`, `Degraded`, `Failing`.
Each has failure probability per step or hazard `λ_state`.

**Simulation**
- propagate state via transition matrix `A`
- sample failure with state emission rate

### 5.3 Model B: MMPP (Markov‑Modulated Poisson Process)
If you want failures as a Poisson count process whose rate depends on a CTMC regime.
Useful for “burstiness”.

### 5.4 Model C: 2‑Weibull Segmented Lifetime
Simpler alternative without hidden states.
- segment 1: Weibull with shape `<1` (decreasing hazard)
- segment 2: Weibull with shape `≈1` (constant hazard)
(Optionally add segment 3 for wear-out.)

**Implementation**
- sample failure time from piecewise survival function
- or implement hazard-based step test.

---

## 6) Parameter Calibration with Minimal Data

### 6.1 Use priors + update later (Bayesian posture)
Treat correlation parameters as uncertain:
- `β_ccf ~ Beta(a,b)` (plane-level)
- `ρ_storm ~ Beta(a,b)` (storm-induced correlation)
- copula τ interval `[τ_low, τ_high]`

Then do Bayesian updating when ops data arrives.

### 6.2 “Historical analogy” bootstraps
Start with:
- similar mission families
- component vendor FIT/MTBF data
- published anomaly/space weather correlations

### 6.3 Practical calibration checklist
For each model you turn on, define:
- **what you can observe** in ops (failures, reboots, degraded links)
- an **observable proxy** for latent causes (e.g., wheel speed jitter, temperature)
- how you’ll update priors every N weeks.

---

## 7) Simulation Wiring Patterns

### 7.1 Event-driven pipeline
At each `dt`:
1. `events += shock_process.step(t,dt)`
2. `events += plane_failure_process.step(t,dt)`
3. `events += mmpp/hmm_process.step(t,dt)`
4. `events += link_failure_process.step(t,dt)`
5. Apply events to `SimState` (mutate node/edge status)
6. Compute KPIs (connectivity, diameter, availability, throughput proxy)

### 7.2 Avoid double-counting
If a node fails, its incident edges implicitly go down — don’t also sample them independently unless you explicitly want “edge issues before node loss”.

### 7.3 Deterministic replay
Store RNG seeds per process and log events. This is non-negotiable for debugging.

---

## 8) Validation: Invariants and Statistical Tests

### 8.1 Structural invariants (always-on)
- Node/edge status is boolean (no tri-state unless you model degraded explicitly)
- Graph symmetry for ISLs (if u↔v links are bidirectional)
- No edge is “up” if either endpoint node is down

### 8.2 Statistical sanity checks (per model)
**Plane CCF**
- Distribution of “multi-sat same-plane failures” matches β target.

**Shock + Beta-Binomial**
- Given shocks, `Var(K)` should exceed binomial variance:
  - `Var_BB(K) = n μ(1-μ) * (1 + (n-1)ρ)`
- Validate mean outage time fraction.

**Copula link failures**
- Empirical Kendall’s τ / tail dependence matches fitted copula.

**HMM/MMPP**
- Regime dwell times roughly match transitions
- Failure bursts align with high-rate regimes

### 8.3 System-level validation
- Compare iid baseline vs correlated models:
  - correlated should produce fatter tail in “constellation unavailable” events
  - verify worst-case outage probabilities increase as expected

---

## 9) Keeping Monte Carlo Runtime Manageable

### 9.1 Importance Sampling (IS)
Bias toward rare correlated disasters:
- increase shock arrival rate in sampling distribution
- increase β_ccf in sampling distribution
Then weight samples by likelihood ratio `w = f / f_tilde`.

### 9.2 Variance reduction
- stratify by plane, shell, or batch
- control variates: compare to iid baseline
- early termination when graph irrecoverably disconnected

### 9.3 Parallelism
Embarrassingly parallel over trials. Keep per-trial RNG streams independent.

---

## 10) Recommended Default Parameter Starting Points (If You Have Zero Data)
These are placeholders to get a simulator running (treat them as priors, not truth).

- Plane-level CCF β: `0.01 – 0.05` (low-to-moderate coupling)
- Storm arrival λ: `2 – 10 severe events/year` (define “severe” explicitly)
- Storm-induced mean failure probability μ: `0.001 – 0.02` depending on shielding
- Storm correlation ρ: `0.01 – 0.10`
- Link correlation (copula τ): `0.1 – 0.4` if ADCS-driven; else `~0.05`
- HMM regimes: 2-state (`Quiet`, `Storm`) before going multi-state

---

## 11) Minimal Code Skeleton (Python)

```python
from dataclasses import dataclass
import random
from typing import List, Dict, Tuple

@dataclass
class Event:
    t: float
    kind: str
    targets: dict
    meta: dict

class SimState:
    def __init__(self, planes: Dict[int, List[int]], edges: List[Tuple[int,int]]):
        self.planes = planes
        self.node_up = {sat: True for sats in planes.values() for sat in sats}
        self.edge_up = {(u,v): True for (u,v) in edges} | {(v,u): True for (u,v) in edges}

    def apply(self, ev: Event):
        if ev.kind == "node_fail":
            for s in ev.targets.get("sat_ids", []):
                self.node_up[s] = False
                # implicit edge down
                for (u,v) in list(self.edge_up.keys()):
                    if u == s or v == s:
                        self.edge_up[(u,v)] = False
        if ev.kind == "edge_fail":
            for e in ev.targets.get("edge_ids", []):
                self.edge_up[e] = False
                self.edge_up[(e[1], e[0])] = False

class FailureProcess:
    def step(self, t: float, dt: float, state: SimState) -> List[Event]:
        return []

class BetaPlaneCCF(FailureProcess):
    def __init__(self, p_node: float, beta: float):
        self.p_node = p_node
        self.beta = beta
    def step(self, t, dt, state):
        out=[]
        for plane_id, sats in state.planes.items():
            if any(not state.node_up[s] for s in sats):
                # optional: skip already-failed sats/planes; depends on your policy
                pass
            p_ccf = self.beta * (1 - (1 - self.p_node)**len(sats))
            if random.random() < p_ccf:
                out.append(Event(t, "node_fail", {"sat_ids": sats}, {"cause":"plane_ccf","plane":plane_id}))
            else:
                p_ind = (1 - self.beta) * self.p_node
                failed=[s for s in sats if state.node_up[s] and random.random() < p_ind]
                if failed:
                    out.append(Event(t, "node_fail", {"sat_ids": failed}, {"cause":"independent","plane":plane_id}))
        return out
```

---

## 12) What to Implement Next (Suggested Build Order)

1. **iid baseline** (so you have a reference)
2. **Plane β-factor CCF** (largest effect, easiest)
3. **Poisson shocks + Beta-Binomial** (radiation storms)
4. **Satellite-level latent jitter model** (cheap correlated link failures)
5. **HMM/MMPP regimes** (time-varying hazard)
6. **Copulas/vines** only if you have telemetry and need fidelity

---

## Appendix A — Parameter “Contracts” to Put in Config

```yaml
failure_models:
  plane_ccf:
    enabled: true
    p_node_per_step: 1.0e-6
    beta: 0.03
  radiation_shock:
    enabled: true
    lambda_per_year: 5
    beta_binomial:
      mean_mu: 0.005
      corr_rho: 0.05
  link_corr_latent:
    enabled: true
    jitter_sigma: 1.0
    thermal_sigma: 0.5
  hmm_regimes:
    enabled: false
    states: ["Quiet","Storm"]
    A: [[0.999,0.001],[0.05,0.95]]
    p_fail: [1.0e-6, 1.0e-4]
```

---

*Source basis: generated from the user-provided report “Correlated Failure Models for Satellite Constellations Beyond iid Bernoulli Assumptions”, preserving model taxonomy, generative steps, and low-data calibration patterns (CCS/MCCV, beta-binomial shocks, copulas, HMM/MMPP) in an implementation-ready format.*
