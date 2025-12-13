# Tier 1 Implementation Plan: Probabilistic ISL Availability Models (LEO–LEO)

**Goal:** Add a probabilistic edge availability / failure model for **LEO–LEO ISLs** (optical 1550 nm and Ka-band 28 GHz) that maps **geometry + nominal margin → P(link up)**, and integrates cleanly with the existing **time-varying topology + routing** pipeline.

This doc is **code-oriented**: equations → functions → data structures → tests.

---

## 0) Scope and Non-Goals

### In-scope
- Deterministic link-budget components needed to compute **effective margin** `M_eff(t)` as satellites move.
- Stochastic impairment models (primarily **pointing jitter** for optical; optional RF fading models for Ka).
- Availability mapping functions: `P_up = 1 - P_out(M_eff, …)`.
- Conversion to an **edge Bernoulli failure** process at each sim timestep.
- Hooks for **Monte Carlo** and/or **analytical** outage probability.
- Validation + invariants for graph-level behavior under probabilistic outages.

### Explicitly out-of-scope (for *LEO–LEO only*)
- Rain / atmospheric attenuation (should be **0** for true LEO–LEO). Keep code paths for completeness if you reuse for GSL later.
- Complex turbulence models (scintillation, beam spread, Strehl) unless you decide to simulate cross-segment (LEO↔ground).
- PHY/MAC details beyond a “link is up / down” and optional latency + capacity scaling.

---

## 1) Core Interfaces

### 1.1 Public API (simulation-facing)
```python
class LinkAvailabilityModel(Protocol):
    def p_up(self, edge: "Edge", t: "Time") -> float:
        ...

    def sample_up(self, edge: "Edge", t: "Time", rng: np.random.Generator) -> bool:
        ...
```

### 1.2 Required inputs per edge at time `t`
- `distance_km(t)` (from propagation)
- `los_clear(t)` (Earth-occlusion / horizon check; for ISL this is the Earth-block LOS test)
- `link_type`: `"optical"` or `"ka"`
- `system_params`: per-link-type parameters (freq/λ, tx power, apertures/antenna gains, impl losses, coding gain, reference conditions)
- `nominal_margin_db`: `M_nominal` defined at **reference geometry** `(d_ref, ... )`

---

## 2) Deterministic Geometry → Losses → Effective Margin

### 2.1 RF Free Space Loss (Ka-band; distance km, frequency GHz)
\[
L_{FS}(d,f) = 20\log_{10}(d) + 20\log_{10}(f) + 92.45
\]

### 2.2 Optical Free Space Loss (distance meters, wavelength meters)
\[
L_{FS}(L,\lambda) = 20\log_{10}\left(\frac{4\pi L}{\lambda}\right)
\]
(For practical coding, you can keep the RF-style constant form if you consistently convert units.)

### 2.3 Effective Margin definition
Define your nominal margin at a **reference distance** `d_ref`:
\[
M_{eff}(t) = M_{nom} - \left(L_{FS}(d(t)) - L_{FS}(d_{ref})\right) - L_{extra}(t)
\]

Where `L_extra(t)` covers deterministic deltas you model (e.g., temperature, pointing bias, etc.).  
For **pure LEO–LEO**, you usually set atmospheric/rain = 0 and keep:
- optical: pointing loss treated stochastically (don’t subtract expected value unless you explicitly model mean loss)
- RF: optional fading treated stochastically; implementation losses already in `M_nom`

### 2.4 LOS gating (hard constraint)
Compute `los_clear(t)` using Earth-occlusion:
\[
\frac{\|x_u \times x_v\|}{\|x_u - x_v\|} > R_E
\]
If `los_clear=False`, then `P_up=0`.

---

## 3) Availability Mapping Strategies (choose one per link type)

You want a **pluggable** mapping from `M_eff` → `P_out`.

### 3.1 Ka-band (LEO–LEO): recommended default
**LEO–LEO Ka-band** is dominated by deterministic FSPL + implementation/coding; atmospheric/rain = 0.  
If you still want stochasticity, pick a simple fading distribution:

#### Option A: “Exponential in margin” model (fast, tunable)
\[
P_{out}(M) = \min(1,\; A \cdot 10^{-M/10})
\]
Where `A` is calibrated so that at your nominal margin you get your desired availability.

**Calibration:** Choose target `P_out_target` at `M_target`:
\[
A = P_{out\_target} \cdot 10^{M_{target}/10}
\]

#### Option B: Rayleigh outage model (use when you truly want multipath-like behavior)
\[
P_{out} = 1 - e^{-\gamma_{th}/\Gamma}
\]
You must map margin to average SNR: `Γ = Γ_ref * 10^(M_eff/10)` and set `γ_th`.

**Note:** For space-space LoS RF, Rayleigh is often pessimistic; consider **Rician/Nakagami-m** if you care.

### 3.2 Optical (1550nm): pointing-jitter dominated
Model outage as probability that **pointing-induced loss** exceeds margin.

You have two realistic implementation tiers:

#### Tier 1 (recommended now): Monte Carlo pointing loss
- Model pointing error angle `θ` as 2D Gaussian with variance `σ_j^2`.
- Convert to received coupling / pointing loss using a Gaussian beam approximation.
- Estimate `P_out` by sampling `N` times.

Pros: simple + extensible.  
Cons: needs CPU (but you can amortize/caching).

#### Tier 2 (analytical): Beta model tail probability
If you model normalized pointing gain `X ~ Beta(α, β)` (often `Beta(β_param, 1)` simplification),
and `Loss_dB = -10 log10(X)`, then:
\[
P_{out}(M) = P(Loss\_dB > M) = P(X < 10^{-M/10}) = F_X(10^{-M/10})
\]
Where `F_X` is the Beta CDF.

**If** `X ~ Beta(k, 1)`, then `F_X(x)=x^k`, so:
\[
P_{out}(M) = (10^{-M/10})^k = 10^{-kM/10}
\]
This yields a very clean exponential model in `M`.

**Compute k (= β_param) from beam divergence + jitter variance:**
\[
\beta_{param} = \frac{\theta_B^2}{4\sigma_j^2}
\]

---

## 4) Data Model

### 4.1 System parameters (typed)
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class KaParams:
    freq_ghz: float = 28.0
    d_ref_km: float = 2000.0
    # availability mapping
    model: str = "exp_margin"  # exp_margin | rayleigh | none
    p_out_target: float = 1e-6
    m_target_db: float = 3.0
    A: float | None = None  # if None, calibrate from target
    gamma_th: float | None = None
    gamma_ref: float | None = None

@dataclass(frozen=True)
class OpticalParams:
    wavelength_m: float = 1550e-9
    d_ref_km: float = 2000.0
    required_margin_db: float = 1.0  # SDA-like
    model: str = "beta_tail"  # beta_tail | mc_pointing | exp_margin
    theta_B_urad: float = 20.0  # beam half-divergence
    sigma_j_urad: float = 2.0   # pointing jitter std
    # MC controls
    mc_samples: int = 256
```

### 4.2 Edge schema
```python
@dataclass
class Edge:
    u: int
    v: int
    kind: str  # "ISL"
    link_type: str  # "optical" | "ka"
    nominal_margin_db: float
    params: KaParams | OpticalParams
```

---

## 5) Implementation: Reference Functions

### 5.1 Free space loss utilities
```python
import numpy as np

def fspl_ka_db(d_km: float, f_ghz: float) -> float:
    return 20*np.log10(d_km) + 20*np.log10(f_ghz) + 92.45

def fspl_optical_db(d_km: float, wavelength_m: float) -> float:
    d_m = d_km * 1000.0
    return 20*np.log10(4*np.pi*d_m / wavelength_m)
```

### 5.2 Effective margin
```python
def effective_margin_db(edge: Edge, d_km: float) -> float:
    if edge.link_type == "ka":
        p: KaParams = edge.params  # type: ignore
        L_ref = fspl_ka_db(p.d_ref_km, p.freq_ghz)
        L = fspl_ka_db(d_km, p.freq_ghz)
    else:
        p: OpticalParams = edge.params  # type: ignore
        L_ref = fspl_optical_db(p.d_ref_km, p.wavelength_m)
        L = fspl_optical_db(d_km, p.wavelength_m)

    L_add = (L - L_ref)
    return edge.nominal_margin_db - L_add
```

### 5.3 Ka-band outage mapping
```python
def p_out_ka(M_db: float, p: KaParams) -> float:
    if p.model == "none":
        return 0.0 if M_db >= 0 else 1.0

    if p.model == "exp_margin":
        A = p.A
        if A is None:
            A = p.p_out_target * 10**(p.m_target_db/10.0)
        return float(min(1.0, A * 10**(-M_db/10.0)))

    if p.model == "rayleigh":
        assert p.gamma_th is not None and p.gamma_ref is not None
        Gamma = p.gamma_ref * 10**(M_db/10.0)
        return float(1.0 - np.exp(-p.gamma_th / Gamma))

    raise ValueError(f"Unknown ka model {p.model}")
```

### 5.4 Optical outage mapping (beta tail)
```python
def beta_param(p: OpticalParams) -> float:
    theta_B = p.theta_B_urad * 1e-6
    sigma_j = p.sigma_j_urad * 1e-6
    return (theta_B**2) / (4.0 * sigma_j**2)

def p_out_optical_beta(M_db: float, p: OpticalParams) -> float:
    k = beta_param(p)
    # If X ~ Beta(k,1), outage is P(X < 10^{-M/10}) = 10^{-kM/10}
    return float(min(1.0, 10**(-k * M_db / 10.0)))
```

### 5.5 Optical outage mapping (MC pointing; simple coupling)
```python
def p_out_optical_mc(M_db: float, p: OpticalParams, rng: np.random.Generator) -> float:
    # Model pointing error magnitude r from 2D Gaussian -> Rayleigh(sigma_j)
    sigma = p.sigma_j_urad * 1e-6
    r = rng.rayleigh(scale=sigma, size=p.mc_samples)
    # Approximate coupling ~ exp(-(r/theta_B)^2). This is a common Gaussian beam overlap proxy.
    theta_B = p.theta_B_urad * 1e-6
    X = np.exp(-(r/theta_B)**2)
    loss_db = -10*np.log10(X + 1e-15)
    return float(np.mean(loss_db > M_db))
```

### 5.6 Unified probability-of-up
```python
def p_up(edge: Edge, d_km: float, los_clear: bool, rng: np.random.Generator | None = None) -> float:
    if not los_clear:
        return 0.0

    M = effective_margin_db(edge, d_km)

    if edge.link_type == "ka":
        p: KaParams = edge.params  # type: ignore
        p_out = p_out_ka(M, p)
        return max(0.0, 1.0 - p_out)

    p: OpticalParams = edge.params  # type: ignore

    if p.model == "beta_tail":
        p_out = p_out_optical_beta(M, p)
    elif p.model == "mc_pointing":
        assert rng is not None
        p_out = p_out_optical_mc(M, p, rng)
    elif p.model == "exp_margin":
        # Treat like generic exponential with k = beta_param for an optical-ish slope
        k = beta_param(p)
        p_out = min(1.0, 10**(-k * M / 10.0))
    else:
        raise ValueError(f"Unknown optical model {p.model}")

    return max(0.0, 1.0 - p_out)
```

### 5.7 Sampling edge up/down
```python
def sample_up(edge: Edge, d_km: float, los_clear: bool, rng: np.random.Generator) -> bool:
    pu = p_up(edge, d_km, los_clear, rng=rng)
    return bool(rng.random() < pu)
```

---

## 6) Integration into Time-Varying Topology

### 6.1 Where to plug in
In your “topology update” step (per timestep):
1. Propagate sat positions.
2. Build candidate ISLs from adjacency rules (+Grid / motif / ILS).
3. For each candidate ISL:
   - compute `distance_km`
   - compute `los_clear`
   - compute `pu = p_up(edge, …)`
   - either:
     - **Deterministic expected graph:** set edge weight to expected penalty (rarely ideal)
     - **Stochastic graph:** draw `up = sample_up(...)` and add edge only if up

### 6.2 Recommendation: two-mode simulation
- **Analysis mode:** compute `P_up` and log it; don’t sample (repeatability).
- **Monte Carlo mode:** sample edges (requires seeds, multiple runs).

### 6.3 Routing implications
- If you sample topology, compute routes on the sampled graph.
- If you want “expected” routing, run multiple MC samples and aggregate:
  - `P(path exists)`
  - mean path length
  - tail latencies

---

## 7) Defaults (Practical, “works immediately”)

### Ka-band LEO–LEO defaults
- `M_nominal_db`: 3–6 dB
- `d_ref_km`: 2000 km
- `model`: `"exp_margin"`
- `p_out_target`: `1e-6` at `m_target_db=3` (≈ 99.9999% up at reference)

### Optical LEO–LEO defaults
- `required_margin_db`: 1–3 dB (SDA-ish guidance)
- `theta_B_urad`: 10–30 µrad
- `sigma_j_urad`: 1–3 µrad
- `model`: `"beta_tail"` first (fast) then `"mc_pointing"` once you want realism.

---

## 8) Validation & Tests

### 8.1 Unit tests (math correctness)
- **FSPL monotonicity:** increasing `d` increases FSPL.
- **Margin delta:** `M_eff(d_ref) == M_nom`.
- **LOS gate:** `los_clear=False → P_up=0`.
- **Ka exp model:** `P_out` decreases 10× for every +10 dB margin.
- **Optical beta model:** with larger `sigma_j` (worse jitter), `beta_param` decreases → outage increases.

### 8.2 Property-based tests (stability)
- `0 ≤ P_up ≤ 1` always.
- For any edge, if `d` increases then `P_up` should non-increase (all else constant).

### 8.3 Simulation-level invariants
- With probabilistic outages disabled (`P_out=0`), results match the deterministic baseline.
- With mild outages, network remains connected > X% of time (for your chosen constellation parameters).
- Link churn rate increases as expected with stronger stochasticity, but doesn’t explode with your hysteresis rules (if applied).

### 8.4 Calibration tests
- Set target `P_out_target` at `M_target`; verify via sampling that empirical outage matches within tolerance.

---

## 9) Performance Considerations

- **Beta-tail model** is O(1) per edge (fast).
- **MC pointing** cost is O(N_samples) per edge; mitigate via:
  - caching `p_out(M_db)` on a grid of margin values
  - using fewer samples per timestep and reusing across edges (if parameters identical)
  - stratified sampling per link class

---

## 10) Extension Hooks (future, but keep seams ready)

### 10.1 Add Rician / Nakagami-m for Ka-band LoS
- Replace Rayleigh with Rician K-factor (space LoS makes more sense).

### 10.2 Add acquisition / PAT state machine for optical
- Separate “link exists” vs “link acquired”:
  - `P_up = P_los * P_acq(t) * P_margin_ok(t)`

### 10.3 Add capacity scaling
Instead of binary up/down:
- compute `SNR(t)` and map to `rate(t)` (ACM), then weight edges by latency/capacity.

---

## 11) Deliverables Checklist

- [ ] `models/availability.py`: `KaParams`, `OpticalParams`, `effective_margin_db`, `p_up`, `sample_up`
- [ ] `tests/test_availability.py`: unit + property tests
- [ ] `topology/update_isls.py`: call `p_up/sample_up` during edge construction
- [ ] `metrics/availability_metrics.py`: aggregate `P_up`, outage events, churn, `P(path exists)`

---

## 12) Quick “copy-paste” usage inside your topology loop

```python
rng = np.random.default_rng(seed)

for (u, v) in candidate_isls:
    d_km = distance_km(u, v, t)
    los = los_clear(u, v, t)

    edge = Edge(
        u=u, v=v,
        kind="ISL",
        link_type="optical",
        nominal_margin_db=3.0,
        params=OpticalParams(model="beta_tail", theta_B_urad=20, sigma_j_urad=2),
    )

    if sample_up(edge, d_km, los, rng):
        G.add_edge(u, v, weight=d_km)
```

---

## Appendix A: Notes on “Elevation Angle” for ISLs

True LEO–LEO ISLs don’t have an elevation angle to a local horizon; their geometry is captured by:
- distance
- Earth-occlusion LOS
- relative angular rates / pointing dynamics

Keep the `elevation` parameter in the generic interface **only** if you plan to reuse this model for ground links later.
