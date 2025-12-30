"""Risk binning and categorization for satellite reliability predictions.

This module implements the **Data Binning Scheme** requirement for the thesis,
providing a standardized methodology for translating continuous model predictions
(reliability scores in the range [0.0, 1.0]) into discrete, actionable risk tiers.

Thesis Context
--------------
In satellite network reliability analysis, raw model outputs (e.g., from a
RandomForest classifier or GNN) produce continuous probability scores representing
the predicted likelihood that a given constellation design will maintain acceptable
connectivity under failure scenarios. However, operational decision-making requires
discrete categorization to:

1. **Prioritize resources**: Allocate engineering attention to high-risk designs.
2. **Trigger automated workflows**: Initiate diagnostics or maneuvers based on tier.
3. **Enable dashboard visualization**: Color-coded risk displays for operators.
4. **Support regulatory compliance**: Demonstrate systematic risk management.

The three-tier scheme implemented here follows industry best practices for
satellite fleet health monitoring, analogous to traffic-light systems used in
aviation and maritime domains.

Tier Definitions
----------------
- **Tier 1 (Healthy)**: Score > 0.8
    - Interpretation: High confidence in sustained connectivity.
    - Action: No intervention required; continue nominal operations.

- **Tier 2 (Watchlist)**: 0.5 ≤ Score ≤ 0.8
    - Interpretation: Moderate risk; design may be vulnerable under stress.
    - Action: Schedule diagnostic review; consider contingency planning.

- **Tier 3 (Critical/Fail)**: Score < 0.5
    - Interpretation: High probability of connectivity degradation or partition.
    - Action: Immediate engineering review; potential orbital maneuver or
      constellation reconfiguration.

Design Principles
-----------------
This module adheres to the project's architectural standards:

1. **Pure Functions**: All functions are stateless and operate only on input data.
   No side effects, no global state, no file I/O.

2. **Type Safety**: Full type hints for Python 3.11+ compatibility.

3. **Determinism**: Given identical inputs, outputs are always identical.
   No randomness or time-dependent behavior.

4. **Separation of Concerns**: This module handles only binning logic.
   Model training, data loading, and visualization are handled elsewhere.

References
----------
- ISO 14300-1: Space systems — Programme management — Part 1: Structuring of a project
- ECSS-M-ST-80C: Space project management — Risk management
- NASA-STD-8719.13: Software Safety Standard

Example Usage
-------------
>>> import pandas as pd
>>> from satnet.metrics.risk_binning import bin_satellite_risk
>>>
>>> # Raw model predictions
>>> df = pd.DataFrame({
...     'satellite_id': ['SAT-001', 'SAT-002', 'SAT-003', 'SAT-004'],
...     'reliability_score': [0.92, 0.65, 0.33, 0.80]
... })
>>>
>>> # Apply binning
>>> result = bin_satellite_risk(df, score_column='reliability_score')
>>> print(result[['satellite_id', 'risk_tier', 'risk_label', 'recommended_action']])
  satellite_id  risk_tier risk_label      recommended_action
0      SAT-001          1    Healthy               No Action
1      SAT-002          2  Watchlist     Schedule Diagnostics
2      SAT-003          3   Critical      Immediate Maneuver
3      SAT-004          2  Watchlist     Schedule Diagnostics

Notes
-----
The threshold values (0.8, 0.5) are configurable via function parameters to
support sensitivity analysis and domain-specific calibration. The defaults
represent a balanced trade-off between false positives (unnecessary alerts)
and false negatives (missed risks) based on empirical analysis of satellite
constellation failure modes.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


# -----------------------------------------------------------------------------
# Constants: Tier Definitions
# -----------------------------------------------------------------------------

TIER_HEALTHY: Literal[1] = 1
TIER_WATCHLIST: Literal[2] = 2
TIER_CRITICAL: Literal[3] = 3

TIER_LABELS: dict[int, str] = {
    TIER_HEALTHY: "Healthy",
    TIER_WATCHLIST: "Watchlist",
    TIER_CRITICAL: "Critical",
}

TIER_ACTIONS: dict[int, str] = {
    TIER_HEALTHY: "No Action",
    TIER_WATCHLIST: "Schedule Diagnostics",
    TIER_CRITICAL: "Immediate Maneuver",
}

# Default thresholds
DEFAULT_HEALTHY_THRESHOLD: float = 0.8
DEFAULT_CRITICAL_THRESHOLD: float = 0.5


# -----------------------------------------------------------------------------
# Core Binning Function
# -----------------------------------------------------------------------------


def bin_satellite_risk(
    df: pd.DataFrame,
    score_column: str = "reliability_score",
    healthy_threshold: float = DEFAULT_HEALTHY_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
    tier_column: str = "risk_tier",
    label_column: str = "risk_label",
    action_column: str = "recommended_action",
) -> pd.DataFrame:
    """Bin satellite reliability scores into discrete risk tiers with action tags.

    This function implements the **Data Binning Scheme** requirement for the thesis,
    transforming continuous reliability predictions into actionable risk categories.

    The binning logic applies the following rules:
        - Tier 1 (Healthy):   score > healthy_threshold
        - Tier 2 (Watchlist): critical_threshold ≤ score ≤ healthy_threshold
        - Tier 3 (Critical):  score < critical_threshold

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw reliability scores. Must include the
        column specified by `score_column`. All other columns are preserved.

    score_column : str, default="reliability_score"
        Name of the column containing reliability scores. Values must be
        floats in the range [0.0, 1.0], where 1.0 indicates maximum reliability
        (lowest risk) and 0.0 indicates minimum reliability (highest risk).

    healthy_threshold : float, default=0.8
        Scores strictly greater than this value are classified as Tier 1 (Healthy).
        Must satisfy: 0.0 < critical_threshold < healthy_threshold ≤ 1.0

    critical_threshold : float, default=0.5
        Scores strictly less than this value are classified as Tier 3 (Critical).
        Scores between critical_threshold and healthy_threshold (inclusive on
        both ends for the watchlist range) are Tier 2 (Watchlist).

    tier_column : str, default="risk_tier"
        Name of the output column for numeric tier values (1, 2, or 3).

    label_column : str, default="risk_label"
        Name of the output column for human-readable tier labels
        ("Healthy", "Watchlist", "Critical").

    action_column : str, default="recommended_action"
        Name of the output column for recommended operational actions
        ("No Action", "Schedule Diagnostics", "Immediate Maneuver").

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with three additional columns:
        - `tier_column`: Integer tier (1, 2, or 3)
        - `label_column`: String label for the tier
        - `action_column`: Recommended action string

    Raises
    ------
    ValueError
        If `score_column` is not present in the DataFrame.
        If threshold values are invalid (not in proper order or out of range).
        If any score values are outside the valid range [0.0, 1.0].

    Examples
    --------
    Basic usage with default thresholds:

    >>> import pandas as pd
    >>> df = pd.DataFrame({'reliability_score': [0.95, 0.72, 0.45, 0.80]})
    >>> result = bin_satellite_risk(df)
    >>> result['risk_tier'].tolist()
    [1, 2, 3, 2]

    Custom thresholds for stricter classification:

    >>> result = bin_satellite_risk(df, healthy_threshold=0.9, critical_threshold=0.6)
    >>> result['risk_tier'].tolist()
    [1, 2, 3, 2]

    See Also
    --------
    compute_tier : Low-level function for single-value tier computation.
    get_tier_label : Map tier integer to human-readable label.
    get_tier_action : Map tier integer to recommended action.
    """
    # --- Input Validation ---
    if score_column not in df.columns:
        raise ValueError(
            f"Score column '{score_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if not (0.0 < critical_threshold < healthy_threshold <= 1.0):
        raise ValueError(
            f"Invalid thresholds: critical_threshold ({critical_threshold}) must be "
            f"less than healthy_threshold ({healthy_threshold}), and both must be "
            f"in the range (0.0, 1.0]."
        )

    scores = df[score_column]

    if scores.isna().any():
        raise ValueError(
            f"Score column '{score_column}' contains NaN values. "
            "All scores must be valid floats in [0.0, 1.0]."
        )

    if (scores < 0.0).any() or (scores > 1.0).any():
        raise ValueError(
            f"Score column '{score_column}' contains values outside [0.0, 1.0]. "
            f"Found min={scores.min()}, max={scores.max()}."
        )

    # --- Compute Tiers ---
    result = df.copy()

    result[tier_column] = scores.apply(
        lambda s: compute_tier(s, healthy_threshold, critical_threshold)
    )
    result[label_column] = result[tier_column].map(TIER_LABELS)
    result[action_column] = result[tier_column].map(TIER_ACTIONS)

    return result


# -----------------------------------------------------------------------------
# Helper Functions (Pure, Stateless)
# -----------------------------------------------------------------------------


def compute_tier(
    score: float,
    healthy_threshold: float = DEFAULT_HEALTHY_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
) -> int:
    """Compute the risk tier for a single reliability score.

    This is a pure function that maps a continuous score to a discrete tier.

    Parameters
    ----------
    score : float
        Reliability score in [0.0, 1.0].

    healthy_threshold : float, default=0.8
        Threshold above which a score is Tier 1 (Healthy).

    critical_threshold : float, default=0.5
        Threshold below which a score is Tier 3 (Critical).

    Returns
    -------
    int
        Risk tier: 1 (Healthy), 2 (Watchlist), or 3 (Critical).

    Examples
    --------
    >>> compute_tier(0.95)
    1
    >>> compute_tier(0.65)
    2
    >>> compute_tier(0.30)
    3
    >>> compute_tier(0.80)  # Boundary: exactly at healthy threshold
    2
    >>> compute_tier(0.50)  # Boundary: exactly at critical threshold
    2
    """
    if score > healthy_threshold:
        return TIER_HEALTHY
    elif score < critical_threshold:
        return TIER_CRITICAL
    else:
        return TIER_WATCHLIST


def get_tier_label(tier: int) -> str:
    """Get the human-readable label for a risk tier.

    Parameters
    ----------
    tier : int
        Risk tier (1, 2, or 3).

    Returns
    -------
    str
        Human-readable label: "Healthy", "Watchlist", or "Critical".

    Raises
    ------
    KeyError
        If tier is not 1, 2, or 3.

    Examples
    --------
    >>> get_tier_label(1)
    'Healthy'
    >>> get_tier_label(3)
    'Critical'
    """
    return TIER_LABELS[tier]


def get_tier_action(tier: int) -> str:
    """Get the recommended action for a risk tier.

    Parameters
    ----------
    tier : int
        Risk tier (1, 2, or 3).

    Returns
    -------
    str
        Recommended action string.

    Raises
    ------
    KeyError
        If tier is not 1, 2, or 3.

    Examples
    --------
    >>> get_tier_action(1)
    'No Action'
    >>> get_tier_action(3)
    'Immediate Maneuver'
    """
    return TIER_ACTIONS[tier]


def summarize_tier_distribution(df: pd.DataFrame, tier_column: str = "risk_tier") -> dict:
    """Compute summary statistics for tier distribution in a DataFrame.

    Useful for generating reports and dashboards showing the overall
    health distribution of a satellite constellation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a tier column (output of `bin_satellite_risk`).

    tier_column : str, default="risk_tier"
        Name of the column containing tier values.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'total': Total number of satellites
        - 'tier_1_count': Count of Tier 1 (Healthy)
        - 'tier_2_count': Count of Tier 2 (Watchlist)
        - 'tier_3_count': Count of Tier 3 (Critical)
        - 'tier_1_pct': Percentage in Tier 1
        - 'tier_2_pct': Percentage in Tier 2
        - 'tier_3_pct': Percentage in Tier 3

    Examples
    --------
    >>> df = pd.DataFrame({'risk_tier': [1, 1, 2, 3, 1, 2]})
    >>> summarize_tier_distribution(df)
    {'total': 6, 'tier_1_count': 3, 'tier_2_count': 2, 'tier_3_count': 1,
     'tier_1_pct': 50.0, 'tier_2_pct': 33.33, 'tier_3_pct': 16.67}
    """
    if tier_column not in df.columns:
        raise ValueError(f"Tier column '{tier_column}' not found in DataFrame.")

    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "tier_1_count": 0,
            "tier_2_count": 0,
            "tier_3_count": 0,
            "tier_1_pct": 0.0,
            "tier_2_pct": 0.0,
            "tier_3_pct": 0.0,
        }

    counts = df[tier_column].value_counts()

    tier_1 = counts.get(TIER_HEALTHY, 0)
    tier_2 = counts.get(TIER_WATCHLIST, 0)
    tier_3 = counts.get(TIER_CRITICAL, 0)

    return {
        "total": total,
        "tier_1_count": tier_1,
        "tier_2_count": tier_2,
        "tier_3_count": tier_3,
        "tier_1_pct": round(100.0 * tier_1 / total, 2),
        "tier_2_pct": round(100.0 * tier_2 / total, 2),
        "tier_3_pct": round(100.0 * tier_3 / total, 2),
    }
