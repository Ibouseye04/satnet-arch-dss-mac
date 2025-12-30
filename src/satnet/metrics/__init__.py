"""Metrics and labeling functions for satellite network analysis.

This module contains pure functions for computing connectivity metrics
and labels from graph state. No simulation or topology dependencies.
"""

from satnet.metrics.labels import (
    aggregate_partition_streaks,
    compute_gcc_frac,
    compute_gcc_size,
    compute_num_components,
    compute_partitioned,
)
from satnet.metrics.risk_binning import (
    bin_satellite_risk,
    compute_tier,
    get_tier_action,
    get_tier_label,
    summarize_tier_distribution,
)

__all__ = [
    # labels.py
    "compute_num_components",
    "compute_gcc_size",
    "compute_gcc_frac",
    "compute_partitioned",
    "aggregate_partition_streaks",
    # risk_binning.py
    "bin_satellite_risk",
    "compute_tier",
    "get_tier_label",
    "get_tier_action",
    "summarize_tier_distribution",
]
