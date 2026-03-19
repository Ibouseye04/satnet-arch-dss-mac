#!/usr/bin/env python3
"""Offline analysis of resilience target distributions and leakage risks.

Loads a tier1_design_runs.csv (or compatible) dataset and reports:
- Per-target summary statistics
- Correlation matrix among canonical resilience targets
- Duplicate/near-duplicate design signature checks
- Train/val/test leakage risk from similar signatures

Usage:
    python tools/analyze_dataset_targets.py data/tier1_design_runs.csv
    python tools/analyze_dataset_targets.py data/tier1_design_runs.csv --output summary.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satnet.metrics.resilience_targets import ALL_TARGETS  # noqa: E402

# Targets we analyse (intersected with what exists in the CSV)
TARGET_COLUMNS = sorted(ALL_TARGETS)

# Design signature columns used for near-duplicate detection
DESIGN_SIG_COLUMNS = [
    "num_planes",
    "sats_per_plane",
    "inclination_deg",
    "altitude_km",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse resilience target distributions and leakage risks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=str, help="Path to runs CSV file")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional CSV summary output path",
    )
    parser.add_argument(
        "--output-corr", type=str, default=None,
        help="Optional CSV path for the correlation matrix",
    )
    parser.add_argument(
        "--sig-precision", type=int, default=1,
        help="Rounding precision (decimal places) for continuous design params in signature",
    )
    return parser.parse_args()


def compute_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute summary stats for each target column."""
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        rows.append({
            "target": col,
            "count": len(s),
            "min": s.min(),
            "max": s.max(),
            "mean": s.mean(),
            "std": s.std(),
            "p25": s.quantile(0.25),
            "p50": s.quantile(0.50),
            "p75": s.quantile(0.75),
            "zeros": (s == 0).sum(),
            "ones": (s == 1).sum() if s.max() <= 1.0 else 0,
        })
    return pd.DataFrame(rows)


def compute_correlation(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute pairwise Spearman correlation among targets."""
    available = [c for c in columns if c in df.columns]
    return df[available].corr(method="spearman")


def make_design_signature(row: pd.Series, precision: int = 1) -> str:
    """Build a stable hash from major design parameters."""
    parts = []
    for col in DESIGN_SIG_COLUMNS:
        val = row.get(col)
        if val is None or pd.isna(val):
            parts.append("NA")
        elif isinstance(val, float):
            parts.append(str(round(val, precision)))
        else:
            parts.append(str(val))
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def check_duplicates(
    df: pd.DataFrame, precision: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check for exact-signature duplicates and cross-split leakage."""
    if not all(c in df.columns for c in DESIGN_SIG_COLUMNS):
        return pd.DataFrame(), pd.DataFrame()

    df = df.copy()
    df["_design_sig"] = df.apply(lambda r: make_design_signature(r, precision), axis=1)

    # Group counts
    sig_counts = df["_design_sig"].value_counts()
    dups = sig_counts[sig_counts > 1].reset_index()
    dups.columns = ["design_signature", "count"]

    # Cross-split check (if 'split' column exists)
    leakage_df = pd.DataFrame()
    if "split" in df.columns:
        cross = df.groupby("_design_sig")["split"].nunique()
        cross_sigs = cross[cross > 1].index
        if len(cross_sigs) > 0:
            leakage_rows = []
            for sig in cross_sigs:
                subset = df[df["_design_sig"] == sig]
                split_counts = subset["split"].value_counts().to_dict()
                leakage_rows.append({"design_signature": sig, **split_counts})
            leakage_df = pd.DataFrame(leakage_rows)

    return dups, leakage_df


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}\n")

    # ── target summary ──────────────────────────────────────────────
    available_targets = [c for c in TARGET_COLUMNS if c in df.columns]
    if not available_targets:
        print("WARNING: No canonical resilience target columns found in CSV.")
        print(f"Expected some of: {TARGET_COLUMNS}")
        sys.exit(0)

    summary = compute_summary(df, available_targets)
    print("=" * 70)
    print("TARGET SUMMARY STATISTICS")
    print("=" * 70)
    print(summary.to_string(index=False))
    print()

    # ── correlation matrix ──────────────────────────────────────────
    if len(available_targets) >= 2:
        corr = compute_correlation(df, available_targets)
        print("=" * 70)
        print("SPEARMAN CORRELATION MATRIX")
        print("=" * 70)
        print(corr.round(3).to_string())
        print()

        if args.output_corr:
            corr.to_csv(args.output_corr)
            print(f"Correlation matrix saved to {args.output_corr}")

    # ── duplicate / leakage checks ──────────────────────────────────
    dups, leakage = check_duplicates(df, precision=args.sig_precision)
    print("=" * 70)
    print("DUPLICATE DESIGN SIGNATURES")
    print("=" * 70)
    if len(dups) == 0:
        print("No exact-signature duplicates found.")
    else:
        print(f"Found {len(dups)} signatures with duplicates:")
        print(dups.head(20).to_string(index=False))
    print()

    if len(leakage) > 0:
        print("WARNING: Cross-split leakage detected:")
        print(leakage.to_string(index=False))
        print()

    # ── save summary ────────────────────────────────────────────────
    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
