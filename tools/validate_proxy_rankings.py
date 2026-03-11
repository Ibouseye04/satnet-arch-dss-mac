#!/usr/bin/env python3
"""Validate whether a cheap proxy preserves full-fidelity rankings.

Takes two result files (reference and proxy) with shared config/experiment
IDs and compares rankings using correlation and overlap metrics.

Usage:
    python tools/validate_proxy_rankings.py reference.csv proxy.csv
    python tools/validate_proxy_rankings.py ref.csv proxy.csv --id-col config_hash --score-col predicted --top-k 10,20,50
    python tools/validate_proxy_rankings.py ref.csv proxy.csv --output report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare proxy vs reference rankings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("reference", type=str, help="Path to reference scores CSV")
    parser.add_argument("proxy", type=str, help="Path to proxy scores CSV")
    parser.add_argument(
        "--id-col", type=str, default="sample_idx",
        help="Column used to join the two files",
    )
    parser.add_argument(
        "--score-col", type=str, default="predicted",
        help="Column containing the score to rank by",
    )
    parser.add_argument(
        "--top-k", type=str, default="5,10,20",
        help="Comma-separated list of k values for top-k overlap",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional JSON output path for the report",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Optional CSV output path for the merged data",
    )
    return parser.parse_args()


# ── ranking metrics ─────────────────────────────────────────────────

def pairwise_ordering_accuracy(ref_scores: np.ndarray, proxy_scores: np.ndarray) -> float:
    """Fraction of pairs where proxy preserves reference ordering."""
    n = len(ref_scores)
    if n < 2:
        return 1.0
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            ref_diff = ref_scores[i] - ref_scores[j]
            proxy_diff = proxy_scores[i] - proxy_scores[j]
            if ref_diff == 0 and proxy_diff == 0:
                concordant += 1
            elif ref_diff * proxy_diff > 0:
                concordant += 1
            total += 1
    return concordant / total if total > 0 else 1.0


def top_k_overlap(
    ref_scores: np.ndarray,
    proxy_scores: np.ndarray,
    ids: np.ndarray,
    k: int,
) -> float:
    """Fraction of reference top-k that also appear in proxy top-k."""
    if k <= 0 or len(ref_scores) < k:
        return float("nan")
    ref_top_k = set(ids[np.argsort(-ref_scores)[:k]])
    proxy_top_k = set(ids[np.argsort(-proxy_scores)[:k]])
    return len(ref_top_k & proxy_top_k) / k


def ndcg_at_k(ref_scores: np.ndarray, proxy_scores: np.ndarray, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if k <= 0 or len(ref_scores) < k:
        return float("nan")

    # Use reference scores as relevance
    proxy_order = np.argsort(-proxy_scores)[:k]
    ideal_order = np.argsort(-ref_scores)[:k]

    def dcg(order: np.ndarray) -> float:
        return sum(
            ref_scores[order[i]] / math.log2(i + 2)
            for i in range(min(k, len(order)))
        )

    dcg_val = dcg(proxy_order)
    idcg_val = dcg(ideal_order)
    if idcg_val == 0:
        return 1.0
    return dcg_val / idcg_val


# ── main ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    ref_path = Path(args.reference)
    proxy_path = Path(args.proxy)
    for p in (ref_path, proxy_path):
        if not p.exists():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    ref_df = pd.read_csv(ref_path)
    proxy_df = pd.read_csv(proxy_path)

    id_col = args.id_col
    score_col = args.score_col

    for label, df in [("reference", ref_df), ("proxy", proxy_df)]:
        for col in (id_col, score_col):
            if col not in df.columns:
                print(f"ERROR: Column '{col}' not found in {label} file. "
                      f"Available: {list(df.columns)}")
                sys.exit(1)

    # Merge on shared IDs
    merged = ref_df[[id_col, score_col]].merge(
        proxy_df[[id_col, score_col]],
        on=id_col,
        suffixes=("_ref", "_proxy"),
    )
    print(f"Merged: {len(merged)} shared IDs "
          f"(ref={len(ref_df)}, proxy={len(proxy_df)})")

    if len(merged) < 2:
        print("ERROR: Need at least 2 shared IDs to compute ranking metrics.")
        sys.exit(1)

    ref_scores = merged[f"{score_col}_ref"].values.astype(float)
    proxy_scores = merged[f"{score_col}_proxy"].values.astype(float)
    ids = merged[id_col].values

    # ── compute metrics ─────────────────────────────────────────
    sp_corr, sp_p = spearmanr(ref_scores, proxy_scores)
    kt_corr, kt_p = kendalltau(ref_scores, proxy_scores)

    # Pairwise accuracy (expensive for large N; sample if needed)
    n = len(ref_scores)
    if n > 2000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n, 2000, replace=False)
        poa = pairwise_ordering_accuracy(ref_scores[sample_idx], proxy_scores[sample_idx])
        poa_note = "sampled (n=2000)"
    else:
        poa = pairwise_ordering_accuracy(ref_scores, proxy_scores)
        poa_note = f"exact (n={n})"

    # Top-k metrics
    k_values = [int(k) for k in args.top_k.split(",")]
    top_k_results = {}
    ndcg_results = {}
    for k in k_values:
        top_k_results[f"top_{k}_overlap"] = top_k_overlap(ref_scores, proxy_scores, ids, k)
        ndcg_results[f"ndcg@{k}"] = ndcg_at_k(ref_scores, proxy_scores, k)

    # ── print report ────────────────────────────────────────────
    report = {
        "n_shared": len(merged),
        "spearman_rho": round(float(sp_corr), 4),
        "spearman_p": float(sp_p),
        "kendall_tau": round(float(kt_corr), 4),
        "kendall_p": float(kt_p),
        "pairwise_ordering_accuracy": round(poa, 4),
        "pairwise_note": poa_note,
        **{k: round(v, 4) for k, v in top_k_results.items()},
        **{k: round(v, 4) for k, v in ndcg_results.items()},
    }

    print("\n" + "=" * 60)
    print("PROXY RANKING VALIDATION REPORT")
    print("=" * 60)
    for key, val in report.items():
        print(f"  {key:35s}: {val}")
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to {out_path}")

    if args.output_csv:
        merged.to_csv(args.output_csv, index=False)
        print(f"Merged data saved to {args.output_csv}")


if __name__ == "__main__":
    main()
