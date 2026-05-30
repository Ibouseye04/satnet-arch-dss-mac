"""Generate architecture diagrams for the SatNet ML models explainer doc.

Renders three PNGs into docs/diagrams/:
    1. ml_pipeline_overview.png  - how RF and the Temporal GNN share data & outputs
    2. random_forest.png         - Random Forest concept + repo usage
    3. temporal_gnn.png          - GCLSTM forward-pass flow

Pure matplotlib (no graphviz needed). Run:
    python docs/diagrams/generate_diagrams.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt

OUT = Path(__file__).parent

# ---- shared palette -------------------------------------------------------
BLUE = "#2f6fb0"
LBLUE = "#dceaf6"
GREEN = "#2e8b57"
LGREEN = "#d8efe1"
ORANGE = "#d9822b"
LORANGE = "#fbe6d2"
GREY = "#555555"
LGREY = "#ececec"
PURPLE = "#7b5ea7"
LPURPLE = "#e8e0f2"


def box(ax, x, y, w, h, text, face, edge, fontsize=10, weight="normal", text_color="#1a1a1a"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize, weight=weight,
        color=text_color, zorder=3,
    )
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, color=GREY, style="-|>", lw=1.8, rad=0.0):
    a = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=16,
        linewidth=lw, color=color, zorder=1,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)


def base_ax(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# ===========================================================================
# 1. PIPELINE OVERVIEW
# ===========================================================================
def pipeline_overview():
    fig, ax = base_ax((11, 7.0), (0, 11), (0, 11))
    ax.text(5.5, 10.5, "SatNet ML Pipeline: Two Models, One Question",
            ha="center", fontsize=15, weight="bold")
    ax.text(5.5, 9.95, "Predict satellite-network resilience (will the network partition?)",
            ha="center", fontsize=10, color=GREY, style="italic")

    box(ax, 3.6, 8.4, 3.8, 1.0,
        "data/tier1_design_runs.csv\n(constellation design runs)",
        LGREY, GREY, fontsize=9.5, weight="bold")

    # Random Forest branch (left)
    box(ax, 0.4, 6.3, 4.2, 1.1,
        "Flat feature vector\nnum_planes, altitude_km,\nnode_failure_prob, ...",
        LORANGE, ORANGE, fontsize=9)
    box(ax, 0.4, 4.4, 4.2, 1.2,
        "RANDOM FOREST\nRandomForestClassifier /\nRegressor (200-300 trees)",
        LORANGE, ORANGE, fontsize=10, weight="bold")
    ax.text(2.5, 4.05, "src/satnet/models/risk_model.py", ha="center",
            fontsize=7.5, color=GREY, family="monospace")

    # Temporal GNN branch (right)
    box(ax, 6.4, 6.3, 4.2, 1.1,
        "Sequence of graph snapshots\n(network 'movie' over time,\n3 features per satellite)",
        LGREEN, GREEN, fontsize=9)
    box(ax, 6.4, 4.4, 4.2, 1.2,
        "TEMPORAL GNN\nSatelliteGNN: GCLSTM +\nmean pool + linear head",
        LGREEN, GREEN, fontsize=10, weight="bold")
    ax.text(8.5, 4.05, "src/satnet/models/gnn_model.py", ha="center",
            fontsize=7.5, color=GREY, family="monospace")

    # converging predictions
    box(ax, 2.6, 2.0, 5.8, 1.1,
        "Predictions CSV (identical schema)\nconfig_hash, target_name, y_true, y_pred, model_type",
        LBLUE, BLUE, fontsize=9, weight="bold")
    box(ax, 3.9, 0.5, 3.2, 0.95,
        "Head-to-head comparison\n& resilience ranking",
        LPURPLE, PURPLE, fontsize=9.5, weight="bold")

    arrow(ax, (4.6, 8.4), (2.5, 7.4), color=ORANGE)
    arrow(ax, (6.4, 8.4), (8.5, 7.4), color=GREEN)
    arrow(ax, (2.5, 6.3), (2.5, 5.6), color=ORANGE)
    arrow(ax, (8.5, 6.3), (8.5, 5.6), color=GREEN)
    arrow(ax, (2.5, 4.4), (4.3, 3.1), color=ORANGE)
    arrow(ax, (8.5, 4.4), (6.7, 3.1), color=GREEN)
    arrow(ax, (5.5, 2.0), (5.5, 1.45), color=BLUE)

    fig.savefig(OUT / "ml_pipeline_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 2. RANDOM FOREST
# ===========================================================================
def random_forest():
    fig, ax = base_ax((11, 6.4), (0, 11), (0, 10))
    ax.text(5.5, 9.5, "Random Forest", ha="center", fontsize=15, weight="bold")
    ax.text(5.5, 9.0, "Many decision trees vote; the majority / average wins",
            ha="center", fontsize=10, color=GREY, style="italic")

    box(ax, 0.3, 5.3, 2.5, 2.4,
        "INPUT\nDesign features\n\nnum_planes\nsats_per_plane\naltitude_km\ninclination_deg\nnode_failure_prob\nedge_failure_prob",
        LORANGE, ORANGE, fontsize=8.5)

    tree_x = [3.5, 5.1, 6.7]
    labels = ["Tree 1", "Tree 2", "Tree N\n(200-300)"]
    for tx, lab in zip(tree_x, labels):
        box(ax, tx, 6.2, 1.2, 1.3, lab, "#ffffff", ORANGE, fontsize=8.5, weight="bold")
        ax.plot([tx + 0.6, tx + 0.25], [6.2, 5.6], color=ORANGE, lw=1)
        ax.plot([tx + 0.6, tx + 0.95], [6.2, 5.6], color=ORANGE, lw=1)
        ax.plot([tx + 0.25, tx + 0.95], [5.6, 5.6], color=ORANGE, lw=1)
        arrow(ax, (2.8, 6.6), (tx, 6.85), color=GREY, lw=1.2)
    ax.text(5.6, 5.25, ". . . each tree trained on a random data / feature subset . . .",
            ha="center", fontsize=8.5, color=GREY, style="italic")

    box(ax, 8.3, 6.2, 2.4, 1.3,
        "AGGREGATE\nmajority vote (class)\nor average (number)",
        LORANGE, ORANGE, fontsize=9, weight="bold")
    for tx in tree_x:
        arrow(ax, (tx + 1.2, 6.85), (8.3, 6.85), color=GREY, lw=1.2, rad=0.05)

    box(ax, 8.3, 3.8, 2.4, 1.4,
        "OUTPUT\npartition_any (yes/no)\nor partition_fraction",
        LBLUE, BLUE, fontsize=9, weight="bold")
    arrow(ax, (9.5, 6.2), (9.5, 5.2), color=BLUE)

    box(ax, 0.3, 2.2, 7.2, 2.3,
        "What the trained model learned\n(models/design_risk_model_metrics.json):\n"
        "  - node_failure_prob  ->  importance ~0.68  (dominant)\n"
        "  - edge_failure_prob  ->  importance ~0.10\n"
        "  Accuracy ~0.68    |    ROC-AUC ~0.73",
        LGREY, GREY, fontsize=9)

    fig.savefig(OUT / "random_forest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 3. TEMPORAL GNN (GCLSTM)
# ===========================================================================
def temporal_gnn():
    fig, ax = base_ax((11.5, 6.8), (0, 11.5), (0, 10))
    ax.text(5.75, 9.5, "Temporal GNN  -  SatelliteGNN (GCLSTM)",
            ha="center", fontsize=15, weight="bold")
    ax.text(5.75, 9.0,
            "Walks through a sequence of network snapshots, carrying memory forward",
            ha="center", fontsize=10, color=GREY, style="italic")

    snap_x = [0.5, 3.0, 5.5]
    tlabels = ["t = 0", "t = 1", "t = T"]
    for i, (sx, tl) in enumerate(zip(snap_x, tlabels)):
        box(ax, sx, 6.6, 1.9, 1.9, "", LGREEN, GREEN)
        ax.text(sx + 0.95, 8.25, tl, ha="center", fontsize=9, weight="bold", color=GREEN)
        cx, cy = sx + 0.95, 7.3
        pts = [(cx + 0.55 * math.cos(a), cy + 0.42 * math.sin(a))
               for a in [0.4, 2.0, 3.5, 5.2]]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)] if i != 1 else [(0, 1), (1, 2), (3, 0)]
        for a, b in edges:
            ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=GREEN, lw=1, zorder=3)
        for px, py in pts:
            ax.plot(px, py, "o", color=BLUE, markersize=7, zorder=4)
    ax.text(4.0, 6.25, "links form / break as satellites move", ha="center",
            fontsize=8.5, color=GREY, style="italic")

    box(ax, 8.0, 6.4, 3.0, 2.0,
        "GCLSTM cell\n(K=1, hidden=64)\n\nGraph Conv (space)\n+ LSTM (time)",
        LPURPLE, PURPLE, fontsize=9.5, weight="bold")
    for sx in snap_x:
        arrow(ax, (sx + 1.9, 7.4), (8.0, 7.4), color=GREEN, lw=1.2, rad=-0.12)

    # memory loop
    ax.add_patch(FancyArrowPatch((10.7, 6.4), (10.7, 8.4),
                 arrowstyle="-|>", mutation_scale=14, color=PURPLE,
                 connectionstyle="arc3,rad=-0.55", lw=1.6))
    ax.text(11.25, 7.4, "memory\n(h, c)", ha="center", fontsize=8,
            color=PURPLE, style="italic", rotation=90)

    # pooling -> linear -> output (bottom row)
    box(ax, 0.8, 3.0, 2.6, 1.3,
        "Global mean pool\nnodes -> 1 graph\nembedding [1, 64]",
        LGREEN, GREEN, fontsize=9, weight="bold")
    box(ax, 4.2, 3.0, 2.6, 1.3,
        "Linear head\n[64] -> out_channels\n(2 = class, 1 = scalar)",
        LGREEN, GREEN, fontsize=9, weight="bold")
    box(ax, 7.8, 3.0, 3.0, 1.3,
        "OUTPUT\npartition_any (yes/no)\nor numeric resilience",
        LBLUE, BLUE, fontsize=9, weight="bold")

    arrow(ax, (9.3, 6.4), (2.1, 4.3), color=PURPLE, rad=0.18)
    ax.text(6.2, 5.15, "final-step node embeddings", ha="center", fontsize=8,
            color=PURPLE, style="italic")
    arrow(ax, (3.4, 3.65), (4.2, 3.65), color=GREEN)
    arrow(ax, (6.8, 3.65), (7.8, 3.65), color=BLUE)

    ax.text(5.75, 1.9,
            "Model: src/satnet/models/gnn_model.py  (forward pass lines 65-110)    |    "
            "graph 'movie' built by src/satnet/models/gnn_dataset.py",
            ha="center", fontsize=8, color=GREY, family="monospace")

    fig.savefig(OUT / "temporal_gnn.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pipeline_overview()
    random_forest()
    temporal_gnn()
    print("wrote 3 diagrams to", OUT)
