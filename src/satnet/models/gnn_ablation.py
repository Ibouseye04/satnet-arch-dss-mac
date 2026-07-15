from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data


TGNN_INPUT_MODE_FULL = "full"
TGNN_INPUT_MODE_TOPOLOGY_ONLY = "topology_only"
TGNN_INPUT_MODE_NODE_STATE_ONLY = "node_state_only"
TGNN_INPUT_MODE_REGISTRY = frozenset(
    {
        TGNN_INPUT_MODE_FULL,
        TGNN_INPUT_MODE_TOPOLOGY_ONLY,
        TGNN_INPUT_MODE_NODE_STATE_ONLY,
    }
)
TGNN_NODE_FEATURE_NAMES = [
    "plane_idx_normalized",
    "sat_in_plane_normalized",
    "node_exists_constant",
]
TGNN_NODE_FEATURE_MEANINGS = {
    "plane_idx_normalized": "Walker plane index divided by max(num_planes - 1, 1)",
    "sat_in_plane_normalized": "satellite index within plane divided by max(sats_per_plane - 1, 1)",
    "node_exists_constant": "constant 1.0 for each surviving node in the effective graph",
}
TGNN_NODE_FEATURE_DIM = len(TGNN_NODE_FEATURE_NAMES)


def validate_tgnn_input_mode(mode: str) -> None:
    if mode not in TGNN_INPUT_MODE_REGISTRY:
        raise ValueError(
            f"Unknown TGNN input mode '{mode}'. "
            f"Allowed values: {sorted(TGNN_INPUT_MODE_REGISTRY)}"
        )


def apply_tgnn_input_mode(data_sequence: list[Data], mode: str) -> list[Data]:
    validate_tgnn_input_mode(mode)
    transformed: list[Data] = []
    for data in data_sequence:
        cloned = data.clone()
        if mode == TGNN_INPUT_MODE_TOPOLOGY_ONLY:
            cloned.x = torch.ones_like(cloned.x)
        elif mode == TGNN_INPUT_MODE_NODE_STATE_ONLY:
            cloned.edge_index = make_self_loop_edge_index(
                int(cloned.num_nodes),
                device=cloned.edge_index.device,
            )
            if hasattr(cloned, "edge_attr") and cloned.edge_attr is not None:
                edge_attr_dim = int(cloned.edge_attr.shape[1]) if cloned.edge_attr.dim() == 2 else 1
                cloned.edge_attr = torch.zeros(
                    (int(cloned.num_nodes), edge_attr_dim),
                    dtype=cloned.edge_attr.dtype,
                    device=cloned.edge_attr.device,
                )
            if hasattr(cloned, "edge_weight") and cloned.edge_weight is not None:
                cloned.edge_weight = torch.ones(
                    int(cloned.num_nodes),
                    dtype=cloned.edge_weight.dtype,
                    device=cloned.edge_weight.device,
                )
        transformed.append(cloned)
    return transformed


def make_self_loop_edge_index(num_nodes: int, *, device: torch.device | None = None) -> torch.Tensor:
    indices = torch.arange(num_nodes, dtype=torch.long, device=device)
    return torch.stack([indices, indices], dim=0)


def tgnn_input_mode_metadata(
    *,
    mode: str,
    baseline_feature_dim: int,
    transformed_feature_dim: int | None = None,
) -> dict[str, Any]:
    validate_tgnn_input_mode(mode)
    transformed_dim = baseline_feature_dim if transformed_feature_dim is None else transformed_feature_dim
    return {
        "input_mode": mode,
        "baseline_node_feature_names": list(TGNN_NODE_FEATURE_NAMES),
        "baseline_node_feature_meanings": dict(TGNN_NODE_FEATURE_MEANINGS),
        "baseline_feature_dim": int(baseline_feature_dim),
        "transformed_feature_dim": int(transformed_dim),
        "temporal_inter_node_edges_preserved": mode != TGNN_INPUT_MODE_NODE_STATE_ONLY,
        "uses_self_loops": mode == TGNN_INPUT_MODE_NODE_STATE_ONLY,
        "node_features_preserved": mode != TGNN_INPUT_MODE_TOPOLOGY_ONLY,
        "topology_preserved": mode != TGNN_INPUT_MODE_NODE_STATE_ONLY,
        "node_state_only_graph_representation": "one deterministic self-loop per surviving node",
    }
