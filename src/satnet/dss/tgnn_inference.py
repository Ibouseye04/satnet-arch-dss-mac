"""Fail-closed adapter for the frozen operational TGNN regression model."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

FROZEN_TGNN_CHECKPOINT_SHA256 = "22cabef076428ba5c118b10fa230c5930af1b1c2da53bdd5c51b118dc0c9960a"
FROZEN_TGNN_TASK = "tgnn_space_regression"
FROZEN_TGNN_TARGET = "space_gcc_fraction_original_min"
FROZEN_TGNN_CONFIG = {
    "node_features": 3,
    "hidden_dim": 64,
    "out_channels": 1,
    "task_type": "regression",
    "cheb_k": 2,
    "num_layers": 1,
}


class DSSCheckpointError(RuntimeError):
    """Raised when the exact operational checkpoint cannot be loaded."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DSSCheckpointError(f"Unable to read TGNN checkpoint: {path}") from exc
    return digest.hexdigest()


def resolve_checkpoint_path(checkpoint_path: str | os.PathLike[str] | None = None) -> Path:
    candidate = checkpoint_path or os.environ.get("SATNET_DSS_TGNN_CHECKPOINT")
    if not candidate:
        raise DSSCheckpointError(
            "SATNET_DSS_TGNN_CHECKPOINT is required; the DSS has no checkpoint fallback"
        )
    path = Path(candidate)
    if not path.is_file():
        raise DSSCheckpointError(f"Frozen TGNN checkpoint does not exist: {path}")
    observed = sha256_file(path)
    if observed != FROZEN_TGNN_CHECKPOINT_SHA256:
        raise DSSCheckpointError(
            "Frozen TGNN checkpoint SHA-256 mismatch: "
            f"expected {FROZEN_TGNN_CHECKPOINT_SHA256}, observed {observed}"
        )
    return path


def load_frozen_tgnn(checkpoint_path: str | os.PathLike[str] | None = None):
    import torch

    from satnet.models.gnn_model import SatelliteGNN

    path = resolve_checkpoint_path(checkpoint_path)
    try:
        torch.set_num_threads(8)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        checkpoint_config = checkpoint.get("configuration", {})
        for key in ("hidden_dim", "cheb_k", "num_layers"):
            if key in checkpoint_config and checkpoint_config[key] != FROZEN_TGNN_CONFIG[key]:
                raise DSSCheckpointError(
                    f"Frozen TGNN checkpoint configuration mismatch for {key}"
                )
        model = SatelliteGNN(
            node_features=FROZEN_TGNN_CONFIG["node_features"],
            hidden_channels=FROZEN_TGNN_CONFIG["hidden_dim"],
            out_channels=FROZEN_TGNN_CONFIG["out_channels"],
            task_type=FROZEN_TGNN_CONFIG["task_type"],
            cheb_k=FROZEN_TGNN_CONFIG["cheb_k"],
        )
        model.load_state_dict(state_dict)
    except (KeyError, RuntimeError, OSError, TypeError, ValueError) as exc:
        raise DSSCheckpointError(f"Unable to load frozen TGNN checkpoint: {path}") from exc
    model.eval()
    return model, FROZEN_TGNN_CHECKPOINT_SHA256


def graph_to_pyg_data(graph, *, num_planes: int, sats_per_plane: int, time_step: int):
    """Build the frozen final-evaluation tensor representation.

    The final external-validation contract defines optical ``link_mode_binary``
    as 1.0. This deliberately avoids the legacy dataset helper's opposite
    polarity while preserving its node/edge ordering and four-field contract.
    """
    import torch
    from torch_geometric.data import Data

    nodes = sorted(graph.nodes())
    node_mapping = {node_id: index for index, node_id in enumerate(nodes)}
    x = torch.zeros((len(nodes), 3), dtype=torch.float32)
    for node_id in nodes:
        node_data = graph.nodes[node_id]
        plane_idx = node_data.get("plane", node_id // sats_per_plane)
        sat_in_plane = node_data.get("sat_in_plane", node_id % sats_per_plane)
        index = node_mapping[node_id]
        x[index, 0] = plane_idx / max(num_planes - 1, 1)
        x[index, 1] = sat_in_plane / max(sats_per_plane - 1, 1)
        x[index, 2] = 1.0

    edges = list(graph.edges())
    edge_pairs = [(node_mapping[first], node_mapping[second]) for first, second in edges]
    edge_index = torch.tensor(
        [
            [first for first, _ in edge_pairs] + [second for _, second in edge_pairs],
            [second for _, second in edge_pairs] + [first for first, _ in edge_pairs],
        ],
        dtype=torch.long,
    ).reshape(2, -1) if edge_pairs else torch.zeros((2, 0), dtype=torch.long)
    edge_attributes: list[list[float]] = []
    for first, second in edges:
        edge_data = graph.edges[first, second]
        link_type_code = {
            "intra_plane": 0.0,
            "inter_plane": 0.5,
            "seam_link": 1.0,
        }.get(edge_data.get("link_type", "unknown"), -0.5)
        edge_attributes.append(
            [
                float(edge_data.get("distance_km", 0.0)) / 10_000.0,
                float(edge_data.get("margin_db", 0.0)) / 100.0,
                link_type_code,
                1.0 if edge_data.get("link_mode", "unknown") == "optical" else 0.0,
            ]
        )
    edge_attr = torch.tensor(
        edge_attributes + edge_attributes, dtype=torch.float32
    ).reshape(-1, 4) if edge_attributes else torch.zeros((0, 4), dtype=torch.float32)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        time_step=torch.tensor([time_step], dtype=torch.long),
        num_nodes=len(nodes),
    )
    data.edge_weight = data.edge_attr[:, 0]
    return data


def sequence_to_pyg(
    graphs: Sequence, *, num_planes: int, sats_per_plane: int
) -> list:
    import torch

    if len(graphs) != 11:
        raise ValueError(f"TGNN sequence must contain exactly 11 snapshots, got {len(graphs)}")
    sequence = [
        graph_to_pyg_data(
            graph,
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            time_step=time_step,
        )
        for time_step, graph in enumerate(graphs)
    ]
    for data in sequence:
        if data.x.ndim != 2 or data.x.shape[1] != 3:
            raise ValueError("TGNN node feature dimension must be 3")
        if data.edge_attr.ndim != 2 or data.edge_attr.shape[1] != 4:
            raise ValueError("TGNN edge feature dimension must be 4")
        if not torch.equal(data.edge_weight, data.edge_attr[:, 0]):
            raise ValueError("TGNN edge_weight must equal edge_attr[:, 0]")
    return sequence


def predict_sequence(model, data_list: Sequence) -> float:
    import torch

    if len(data_list) != 11:
        raise ValueError("TGNN inference requires exactly 11 temporal snapshots")
    for data in data_list:
        if data.x.shape[1] != 3 or data.edge_attr.shape[1] != 4:
            raise ValueError("TGNN feature dimensions do not match the frozen contract")
        data.edge_weight = data.edge_attr[:, 0]
    model.eval()
    with torch.no_grad():
        output = model(list(data_list))
    return float(output.squeeze().item())


def predict_graph_sequence(
    model,
    graphs: Sequence,
    *,
    num_planes: int,
    sats_per_plane: int,
) -> tuple[float, list[tuple[int, int, int]]]:
    data_list = sequence_to_pyg(
        graphs, num_planes=num_planes, sats_per_plane=sats_per_plane
    )
    shapes = [
        (int(data.x.shape[1]), int(data.edge_attr.shape[1]), int(data.edge_index.shape[1]))
        for data in data_list
    ]
    return predict_sequence(model, data_list), shapes
