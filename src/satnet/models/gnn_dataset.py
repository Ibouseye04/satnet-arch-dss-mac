"""
PyTorch Geometric Dataset for SatNet Temporal GNN Training.

This module provides a Dataset class that regenerates graph topology on-the-fly
from the tier1_design_runs.csv configuration file using the HypatiaAdapter.

Graph Reconstruction Contract (Step 3):
    This dataset implements the Tier 1 graph reconstruction contract:
    1. Requires epoch_iso and all graph-defining metadata from schema v2
    2. Uses duration_minutes and step_seconds from CSV for temporal iteration
    3. Applies failed_nodes_json and failed_edges_json to match original labels
    
    This ensures regenerated graphs match the partition_any labels from simulation.

Target Model: GCLSTM (Thesis Model)

Usage:
    from satnet.models.gnn_dataset import SatNetTemporalDataset
    
    dataset = SatNetTemporalDataset(
        root="data/",
        csv_file="tier1_design_runs.csv",
    )
    
    # Get temporal sequence for run idx
    data_list = dataset[idx]  # List[Data] for each time step
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx

from datetime import datetime

from satnet.network.hypatia_adapter import HypatiaAdapter
from satnet.simulation.tier1_rollout import (
    DATASET_VERSION,
    SCHEMA_VERSION,
    Tier1FailureRealization,
)
from satnet.utils.graph_cache import (
    extract_cache_key_config,
    load_graph_sequence,
    make_cache_metadata,
    make_sample_cache_key,
    save_graph_sequence,
    validate_cache_entry,
)

logger = logging.getLogger(__name__)

GRAPH_SEQUENCE_GENERATOR_PROVENANCE = (
    "SatNetTemporalDataset:hypatia_temporal_graph_sequence:v2"
)
RECONSTRUCTION_REQUIRED_COLUMNS = frozenset(
    {
        "num_planes",
        "sats_per_plane",
        "inclination_deg",
        "altitude_km",
        "phasing_factor",
        "duration_minutes",
        "step_seconds",
        "num_steps",
        "max_isl_distance_km",
        "orbital_engine",
        "epoch_iso",
        "isl_policy",
        "adjacent_search_k",
        "max_inter_plane_links_per_sat",
        "failure_model",
        "failed_nodes_json",
        "failed_edges_json",
        "num_failed_nodes",
        "num_failed_edges",
        "schema_version",
        "dataset_version",
    }
)


def _format_temporal_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _temporal_sort_key(value: Any) -> tuple[int, float | str]:
    try:
        return 0, float(value)
    except (TypeError, ValueError):
        return 1, str(value)


def _format_temporal_values(values: list[Any]) -> str:
    return "[" + ", ".join(_format_temporal_value(value) for value in values) + "]"


def _unique_sorted_temporal_values(df: pd.DataFrame, column: str) -> list[Any]:
    if column not in df.columns:
        return []
    values = {
        int(value) if isinstance(value, float) and value.is_integer() else value
        for value in df[column].dropna().tolist()
    }
    return sorted(values, key=_temporal_sort_key)


def format_dataset_temporal_metadata_summary(
    df: pd.DataFrame,
    *,
    fallback_duration_minutes: int,
    fallback_step_seconds: int,
) -> str:
    durations = _unique_sorted_temporal_values(df, "duration_minutes")
    steps = _unique_sorted_temporal_values(df, "step_seconds")
    num_steps = _unique_sorted_temporal_values(df, "num_steps")

    has_multiple_values = any(len(values) > 1 for values in (durations, steps, num_steps))
    if has_multiple_values:
        return (
            f"multiple durations={_format_temporal_values(durations)}, "
            f"steps={_format_temporal_values(steps)}, "
            f"num_steps={_format_temporal_values(num_steps)}"
        )

    duration_text = (
        f"duration={_format_temporal_value(durations[0])} min"
        if durations else f"duration={fallback_duration_minutes} min fallback"
    )
    step_text = (
        f"step={_format_temporal_value(steps[0])} s"
        if steps else f"step={fallback_step_seconds} s fallback"
    )
    num_steps_text = (
        f"num_steps={_format_temporal_value(num_steps[0])}"
        if num_steps else "num_steps=unknown fallback"
    )
    return f"{duration_text}, {step_text}, {num_steps_text}"


class SatNetTemporalDataset(Dataset):
    """
    PyTorch Geometric Dataset for satellite network temporal graphs.
    
    This dataset regenerates graph topology on-the-fly from constellation
    configurations stored in a CSV file. Each sample is a temporal sequence
    of PyG Data objects representing the network at each time step.
    
    Designed for EvolveGCN-O training where we need:
    - Node features (x): Satellite properties
    - Edge indices: ISL connectivity at each time step
    - Labels (y): partition_any from the CSV
    
    Uses standard Dataset (not InMemoryDataset) to avoid RAM issues with
    large numbers of runs (~2000).
    
    Attributes:
        root: Root directory containing the CSV file
        csv_file: Name of the CSV file within root
        duration_minutes: Simulation duration for ISL calculation
        step_seconds: Time step interval for ISL calculation
        transform: Optional transform to apply to Data objects
        pre_transform: Optional pre-transform (not used in on-the-fly mode)
    """
    
    def __init__(
        self,
        root: str,
        csv_file: str = "tier1_design_runs.csv",
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
        duration_minutes: int = 10,
        step_seconds: int = 60,
        use_cache: bool = False,
        write_cache: bool = False,
        cache_dir: Optional[str] = None,
        target_name: str = "partition_any",
    ):
        """
        Initialize the SatNet Temporal Dataset.

        Args:
            root: Root directory containing the CSV file.
            csv_file: Name of the CSV file within root (default: tier1_design_runs.csv).
            transform: Transform to apply to each Data object.
            pre_transform: Pre-transform (not used in on-the-fly mode).
            duration_minutes: Duration for ISL calculation (default: 10 min).
            step_seconds: Time step interval (default: 60 sec).
            use_cache: If True, attempt to load cached graph sequences.
            write_cache: If True, write generated sequences to cache.
            cache_dir: Directory for cached .pt files (default: artifacts/graph_cache).
            target_name: Column name for the target variable (default: partition_any).

        Raises:
            FileNotFoundError: If the CSV file does not exist at root/csv_file.
        """
        self.csv_file = csv_file
        self.duration_minutes = duration_minutes
        self.step_seconds = step_seconds
        self.use_cache = use_cache
        self.write_cache = write_cache
        self._cache_dir = cache_dir
        self.target_name = target_name
        
        # Validate that the CSV file exists before proceeding
        csv_path = os.path.join(root, csv_file)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}. "
                f"Please ensure the file exists at the specified location."
            )
        
        # Initialize parent class (creates raw/processed dirs)
        super().__init__(root, transform, pre_transform)
        
        # Load the CSV with run configurations
        self._df = pd.read_csv(self.raw_paths[0])
        self._validate_csv()
        
        temporal_metadata_summary = format_dataset_temporal_metadata_summary(
            self._df,
            fallback_duration_minutes=duration_minutes,
            fallback_step_seconds=step_seconds,
        )
        logger.info(
            "Loaded %d runs from %s (%s)",
            len(self._df), self.raw_paths[0], temporal_metadata_summary,
        )
    
    def _validate_csv(self) -> None:
        """Validate that required columns exist in the CSV."""
        required_cols = set(RECONSTRUCTION_REQUIRED_COLUMNS)
        required_cols.add(self.target_name)
        missing = sorted(required_cols - set(self._df.columns))
        if missing:
            raise ValueError(
                f"CSV missing required reconstruction columns: {missing}. "
                f"Available columns: {list(self._df.columns)}"
            )

        null_columns = sorted(
            column for column in required_cols if self._df[column].isna().any()
        )
        if null_columns:
            raise ValueError(
                f"CSV contains null reconstruction metadata: {null_columns}"
            )

        schema_versions = {int(value) for value in self._df["schema_version"].tolist()}
        if schema_versions != {SCHEMA_VERSION}:
            raise ValueError(
                f"CSV schema_version must be {SCHEMA_VERSION}; found {sorted(schema_versions)}"
            )
        dataset_versions = {
            str(value) for value in self._df["dataset_version"].tolist()
        }
        if dataset_versions != {DATASET_VERSION}:
            raise ValueError(
                f"CSV dataset_version must be '{DATASET_VERSION}'; "
                f"found {sorted(dataset_versions)}"
            )
    
    @property
    def raw_dir(self) -> str:
        """Return root directory directly (CSV lives in data/, not data/raw/)."""
        return self.root
    
    @property
    def processed_dir(self) -> str:
        """Return processed directory for caching."""
        return os.path.join(self.root, "processed")
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names (the CSV)."""
        return [self.csv_file]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return empty list - we generate on-the-fly, no processed files."""
        return []
    
    def download(self) -> None:
        """No download needed - CSV must be provided."""
        pass
    
    def process(self) -> None:
        """No pre-processing - we generate on-the-fly."""
        pass
    
    def len(self) -> int:
        """Return the number of runs in the dataset."""
        return len(self._df)
    
    @property
    def _resolved_cache_dir(self) -> str:
        if self._cache_dir is not None:
            return self._cache_dir
        # default: artifacts/graph_cache relative to root
        return os.path.join(self.root, "..", "artifacts", "graph_cache")

    def _materialize_sequence(
        self,
        structural_data_list: List[Data],
        *,
        label: float,
        run_id: int,
        failure_model: str | None = None,
    ) -> List[Data]:
        """Attach target-specific fields to a target-agnostic graph sequence."""
        data_list: List[Data] = []
        for structural_data in structural_data_list:
            data = structural_data.clone()
            data.y = torch.tensor([label], dtype=torch.float)
            data.run_id = torch.tensor([run_id], dtype=torch.long)
            if failure_model is not None:
                data.failure_model = failure_model

            if self.transform is not None:
                data = self.transform(data)

            data_list.append(data)
        return data_list

    def get(self, idx: int) -> List[Data]:
        """
        Get the temporal graph sequence for a single run.

        This method:
        1. Reads row `idx` from the CSV to get constellation config
        2. (Optional) Checks cache for a pre-built sequence
        3. Instantiates HypatiaAdapter with those parameters
        4. Calls calculate_isls() to generate temporal topology
        5. Converts each time step's NetworkX graph to PyG Data
        6. (Optional) Writes the sequence to cache
        7. Returns a list of Data objects (one per time step)

        Args:
            idx: Index of the run (0-indexed)

        Returns:
            List of PyG Data objects, one per time step. Each Data has:
                - x: Node features [num_sats, num_features]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge attributes (distance, link_type, etc.)
                - y: Label (partition_any from CSV)
                - run_id: Index of this run
                - time_step: Time step index
        """
        if idx < 0 or idx >= len(self._df):
            raise IndexError(f"Index {idx} out of range [0, {len(self._df)})")

        row = self._df.iloc[idx]

        # Extract constellation parameters
        num_planes = int(row["num_planes"])
        sats_per_plane = int(row["sats_per_plane"])
        inclination_deg = float(row["inclination_deg"])
        altitude_km = float(row["altitude_km"])

        phasing_factor = int(row["phasing_factor"])

        # Get the label (supports any target column)
        label = float(row[self.target_name])

        epoch = datetime.fromisoformat(str(row["epoch_iso"]))
        duration_minutes = int(row["duration_minutes"])
        step_seconds = int(row["step_seconds"])
        expected_num_steps = int(row["num_steps"])
        max_isl_distance_km = float(row["max_isl_distance_km"])
        orbital_engine = str(row["orbital_engine"])
        isl_policy = str(row["isl_policy"])
        adjacent_search_k = int(row["adjacent_search_k"])
        max_inter_plane_links_per_sat = int(row["max_inter_plane_links_per_sat"])
        failure_model = str(row["failure_model"])

        failures = Tier1FailureRealization.from_json_strings(
            str(row["failed_nodes_json"]),
            str(row["failed_edges_json"]),
        )
        if len(failures.failed_nodes) != int(row["num_failed_nodes"]):
            raise ValueError("failed_nodes_json does not match num_failed_nodes")
        if len(failures.failed_edges) != int(row["num_failed_edges"]):
            raise ValueError("failed_edges_json does not match num_failed_edges")

        # ── cache lookup ────────────────────────────────────────────
        cache_key = None
        generator_config: dict[str, Any] | None = None
        expected_cache_metadata: dict[str, Any] | None = None
        if self.use_cache or self.write_cache:
            sample_config = row.to_dict()
            sample_config["isl_policy"] = isl_policy
            sample_config["adjacent_search_k"] = adjacent_search_k
            sample_config["failure_model"] = failure_model
            sample_config["max_inter_plane_links_per_sat"] = max_inter_plane_links_per_sat
            generator_config = extract_cache_key_config(sample_config)
            cache_key = make_sample_cache_key(sample_config)
            expected_cache_metadata = make_cache_metadata(
                sample_cache_key=cache_key,
                generator_provenance=GRAPH_SEQUENCE_GENERATOR_PROVENANCE,
                generator_config=generator_config,
            )

        if self.use_cache and cache_key is not None:
            cached, _, cached_metadata = load_graph_sequence(
                self._resolved_cache_dir, cache_key
            )
            if cached is not None:
                validate_cache_entry(
                    cached,
                    cached_metadata,
                    expected_sample_cache_key=cache_key,
                    expected_generator_provenance=GRAPH_SEQUENCE_GENERATOR_PROVENANCE,
                    expected_generator_config=generator_config,
                )
                return self._materialize_sequence(
                    cached,
                    label=label,
                    run_id=idx,
                    failure_model=failure_model,
                )

        # ── generate graphs (cache miss or caching disabled) ────────
        # Instantiate HypatiaAdapter with explicit epoch for reproducibility
        adapter = HypatiaAdapter(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            inclination_deg=inclination_deg,
            altitude_km=altitude_km,
            phasing_factor=phasing_factor,
            epoch=epoch,
            orbital_engine=orbital_engine,
        )

        # Calculate ISLs for the specified duration
        adapter.calculate_isls(
            duration_minutes=duration_minutes,
            step_seconds=step_seconds,
            max_isl_distance_km=max_isl_distance_km,
            isl_policy=isl_policy,
            adjacent_search_k=adjacent_search_k,
            max_inter_plane_links_per_sat=max_inter_plane_links_per_sat,
        )

        # Convert each time step to PyG Data
        structural_data_list: List[Data] = []

        for time_step, G in adapter.iter_graphs():
            # Apply persistent failures (Step 3 contract)
            G_eff = G.copy()
            nodes_to_remove = [n for n in failures.failed_nodes if G_eff.has_node(n)]
            G_eff.remove_nodes_from(nodes_to_remove)
            for u, v in failures.failed_edges:
                if G_eff.has_edge(u, v):
                    G_eff.remove_edge(u, v)
            data = self._networkx_to_pyg_data(
                G=G_eff,
                time_step=time_step,
                num_planes=num_planes,
                sats_per_plane=sats_per_plane,
            )
            data.isl_policy = isl_policy
            data.adjacent_search_k = adjacent_search_k
            data.max_inter_plane_links_per_sat = max_inter_plane_links_per_sat
            data.failure_model = failure_model
            structural_data_list.append(data)

        if len(structural_data_list) != expected_num_steps:
            raise ValueError(
                "Reconstructed graph sequence length does not match num_steps: "
                f"expected {expected_num_steps}, got {len(structural_data_list)}"
            )

        # ── cache write ─────────────────────────────────────────────
        if self.write_cache and cache_key is not None:
            save_graph_sequence(
                structural_data_list,
                self._resolved_cache_dir,
                cache_key,
                metadata=expected_cache_metadata,
            )

        return self._materialize_sequence(
            structural_data_list,
            label=label,
            run_id=idx,
            failure_model=failure_model,
        )
    
    def _networkx_to_pyg_data(
        self,
        G,
        time_step: int,
        num_planes: int,
        sats_per_plane: int,
        label: Optional[float] = None,
        run_id: Optional[int] = None,
    ) -> Data:
        """Convert a NetworkX graph to a PyG Data object."""
        num_nodes = G.number_of_nodes()

        # Create mapping from original node IDs to contiguous indices [0, num_nodes)
        # This is required because node failures can create non-contiguous node IDs
        # e.g., if nodes {0,1,2,5,6} remain after failure, we map to {0,1,2,3,4}
        node_mapping = {node_id: idx for idx, node_id in enumerate(sorted(G.nodes()))}

        # Build node features
        # For now, use simple features: [plane_idx_normalized, sat_in_plane_normalized, 1.0]
        # This can be extended with orbital parameters later
        x = torch.zeros((num_nodes, 3), dtype=torch.float)

        for node_id in G.nodes():
            idx = node_mapping[node_id]
            node_data = G.nodes[node_id]
            plane_idx = node_data.get("plane", node_id // sats_per_plane)
            sat_in_plane = node_data.get("sat_in_plane", node_id % sats_per_plane)

            # Normalize features to [0, 1] range
            x[idx, 0] = plane_idx / max(num_planes - 1, 1)
            x[idx, 1] = sat_in_plane / max(sats_per_plane - 1, 1)
            x[idx, 2] = 1.0  # Constant feature (node exists)

        # Build edge_index from NetworkX edges, using mapped indices
        edges = list(G.edges())
        if len(edges) > 0:
            # Create bidirectional edges (undirected graph) with mapped indices
            mapped_edges = [(node_mapping[u], node_mapping[v]) for u, v in edges]
            edge_index = torch.tensor(
                [[e[0] for e in mapped_edges] + [e[1] for e in mapped_edges],
                 [e[1] for e in mapped_edges] + [e[0] for e in mapped_edges]],
                dtype=torch.long,
            )
            
            # Build edge attributes (distance, link_type encoded)
            edge_attr_list = []
            for u, v in edges:
                edge_data = G.edges[u, v]
                distance_km = edge_data.get("distance_km", 0.0)
                margin_db = edge_data.get("margin_db", 0.0)
                
                # Encode link_type as numeric
                link_type = edge_data.get("link_type", "unknown")
                link_type_code = {
                    "intra_plane": 0.0,
                    "inter_plane": 1.0,
                    "seam_link": 2.0,
                }.get(link_type, -1.0)
                
                # Encode link_mode as numeric
                link_mode = edge_data.get("link_mode", "unknown")
                link_mode_code = 0.0 if link_mode == "optical" else 1.0
                
                edge_attr_list.append([
                    distance_km / 10000.0,  # Normalize distance
                    margin_db / 100.0,       # Normalize margin
                    link_type_code / 2.0,    # Normalize type
                    link_mode_code,          # Binary mode
                ])
            
            # Duplicate for bidirectional edges
            edge_attr = torch.tensor(
                edge_attr_list + edge_attr_list,
                dtype=torch.float,
            )
        else:
            # No edges - empty tensors
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4), dtype=torch.float)
        
        data_kwargs = dict(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            time_step=torch.tensor([time_step], dtype=torch.long),
            num_nodes=num_nodes,
        )

        if label is not None:
            data_kwargs["y"] = torch.tensor([label], dtype=torch.float)
        if run_id is not None:
            data_kwargs["run_id"] = torch.tensor([run_id], dtype=torch.long)

        # Create PyG Data object
        data = Data(**data_kwargs)

        return data
    
    def get_run_config(self, idx: int) -> dict:
        """
        Get the configuration dictionary for a specific run.
        
        Args:
            idx: Index of the run
        
        Returns:
            Dictionary with constellation parameters and label
        """
        if idx < 0 or idx >= len(self._df):
            raise IndexError(f"Index {idx} out of range [0, {len(self._df)})")
        
        return self._df.iloc[idx].to_dict()
    
    def get_label_distribution(self) -> Tuple[int, int]:
        """Get the distribution of binary target labels.

        Returns:
            Tuple of (num_negative, num_positive) counts.
            For continuous targets, thresholds at 0.5.
        """
        col = self.target_name if self.target_name in self._df.columns else "partition_any"
        binary = (self._df[col] > 0.5).astype(int)
        counts = binary.value_counts()
        return counts.get(0, 0), counts.get(1, 0)


def collate_temporal_sequences(batch: List[List[Data]]) -> List[List[Data]]:
    """
    Custom collate function for temporal sequences.
    
    Since each sample is a list of Data objects (one per time step),
    we need a custom collate function that preserves the temporal structure.
    
    Args:
        batch: List of samples, where each sample is List[Data]
    
    Returns:
        List of samples (unchanged structure for EvolveGCN)
    
    Note:
        For EvolveGCN, you typically process sequences one at a time
        or use a custom batching strategy. This collate function
        preserves the temporal structure.
    """
    return batch


if __name__ == "__main__":
    # Determine the data directory relative to this file
    # This file is at src/satnet/models/gnn_dataset.py
    # Data is at data/ (project root: 3 levels up from models/)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    data_dir = os.path.join(project_root, "data")
    
    try:
        dataset = SatNetTemporalDataset(root=data_dir)
        print(f"Found {len(dataset)} runs")
    except FileNotFoundError as e:
        print(f"Dataset loading failed: {e}")
        print("\nTo use this dataset, ensure tier1_design_runs.csv exists in data/")
