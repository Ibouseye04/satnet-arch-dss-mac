"""
PyTorch Geometric Dataset for SatNet Temporal GNN Training.

This module provides a Dataset class that regenerates graph topology on-the-fly
from the tier1_design_runs.csv configuration file using the HypatiaAdapter.

Target Model: EvolveGCN-O (Thesis Model)

Usage:
    from satnet.models.gnn_dataset import SatNetTemporalDataset
    
    dataset = SatNetTemporalDataset(
        csv_path="data/tier1_design_runs.csv",
        duration_minutes=10,
        step_seconds=60,
    )
    
    # Get temporal sequence for run idx
    data_list = dataset[idx]  # List[Data] for each time step
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx

from satnet.network.hypatia_adapter import HypatiaAdapter

logger = logging.getLogger(__name__)


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
        csv_path: Path to tier1_design_runs.csv
        duration_minutes: Simulation duration for ISL calculation
        step_seconds: Time step interval for ISL calculation
        transform: Optional transform to apply to Data objects
        pre_transform: Optional pre-transform (not used in on-the-fly mode)
    """
    
    def __init__(
        self,
        csv_path: str | Path,
        duration_minutes: int = 10,
        step_seconds: int = 60,
        root: Optional[str] = None,
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
        pre_filter: Optional[callable] = None,
    ):
        """
        Initialize the SatNet Temporal Dataset.
        
        Args:
            csv_path: Path to the tier1_design_runs.csv file containing
                      constellation configurations and labels.
            duration_minutes: Duration for ISL calculation (default: 10 min).
            step_seconds: Time step interval (default: 60 sec).
            root: Root directory for dataset (optional, for caching).
            transform: Transform to apply to each Data object.
            pre_transform: Pre-transform (not used in on-the-fly mode).
            pre_filter: Pre-filter (not used in on-the-fly mode).
        """
        self.csv_path = Path(csv_path)
        self.duration_minutes = duration_minutes
        self.step_seconds = step_seconds
        
        # Load the CSV with run configurations
        # Note: File existence is not checked locally - CSV is expected to exist
        # at runtime on the deployment server (data/tier1_design_runs.csv)
        self._df = pd.read_csv(self.csv_path)
        self._validate_csv()
        
        logger.info(
            "Loaded %d runs from %s (duration=%d min, step=%d s)",
            len(self._df), self.csv_path, duration_minutes, step_seconds,
        )
        
        # Initialize parent class
        super().__init__(root, transform, pre_transform, pre_filter)
    
    def _validate_csv(self) -> None:
        """Validate that required columns exist in the CSV."""
        required_cols = [
            "num_planes",
            "sats_per_plane", 
            "inclination_deg",
            "altitude_km",
            "partition_any",
        ]
        missing = [c for c in required_cols if c not in self._df.columns]
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Available columns: {list(self._df.columns)}"
            )
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names (the CSV)."""
        return [self.csv_path.name]
    
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
    
    def get(self, idx: int) -> List[Data]:
        """
        Get the temporal graph sequence for a single run.
        
        This method:
        1. Reads row `idx` from the CSV to get constellation config
        2. Instantiates HypatiaAdapter with those parameters
        3. Calls calculate_isls() to generate temporal topology
        4. Converts each time step's NetworkX graph to PyG Data
        5. Returns a list of Data objects (one per time step)
        
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
        
        # Optional parameters with defaults
        phasing_factor = int(row.get("phasing_factor", 1))
        
        # Get the label
        label = int(row["partition_any"])
        
        # Get seed if available (for reproducibility)
        seed = row.get("seed", None)
        
        # Instantiate HypatiaAdapter
        adapter = HypatiaAdapter(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            inclination_deg=inclination_deg,
            altitude_km=altitude_km,
            phasing_factor=phasing_factor,
        )
        
        # Calculate ISLs for the specified duration
        adapter.calculate_isls(
            duration_minutes=self.duration_minutes,
            step_seconds=self.step_seconds,
        )
        
        # Convert each time step to PyG Data
        data_list = []
        
        for time_step, G in adapter.iter_graphs():
            data = self._networkx_to_pyg_data(
                G=G,
                label=label,
                run_id=idx,
                time_step=time_step,
                num_planes=num_planes,
                sats_per_plane=sats_per_plane,
            )
            
            # Apply transform if specified
            if self.transform is not None:
                data = self.transform(data)
            
            data_list.append(data)
        
        return data_list
    
    def _networkx_to_pyg_data(
        self,
        G,
        label: int,
        run_id: int,
        time_step: int,
        num_planes: int,
        sats_per_plane: int,
    ) -> Data:
        """
        Convert a NetworkX graph to a PyG Data object.
        
        Args:
            G: NetworkX graph from HypatiaAdapter
            label: partition_any label (0 or 1)
            run_id: Index of the run
            time_step: Time step index
            num_planes: Number of orbital planes
            sats_per_plane: Satellites per plane
        
        Returns:
            PyG Data object with node features, edge_index, and label
        """
        num_nodes = G.number_of_nodes()
        
        # Build node features
        # For now, use simple features: [plane_idx_normalized, sat_in_plane_normalized, 1.0]
        # This can be extended with orbital parameters later
        x = torch.zeros((num_nodes, 3), dtype=torch.float)
        
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            plane_idx = node_data.get("plane", node_id // sats_per_plane)
            sat_in_plane = node_data.get("sat_in_plane", node_id % sats_per_plane)
            
            # Normalize features to [0, 1] range
            x[node_id, 0] = plane_idx / max(num_planes - 1, 1)
            x[node_id, 1] = sat_in_plane / max(sats_per_plane - 1, 1)
            x[node_id, 2] = 1.0  # Constant feature (node exists)
        
        # Build edge_index from NetworkX edges
        edges = list(G.edges())
        if len(edges) > 0:
            # Create bidirectional edges (undirected graph)
            edge_index = torch.tensor(
                [[e[0] for e in edges] + [e[1] for e in edges],
                 [e[1] for e in edges] + [e[0] for e in edges]],
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
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long),
            run_id=torch.tensor([run_id], dtype=torch.long),
            time_step=torch.tensor([time_step], dtype=torch.long),
            num_nodes=num_nodes,
        )
        
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
        """
        Get the distribution of partition_any labels.
        
        Returns:
            Tuple of (num_negative, num_positive) counts
        """
        counts = self._df["partition_any"].value_counts()
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
    # Quick test (will fail without CSV, but validates imports)
    print("SatNetTemporalDataset module loaded successfully.")
    print("Required dependencies: torch, torch_geometric, pandas")
    print("\nUsage:")
    print("  dataset = SatNetTemporalDataset('data/tier1_design_runs.csv')")
    print("  data_list = dataset[0]  # Get temporal sequence for run 0")
