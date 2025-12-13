"""
EvolveGCN-O Model for Satellite Network Temporal Graph Classification.

This module implements the "Thesis Model" - an Evolving Graph Convolutional Network
that processes temporal sequences of satellite network graphs to predict partition risk.

Architecture:
    - EvolveGCN-O: Uses LSTM to evolve GCN weights over time (preserves node_features dim)
    - Global Mean Pooling: Aggregates node embeddings to graph-level
    - Linear Classifier: Binary classification (Robust vs. Partitioned)

Target: Predict `partition_any` label from temporal ISL connectivity patterns.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric_temporal.nn.recurrent import EvolveGCNO


class SatelliteEvolveGCN(nn.Module):
    """
    EvolveGCN-O model for satellite network graph classification.
    
    This model processes temporal sequences of graph snapshots and produces
    a single graph-level prediction (binary classification).
    
    Architecture:
        1. EvolveGCN-O (recurrent): Processes node features through time-evolving GCN
        2. Global Mean Pool: Aggregates node embeddings to graph embedding
        3. Linear: Maps graph embedding to class logits
    
    Attributes:
        node_features: Dimension of input/output node features (default: 3)
        out_channels: Number of output classes (default: 2)
    """
    
    def __init__(
        self,
        node_features: int = 3,
        hidden_channels: int = 64,
        out_channels: int = 2,
    ):
        """
        Initialize the SatelliteEvolveGCN model.
        
        Args:
            node_features: Dimension of node features (sats_per_plane, altitude, inclination).
            hidden_channels: Hidden dimension (unused in simple arch, kept for compatibility).
            out_channels: Number of output classes (2 for binary classification).
        """
        super().__init__()
        
        self.node_features = node_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # EvolveGCN-O: Evolving GCN with LSTM-based weight evolution
        # Note: EvolveGCNO(in_channels) outputs same dimension as input
        self.recurrent = EvolveGCNO(in_channels=node_features)
        
        # Linear classifier: graph embedding -> class logits
        self.linear = nn.Linear(node_features, out_channels)
    
    def forward(self, data_list: List[Data]) -> torch.Tensor:
        """
        Forward pass through the temporal GNN.
        
        Processes a sequence of graph snapshots through EvolveGCN-O,
        then pools and classifies.
        
        Args:
            data_list: List of PyG Data objects, one per time step.
                       Each Data must have:
                       - x: Node features [num_nodes, node_features]
                       - edge_index: Edge connectivity [2, num_edges]
                       - edge_weight (optional): Edge weights [num_edges]
        
        Returns:
            logits: Class logits [1, out_channels] for the graph sequence.
        """
        if len(data_list) == 0:
            raise ValueError("data_list cannot be empty")
        
        # Process each time step through EvolveGCN-O
        # EvolveGCN-O maintains internal state (LSTM) that evolves the GCN weights
        h = None
        for data in data_list:
            x = data.x  # [num_nodes, node_features]
            edge_index = data.edge_index  # [2, num_edges]
            edge_weight = getattr(data, "edge_weight", None)
            
            # EvolveGCN-O forward: updates internal LSTM state and applies GCN
            h = self.recurrent(x, edge_index, edge_weight)  # [num_nodes, node_features]
        
        # h now contains the final time step's node embeddings
        # Global mean pooling: aggregate node embeddings to graph embedding
        num_nodes = h.size(0)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=h.device)
        
        graph_embedding = global_mean_pool(h, batch)  # [1, node_features]
        
        # Classify
        logits = self.linear(graph_embedding)  # [1, out_channels]
        
        return logits
    
    def predict(self, data_list: List[Data]) -> int:
        """
        Make a prediction for a single graph sequence.
        
        Args:
            data_list: List of PyG Data objects (temporal sequence).
        
        Returns:
            Predicted class (0 = Robust, 1 = Partitioned).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data_list)
            return logits.argmax(dim=1).item()
    
    def predict_proba(self, data_list: List[Data]) -> torch.Tensor:
        """
        Get prediction probabilities for a single graph sequence.
        
        Args:
            data_list: List of PyG Data objects (temporal sequence).
        
        Returns:
            Probability tensor [1, out_channels] with softmax probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data_list)
            return F.softmax(logits, dim=1)


if __name__ == "__main__":
    # Quick sanity check
    print("SatelliteEvolveGCN model definition loaded successfully.")
    
    # Create a dummy model
    model = SatelliteEvolveGCN(node_features=3, out_channels=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data sequence (3 time steps, 10 nodes each)
    dummy_sequence = []
    for t in range(3):
        x = torch.randn(10, 3)
        edge_index = torch.randint(0, 10, (2, 20))
        dummy_sequence.append(Data(x=x, edge_index=edge_index))
    
    # Forward pass
    logits = model(dummy_sequence)
    print(f"Output shape: {logits.shape}")  # Should be [1, 2]
    print(f"Logits: {logits}")
