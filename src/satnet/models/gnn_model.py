"""
GCLSTM Model for Satellite Network Temporal Graph Prediction.

This module implements the "Thesis Model" - a Graph Convolutional LSTM Network
that processes temporal sequences of satellite network graphs.

Supports both:
    - Binary classification (out_channels=2, task_type="classification")
    - Scalar regression    (out_channels=1, task_type="regression")

Architecture:
    - GCLSTM: Combines GCN with LSTM for temporal graph learning
    - Global Mean Pooling: Aggregates node embeddings to graph-level
    - Linear Head: Maps graph embedding to output logits/scalar
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric_temporal.nn.recurrent import GCLSTM


class SatelliteGNN(nn.Module):
    """
    GCLSTM model for satellite network graph prediction.

    Supports classification (out_channels=2) and regression (out_channels=1).

    Attributes:
        node_features: Dimension of input node features (default: 3)
        hidden_channels: Dimension of hidden state (default: 64)
        out_channels: Output dimension (2 for classification, 1 for regression)
        task_type: "classification" or "regression"
    """

    def __init__(
        self,
        node_features: int = 3,
        hidden_channels: int = 64,
        out_channels: int = 2,
        task_type: Literal["classification", "regression"] = "classification",
    ):
        super().__init__()

        self.node_features = node_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.task_type = task_type

        if task_type == "regression" and out_channels != 1:
            raise ValueError("Regression requires out_channels=1")

        # GCLSTM: Graph Convolutional LSTM (K=1: 1-hop Chebyshev filter)
        self.recurrent = GCLSTM(in_channels=node_features, out_channels=hidden_channels, K=1)

        # Linear head: graph embedding → class logits or scalar
        self.linear = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data_list: List[Data]) -> torch.Tensor:
        """
        Forward pass through the temporal GNN.
        
        Processes a sequence of graph snapshots through GCLSTM,
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
        
        # CRITICAL: Reset hidden states at the start of each forward pass
        # This ensures each batch is independent and prevents the
        # "backward through graph a second time" error
        h: Optional[torch.Tensor] = None
        c: Optional[torch.Tensor] = None
        
        # Process each time step through GCLSTM
        for data in data_list:
            x = data.x  # [num_nodes, node_features]
            edge_index = data.edge_index  # [2, num_edges]
            edge_weight = getattr(data, "edge_weight", None)
            
            # GCLSTM forward: returns (h, c) hidden and cell states
            h, c = self.recurrent(x, edge_index, edge_weight, h, c)  # h: [num_nodes, hidden_channels]
        
        # h now contains the final time step's node embeddings
        # Global mean pooling: aggregate node embeddings to graph embedding
        num_nodes = h.size(0)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=h.device)
        
        graph_embedding = global_mean_pool(h, batch)  # [1, hidden_channels]
        
        # Classify
        logits = self.linear(graph_embedding)  # [1, out_channels]
        
        return logits
    
    def predict(self, data_list: List[Data]) -> Union[int, float]:
        """Make a prediction for a single graph sequence.

        Returns:
            Classification: predicted class (int).
            Regression: predicted scalar (float).
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(data_list)
            if self.task_type == "regression":
                return out.squeeze().item()
            return out.argmax(dim=1).item()

    def predict_proba(self, data_list: List[Data]) -> torch.Tensor:
        """Get prediction probabilities (classification only)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(data_list)
            return F.softmax(logits, dim=1)


if __name__ == "__main__":
    # Quick sanity check
    print("SatelliteGNN (GCLSTM) model definition loaded successfully.")
    
    # Create a dummy model
    model = SatelliteGNN(node_features=3, hidden_channels=64, out_channels=2)
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
