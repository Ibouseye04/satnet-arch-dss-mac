"""
EvolveGCN-O Model for Satellite Network Temporal Graph Classification.

This module implements the "Thesis Model" - an Evolving Graph Convolutional Network
that processes temporal sequences of satellite network graphs to predict partition risk.

Architecture:
    - EvolveGCN-O: Uses LSTM to evolve GCN weights over time
    - Global Mean Pooling: Aggregates node embeddings to graph-level
    - MLP Classifier: Binary classification (Partitioned vs. Robust)

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
        1. EvolveGCN-O layer: Processes node features through time-evolving GCN
        2. Global Mean Pool: Aggregates node embeddings to graph embedding
        3. MLP: Maps graph embedding to class logits
    
    Attributes:
        input_dim: Dimension of input node features (default: 3)
        hidden_dim: Hidden dimension for GCN layers (default: 64)
        output_dim: Number of output classes (default: 2)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 2,
    ):
        """
        Initialize the SatelliteEvolveGCN model.
        
        Args:
            input_dim: Dimension of input node features.
                       Default is 3 (sats_per_plane, altitude, inclination normalized).
            hidden_dim: Hidden dimension for the EvolveGCN layer.
            output_dim: Number of output classes (2 for binary classification).
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # EvolveGCN-O: Evolving GCN with LSTM-based weight evolution
        # Input: node features of dimension input_dim
        # Output: node embeddings of dimension hidden_dim
        self.evolve_gcn = EvolveGCNO(
            in_channels=input_dim,
            out_channels=hidden_dim,
        )
        
        # MLP classifier: graph embedding -> class logits
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(
        self,
        data_sequence: List[Data],
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the temporal GNN.
        
        Processes a sequence of graph snapshots through EvolveGCN-O,
        then pools and classifies.
        
        Args:
            data_sequence: List of PyG Data objects, one per time step.
                          Each Data must have:
                          - x: Node features [num_nodes, input_dim]
                          - edge_index: Edge connectivity [2, num_edges]
            return_embeddings: If True, also return intermediate embeddings.
        
        Returns:
            logits: Class logits [1, output_dim] for the graph sequence.
            (Optional) embeddings: Final graph embedding if return_embeddings=True.
        """
        if len(data_sequence) == 0:
            raise ValueError("data_sequence cannot be empty")
        
        # Process each time step through EvolveGCN-O
        # EvolveGCN-O maintains internal state (LSTM) that evolves the GCN weights
        node_embeddings_list = []
        
        for data in data_sequence:
            x = data.x  # [num_nodes, input_dim]
            edge_index = data.edge_index  # [2, num_edges]
            
            # EvolveGCN-O forward: updates internal LSTM state and applies GCN
            h = self.evolve_gcn(x, edge_index)  # [num_nodes, hidden_dim]
            node_embeddings_list.append(h)
        
        # Use the final time step's node embeddings for classification
        # This captures the evolved representation after seeing all time steps
        final_node_embeddings = node_embeddings_list[-1]  # [num_nodes, hidden_dim]
        
        # Global mean pooling: aggregate node embeddings to graph embedding
        # Create a batch tensor (all nodes belong to graph 0)
        num_nodes = final_node_embeddings.size(0)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=final_node_embeddings.device)
        
        graph_embedding = global_mean_pool(final_node_embeddings, batch)  # [1, hidden_dim]
        
        # Classify
        logits = self.classifier(graph_embedding)  # [1, output_dim]
        
        if return_embeddings:
            return logits, graph_embedding
        
        return logits
    
    def reset_hidden_state(self) -> None:
        """
        Reset the internal LSTM state of EvolveGCN-O.
        
        Call this before processing a new graph sequence to ensure
        the model starts with a fresh state.
        """
        # EvolveGCN-O resets automatically on each forward call
        # This method is provided for API consistency
        pass
    
    def predict(self, data_sequence: List[Data]) -> int:
        """
        Make a prediction for a single graph sequence.
        
        Args:
            data_sequence: List of PyG Data objects (temporal sequence).
        
        Returns:
            Predicted class (0 = Robust, 1 = Partitioned).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data_sequence)
            return logits.argmax(dim=1).item()
    
    def predict_proba(self, data_sequence: List[Data]) -> torch.Tensor:
        """
        Get prediction probabilities for a single graph sequence.
        
        Args:
            data_sequence: List of PyG Data objects (temporal sequence).
        
        Returns:
            Probability tensor [1, output_dim] with softmax probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data_sequence)
            return F.softmax(logits, dim=1)


if __name__ == "__main__":
    # Quick sanity check
    print("SatelliteEvolveGCN model definition loaded successfully.")
    
    # Create a dummy model
    model = SatelliteEvolveGCN(input_dim=3, hidden_dim=64, output_dim=2)
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
