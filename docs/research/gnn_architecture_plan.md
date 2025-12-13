# /gnn_architecture_plan.md

# EXECUTIVE SUMMARY

This document outlines a comprehensive architecture plan for an Advanced Temporal Graph Neural Network (GNN) designed for a PhD dissertation focused on satellite constellation network analysis. The primary goal is to predict future network partitions and identify critical nodes within constellations characterized by a fixed set of satellites (nodes) and highly dynamic Inter-Satellite Links (ISLs) that change with orbital mechanics (edges).

Based on an extensive literature review, the primary recommendation is to implement the **EvolveGCN-O** architecture [^1](https://arxiv.org/abs/1902.10191). This model is uniquely suited for this problem as it evolves the GNN's weight parameters over time using an LSTM, rather than relying on node embeddings. This approach is robust to the frequent and predictable changes in the graph's edge structure (ISLs) while the node set (satellites) remains constant. Its computational complexity, O(|V|Th²), is generally more favorable for long time sequences (T) compared to attention-based models. As a strong secondary alternative, **DySAT** [^2](https://arxiv.org/abs/1812.09430) is recommended due to its high performance in link prediction and its design for parallelization on GPUs. The implementation should begin with the PyTorch Geometric Temporal library for rapid prototyping and transition to a custom PyTorch Geometric implementation to incorporate advanced optimizations like batch-insensitive training (BADGNN) [^3](https://arxiv.org/html/2506.19282v1) and physics-informed constraints derived from orbital mechanics [^4](https://www.arxiv.org/pdf/2507.22279).

# 1. ARCHITECTURE COMPARISON

## 1.1 EvolveGCN

EvolveGCN adapts Graph Convolutional Networks (GCNs) for dynamic graphs by evolving the model parameters themselves over time using a Recurrent Neural Network (RNN) [^1](https://arxiv.org/pdf/1902.10191). This makes it inherently suitable for graphs where the underlying structure changes but the GNN's task remains the same.

-   **EvolveGCN-O vs EvolveGCN-H variants**:
    -   **EvolveGCN-H (Hidden)**: Treats the GCN weight matrices as the hidden states of a GRU. At each timestep, the GRU takes the current node embeddings as input to compute the new GCN weights for that step [^1](https://arxiv.org/pdf/1902.10191). This variant adapts based on the graph's current state.
    -   **EvolveGCN-O (Output)**: Treats the GCN weights as the input and output of the recurrent model. It uses an LSTM to evolve the GCN weights directly from the weights of the previous timestep, without needing node embeddings as input to the recurrent part [^1](https://arxiv.org/pdf/1902.10191). This variant is particularly powerful for scenarios with changing node sets or when focusing purely on structural evolution.

-   **Handling fixed nodes with dynamic edges**: EvolveGCN is exceptionally well-suited for this scenario. By evolving the GCN parameters (the filters that learn local neighborhood structures), it directly models how the relationships (ISLs) change over time, without being dependent on learning stable embeddings for nodes whose connectivity is constantly in flux [^1](https://arxiv.org/pdf/1902.10191).

-   **Temporal modeling approach**: It uses an RNN (specifically GRU for EvolveGCN-H and LSTM for EvolveGCN-O) to update the GCN weight matrices at each timestep. This captures temporal dependencies by learning a motion model for the GCN parameters themselves [^1](https://arxiv.org/pdf/1902.10191).

-   **Scalability analysis**: The number of trainable parameters does not grow with the number of timesteps, as only the RNN parameters are trained [^1](https://arxiv.org/pdf/1902.10191). The computational complexity per timestep is dominated by the GCN and RNN operations, generally O(|V|Th²) where h is the hidden dimension. This makes it manageable for long temporal sequences.

-   **Prediction capabilities**: The model has been successfully evaluated on dynamic link prediction, edge classification, and node classification tasks across multiple benchmarks [^1](https://arxiv.org/pdf/1902.10191).

-   **Pros and cons for satellite constellation use case**:
    -   **Pros**: Directly models the evolution of graph structure, which is the core of the ISL problem. EvolveGCN-O avoids reliance on node embeddings, which can be unstable in highly dynamic topologies. Its scalability with respect to time is excellent.
    -   **Cons**: May be less effective at capturing long-range dependencies compared to attention mechanisms. Performance on link prediction can sometimes be lower than unsupervised graph autoencoding methods [^1](https://arxiv.org/pdf/1902.10191).

-   **Citation**: Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T., Schardl, T. B., & Leiserson, C. E. (2020). Evolvegcn: Evolving graph convolutional networks for dynamic graphs. In *Proceedings of the AAAI conference on artificial intelligence* (Vol. 34, No. 04, pp. 5363-5370) [^1](https://arxiv.org/abs/1902.10191).

## 1.2 DySAT (Dynamic Self-Attention Network)

DySAT learns node representations by employing self-attention mechanisms along two dimensions: the structural neighborhood at each timestep and the temporal dynamics across timesteps [^2](https://arxiv.org/pdf/1812.09430).

-   **Architecture details**:
    -   **Structural Self-Attention**: Uses a Graph Attention Network (GAT)-like mechanism to attend over a node's immediate neighbors within a single graph snapshot, capturing spatial relationships [^2](https://arxiv.org/pdf/1812.09430).
    -   **Temporal Self-Attention**: Uses a scaled dot-product attention mechanism (similar to Transformers) to attend over a node's representations from previous timesteps, capturing its evolutionary patterns [^2](https://arxiv.org/pdf/1812.09430).

-   **Handling fixed nodes with dynamic edges**: DySAT is explicitly designed for dynamic graphs with a shared, fixed node set across snapshots [^2](https://arxiv.org/pdf/1812.09430). The structural attention adapts to the changing neighborhood (dynamic edges) at each step, while the temporal attention integrates these changes over time.

-   **Temporal modeling approach**: The temporal modeling is purely attention-based, which allows for parallel computation across the time dimension and is effective at capturing long-range dependencies, unlike RNNs which process sequentially [^2](https://arxiv.org/pdf/1812.09430).

-   **Scalability analysis**: The computational complexity is dominated by the temporal attention component, at O(|V|T²D) [^2](https://arxiv.org/pdf/1812.09430). While quadratic in the number of timesteps (T), the high parallelizability of self-attention makes DySAT empirically around 10 times faster than RNN-based methods on GPUs [^2](https://arxiv.org/pdf/1812.09430).

-   **Prediction capabilities**: DySAT has demonstrated significant performance gains on dynamic link prediction tasks, achieving 3-4% higher Macro-AUC over state-of-the-art baselines on several benchmarks [^2](https://arxiv.org/pdf/1812.09430).

-   **Pros and cons for satellite constellation use case**:
    -   **Pros**: Excellent at link prediction. The attention mechanism is powerful for capturing complex temporal dependencies. Highly parallelizable on GPU hardware.
    -   **Cons**: The O(T²) complexity can become a bottleneck for very long time sequences. It focuses on learning node embeddings, which might be less stable than evolving model parameters directly for this use case.

-   **Citation**: Sankar, A., Wu, Y., Gou, L., Zhang, W., & Yang, H. (2020). Dysat: Deep neural representation learning on dynamic graphs via self-attention networks. In *Proceedings of the 13th international conference on web search and data mining* (pp. 520-528) [^2](https://arxiv.org/abs/1812.09430).

## 1.3 Graph WaveNet

Graph WaveNet is a spatial-temporal GNN that combines graph convolutions with dilated causal convolutions (a type of Temporal Convolutional Network, or TCN) to capture spatial and temporal dependencies [^5](https://arxiv.org/pdf/1906.00121).

-   **Architecture details**: It introduces a self-adaptive adjacency matrix, computed as `SoftMax(ReLU(E₁E₂ᵀ))`, where `E₁` and `E₂` are learnable node embeddings. This allows the model to learn hidden spatial dependencies without relying on an explicit graph structure. Its temporal component uses stacked dilated 1D convolutions, enabling an exponentially large receptive field to capture long-range temporal patterns [^5](https://arxiv.org/pdf/1906.00121).

-   **Handling dynamic graphs**: The adaptive adjacency matrix allows the model to infer and adapt to the underlying graph structure, making it suitable for dynamic graphs where relationships change or are not explicitly known [^5](https://arxiv.org/pdf/1906.00121).

-   **Temporal modeling approach**: It uses a TCN instead of an RNN. TCNs are non-recurrent, allowing for parallel computation over the time dimension and avoiding issues like vanishing gradients [^5](https://arxiv.org/pdf/1906.00121).

-   **Scalability analysis**: Graph WaveNet is highly efficient. In training, it is reported to be five times faster than DCRNN, and it is the most efficient during inference as it can predict an entire sequence of future steps in a single forward pass [^5](https://arxiv.org/pdf/1906.00121).

-   **Prediction capabilities**: It is designed for future state prediction (forecasting) and outputs an entire sequence of future values at once, which is beneficial for tasks like traffic forecasting [^5](https://arxiv.org/pdf/1906.00121).

-   **Pros and cons for satellite constellation use case**:
    -   **Pros**: Very fast inference. Adaptive adjacency matrix is useful for learning implicit relationships. TCNs are efficient and effective for long sequences.
    -   **Cons**: Primarily designed for forecasting on graphs with time-varying node features, rather than graphs with a fundamentally changing topology. The adaptive matrix learns a single static graph structure to use across all timesteps, which may not be suitable for the continuously changing ISL topology.

-   **Citation**: Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph wavenet for deep spatial-temporal graph modeling. In *Proceedings of the 28th international joint conference on artificial intelligence* (pp. 1907-1913) [^5](https://arxiv.org/abs/1906.00121).

## 1.4 Additional Relevant Architectures

-   **BADGNN (2025)**: A framework for batch-insensitive training of dynamic GNNs. It uses Temporal Lipschitz Regularization and Adaptive Attention Adjustment to maintain performance with very large batch sizes (up to 11x larger), enabling significant training speedups (up to 2x) [^3](https://arxiv.org/html/2506.19282v1). This is critical for scaling to ~2000 design runs.
-   **Physics-Informed EvolveGCN (2025)**: An extension of EvolveGCN that incorporates a physics-informed loss function based on the Clohessy-Wiltshire (CW) equations for orbital mechanics. This ensures that predictions of satellite positions and velocities are physically plausible, leading to more stable and accurate forecasts [^4](https://www.arxiv.org/pdf/2507.22279).
-   **HeteroGCLSTM (2025)**: A recurrent GNN layer designed for heterogeneous graphs, specifically demonstrated on GNSS satellite-receiver networks. It models spatial and temporal dynamics simultaneously, making it suitable for graphs with different node types [^6](https://arxiv.org/html/2509.14000v1).
-   **DGCN (2021)**: A framework combining GCNs and LSTMs specifically for the task of predicting critical nodes in temporal networks. It provides a strong precedent and evaluation methodology (Kendall τ coefficient, top-k hit rate) for the critical node identification goal of this dissertation [^7](https://arxiv.org/pdf/2106.10419).

## 1.5 Architecture Comparison Table

| Architecture                  | Temporal Modeling         | Fixed Nodes/Dynamic Edges                   | Complexity           | Scalability (2000 runs)                                      | Prediction Type                              | Best For                                                                  |
| ----------------------------- | ------------------------- | ------------------------------------------- | -------------------- | ------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------------------- |
| **EvolveGCN-O**               | RNN (LSTM)                | Excellent fit; evolves model params         | O(\|V\|Th²)          | High; parameters don't grow with time, good for long sequences | Link/Node/Edge Classification                | Modeling pure structural evolution in highly dynamic topologies.          |
| **DySAT**                     | Self-Attention            | Excellent fit; designed for this scenario     | O(\|V\|T²D)          | Medium; O(T²) complexity but highly parallelizable on GPU (10x faster than RNN) | Link Prediction (SOTA performance)           | Capturing complex long-range dependencies for link prediction tasks.      |
| **Graph WaveNet**             | TCN (Dilated Convolutions) | Moderate fit; learns a single adaptive graph | Linear in T           | High; fastest inference, parallel training                    | Future State Forecasting                     | Forecasting node features on a slowly changing or unknown graph structure. |
| **BADGNN** (Framework)        | N/A (Agnostic)            | N/A                                         | N/A                  | Very High; enables massive batch sizes and 2x speedup        | N/A (Training Optimization)                  | Scaling training efficiently across thousands of design runs.             |
| **Physics-Informed EvolveGCN** | RNN (GRU)                 | Excellent fit; enhances EvolveGCN           | O(\|V\|Th²)          | High; adds physics loss, improving stability                 | Node State Prediction (Position/Velocity)    | High-fidelity satellite position forecasting with physical constraints. |
| **DGCN** (Framework)          | RNN (LSTM) + GCN          | Good fit; snapshot-based approach           | Varies               | Medium; specific implementation details matter               | Critical Node Prediction                     | Directly addressing the critical node identification task.                |

# 2. PYTORCH GEOMETRIC TEMPORAL EVALUATION

## 2.1 Library Overview

PyTorch Geometric Temporal is an extension library for PyTorch Geometric dedicated to deep learning on dynamic and temporal graphs [^8](https://pytorch-geometric-temporal.readthedocs.io/).
-   **Available models**: The library includes implementations of many state-of-the-art temporal GNNs, including **EvolveGCN** (both H and O variants), TGCN, A3TGCN, GCLSTM, and others [^8](https://pytorch-geometric-temporal.readthedocs.io/). While a direct DySAT implementation is not listed in the main docs, its components (attention layers) are available within the PyTorch Geometric ecosystem.
-   **Documentation and community support**: The library has comprehensive documentation with installation guides, tutorials, and an API reference. It is built upon the widely adopted PyTorch Geometric, benefiting from its large user base and community support.
-   **Integration with PyTorch Geometric ecosystem**: It seamlessly integrates with PyG's `DataLoader` and `Data` objects, making it easy to use for those familiar with the core library.

## 2.2 Production Readiness Assessment

The library is mature and suitable for production-level research.
-   **TGB (Temporal Graph Benchmark) integration**: The official Temporal Graph Benchmark, a key standard for evaluating temporal GNNs, provides datasets as `PyG` compatible `TemporalData` objects, demonstrating the library's format is a de facto standard for serious research [^9](https://tgb.complexdatalab.com/). This signals that the library is robust enough for rigorous, reproducible benchmarking.
-   **Support for PyG TemporalData objects**: The use of a standardized data format simplifies data handling and ensures compatibility with a wider ecosystem of tools and benchmarks [^9](https://tgb.complexdatalab.com/).
-   **Mentioned in 2024 Temporal Graph Learning survey**: A 2024 survey of the field lists PyTorch Geometric Temporal as one of the key libraries for researchers and practitioners, indicating its relevance and adoption within the academic community [^10](https://towardsdatascience.com/temporal-graph-learning-in-2024-feaa9371b8e2/).
-   **Community adoption indicators**: The library is cited in numerous research papers, showing it is actively used for developing and benchmarking new models.

## 2.3 PyG Temporal vs Custom Implementation

**PyTorch Geometric Temporal Pros:**
-   **Pre-implemented models**: Provides immediate access to battle-tested implementations of models like EvolveGCN, saving significant development time.
-   **Standardized data formats**: Simplifies data pipeline construction and ensures compatibility with benchmarks like TGB.
-   **Integration with TGB benchmarks**: Facilitates rigorous evaluation and comparison against state-of-the-art models.
-   **Active research community use**: Benefits from continuous updates and a community of users who can provide support.

**Custom PyTorch Geometric + LSTM/GRU Wrapper Pros:**
-   **More flexibility**: Allows for easier modification of core architectures to add novel components like physics-informed loss functions [^4](https://www.arxiv.org/pdf/2507.22279).
-   **Easier to add physics-informed constraints**: As demonstrated in the Physics-Informed EvolveGCN paper, a custom implementation provides direct access to the loss computation where physical constraints can be injected.
-   **Better control over training pipeline**: Enables the integration of advanced, custom training optimization techniques like the ETC framework or BADGNN's specific regularization terms [^3](https://arxiv.org/html/2506.19282v1).
-   **Can incorporate BADGNN techniques for scalability**: The specific Lipschitz regularization and attention adjustment in BADGNN would likely require a custom implementation to integrate properly.

## 2.4 Recommendation

For this dissertation, a **hybrid approach is recommended**.
1.  **Phase 1 (Prototyping)**: Begin with **PyTorch Geometric Temporal**. Its pre-built EvolveGCN-O model will enable rapid development of the baseline link prediction and critical node prediction models. This will accelerate initial experiments and ensure the core data pipeline is robust.
2.  **Phase 2 (Optimization & Extension)**: Transition to a **custom implementation** based on PyTorch Geometric. This will be necessary for the more advanced stages of the dissertation, specifically for integrating physics-informed loss functions and implementing the BADGNN framework to efficiently scale training across the ~2000 design runs. The initial prototype will provide a validated foundation for this custom model.

# 3. NODE FEATURE ENGINEERING

## 3.1 Satellite Network Features from Literature

The literature on GNNs for satellite networks has identified several classes of informative features.

**Spatial/Geometric Features:**
-   **Degree centrality**: The number of active ISLs for a satellite at a given timestep. A fundamental measure of connectivity [^4](https://www.arxiv.org/pdf/2507.22279).
-   **Betweenness centrality**: A measure of a satellite's importance as a bridge in the network's shortest paths. Crucial for identifying critical nodes.
-   **Geographic coordinates**: The satellite's latitude and longitude over the Earth's surface [^6](https://arxiv.org/html/2509.14000v1).
-   **Azimuth and elevation angles**: The angles to other visible satellites from a given satellite's perspective [^6](https://arxiv.org/html/2509.14000v1).

**Orbital Parameters:**
-   **Position vectors**: (x, y, z) coordinates in a standard frame like Earth-Centered Inertial (ECI) [^4](https://www.arxiv.org/pdf/2507.22279).
-   **Velocity vectors**: (vx, vy, vz) velocity components in the ECI frame [^4](https://www.arxiv.org/pdf/2507.22279).
-   **Keplerian Elements**: The set of six classical orbital elements: semi-major axis (a), eccentricity (e), inclination (i), Right Ascension of Ascending Node (RAAN, Ω), argument of periapsis (ω), and mean anomaly (M). These are used as direct inputs for ISL prediction [^11](https://link.springer.com/article/10.1007/s44196-024-00610-9).

**System State Features:**
-   **Battery state/energy level**: The current charge level of the satellite's batteries, which affects its operational capability [^12](https://arxiv.org/html/2303.13773v4).
-   **Power consumption/generation**: Instantaneous power draw and generation from solar panels.
-   **Link quality metrics**: Signal-to-Noise Ratio (SNR) of active communication links [^6](https://arxiv.org/html/2509.14000v1).
-   **Communication bandwidth available**: The current data throughput capacity of the satellite's transponders.

**Network Traffic Features:**
-   **Traffic flow through satellite**: The volume of data being routed through a satellite node [^13](https://www.mdpi.com/2076-3417/14/9/3840).
-   **Link capacity and utilization**: The maximum capacity of an ISL and its current usage percentage [^13](https://www.mdpi.com/2076-3417/14/9/3840).
-   **End-to-end delay**: Latency measurements for data packets traversing the network [^13](https://www.mdpi.com/2076-3417/14/9/3840).
-   **Packet loss rates**: The percentage of dropped packets on communication links.

## 3.2 Feature Normalization Best Practices

-   **GRANOLA approach**: A 2024 paper introduced GRANOLA, an adaptive normalization layer for GNNs. It generates normalization parameters (gamma, beta) using a secondary GNN (`GNNnorm`) that takes the graph structure and Random Node Features (RNF) as input. This allows the normalization to adapt to each specific graph instance, leading to faster convergence and better performance [^14](https://arxiv.org/html/2404.13344v1). This is highly recommended for handling the diverse constellation designs.
-   **Temporal vs spatial feature normalization**:
    -   **Spatial Features** (e.g., degree centrality) should be re-normalized at each timestep based on the statistics of that specific graph snapshot.
    -   **Temporal Features** (e.g., battery depletion) should be normalized over the entire time horizon of a single simulation run to preserve their relative trend.
-   **Handling time-varying features**: For features like battery level, which have a clear temporal pattern (depleting in shadow, charging in sun), it can be beneficial to use their raw values or a normalized derivative to capture the rate of change.
-   **Standardization vs min-max scaling**:
    -   **Standardization (Z-score)** is recommended for features with a Gaussian-like distribution (e.g., link quality metrics).
    -   **Min-max scaling** is suitable for features with defined bounds (e.g., battery level from 0 to 100, latitude from -90 to 90).

## 3.3 Recommended Feature Set for HypatiaAdapter

Assuming the dissertation dataset contains orbital and connectivity information:

-   **Minimum viable features**:
    -   `degree_centrality`: Calculated from the adjacency matrix at each step.
    -   `position_x, position_y, position_z`: ECI coordinates.
    -   `velocity_vx, velocity_vy, velocity_vz`: ECI velocity components.
    *These features are fundamental and should be directly available or derivable from `tier1_design_steps.csv`.*

-   **Extended feature set**:
    -   `betweenness_centrality`: Calculated at each step (computationally intensive).
    -   `latitude`, `longitude`: Derived from ECI position.
    -   `is_in_sunlight`: Binary feature derived from position and time (proxy for power generation state).
    *This set adds more topological and system state information.*

-   **Advanced features**:
    -   `Keplerian Elements`: If available or derivable, these are powerful predictors for ISLs [^11](https://link.springer.com/article/10.1007/s44196-024-00610-9).
    -   `relative_position_cw`, `relative_velocity_cw`: Physics-informed features derived from Clohessy-Wiltshire equations relative to a chief satellite in the constellation plane.
    *These features require more domain knowledge and pre-processing but can significantly improve model accuracy.*

# 4. INPUT TENSOR SPECIFICATIONS

## 4.1 EvolveGCN Tensor Format

-   **Input**: EvolveGCN processes a sequence of graph snapshots. For each timestep `t`, it requires:
    -   Adjacency matrices `At` ∈ ℝ^(n×n), where n is the number of nodes (satellites).
    -   Node features `Xt` ∈ ℝ^(n×d), where d is the number of node features.
-   **Batch dimension handling**: For batch training, these are combined into a tensor.
-   **Specific shape**: `(Batch_size, Time_steps, Nodes, Features)`. The `Nodes` and `Features` dimensions are processed by the GCN layers, while the `Time_steps` dimension is processed by the recurrent component (LSTM/GRU).
-   **Example for satellite constellation**: For a batch of 32 simulation runs, each with 10 timesteps, for a 66-satellite constellation with 12 features per satellite, the shape would be `(32, 10, 66, 12)`.

## 4.2 DySAT Tensor Format

-   **Input**: DySAT also takes a sequence of graph snapshots, `G₁, G₂, ..., GT`, where the node set `V` is fixed [^2](https://arxiv.org/pdf/1812.09430).
-   **Structural attention input shape**: For each snapshot, the input is a node feature matrix of shape `(Nodes, Features)`.
-   **Temporal attention input shape**: After processing by the structural attention layers, the representations for a single node across all timesteps are collected into a sequence of shape `(Time_steps, Hidden_Features)`, which is then processed by the temporal attention mechanism.
-   **Specific shape recommendation**: The initial input data can be structured identically to EvolveGCN: `(Batch_size, Time_steps, Nodes, Features)`. The model's internal layers will then reshape and process this data accordingly.

## 4.3 Graph WaveNet Tensor Format

-   **Input**: Graph WaveNet is designed for a three-dimensional tensor of shape `[N, C, L]`, where `N` is the number of nodes, `C` is the number of channels (features), and `L` is the sequence length (timesteps) [^5](https://arxiv.org/pdf/1906.00121).
-   **Adaptive adjacency matrix**: It also requires two learnable embedding matrices, `E₁` and `E₂`, of shape `(Nodes, Embedding_dim)` to compute the adaptive adjacency matrix.
-   **Output shape**: It produces an output tensor of shape `[N, C, L_out]` in a single pass.
-   **Specific recommendation**: For the satellite scenario, the input data would need to be permuted to `(Batch_size, Nodes, Features, Time_steps)`.

## 4.4 Recommended Input Shape for Dissertation

The specific recommended tensor shape is `(Batch_size, Time_steps, Nodes, Features)`.

-   **Batch**: Start with **32**. Using the findings from the BADGNN paper, this can be scaled up to **256 or higher** during optimization to significantly reduce training time without sacrificing performance [^3](https://arxiv.org/html/2506.19282v1).
-   **Time**: A window of **10-20 timesteps** is recommended. This is a balance between capturing meaningful temporal dynamics and managing GPU memory. The prediction horizon would typically be 1-5 steps into the future.
-   **Nodes**: The number of satellites varies (50-550+). To handle this in batches, all graphs within a batch should be padded to the size of the largest graph in that batch. An attention mask must be used during message passing and loss calculation to ignore the padded nodes. A more advanced strategy involves batching together simulations with similar constellation sizes.
-   **Features**: Based on the recommended minimum viable feature set (Section 3.3), the initial number of features `d` would be **7** (1 for degree centrality, 3 for position, 3 for velocity).

# 5. IMPLEMENTATION STRATEGY

## 5.1 Recommended Architecture

**PRIMARY RECOMMENDATION: EvolveGCN-O**

This architecture is the best fit for the dissertation's goals, justified by:
-   **Fixed nodes + dynamic edges requirement**: It is explicitly designed to model structural evolution by adapting GCN parameters, which perfectly matches the satellite problem [^1](https://arxiv.org/pdf/1902.10191).
-   **Critical node identification capability**: Its ability to perform node classification makes it directly applicable to identifying critical nodes.
-   **Partition prediction capability**: By predicting future links, the model's output can be used to construct the future graph and analyze its partitions.
-   **Scalability to ~2000 design runs**: Its O(|V|Th²) complexity is favorable for long time sequences, and the number of trainable parameters is constant with respect to time, making it memory efficient [^1](https://arxiv.org/pdf/1902.10191). When combined with BADGNN/ETC techniques, it can be trained efficiently on the large dataset.
-   **Implementation complexity**: It is a well-established architecture with available implementations in PyG Temporal, making the baseline model straightforward to implement within the dissertation timeline.

## 5.2 Implementation Steps

### Phase 1: Data Preparation (Weeks 1-2)
-   Load `tier1_design_runs.csv` and `tier1_design_steps.csv`.
-   Extract graph snapshots from HypatiaAdapter at regular intervals (e.g., every 5 minutes of simulation time).
-   Construct sparse adjacency matrices `At` for each timestep based on ISL connectivity.
-   Engineer node features `Xt` (start with degree centrality, ECI position, and ECI velocity).
-   Create train/validation/test splits, ensuring stratification by constellation size to prevent distribution shift.

### Phase 2: Baseline Model (Weeks 3-4)
-   Implement EvolveGCN-O using PyTorch Geometric Temporal.
-   Develop a training loop for dynamic link prediction, using Binary Cross-Entropy loss.
-   Evaluate on TGB-style metrics: Average Precision (AP) and AUC-ROC for predicting future ISL formation.

### Phase 3: Critical Node Prediction (Weeks 5-6)
-   Adapt the EvolveGCN-O output layer for node classification (a linear layer followed by a Sigmoid/Softmax).
-   Define the ground-truth criticality metric: compute the betweenness centrality for each node in future graph snapshots.
-   Train the model to predict the top-k most central nodes.
-   Evaluate using the Kendall τ rank correlation coefficient and top-k hit rate, as established in the DGCN paper [^7](https://arxiv.org/pdf/2106.10419).

### Phase 4: Partition Prediction (Weeks 7-8)
-   Use the trained link prediction model to generate a predicted adjacency matrix for a future timestep.
-   Apply a community detection algorithm (e.g., Louvain) to the predicted graph to identify partitions.
-   Evaluate the predicted partitions against the ground-truth partitions using graph-level metrics like modularity and Normalized Mutual Information (NMI).

### Phase 5: Optimization & Ablation (Weeks 9-10)
-   Transition to a custom implementation to apply BADGNN techniques for training with larger batch sizes [^3](https://arxiv.org/html/2506.19282v1).
-   Integrate the GRANOLA adaptive normalization layer to improve convergence and performance [^14](https://arxiv.org/html/2404.13344v1).
-   Conduct ablation studies on the importance of different node features (e.g., orbital vs. geometric).
-   Perform hyperparameter tuning on learning rate, hidden dimensions, and the temporal window size.

### Phase 6: Physics-Informed Extension (Optional, Weeks 11-12)
-   Add a physics-informed loss term based on Clohessy-Wiltshire equations to the main loss function, penalizing physically implausible trajectory predictions [^4](https://www.arxiv.org/pdf/2507.22279).
-   Compare the performance and prediction stability of the physics-informed model versus the pure data-driven approach.

## 5.3 Code Structure
```
satellite_gnn/
├── data/
│   ├── tier1_design_runs.csv
│   ├── tier1_design_steps.csv
│   └── hypatia_adapter.py      # Script to process raw data into graph snapshots
├── models/
│   ├── evolvegcn.py            # EvolveGCN-O model definition
│   ├── layers.py               # Custom layers (e.g., GRANOLA)
│   └── loss_functions.py       # Physics-informed loss, focal loss
├── training/
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Evaluation script for all tasks
│   └── utils.py                # Helper functions, masking, padding
├── experiments/
│   ├── link_prediction.py      # Experiment runner for link prediction
│   ├── critical_nodes.py       # Experiment runner for critical node ID
│   └── partition_prediction.py   # Experiment runner for partition prediction
└── config/
    └── model_config.yaml       # Hyperparameters and settings
```

## 5.4 Training Optimization
-   **Batch size**: Start at 32. With a custom implementation, apply BADGNN principles to scale up to 256 or 512 for massive throughput [^3](https://arxiv.org/html/2506.19282v1).
-   **Learning rate**: 0.001 with an Adam optimizer and a learning rate scheduler (e.g., ReduceLROnPlateau).
-   **Early stopping**: Monitor validation loss with a patience of 20 epochs to prevent overfitting.
-   **ETC framework**: Implement the information-loss-bounded batching scheme from the ETC paper for a more principled approach to creating large batches, which can yield speedups from 1.6x to 62.4x [^15](https://www.vldb.org/pvldb/vol17/p1060-gao.pdf).
-   **GPU utilization**: If switching to DySAT, leverage its high parallelizability, which has shown up to 10x speedups over RNN-based methods on GPUs [^2](https://arxiv.org/pdf/1812.09430).

## 5.5 Evaluation Protocol
-   **Link Prediction**: Average Precision (AP), AUC-ROC on future ISL formation.
-   **Critical Nodes**: Kendall τ rank correlation, top-5/top-10 hit rate.
-   **Partition Prediction**: Modularity, Normalized Mutual Information (NMI).
-   **TGB Benchmarks**: Use the evaluation protocols from TGB for link prediction to ensure rigorous and standardized assessment [^9](https://tgb.complexdatalab.com/).
-   **Ablation Studies**: Systematically remove feature groups or model components (e.g., GRANOLA, physics-loss) to quantify their contribution.

# 6. SATELLITE-SPECIFIC CONSIDERATIONS

## 6.1 Handling Orbital Dynamics
-   **ISL visibility windows**: Adjacency matrices must be constructed based on line-of-sight and maximum range constraints between satellites at each timestep.
-   **Orbital period synchronization**: The temporal window for the GNN should be chosen carefully to capture phenomena within and across orbital periods.
-   **Day/night cycle effects**: The `is_in_sunlight` feature can model the battery charging/discharging cycle, which is a critical system state.
-   **Doppler shift and link quality**: Advanced models can use predicted Doppler shifts to inform features related to link quality.

## 6.2 Constellation-Specific Patterns
-   **Polar orbit clustering**: The GNN must learn that satellites in polar orbits have significantly higher connectivity (degree centrality) near the poles.
-   **Equatorial gaps**: The model should be able to identify potential network partitions that arise from coverage gaps over the equator in certain constellation designs.
-   **Walker Delta/Star geometries**: The GNN's learned filters should be able to capture the regular, repeating structural patterns inherent in these common constellation geometries.
-   **Plane phasing**: Features describing a satellite's plane ID and its phase within that plane can help the model distinguish between intra-plane and inter-plane ISLs.

## 6.3 Physics-Informed Constraints
-   **Clohessy-Wiltshire equations**: As shown in the 2025 Physics-Informed EvolveGCN paper, these equations of relative motion can be incorporated into the loss function to enforce physically correct trajectories for co-planar satellites [^4](https://www.arxiv.org/pdf/2507.22279).
-   **Kepler vs SGP4 orbit propagation**: The choice of orbit propagator for generating ground-truth data is critical. SGP4 is more realistic as it accounts for perturbations like atmospheric drag [^11](https://link.springer.com/article/10.1007/s44196-024-00610-9).
-   **Conservation of energy**: For advanced physics constraints, the total energy of the satellite (kinetic + potential) should remain constant, which can be used as another loss term.
-   **Maximum ISL range**: A hard constraint (e.g., 2000-5000 km) should be applied when constructing the adjacency matrix to reflect physical limitations.

# 7. EXPECTED OUTCOMES & SUCCESS METRICS

## 7.1 Baseline Performance Targets
Based on state-of-the-art results from the literature:
-   **Link Prediction AP**: >0.85 (DySAT achieved 3-4% improvement over strong baselines, setting a high bar [^2](https://arxiv.org/pdf/1812.09430)).
-   **Critical Node top-10 hit rate**: >0.70 (Based on performance of DGCN on real-world temporal networks [^7](https://arxiv.org/pdf/2106.10419)).
-   **Partition Prediction NMI**: >0.75 (A standard target for good community detection performance).
-   **Training time**: <2 hours per epoch on a single V100/A100 GPU for a large constellation, leveraging BADGNN/ETC optimizations [^3](https://arxiv.org/html/2506.19282v1).

## 7.2 Dissertation Contributions
-   A novel application of advanced temporal GNNs to the critical problem of satellite constellation partitioning and resilience analysis.
-   A rigorous, empirical comparison of leading architectures (EvolveGCN, DySAT) on a large-scale aerospace network dataset.
-   The development of a physics-informed GNN for satellite systems that combines data-driven learning with orbital mechanics.
-   A demonstration of scalable training methodologies for temporal GNNs on a dataset of ~2000 complex constellation designs.
-   A comprehensive study on feature engineering for GNNs in orbital networks, identifying key predictive features.

## 7.3 Production Readiness
-   **TGB benchmark integration**: Adherence to TGB evaluation protocols will ensure the research is reproducible and comparable to other SOTA models [^9](https://tgb.complexdatalab.com/).
-   **Model checkpointing and versioning**: All models and results will be saved and tracked for reproducibility.
-   **Inference time**: Target <100ms per multi-step prediction on a single GPU to be viable for near-real-time operational scenarios.
-   **Generalization to unseen constellation sizes**: The model must demonstrate strong performance on test sets with constellation sizes not seen during training.

# 8. IMPLEMENTATION RESOURCES

## 8.1 Code Repositories
-   **EvolveGCN**: [https://github.com/IBM/EvolveGCN](https://github.com/IBM/EvolveGCN) [^1](https://arxiv.org/abs/1902.10191)
-   **DySAT PyTorch**: [https://github.com/FeiGSSS/DySAT_pytorch](https://github.com/FeiGSSS/DySAT_pytorch)
-   **PyG Temporal**: [https://github.com/benedekrozemberczki/pytorch_geometric_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) [^8](https://pytorch-geometric-temporal.readthedocs.io/)
-   **TGB Benchmarks**: [https://github.com/shenyangHuang/TGB](https://github.com/shenyangHuang/TGB) [^9](https://tgb.complexdatalab.com/)
-   **ETC Training**: [https://github.com/eddiegaoo/ETC](https://github.com/eddiegaoo/ETC) [^15](https://www.vldb.org/pvldb/vol17/p1060-gao.pdf)
-   **Satellite GNN Example**: [https://github.com/daddydrac/Combinatorial-Optimization-and-Reasoning-for-GNNs](https://github.com/daddydrac/Combinatorial-Optimization-and-Reasoning-for-GNNs) [^16](https://medium.com/@joehoeller/optimizing-satellite-constellations-with-graph-neural-networks-6ce87d50a29f)

## 8.2 Key Papers (Full Citations)

**Core Architectures:**
-   Pareja, A., et al. (2020). "Evolvegcn: Evolving graph convolutional networks for dynamic graphs." *AAAI*. [^1](https://arxiv.org/abs/1902.10191)
-   Sankar, A., et al. (2020). "Dysat: Deep neural representation learning on dynamic graphs via self-attention networks." *WSDM*. [^2](https://arxiv.org/abs/1812.09430)
-   Wu, Z., et al. (2019). "Graph wavenet for deep spatial-temporal graph modeling." *IJCAI*. [^5](https://arxiv.org/abs/1906.00121)
-   Yu, E., et al. (2021). "Predicting critical nodes in temporal networks by dynamic graph convolutional networks." *arXiv preprint arXiv:2106.10419*. [^7](https://arxiv.org/abs/2106.10419)

**Satellite Applications:**
-   Mehta, P., et al. (2025). "Physics-Informed EvolveGCN: Satellite Prediction for Multi Agent Systems." *arXiv preprint arXiv:2507.22279*. [^4](https://www.arxiv.org/pdf/2507.22279)
-   Borio, D., et al. (2025). "Deep Temporal Graph Networks for Real-Time Correction of GNSS Jamming-Induced Deviations." *arXiv preprint arXiv:2509.14000v1*. [^6](https://arxiv.org/html/2509.14000v1)
-   Zhu, M., et al. (2024). "Low Earth Orbit Satellite Network Routing Algorithm Based on Graph Neural Network and Deep Reinforcement Learning." *Applied Sciences*. [^13](https://www.mdpi.com/2076-3417/14/9/3840)

**Training Optimization:**
-   Zhou, Y., & Ren, X. (2025). "A Batch-Insensitive Dynamic GNN Approach to Address Temporal Discontinuity in Graph Streams." *arXiv preprint arXiv:2506.19282v1*. [^3](https://arxiv.org/html/2506.19282v1)
-   Gao, Z., et al. (2024). "ETC: Efficient Training of Temporal Graph Neural Networks on Large-Scale Dynamic Graphs." *VLDB*. [^15](https://www.vldb.org/pvldb/vol17/p1060-gao.pdf)
-   Beaini, D., et al. (2024). "GRANOLA: Adaptive Normalization for Graph Neural Networks." *arXiv preprint arXiv:2404.13344v1*. [^14](https://arxiv.org/html/2404.13344v1)

**Surveys:**
-   Kazemi, S. M., et al. (2024). "A Comprehensive Survey of Dynamic Graph Neural Networks." *arXiv preprint arXiv:2404.18211*.
-   Huang, S., et al. (2024). "Temporal Graph Learning in 2024." *Towards Data Science*. [^10](https://towardsdatascience.com/temporal-graph-learning-in-2024-feaa9371b8e2/)

**Feature Engineering:**
-   Perez-Nieto, C., et al. (2024). "Inter-Satellite Link Prediction with Supervised Learning Based on Kepler and SGP4 Orbits." *International Journal of Computational Intelligence Systems*. [^11](https://link.springer.com/article/10.1007/s44196-024-00610-9)
-   Lesage, J., et al. (2023). "Graph Neural Networks for the Offline Nanosatellite Task Scheduling Problem." *arXiv preprint arXiv:2303.13773v4*. [^12](https://arxiv.org/html/2303.13773v4)

## 8.3 Datasets & Benchmarks
-   **TGB**: The standard for temporal graph benchmarks. URL: [https://tgb.complexdatalab.com/](https://tgb.complexdatalab.com/) [^9](https://tgb.complexdatalab.com/)
-   **Celestrak**: Source for real-world satellite Two-Line Element (TLE) sets for orbit propagation. URL: [https://celestrak.org/](https://celestrak.org/) [^11](https://link.springer.com/article/10.1007/s44196-024-00610-9)
-   **HypatiaAdapter**: The primary dissertation dataset, providing graph snapshots from the simulation environment.

# 9. RISK MITIGATION & ALTERNATIVES

## 9.1 Potential Challenges
-   **Varying constellation sizes (50-550+ satellites)**: This is a major challenge for batching. **Solution**: Pad node and adjacency matrices to the maximum size within a batch and use an attention mask to ignore padded nodes during all computations.
-   **Memory constraints with ~2000 runs**: Loading and processing all snapshots can exceed GPU memory. **Solution**: Implement the large-batch training strategies from BADGNN [^3](https://arxiv.org/html/2506.19282v1) and efficient data loading from the ETC framework [^15](https://www.vldb.org/pvldb/vol17/p1060-gao.pdf).
-   **Imbalanced ISL formation (mostly no links)**: In any given snapshot, the graph is sparse, leading to a class imbalance problem for link prediction. **Solution**: Use focal loss or class weighting during training to give more importance to the minority class (link formation).
-   **Long-range temporal dependencies**: Events like battery depletion over multiple orbits require a long temporal receptive field. **Solution**: Increase the temporal window size (`Time_steps`) or use an attention-based model like DySAT which is better at capturing long-range dependencies.

## 9.2 Fallback Strategies
If the primary architecture (EvolveGCN-O) underperforms:
-   **Alternative 1: Switch to DySAT**: If link prediction performance is paramount and long-range dependencies are key, switch to DySAT. Its parallelizability can offset the higher theoretical complexity.
-   **Alternative 2: Ensemble Model**: Combine the outputs of EvolveGCN (strong on structural evolution) and DySAT (strong on link prediction) for a more robust prediction.
-   **Alternative 3: Hybrid Physics-Based + GNN**: If the GNN struggles to learn orbital dynamics, simplify its task. Use an orbit propagator (e.g., SGP4) to predict the future topology and use a GNN only to predict system-state features (e.g., link quality, congestion).
-   **Alternative 4: Simplify to Static GNN**: If temporal dynamics prove too complex, aggregate features over a time window (e.g., average degree, max betweenness) and apply a simpler static GNN (like GCN or GAT) to predict future criticality.

# 10. TIMELINE & NEXT STEPS

## Immediate Next Steps (Week 1)
1.  Load `tier1_design_runs.csv` and `tier1_design_steps.csv` into a data frame.
2.  Inspect the data schema to confirm availability of position, velocity, and connectivity data.
3.  Implement the `hypatia_adapter.py` script to extract the first full graph snapshot for a single simulation run.
4.  Implement the function to construct a sparse adjacency matrix from the ISL data.
5.  Verify that the generated tensors match the recommended `(Batch, Time, Nodes, Features)` format.

## Short-term Goals (Weeks 2-4)
1.  Implement the baseline EvolveGCN-O model using PyTorch Geometric Temporal.
2.  Train the model on the binary link prediction task for a single future timestep.
3.  Achieve a baseline validation AP > 0.80.
4.  Document initial results, training curves, and model performance in the dissertation draft.

## Medium-term Goals (Weeks 5-8)
1.  Extend the model architecture to predict node-level criticality scores.
2.  Implement the partition prediction pipeline using the outputs of the link prediction model.
3.  Conduct ablation studies on the minimal viable feature set vs. the extended feature set.
4.  Compare GNN performance against the 92% accuracy Random Forest baseline from prior work.

## Long-term Goals (Weeks 9-12)
1.  Implement the optional physics-informed loss extension.
2.  Refactor the training loop to incorporate BADGNN/ETC optimization techniques for final, large-scale experiments.
3.  Run final experiments on the full test set across all tasks.
4.  Analyze results, generate figures, and write the main experiment chapter of the dissertation.

# APPENDIX: TECHNICAL SPECIFICATIONS

-   **Hardware requirements**:
    -   GPU: NVIDIA A100 or V100 with at least 32GB of VRAM recommended for handling large batches and padded tensors.
    -   CPU: 16+ cores for parallel data preprocessing.
    -   RAM: 128GB+ to hold intermediate data structures for ~2000 runs.
-   **Software dependencies**:
    -   PyTorch 2.0+
    -   CUDA 11.8+
    -   PyTorch Geometric 2.4+
    -   PyTorch Geometric Temporal 0.5+
    -   `tgb`, `pandas`, `numpy`, `scikit-learn`
-   **Estimated training time per architecture**:
    -   EvolveGCN-O: ~2-3 hours per epoch on a large constellation run (without optimization). With BADGNN/ETC, this should reduce to <1 hour.
    -   DySAT: ~1.5-2.5 hours per epoch (benefits from GPU parallelism).
-   **Memory footprint analysis**: The largest memory consumer will be the padded feature and adjacency tensors. For a batch of 32 runs, padded to 550 nodes, with a time window of 20 and 12 features, the feature tensor alone would be `32 * 20 * 550 * 12 * 4 bytes ≈ 16 GB`. This underscores the need for memory-efficient data loading and large-batch optimizations.

-   **Computational Complexity Comparison Table**:

| Architecture  | Time Complexity       | Space Complexity (Parameters) | Notes                                           |
|---------------|-----------------------|-------------------------------|-------------------------------------------------|
| EvolveGCN     | O(T * (|E| + |V|d)h)   | O(h² + dh)                    | Linear in T, efficient for long sequences.      |
| DySAT         | O(T²|V|d + T|E|d)     | O(d²)                         | Quadratic in T, but highly parallelizable.      |
| Graph WaveNet | O(T * (|E| + |V|d²)L) | O(d²L)                        | L = number of layers. Very fast inference.      |