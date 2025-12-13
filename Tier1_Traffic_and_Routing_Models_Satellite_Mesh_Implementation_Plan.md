# Traffic and Routing Models for Satellite Mesh Networks: A Comprehensive Research Report

## 1. Executive Summary

This report provides a comprehensive analysis of traffic and routing models applicable to dynamic satellite mesh networks. The research covers foundational routing algorithms, advanced multi-path techniques, demand matrix computation, performance metrics, and state-of-the-art methods for accelerating large-scale simulations.

The key findings indicate that a robust and efficient routing model for satellite networks must account for rapidly changing topologies and stringent latency requirements. While classic algorithms like Dijkstra's remain fundamental for shortest-path computation, they are insufficient on their own. Techniques like Equal-Cost Multi-Path (ECMP) are crucial for load balancing but must be implemented carefully to avoid hash collisions that can reduce network bisection bandwidth by up to 50% [^19](https://raghavan.usc.edu/papers/hedera-nsdi10.pdf).

For generating large-scale training or evaluation datasets, traditional flow-level simulators are prohibitively slow. The report details several fast approximation methods that offer significant speedups. **SaTE**, a Graph Neural Network (GNN)-based approach, provides a 2738x speedup over commercial solvers with a computation latency of just 17ms [^14](https://fardatalab.org/sigcomm26-wu.pdf). Other methods like the Data-Oriented Network Simulator (**DONS**) achieve a 65x speedup over OMNET++ [^12](https://vincen.tl/files/gao23dons.pdf).

**Key Recommendation:**
For building a minimal-yet-defensible routing model, a hybrid approach is recommended. Use established Python libraries like **NetworkX** for graph representation and implementing core routing logic (Shortest Path, K-Shortest Path, ECMP). Generate traffic using various demand matrix patterns (uniform, hotspot, GS-to-sat). To overcome the performance bottleneck of simulating millions of flows, avoid full flow solvers for large dataset generation. Instead, leverage the principles from fast approximation methods by pre-calculating paths and directly computing edge utilization based on aggregate demand. This approach balances accuracy with the speed required for large-scale analysis and dataset generation.

---

## 2. Shortest Path Routing Algorithms

The foundation of any network routing model is the ability to find the shortest path between two nodes. The choice of algorithm depends on the graph's properties, such as the presence of edge weights and whether they can be negative.

### Dijkstra's Algorithm
Dijkstra's algorithm is a greedy algorithm that finds the shortest path between a source node and all other nodes in a graph with non-negative edge weights.
*   **Complexity:** O((V+E)log V) using a priority queue [^20](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html).
*   **Use Cases:** Ideal for calculating shortest paths where the cost metric (e.g., latency, distance) is always positive. This is the most common scenario in satellite networks.
*   **Implementation:** Available in `NetworkX` via functions like `networkx.dijkstra_path` and `networkx.single_source_dijkstra` [^20](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html).

### Bellman-Ford Algorithm
The Bellman-Ford algorithm computes shortest paths from a single source vertex to all other vertices in a weighted digraph. It is more versatile than Dijkstra's as it can handle graphs with negative edge weights.
*   **Complexity:** O(VE) [^20](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html).
*   **Use Cases:** Necessary if edge weights can be negative (e.g., in economic routing models). It can also detect negative weight cycles.
*   **Implementation:** Available in `NetworkX` via `networkx.bellman_ford_path` and `networkx.single_source_bellman_ford` [^20](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html).

### Breadth-First Search (BFS) for Unweighted Graphs
For graphs where all edge weights are equal (or unweighted), a simple Breadth-First Search (BFS) is the most efficient algorithm for finding the shortest path.
*   **Complexity:** O(V+E) [^20](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html).
*   **Use Cases:** Best for finding the path with the minimum number of hops.
*   **Implementation:** `NetworkX` uses BFS for its unweighted shortest path functions, such as `networkx.shortest_path` [^20](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html).

### K-Shortest Path Algorithms
Finding a single shortest path is often insufficient for traffic engineering, which may require multiple diverse paths for load balancing or resilience.
*   **Yen's Algorithm:** A widely known algorithm to find K shortest loopless paths. It is used in the **LEOCraft** framework for LEO network simulation [^9](https://www.usenix.org/system/files/atc25-basak.pdf).
*   **Eppstein's Algorithm:** A more efficient algorithm that improves upon previous bounds by achieving constant time per path after an initial computation. It is faster than Yen's algorithm for large K [^15](https://ics.uci.edu/~eppstein/pubs/Epp-SJC-98.pdf).
    *   **Complexity:** O(m + n log n + k) to find k paths between two vertices [^15](https://ics.uci.edu/~eppstein/pubs/Epp-SJC-98.pdf).

### All-Pairs Shortest Path Algorithms
These algorithms compute the shortest paths between all pairs of nodes in the graph.
*   **Floyd-Warshall Algorithm:** A dynamic programming algorithm that is efficient for dense graphs.
    *   **Complexity:** O(V³) [^20](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html).
*   **Johnson's Algorithm:** Combines Bellman-Ford with Dijkstra's algorithm. It is more efficient for sparse graphs and can handle negative edge weights.
    *   **Complexity:** O(V(V+E)log V) [^20](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html).

### Practical Libraries

| Library | Logo | Description |
| :--- | :--- | :--- |
| **NetworkX** | ![NetworkX Logo](https://networkx.org/documentation/stable/_static/networkx_banner.svg) [^20](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html) | A pure Python library for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. It provides a comprehensive suite of shortest path algorithms [^20](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html). |
| **igraph** | ![igraph Visualization](https://python.igraph.org/en/main/_images/sphx_glr_shortest_path_visualisation_001.png) [^21](https://python.igraph.org/en/main/tutorials/shortest_path_visualisation.html) | A high-performance graph library with bindings for Python, R, and C++. It is often faster than NetworkX due to its C core. It provides functions like `get_shortest_paths` and `get_all_shortest_paths` [^21](https://python.igraph.org/en/main/tutorials/shortest_path_visualisation.html). |

---

## 3. ECMP (Equal-Cost Multi-Path) Routing

ECMP is a routing strategy that allows a router to forward packets to a single destination over multiple paths of equal cost. This enables load balancing and increases effective bandwidth.

*   **Modified Dijkstra:** To support ECMP, a shortest path algorithm like Dijkstra's must be modified to store a list of all next-hops for a destination if they belong to paths of equal cost, rather than selecting only one [^16](https://cs.brown.edu/research/pubs/theses/capstones/2015/pak.pdf).
*   **Hash-Based Load Balancing:** Traffic is typically distributed across the equal-cost paths using a hash function on packet headers (e.g., source/destination IP addresses and ports). This ensures that packets of the same flow follow the same path, preventing reordering [^16](https://cs.brown.edu/research/pubs/theses/capstones/2015/pak.pdf).
*   **Port Utilization Tracking:** For more intelligent load balancing, especially in SDN environments, the controller can track the utilization of switch ports to make fairer path selection decisions, choosing the least-used port for new flows [^16](https://cs.brown.edu/research/pubs/theses/capstones/2015/pak.pdf).
*   **SDN Implementation:** In an SDN context, a central controller (e.g., Floodlight) computes the equal-cost paths and installs forwarding rules into the data plane switches (e.g., Open vSwitch) [^16](https://cs.brown.edu/research/pubs/theses/capstones/2015/pak.pdf).
*   **ECMP Collision Issues:** Static hashing in ECMP can lead to "collisions," where multiple large flows are hashed to the same path, creating a bottleneck even when other paths are underutilized. Research presented in the **Hedera** paper showed that such collisions can result in a **50% loss of bisection bandwidth** [^19](https://raghavan.usc.edu/papers/hedera-nsdi10.pdf).
*   **Use in Satellite Networks:** The **SKYFALL** framework proposes using ECMP as a countermeasure against Link-Flooding Attacks (LFAs) in LEO networks. By distributing traffic across multiple Inter-Satellite Links (ISLs), ECMP can prevent single-link congestion and balance the load under attack [^8](https://www.ndss-symposium.org/wp-content/uploads/2025-109-paper.pdf).

---

## 4. Edge Utilization Computation with Demand Matrices

To simulate network performance, it is essential to model the traffic flowing through it. This is typically done using a demand matrix (or traffic matrix).

*   **Demand Matrix Representation:** An N×N matrix where N is the number of nodes in the network. Each element `(i, j)` represents the traffic demand from source node `i` to destination node `j` [^19](https://raghavan.usc.edu/papers/hedera-nsdi10.pdf).
*   **Demand Estimation:** The **Hedera** paper proposes an iterative algorithm to estimate the "natural demand" of flows. The algorithm has a time complexity of **O(|F|)**, where F is the number of active large flows, and typically runs in 50-200ms [^19](https://raghavan.usc.edu/papers/hedera-nsdi10.pdf).

### Traffic Patterns for Simulation
Different traffic patterns are used to evaluate routing performance under various conditions.
*   **Uniform All-to-All:** Every node sends an equal amount of traffic to every other node.
*   **Ground-Station-to-Satellite (GS-to-Sat):** Models the asymmetric nature of traffic entering the satellite mesh from ground stations.
*   **Hotspot Patterns:** A many-to-one or many-to-few pattern where a large number of nodes send traffic to a small number of destination nodes (e.g., popular data centers).
*   **Additional Patterns (from Hedera [^19](https://raghavan.usc.edu/papers/hedera-nsdi10.pdf)):**
    *   **Stride(i):** Host `x` sends to host `(x + i) mod N`.
    *   **Staggered Prob:** Hosts send traffic to others in the same edge switch, same pod, or elsewhere with varying probabilities.
    *   **Random:** Uniform probability for any source-destination pair.
    *   **Data Shuffle:** An all-to-all pattern modeling large data transfers like those in MapReduce.

### Algorithms for Flow Scheduling
To avoid the pitfalls of static ECMP, dynamic flow scheduling can be used.
*   **Global First Fit (GFF):** A greedy algorithm that linearly searches for an available path that can accommodate a large flow's demand [^19](https://raghavan.usc.edu/papers/hedera-nsdi10.pdf).
*   **Simulated Annealing (SA):** A probabilistic optimization algorithm that computes near-optimal paths for flows to maximize overall network utilization [^19](https://raghavan.usc.edu/papers/hedera-nsdi10.pdf).

### Flow Threshold Considerations
Dynamically scheduling every single flow is computationally expensive. A common strategy is to only manage "elephant flows" that are responsible for the majority of data transfer. The Hedera paper suggests a threshold of **100 Mbps** (representing 10% of a 1GbE link) to classify a flow as large enough to warrant dynamic scheduling [^19](https://raghavan.usc.edu/papers/hedera-nsdi10.pdf).

---

## 5. Performance Metrics

A comprehensive set of metrics is required to evaluate the performance of a routing model.

### A. Max Utilization
This metric identifies the most congested link in the network, which is often the bottleneck that limits overall performance.
*   **Definition:** The maximum utilization across all edges in the network, calculated as `(demand on edge) / (capacity of edge)`.
*   **Implementation:** Tracked in frameworks like **Hedera** to guide flow placement [^19](https://raghavan.usc.edu/papers/hedera-nsdi10.pdf) and **SKYFALL** to identify bottleneck links under attack [^8](https://www.ndss-symposium.org/wp-content/uploads/2025-109-paper.pdf).

### B. Edge Threshold Violations
This metric counts the number of links whose utilization exceeds a predefined threshold.
*   **Definition:** The number of edges `e` where `utilization(e) > ρ`.
*   **Common Thresholds:** Typical values for `ρ` are 0.7, 0.8, or 0.9, representing 70%, 80%, or 90% of link capacity.

### C. Average Hop Count
This metric measures the average path length in terms of the number of hops.
*   **Definition:** The mean number of hops across all source-destination paths.
*   **Implementation:** Can be calculated from the paths generated by routing algorithms. Papers like RFC 6687 [^24](https://datatracker.ietf.org/doc/html/rfc6687) and LEOCraft [^9](https://www.usenix.org/system/files/atc25-basak.pdf) provide detailed Cumulative Distribution Function (CDF) analyses of hop counts in different network scenarios.

### D. Path Stretch
Path stretch (or routing stretch) quantifies how much longer a path taken by a routing algorithm is compared to the optimal shortest path.
*   **Definition:** The ratio of the actual path cost to the optimal (shortest) path cost [^23](https://www.cs.utexas.edu/~lili/papers/pub/NSDI07.pdf).
*   **Formulas from RFC 6687 [^24](https://datatracker.ietf.org/doc/html/rfc6687):**
    *   `Hop Distance Stretch = Hops_actual - Hops_shortest`
    *   `ETX Path Cost Stretch = ETX_actual - ETX_shortest`
    *   `ETX Fractional Stretch = ETX_stretch / ETX_shortest`
    *   `Fractional Hop Distance Stretch = Hop_stretch / Hop_shortest`
*   **S4 Protocol:** The S4 routing protocol is designed to achieve a low worst-case stretch of 3, with an average stretch close to 1. It provides a theoretical worst-case stretch bound of `1 + 2/k` [^23](https://www.cs.utexas.edu/~lili/papers/pub/NSDI07.pdf).
*   **Benchmark Values:** In a performance evaluation of the RPL protocol, the 90th percentile CDF showed a hop count of 4.89 for RPL paths compared to 4.0 for optimal shortest paths [^24](https://datatracker.ietf.org/doc/html/rfc6687).

### E. Latency Proxy Models
Modeling the precise latency of every packet is computationally intensive. Proxy models provide a good estimate.
*   **Distance-Based Propagation Delay:** The fundamental component of latency is the time it takes for a signal to travel across a link.
    *   `delay = distance / speed_of_light` (where speed_of_light is ~299,792 km/s in a vacuum).
*   **M/M/1 Queueing Models:** Multi-hop satellite networks can be modeled as a series of M/M/1 queues to analyze queueing delays. The total network delay increases with system utilization (`ρ`) and the number of hops (`K`) [^22](https://arxiv.org/pdf/1910.12767).
*   **Latency Components:** Total end-to-end latency is the sum of:
    *   Propagation delay on all Ground-to-Satellite Links (GSLs) and Inter-Satellite Links (ISLs).
    *   Processing and queueing delay at each satellite.
*   **Age of Information (AoI):** A metric that measures the "freshness" of data, defined as the time elapsed since the generation of the last received update. Research shows that the optimal system utilization (`ρ`) that minimizes AoI decreases as the number of satellite hops increases [^22](https://arxiv.org/pdf/1910.12767).

---

## 6. Fast Approximation Methods for Dataset Generation

Full-fidelity, flow-level simulation is too slow for generating the large datasets needed for modern network analysis and machine learning. The following methods provide significant speedups.

### A. SaTE (SIGCOMM 2026)
SaTE is a low-latency Traffic Engineering (TE) framework designed specifically for dynamic satellite networks.
*   **Approach:** Uses Graph Neural Networks (GNNs) on a heterogeneous graph model to perform TE inference [^14](https://fardatalab.org/sigcomm26-wu.pdf).
*   **Performance:**
    *   **2738x speedup** compared to commercial solvers (e.g., Gurobi).
    *   **17ms** average computation latency.
    *   **23.5% improvement** in satisfied network demand.
*   **Features:** Employs dataset pruning (topology, traffic, path) to train efficiently on a small, representative subset of data. It was evaluated on a simulated 4,236-node Starlink constellation [^14](https://fardatalab.org/sigcomm26-wu.pdf).

### B. Machine Learning Approximation (HotNets 2018)
This approach replaces computationally intensive regions of a network simulation with trained Machine Learning models.
*   **Approach:** Uses Long Short-Term Memory (LSTM) models to predict packet drops and latency, avoiding the need for a full flow solver [^11](https://www.cs.swarthmore.edu/~ckazer/hotnets18-kazer.pdf).
*   **Performance:** Achieves **orders of magnitude speedup** over traditional simulators like OMNET++.
*   **Features:** Offers an adjustable tradeoff between simulation speed and accuracy. The framework uses **OMNET++** for the full-fidelity components and **PyTorch** for the ML models [^11](https://www.cs.swarthmore.edu/~ckazer/hotnets18-kazer.pdf).

### C. DONS (Data-Oriented Network Simulator)
DONS is a high-speed Discrete Event Simulator that rethinks the fundamental design of network simulators.
*   **Approach:** Employs a Data-Oriented Design (DOD) instead of the traditional Object-Oriented Design (OOD). This improves cache and memory efficiency dramatically [^12](https://vincen.tl/files/gao23dons.pdf).
*   **Performance:** Achieves up to a **65x speedup** compared to OMNET++.
*   **Features:** Supports automatic parallelization across multiple cores and multiple servers. Its memory efficiency allows it to simulate much larger networks on a single machine.
*   **GitHub:** [github.com/dons2023/Data-Oriented-Network-Simulator](https://github.com/dons2023/Data-Oriented-Network-Simulator) [^12](https://vincen.tl/files/gao23dons.pdf)

### D. OpenSN (APNET 2024)
OpenSN is an open-source library for emulating large-scale LEO satellite networks with high efficiency.
*   **Approach:** Uses container-based virtualization (**Docker**) to emulate satellite nodes, allowing real applications and routing software to run on the emulated network [^26](https://conferences.sigcomm.org/events/apnet2024/papers/OpenSNAnOpenSourceLibraryforEmulatingLEOSatelliteNetworks.pdf).
*   **Performance:**
    *   **5-10x faster** constellation construction than StarryNet.
    *   **2-4x faster** link state updates than Mininet.
*   **Features:** Supports multi-machine scalability for very large constellations using **etcd** for control plane coordination and **VXLAN** for the data plane [^26](https://conferences.sigcomm.org/events/apnet2024/papers/OpenSNAnOpenSourceLibraryforEmulatingLEOSatelliteNetworks.pdf).
*   **GitHub:** [github.com/OpenSN-Library](https://github.com/OpenSN-Library) [^26](https://conferences.sigcomm.org/events/apnet2024/papers/OpenSNAnOpenSourceLibraryforEmulatingLEOSatelliteNetworks.pdf)

---

## 7. Practical Implementations & Open-Source Libraries

| Name | Description | Key Features | GitHub / URL | Language / Framework |
| :--- | :--- | :--- | :--- | :--- |
| **LEOCraft** | A flow-level LEO network simulator and design framework. | Yen's k-shortest path, performance metrics (throughput, stretch, hop count), visualization. | [github.com/suvambasak/LEOCraft.git](https://github.com/suvambasak/LEOCraft.git) [^9](https://www.usenix.org/system/files/atc25-basak.pdf) | Python |
| **SKYFALL** | A framework for analyzing Link-Flooding Attack (LFA) risks in LEO networks. | Time-varying bottleneck detection, attack simulation, ECMP as a countermeasure. | [github.com/SpaceNetLab/SKYFALL](https://github.com/SpaceNetLab/SKYFALL) [^8](https://www.ndss-symposium.org/wp-content/uploads/2025-109-paper.pdf) | Python |
| **Hypatia** | A popular open-source simulator for LEO satellite networks. | Supports various constellations, routing algorithms, and performance analysis. | [github.com/snkas/hypatia](https://github.com/snkas/hypatia) | Python |
| **OpenSN** | A library for emulating large-scale LEO satellite networks with high efficiency. | Container-based (Docker), multi-machine scalability, faster than StarryNet and Mininet. | [github.com/OpenSN-Library](https://github.com/OpenSN-Library) [^26](https://conferences.sigcomm.org/events/apnet2024/papers/OpenSNAnOpenSourceLibraryforEmulatingLEOSatelliteNetworks.pdf) | Python, Docker |
| **LeoEM** | A real-time emulator for LEO satellite networks. | Emulates dynamic topology changes, supports real-time application testing. | [github.com/XuyangCaoUCSD/LeoEM](https://github.com/XuyangCaoUCSD/LeoEM) | Python, Mininet |
| **DONS** | A fast Discrete Event Network Simulator with automatic parallelization. | Data-Oriented Design, 65x faster than OMNET++, low memory usage. | [github.com/dons2023/Data-Oriented-Network-Simulator](https://github.com/dons2023/Data-Oriented-Network-Simulator) [^12](https://vincen.tl/files/gao23dons.pdf) | C++, Unity |
| **starsim** | A Python-based simulation of LEO internet satellite constellations. | Simulates satellite motion, connectivity, and basic routing. | [github.com/sidharthrajaram/starsim](https://github.com/sidharthrajaram/starsim) | Python |
| **leosatellites** | An OMNET++ based simulator for LEO satellite networks. | Detailed physical and MAC layer modeling. | [github.com/Avian688/leosatellites](https://github.com/Avian688/leosatellites) | OMNET++, C++ |
| **MA-DRL Router** | A simulator for multi-agent deep reinforcement learning routing in LEO networks. | Focuses on intelligent, adaptive routing strategies. | [github.com/SatCom-TELMA/MA-DRL_Routing_Simulator](https://github.com/SatCom-TELMA/MA-DRL_Routing_Simulator) | Python, TensorFlow |
| **Motif Topology** | Code and data for designing LEO topologies using repetitive patterns ("motifs"). | Improves network capacity by up to 54% over +Grid for Starlink. | [satnetwork.github.io](https://satnetwork.github.io/) [^10](https://bdebopam.github.io/papers/conext19_LEO_topology.pdf) | Python |
| **NetworkX** | A comprehensive Python library for network analysis. | Implements a wide range of graph and shortest path algorithms. | [networkx.org](https://networkx.org/documentation/stable/index.html) [^20](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html) | Python |
| **igraph** | A high-performance graph analysis library with bindings for multiple languages. | Faster than NetworkX for many operations due to its C core. | [python.igraph.org](https://python.igraph.org/en/main/index.html) [^21](https://python.igraph.org/en/main/tutorials/shortest_path_visualisation.html) | Python, C++ |

---

## 8. Research Papers on LEO/Satellite Routing

| Paper Title | Conference/Journal | Year | Key Contributions | PDF URL |
| :--- | :--- | :--- | :--- | :--- |
| SaTE: Low-Latency Traffic Engineering for Satellite Networks | SIGCOMM | 2026 (projected) | GNN-based TE, 17ms latency, 2738x speedup. | [fardatalab.org](https://fardatalab.org/sigcomm26-wu.pdf) [^14](https://fardatalab.org/sigcomm26-wu.pdf) |
| Time-varying Bottleneck Links in LEO Satellite Networks | NDSS | 2025 | Identifies time-varying bottlenecks; proposes SKYFALL for risk analysis. | [ndss-symposium.org](https://www.ndss-symposium.org/wp-content/uploads/2025-109-paper.pdf) [^8](https://www.ndss-symposium.org/wp-content/uploads/2025-109-paper.pdf) |
| LEOCraft: Towards Designing Performant LEO Networks | USENIX ATC | 2025 | Open-source LEO network simulator; uses Yen's algorithm for k-shortest paths. | [usenix.org](https://www.usenix.org/system/files/atc25-basak.pdf) [^9](https://www.usenix.org/system/files/atc25-basak.pdf) |
| OpenSN: An Open Source Library for Emulating LEO Satellite Networks | APNet | 2024 | Efficient container-based LEO network emulation, 5-10x faster than StarryNet. | [sigcomm.org](https://conferences.sigcomm.org/events/apnet2024/papers/OpenSNAnOpenSourceLibraryforEmulatingLEOSatelliteNetworks.pdf) [^26](https://conferences.sigcomm.org/events/apnet2024/papers/OpenSNAnOpenSourceLibraryforEmulatingLEOSatelliteNetworks.pdf) |
| DONS: Fast and Affordable Discrete Event Network Simulation | USENIX ATC | 2023 | Data-Oriented Design for simulators, 65x speedup over OMNET++. | [vincen.tl](https://vincen.tl/files/gao23dons.pdf) [^12](https://vincen.tl/files/gao23dons.pdf) |
| Network topology design at 27,000 km/hour | CoNEXT | 2019 | Proposes "motifs" for LEO topology design, improving capacity up to 54%. | [bdebopam.github.io](https://bdebopam.github.io/papers/conext19_LEO_topology.pdf) [^10](https://bdebopam.github.io/papers/conext19_LEO_topology.pdf) |
| Fast Network Simulation Through Approximation | HotNets | 2018 | Uses ML (LSTMs) to approximate network regions for orders-of-magnitude speedup. | [cs.swarthmore.edu](https://www.cs.swarthmore.edu/~ckazer/hotnets18-kazer.pdf) [^11](https://www.cs.swarthmore.edu/~ckazer/hotnets18-kazer.pdf) |
| Hedera: Dynamic Flow Scheduling for Data Center Networks | NSDI | 2010 | Dynamic scheduling for large flows; highlights ECMP collision issues (50% bandwidth loss). | [usc.edu](https://raghavan.usc.edu/papers/hedera-nsdi10.pdf) [^19](https://raghavan.usc.edu/papers/hedera-nsdi10.pdf) |
| S4: Small State and Small Stretch Routing Protocol | NSDI | 2007 | A routing protocol achieving worst-case stretch of 3 with O(sqrt(N)) state. | [cs.utexas.edu](https://www.cs.utexas.edu/~lili/papers/pub/NSDI07.pdf) [^23](https://www.cs.utexas.edu/~lili/papers/pub/NSDI07.pdf) |
| Finding the k Shortest Paths | J. ACM | 1998 | Details Eppstein's algorithm for k-shortest paths (O(m + n log n + k)). | [ics.uci.edu](https://ics.uci.edu/~eppstein/pubs/Epp-SJC-98.pdf) [^15](https://ics.uci.edu/~eppstein/pubs/Epp-SJC-98.pdf) |
| Latency and timeliness in multi-hop satellite networks | arXiv | 2019 | Models satellite networks as M/M/1 queues, analyzes Age of Information (AoI). | [arxiv.org](https://arxiv.org/pdf/1910.12767) [^22](https://arxiv.org/pdf/1910.12767) |
| Performance Evaluation of RPL | RFC 6687 | 2012 | Defines and provides benchmark values for path stretch metrics (hop, ETX). | [datatracker.ietf.org](https://datatracker.ietf.org/doc/html/rfc6687) [^24](https://datatracker.ietf.org/doc/html/rfc6687) |
| Equal-Cost Multi-Path (ECMP) Routing in SDN | Brown Univ. Capstone | 2015 | Details the implementation of ECMP in an SDN environment. | [cs.brown.edu](https://cs.brown.edu/research/pubs/theses/capstones/2015/pak.pdf) [^16](https://cs.brown.edu/research/pubs/theses/capstones/2015/pak.pdf) |

---

## 9. Recommended Implementation Approach

This section provides a practical, step-by-step recommendation for building a satellite mesh routing model that balances accuracy and speed, avoiding slow full-flow solvers for dataset generation.

**Step 1: Graph and Topology Representation**
*   **Tool:** Use **NetworkX** or **igraph** in Python.
*   **Action:** Create a graph object where nodes are satellites and ground stations, and edges are Inter-Satellite Links (ISLs) and Ground-to-Satellite Links (GSLs). Assign capacity to edges and propagation delay (calculated from distance) as the primary weight.

**Step 2: Demand Matrix Generation**
*   **Action:** Implement functions to generate N×N demand matrices based on various traffic patterns:
    *   `generate_uniform_demand(nodes, total_demand)`
    *   `generate_hotspot_demand(nodes, hotspots, demand_per_flow)`
    *   `generate_gs_to_sat_demand(ground_stations, satellites, demand)`
*   This allows for testing the routing model under diverse and realistic load conditions.

**Step 3: Routing Logic Implementation**
*   **Shortest Path:** Use `networkx.single_source_dijkstra` with edge weight set to latency. This will be the baseline.
*   **K-Shortest Paths / ECMP:**
    *   To get multiple paths, use an implementation of Yen's or Eppstein's algorithm. Some libraries have this built-in.
    *   For ECMP, find all paths with a cost equal to the shortest path cost.
    *   Implement a simple hash-based selection function to distribute flows across the K equal-cost paths.

**Step 4: Edge Utilization Calculation (Fast Approximation)**
*   **Action:** This is the core of the fast simulation. Instead of simulating individual packets, calculate utilization directly.
    1.  For each demand `(source, dest, demand_value)` in the demand matrix:
    2.  Use the routing logic from Step 3 to get the path(s) for this source-destination pair.
    3.  If using ECMP, distribute `demand_value` across the `k` paths (e.g., `demand_value / k` per path).
    4.  For each edge in the chosen path(s), add the allocated demand to that edge's total load.
*   This approach avoids a flow solver and directly computes the steady-state link loads.

**Step 5: Metric Computation**
*   **Action:** After running the utilization calculation (Step 4), compute all required performance metrics:
    *   **Max Utilization:** Find `max(edge_load / edge_capacity)` across all edges.
    *   **Edge Threshold Violations:** Count edges where `(edge_load / edge_capacity) > threshold`.
    *   **Average Hop Count:** Calculate the mean hop count of all paths used.
    *   **Path Stretch:** For each path, compare its latency (sum of edge weights) to the baseline Dijkstra shortest path latency. Compute the stretch metrics.
    *   **Latency:** The path cost from the weighted graph serves as the latency proxy.

**Step 6: Dataset Generation**
*   **Action:** Wrap the entire process (Steps 1-5) in a script. Iterate over:
    *   Different topology snapshots (as satellites move).
    *   Different demand matrices.
    *   Different routing algorithms (Shortest Path vs. ECMP).
*   Save the results (topology, demand matrix, routing choice, and all computed metrics) for each run. This creates a comprehensive dataset for analysis or training ML models.

---

## 10. Code Examples & Pseudocode

### Pseudocode: K-Shortest Paths / ECMP Route Selection
```python
def get_ecmp_paths(graph, source, dest, k):
    # Find all simple paths up to a certain depth limit to avoid infinite loops
    all_paths = list(nx.all_simple_paths(graph, source, dest, cutoff=len(graph)))

    # Calculate cost (e.g., latency) for each path
    path_costs = []
    for path in all_paths:
        cost = sum(graph[u][v]['weight'] for u, v in zip(path, path[1:]))
        path_costs.append((path, cost))

    # Sort paths by cost
    path_costs.sort(key=lambda x: x[1])

    # Find the cost of the shortest path
    if not path_costs:
        return []
    shortest_cost = path_costs[0][1]

    # Filter for all paths that have the same cost as the shortest one
    equal_cost_paths = [p for p, c in path_costs if c == shortest_cost]

    # Return up to k of these paths
    return equal_cost_paths[:k]
```

### Pseudocode: Edge Utilization Calculation
```python
def calculate_edge_utilization(graph, demand_matrix, routing_function):
    edge_loads = {edge: 0.0 for edge in graph.edges()}

    for source in demand_matrix:
        for dest in demand_matrix[source]:
            demand = demand_matrix[source][dest]
            if demand == 0:
                continue

            # Get paths based on the chosen routing logic (e.g., ECMP)
            paths = routing_function(graph, source, dest)
            if not paths:
                continue

            # Distribute demand across the paths
            demand_per_path = demand / len(paths)

            for path in paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    edge = (u, v)
                    if edge in edge_loads:
                        edge_loads[edge] += demand_per_path
                    else: # Handle directed vs undirected graphs
                        edge_loads[(v, u)] += demand_per_path

    # Convert loads to utilization percentage
    utilization = {edge: load / graph.edges[edge]['capacity'] for edge, load in edge_loads.items()}
    return utilization
```

### Pseudocode: Path Stretch Computation
```python
def compute_path_stretch(graph, actual_path):
    source, dest = actual_path[0], actual_path[-1]

    # Calculate cost of the actual path taken
    actual_cost = sum(graph[u][v]['weight'] for u, v in zip(actual_path, actual_path[1:]))

    # Calculate cost of the optimal shortest path
    optimal_cost = nx.dijkstra_path_length(graph, source, dest, weight='weight')

    # Calculate stretch
    if optimal_cost == 0:
        return 1.0 # Or handle as a special case
    stretch = actual_cost / optimal_cost

    return stretch
```
