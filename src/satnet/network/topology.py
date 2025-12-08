import networkx as nx
from dataclasses import dataclass


@dataclass
class TopologyConfig:
    num_satellites: int = 20
    num_ground_stations: int = 4
    isl_degree: int = 2  # target satellite ISL degree (even number preferred)


def generate_topology(cfg: TopologyConfig) -> nx.Graph:
    """
    Build a simple example topology:

    - N satellites arranged in a ring, plus additional ISLs up to isl_degree.
    - M ground stations, each connected to 2 satellites.
    - Edges have 'capacity' attributes (arbitrary units).
    """
    G = nx.Graph()

    # Satellites
    for i in range(cfg.num_satellites):
        G.add_node(f"SAT-{i}", type="satellite")

    # Ground stations
    for i in range(cfg.num_ground_stations):
        G.add_node(f"GS-{i}", type="ground")

    sats = [n for n, d in G.nodes(data=True) if d["type"] == "satellite"]
    n = len(sats)

    # Basic ring (degree 2)
    for i in range(n):
        u = sats[i]
        v = sats[(i + 1) % n]
        G.add_edge(u, v, capacity=10.0)

    # Extra ISLs to reach approx isl_degree
    # For isl_degree > 2, add "chord" edges with increasing offset.
    target_deg = max(2, cfg.isl_degree)
    extra_per_node = max(0, target_deg // 2 - 1)

    for i in range(n):
        u = sats[i]
        for d in range(2, 2 + extra_per_node):
            v = sats[(i + d) % n]
            if not G.has_edge(u, v):
                G.add_edge(u, v, capacity=10.0)

    # Each GS connects to 2 satellites
    for i in range(cfg.num_ground_stations):
        gs = f"GS-{i}"
        G.add_edge(gs, sats[i % n], capacity=50.0)
        G.add_edge(gs, sats[(i + 1) % n], capacity=50.0)

    return G