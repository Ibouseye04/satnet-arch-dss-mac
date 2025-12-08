from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import List

import networkx as nx

from satnet.network.topology import TopologyConfig, generate_topology
from satnet.simulation.failures import (
    FailureConfig,
    sample_failures,
    apply_failures,
    compute_impact,
)

# ---------------------------------------------------------------------------
# Monte Carlo config + outcome-aware samples (Model A / sanity check)
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloConfig:
    num_runs: int = 1000
    node_failure_prob: float = 0.02
    edge_failure_prob: float = 0.05
    seed: int | None = 123


@dataclass
class MonteCarloSample:
    """
    One row for ML / analysis: includes post-failure outcomes.

    This is the "oracle" / outcome-aware dataset used for sanity checks
    (Model A) – not the design-time model for the thesis.
    """

    run_id: int

    # topology-level parameters
    num_nodes_0: int
    num_edges_0: int
    avg_degree_0: float
    num_satellites: int
    num_ground_stations: int

    # failure config
    node_failure_prob: float
    edge_failure_prob: float

    # realized failures
    failed_nodes: int
    failed_edges: int

    # structural impact
    nodes_after: int
    edges_after: int
    num_components_after: int
    largest_component_after: int
    largest_component_ratio: float  # largest_after / num_nodes_0

    # label
    partitioned: int  # 1 if partition / significant shrink, else 0


@dataclass
class MonteCarloStats:
    num_runs: int
    mean_failed_nodes: float
    mean_failed_edges: float
    prob_partition: float
    mean_largest_component_ratio: float


def generate_failure_dataset(cfg: MonteCarloConfig) -> List[MonteCarloSample]:
    """
    Outcome-aware dataset:
    - Build one base topology
    - Run num_runs failure scenarios
    - Record BOTH pre- and post-failure metrics
    """
    topo_cfg = TopologyConfig()
    G_base: nx.Graph = generate_topology(topo_cfg)

    n0 = G_base.number_of_nodes()
    e0 = G_base.number_of_edges()
    avg_deg0 = (2 * e0 / n0) if n0 > 0 else 0.0

    samples: List[MonteCarloSample] = []

    for i in range(cfg.num_runs):
        seed_i = None if cfg.seed is None else cfg.seed + i

        fail_cfg = FailureConfig(
            node_failure_prob=cfg.node_failure_prob,
            edge_failure_prob=cfg.edge_failure_prob,
            seed=seed_i,
        )

        failures = sample_failures(G_base, fail_cfg)
        G_failed = apply_failures(G_base, failures)
        impact = compute_impact(G_base, G_failed)

        partitioned = int(
            (impact.num_components_after > impact.num_components_before)
            or (
                impact.largest_component_after
                < impact.largest_component_before
            )
        )

        largest_ratio = (
            impact.largest_component_after / float(n0) if n0 > 0 else 0.0
        )

        sample = MonteCarloSample(
            run_id=i,
            num_nodes_0=n0,
            num_edges_0=e0,
            avg_degree_0=avg_deg0,
            num_satellites=topo_cfg.num_satellites,
            num_ground_stations=topo_cfg.num_ground_stations,
            node_failure_prob=cfg.node_failure_prob,
            edge_failure_prob=cfg.edge_failure_prob,
            failed_nodes=len(failures.failed_nodes),
            failed_edges=len(failures.failed_edges),
            nodes_after=impact.nodes_after,
            edges_after=impact.edges_after,
            num_components_after=impact.num_components_after,
            largest_component_after=impact.largest_component_after,
            largest_component_ratio=largest_ratio,
            partitioned=partitioned,
        )
        samples.append(sample)

    return samples


def summarize_samples(samples: List[MonteCarloSample]) -> MonteCarloStats:
    n = len(samples)
    if n == 0:
        return MonteCarloStats(0, 0.0, 0.0, 0.0)

    mean_failed_nodes = fsum(s.failed_nodes for s in samples) / n
    mean_failed_edges = fsum(s.failed_edges for s in samples) / n
    prob_partition = fsum(s.partitioned for s in samples) / n
    mean_largest_ratio = (
        fsum(s.largest_component_ratio for s in samples) / n
    )

    return MonteCarloStats(
        num_runs=n,
        mean_failed_nodes=mean_failed_nodes,
        mean_failed_edges=mean_failed_edges,
        prob_partition=prob_partition,
        mean_largest_component_ratio=mean_largest_ratio,
    )


def run_failure_sweep(cfg: MonteCarloConfig) -> MonteCarloStats:
    """
    Backward-compatible helper used by scripts/failure_sweep.py.
    """
    samples = generate_failure_dataset(cfg)
    return summarize_samples(samples)


# ---------------------------------------------------------------------------
# Design-time dataset (Model B / thesis model)
# ---------------------------------------------------------------------------


@dataclass
class DesignScenarioConfig:
    """
    One design *configuration* to sample around.
    We will simulate many runs per scenario with the same design parameters.
    """

    scenario_id: int
    num_satellites: int
    num_ground_stations: int
    isl_degree: int
    node_failure_prob: float
    edge_failure_prob: float
    runs_per_scenario: int = 100


@dataclass
class DesignSample:
    """
    Single Monte Carlo run for a given design scenario.

    Features: only design-time quantities (architecture + failure probs).
    Label: whether that run produced a partition.

    This is the dataset we train the thesis model on.
    """

    scenario_id: int
    run_id: int

    # architecture features
    num_nodes_0: int
    num_edges_0: int
    avg_degree_0: float
    num_satellites: int
    num_ground_stations: int
    isl_degree: int

    # design-time failure assumptions
    node_failure_prob: float
    edge_failure_prob: float

    # label
    partitioned: int


def generate_designtime_dataset(
    scenarios: List[DesignScenarioConfig],
    base_seed: int = 1000,
) -> List[DesignSample]:
    """
    Generate a dataset for *design-time* prediction.

    For each DesignScenarioConfig:
      - Build a topology for that design
      - Run runs_per_scenario random failure realizations
      - Label each run as partitioned / not

    NOTE: Features here are *only* pre-failure knowledge.
    No post-failure leakage (no failed_nodes, largest_component_ratio, etc.).
    """
    samples: List[DesignSample] = []
    global_run_id = 0

    for sc in scenarios:
        topo_cfg = TopologyConfig(
            num_satellites=sc.num_satellites,
            num_ground_stations=sc.num_ground_stations,
            isl_degree=sc.isl_degree,
        )

        G_base = generate_topology(topo_cfg)

        n0 = G_base.number_of_nodes()
        e0 = G_base.number_of_edges()
        avg_deg0 = (2 * e0 / n0) if n0 > 0 else 0.0

        for _ in range(sc.runs_per_scenario):
            seed_i = base_seed + global_run_id

            fail_cfg = FailureConfig(
                node_failure_prob=sc.node_failure_prob,
                edge_failure_prob=sc.edge_failure_prob,
                seed=seed_i,
            )

            failures = sample_failures(G_base, fail_cfg)
            G_failed = apply_failures(G_base, failures)
            impact = compute_impact(G_base, G_failed)

            partitioned = int(
                (impact.num_components_after > impact.num_components_before)
                or (
                    impact.largest_component_after
                    < impact.largest_component_before
                )
            )

            sample = DesignSample(
                scenario_id=sc.scenario_id,
                run_id=global_run_id,
                num_nodes_0=n0,
                num_edges_0=e0,
                avg_degree_0=avg_deg0,
                num_satellites=sc.num_satellites,
                num_ground_stations=sc.num_ground_stations,
                isl_degree=sc.isl_degree,
                node_failure_prob=sc.node_failure_prob,
                edge_failure_prob=sc.edge_failure_prob,
                partitioned=partitioned,
            )
            samples.append(sample)
            global_run_id += 1

    return samples