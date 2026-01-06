"""Legacy modules quarantined from Tier 1 production paths.

This namespace contains deprecated simulation code that uses static t=0 snapshots
and/or lacks explicit epoch handling. These modules are preserved for reference
but should NOT be imported by Tier 1 scripts or production code.

Quarantined modules:
- engine.py: SimulationEngine with static graph_at_t0 behavior
- failures.py: Generic i.i.d. failure sampling (Tier 1 uses tier1_rollout inline)

For Tier 1 temporal evaluation, use:
- satnet.simulation.tier1_rollout.run_tier1_rollout()
- satnet.simulation.monte_carlo.generate_tier1_temporal_dataset()
"""

__all__ = ["engine", "failures"]
