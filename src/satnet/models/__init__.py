"""SatNet Models - Risk scoring and GNN dataset utilities."""

# Lazy imports to avoid requiring ML dependencies for non-ML usage
__all__ = [
    "SatNetTemporalDataset",
    "collate_temporal_sequences",
]


def __getattr__(name: str):
    """Lazy import for ML dependencies."""
    if name in ("SatNetTemporalDataset", "collate_temporal_sequences"):
        from satnet.models.gnn_dataset import (
            SatNetTemporalDataset,
            collate_temporal_sequences,
        )
        if name == "SatNetTemporalDataset":
            return SatNetTemporalDataset
        return collate_temporal_sequences
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
