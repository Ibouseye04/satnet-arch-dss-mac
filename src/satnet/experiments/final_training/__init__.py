"""Strict, no-training compatibility infrastructure for the frozen SATNET 10k exports."""

from .contracts import AUTHORIZED_TASKS, DATASET_BUNDLE_HASH, TRAINING_PLAN_BUNDLE_HASH

__all__ = ["AUTHORIZED_TASKS", "DATASET_BUNDLE_HASH", "TRAINING_PLAN_BUNDLE_HASH"]
