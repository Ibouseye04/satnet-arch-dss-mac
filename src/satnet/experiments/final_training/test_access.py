from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import DATASET_BUNDLE_HASH, FINAL_SEEDS, TRAINING_PLAN_BUNDLE_HASH


class TestTargetAccessError(PermissionError):
    __test__ = False
    """Raised whenever TEST targets are requested before final authorization."""


@dataclass(frozen=True)
class FinalEvaluationAuthorization:
    selected_configuration_frozen: bool
    validation_selection_complete: bool
    final_seed_set_fixed: bool
    model_manifest_exists: bool
    dataset_bundle_hash: str
    training_plan_bundle_hash: str
    final_seeds: tuple[int, ...] = FINAL_SEEDS

    def validate(self) -> None:
        if not all((self.selected_configuration_frozen, self.validation_selection_complete, self.final_seed_set_fixed, self.model_manifest_exists)):
            raise TestTargetAccessError("Final evaluation authorization evidence is incomplete")
        if self.dataset_bundle_hash != DATASET_BUNDLE_HASH or self.training_plan_bundle_hash != TRAINING_PLAN_BUNDLE_HASH:
            raise TestTargetAccessError("Final evaluation authorization is bound to the wrong immutable bundle")
        if tuple(self.final_seeds) != FINAL_SEEDS:
            raise TestTargetAccessError("Final evaluation seed set is not frozen")


def require_test_authorization(authorization: FinalEvaluationAuthorization | None) -> None:
    if authorization is None:
        raise TestTargetAccessError("TEST targets are blocked until explicit final-evaluation authorization")
    authorization.validate()


@dataclass(frozen=True)
class MetadataView:
    rows: tuple[Mapping[str, Any], ...]


class TargetGate:
    """Small capability gate shared by RF and TGNN loaders."""

    def __init__(self, *, split: str, metadata: MetadataView, target_values: tuple[Any, ...] | None) -> None:
        self.split = split
        self.metadata = metadata
        self._target_values = target_values

    @property
    def targets(self) -> tuple[Any, ...]:
        if self.split == "test":
            raise TestTargetAccessError("TEST targets are metadata-only before final evaluation")
        if self._target_values is None:
            raise TestTargetAccessError("Targets are unavailable for this view")
        return self._target_values

    def authorized_targets(self, authorization: FinalEvaluationAuthorization) -> tuple[Any, ...]:
        if self.split != "test":
            return self.targets
        require_test_authorization(authorization)
        if self._target_values is None:
            raise TestTargetAccessError("TEST target payload was not retained")
        return self._target_values
