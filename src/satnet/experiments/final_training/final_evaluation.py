from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import DATASET_BUNDLE_HASH, TRAINING_PLAN_BUNDLE_HASH, FINAL_SEEDS
from .test_access import FinalEvaluationAuthorization, require_test_authorization


@dataclass(frozen=True)
class FinalEvaluationEvidence:
    selected_configuration_frozen: bool
    validation_selection_complete: bool
    final_seed_set_fixed: bool
    model_manifest_exists: bool
    dataset_bundle_hash: str = DATASET_BUNDLE_HASH
    training_plan_bundle_hash: str = TRAINING_PLAN_BUNDLE_HASH
    final_seeds: tuple[int, ...] = FINAL_SEEDS

    def authorization(self) -> FinalEvaluationAuthorization:
        authorization = FinalEvaluationAuthorization(
            self.selected_configuration_frozen,
            self.validation_selection_complete,
            self.final_seed_set_fixed,
            self.model_manifest_exists,
            self.dataset_bundle_hash,
            self.training_plan_bundle_hash,
            self.final_seeds,
        )
        authorization.validate()
        return authorization


def require_final_evaluation_evidence(evidence: FinalEvaluationEvidence) -> FinalEvaluationAuthorization:
    """The only control-plane entry point that can create final TEST capability."""
    authorization = evidence.authorization()
    require_test_authorization(authorization)
    return authorization
