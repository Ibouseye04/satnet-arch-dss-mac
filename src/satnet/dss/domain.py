"""Result contracts for the SATNET Phase 1 DSS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DSSModelMetadata:
    family: str
    task: str
    target: str
    checkpoint_sha256: str
    realization_count: int


@dataclass(frozen=True)
class DSSSpaceResilience:
    expected_minimum_gcc: float
    lowest_modeled_gcc: float
    highest_modeled_gcc: float
    required_minimum_connectivity: float
    expected_margin: float
    lowest_margin: float
    expected_assessment: str
    realizations_meeting_requirement: int
    realization_count: int
    realization_risk_flag: bool


@dataclass(frozen=True)
class DSSSystemContext:
    mean_ground_service_fraction: float | None
    minimum_ground_service_fraction: float | None
    mean_overall_service_fraction: float | None
    minimum_overall_service_fraction: float | None
    limiting_segment: str | None
    ground_provenance: str
    overall_provenance: str
    status: str
    blocked_reason: str | None = None


@dataclass(frozen=True)
class DSSAnalysisResult:
    architecture: dict[str, object]
    model: DSSModelMetadata
    space_resilience: DSSSpaceResilience
    system_context: DSSSystemContext
    provenance: dict[str, object]
    analysis_details: dict[str, object]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
