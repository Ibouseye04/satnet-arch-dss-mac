"""HTTP request models for the SATNET DSS API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt

from satnet.dss.schemas import DSSArchitectureRequest


Number = StrictInt | StrictFloat


class DSSArchitectureRequestDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_planes: StrictInt
    sats_per_plane: StrictInt
    altitude_km: Number
    inclination_deg: Number
    satellite_node_failure_probability: Number
    satellite_edge_failure_probability: Number
    civilian_count: StrictInt
    government_count: StrictInt
    military_count: StrictInt
    ground_station_failure_probability: Number
    required_minimum_connectivity: Number = 0.80

    def to_domain(self) -> DSSArchitectureRequest:
        dump = getattr(self, "model_dump", self.dict)
        payload: dict[str, Any] = dump()
        return DSSArchitectureRequest.from_mapping(payload)


class ErrorBody(BaseModel):
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
