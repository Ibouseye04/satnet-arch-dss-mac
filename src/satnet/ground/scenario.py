from __future__ import annotations

from dataclasses import dataclass, field

from satnet.ground.canonical import canonical_hash
from satnet.ground.persistence import GroundRunDesignRecord
from satnet.simulation.tier1_rollout import Tier1RolloutConfig

SCENARIO_DESIGN_IDENTITY_DOMAIN = "satnet_scenario_design"
SCENARIO_DESIGN_IDENTITY_VERSION = "1"


@dataclass(frozen=True)
class ScenarioDesign:
    satellite_config: Tier1RolloutConfig
    ground_design: GroundRunDesignRecord
    scenario_design_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.satellite_config, Tier1RolloutConfig):
            raise TypeError("satellite_config must be a Tier1RolloutConfig")
        if not isinstance(self.ground_design, GroundRunDesignRecord):
            raise TypeError("ground_design must be a GroundRunDesignRecord")
        satellite_hash = self.satellite_config.config_hash()
        if satellite_hash != self.ground_design.satellite_config_hash:
            raise ValueError(
                "Ground design satellite_config_hash does not match satellite configuration"
            )
        payload = {
            "ground_design_hash": self.ground_design.ground_design_hash,
            "identity_domain": SCENARIO_DESIGN_IDENTITY_DOMAIN,
            "identity_version": SCENARIO_DESIGN_IDENTITY_VERSION,
            "satellite_config_hash": satellite_hash,
        }
        object.__setattr__(self, "scenario_design_hash", canonical_hash(payload))
