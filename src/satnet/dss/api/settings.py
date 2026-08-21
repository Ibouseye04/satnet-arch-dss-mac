"""Environment-backed settings for the optional DSS HTTP service."""

from __future__ import annotations

from dataclasses import dataclass
import os

DEFAULT_CORS_ORIGIN = "http://localhost:5173"


@dataclass(frozen=True)
class DSSApiSettings:
    checkpoint_path: str | None
    ground_catalog_path: str | None
    cors_origins: tuple[str, ...]

    @classmethod
    def from_environment(cls) -> "DSSApiSettings":
        configured_origins = os.environ.get("SATNET_DSS_CORS_ORIGINS", DEFAULT_CORS_ORIGIN)
        origins = tuple(
            origin.strip()
            for origin in configured_origins.split(",")
            if origin.strip() and origin.strip() != "*"
        )
        return cls(
            checkpoint_path=os.environ.get("SATNET_DSS_TGNN_CHECKPOINT"),
            ground_catalog_path=os.environ.get("SATNET_DSS_GROUND_CATALOG"),
            cors_origins=origins or (DEFAULT_CORS_ORIGIN,),
        )
