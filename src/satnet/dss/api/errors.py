"""Structured HTTP error helpers for the DSS API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DSSApiError(Exception):
    status_code: int
    code: str
    message: str
    field: str | None = None

    def __str__(self) -> str:
        return self.message
