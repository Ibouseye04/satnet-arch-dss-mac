from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any
import unicodedata


def canonical_float_string(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Canonical floats must be finite")
    normalized = 0.0 if value == 0.0 else value
    return format(normalized, ".17g")


def canonical_utc_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("Timestamp must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("Timestamp must be normalized to UTC")
    normalized = value.astimezone(timezone.utc)
    return normalized.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def canonical_station_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Station name must be a string")
    canonical = unicodedata.normalize("NFC", value.strip())
    if not canonical:
        raise ValueError("Station name must not be empty")
    if len(canonical) > 128:
        raise ValueError("Station name must contain at most 128 Unicode code points")
    return canonical


def _validate_canonical_value(value: Any, path: str = "payload") -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        raise TypeError(
            f"Canonical hash value at {path} is a float; encode it with canonical_float_string"
        )
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_canonical_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Canonical hash key at {path} must be a string")
            _validate_canonical_value(item, f"{path}.{key}")
        return
    raise TypeError(f"Unsupported canonical hash value at {path}: {type(value).__name__}")


def canonical_json(value: dict[str, Any]) -> str:
    _validate_canonical_value(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
