from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping

from satnet.ground.canonical import canonical_float_string


class CanonicalAttributeType(str, Enum):
    NONE = "none"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"


@dataclass(frozen=True)
class CanonicalAttribute:
    name: str
    value_type: CanonicalAttributeType
    value: str | int | bool | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Canonical attribute name must be a nonempty string")
        if not isinstance(self.value_type, CanonicalAttributeType):
            raise TypeError("value_type must be a CanonicalAttributeType")
        if self.value_type is CanonicalAttributeType.NONE:
            if self.value is not None:
                raise TypeError("NONE attribute value must be None")
        elif self.value_type is CanonicalAttributeType.BOOLEAN:
            if type(self.value) is not bool:
                raise TypeError("BOOLEAN attribute value must be a Boolean")
        elif self.value_type is CanonicalAttributeType.INTEGER:
            if type(self.value) is not int:
                raise TypeError("INTEGER attribute value must be an integer")
        elif self.value_type is CanonicalAttributeType.FLOAT:
            if not isinstance(self.value, str):
                raise TypeError("FLOAT attribute value must be a canonical float string")
            try:
                parsed = float(self.value)
            except ValueError as exc:
                raise ValueError("FLOAT attribute value is invalid") from exc
            if canonical_float_string(parsed) != self.value:
                raise ValueError("FLOAT attribute value is not canonical")
        elif self.value_type is CanonicalAttributeType.STRING:
            if type(self.value) is not str:
                raise TypeError("STRING attribute value must be a string")

    @classmethod
    def from_value(cls, name: str, value: object) -> "CanonicalAttribute":
        if value is None:
            return cls(name, CanonicalAttributeType.NONE, None)
        if type(value) is bool:
            return cls(name, CanonicalAttributeType.BOOLEAN, value)
        if type(value) is int:
            return cls(name, CanonicalAttributeType.INTEGER, value)
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError(f"Attribute '{name}' float value must be finite")
            return cls(name, CanonicalAttributeType.FLOAT, canonical_float_string(value))
        if type(value) is str:
            return cls(name, CanonicalAttributeType.STRING, value)
        raise TypeError(
            f"Attribute '{name}' has unsupported type {type(value).__name__}; "
            "G3 supports only None, Boolean, integer, finite float, and string"
        )

    def to_value(self) -> object:
        if self.value_type is CanonicalAttributeType.FLOAT:
            return float(self.value)
        return self.value

    def canonical_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "value_type": self.value_type.value,
        }


def canonicalize_attributes(
    attributes: Mapping[str, object],
) -> tuple[CanonicalAttribute, ...]:
    if not isinstance(attributes, Mapping):
        raise TypeError("attributes must be a mapping")
    if any(not isinstance(name, str) or not name for name in attributes):
        raise ValueError("Attribute names must be nonempty strings")
    return tuple(
        CanonicalAttribute.from_value(name, attributes[name])
        for name in sorted(attributes)
    )


def validate_canonical_attributes(
    attributes: tuple[CanonicalAttribute, ...],
) -> None:
    if not isinstance(attributes, tuple):
        raise TypeError("Canonical attributes must be a tuple")
    if any(not isinstance(attribute, CanonicalAttribute) for attribute in attributes):
        raise TypeError("Every canonical attribute must be a CanonicalAttribute")
    names = [attribute.name for attribute in attributes]
    if names != sorted(names):
        raise ValueError("Canonical attributes must be ordered by name")
    if len(names) != len(set(names)):
        raise ValueError("Canonical attributes contain duplicate names")


def attributes_to_dict(
    attributes: tuple[CanonicalAttribute, ...],
) -> dict[str, object]:
    validate_canonical_attributes(attributes)
    return {attribute.name: attribute.to_value() for attribute in attributes}
