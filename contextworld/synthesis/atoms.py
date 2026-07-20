from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


PIXEL_EFFECT_CONTRACTS: dict[str, tuple[str, ...]] = {
    "single_frame_geometry": (
        "factor_readback",
        "state_invariance",
        "single_frame_pixel_semantics",
        "unchanged_pixels",
    ),
    "single_frame_appearance": (
        "factor_readback",
        "state_invariance",
        "single_frame_pixel_semantics",
        "unchanged_pixels",
    ),
    "temporal_dynamics": (
        "factor_readback",
        "state_transition",
        "pixel_transition",
        "temporal_alignment",
    ),
    "contact_dynamics": (
        "factor_readback",
        "state_transition",
        "pixel_transition",
        "contact_semantics",
    ),
    "camera_projection": (
        "factor_readback",
        "state_invariance",
        "single_frame_pixel_semantics",
        "projection_semantics",
    ),
}


class AtomValidationError(ValueError):
    """Raised when a requested atom cannot be safely compiled."""


@dataclass(frozen=True)
class CompiledAtom:
    kind: str
    factor_key: str
    factor_value: Any
    variation_value: Any


class VariationAtom(ABC):
    """Adapter from a portable atom name to one environment parameter."""

    kind: str
    factor_key: str
    pixel_effect: str
    required_oracles: tuple[str, ...]

    @abstractmethod
    def compile(self, value: Any) -> CompiledAtom:
        raise NotImplementedError


class AgentSpeedAtom(VariationAtom):
    kind = "agent_speed"
    factor_key = "agent.speed"
    pixel_effect = "temporal_dynamics"
    required_oracles = ("speed_frame_skip_oracle",)
    minimum = 1.75
    maximum = 10.5

    def compile(self, value: Any) -> CompiledAtom:
        try:
            speed = float(value)
        except (TypeError, ValueError) as exc:
            raise AtomValidationError("agent_speed must be numeric") from exc
        if not np.isfinite(speed) or not self.minimum <= speed <= self.maximum:
            raise AtomValidationError(
                f"agent_speed={speed!r} is outside "
                f"[{self.minimum}, {self.maximum}]"
            )
        return CompiledAtom(
            kind=self.kind,
            factor_key=self.factor_key,
            factor_value=speed,
            variation_value=np.asarray([speed], dtype=np.float32),
        )


class DoorPositionAtom(VariationAtom):
    kind = "door_position"
    factor_key = "door.position"
    pixel_effect = "single_frame_geometry"
    required_oracles = (
        "door_position_pixel_oracle",
        "door_position_passage_oracle",
    )
    minimum = 24
    maximum = 199

    def compile(self, value: Any) -> CompiledAtom:
        if isinstance(value, bool):
            raise AtomValidationError("door_position must be an integer")
        try:
            position = int(value)
        except (TypeError, ValueError) as exc:
            raise AtomValidationError("door_position must be an integer") from exc
        if float(value) != position:
            raise AtomValidationError("door_position must be an integer")
        if not self.minimum <= position <= self.maximum:
            raise AtomValidationError(
                f"door_position={position!r} is outside the smoke-safe "
                f"range [{self.minimum}, {self.maximum}]"
            )
        return CompiledAtom(
            kind=self.kind,
            factor_key=self.factor_key,
            factor_value=position,
            variation_value=np.asarray([position] * 3, dtype=np.int64),
        )


def tworoom_atom_registry() -> dict[str, VariationAtom]:
    atoms: tuple[VariationAtom, ...] = (AgentSpeedAtom(), DoorPositionAtom())
    for atom in atoms:
        if atom.pixel_effect not in PIXEL_EFFECT_CONTRACTS:
            raise RuntimeError(
                f"Atom {atom.kind!r} has unsupported pixel_effect "
                f"{atom.pixel_effect!r}"
            )
        if not atom.required_oracles:
            raise RuntimeError(
                f"Atom {atom.kind!r} must declare pixel_effect and required_oracles"
            )
    return {atom.kind: atom for atom in atoms}
