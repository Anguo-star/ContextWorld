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


class PassageOpenAtom(VariationAtom):
    """Hidden binary rule controlling whether the visible doorway is usable."""

    kind = "passage_open"
    factor_key = "passage.open"
    pixel_effect = "contact_dynamics"
    required_oracles = ("hidden_passage_contact_oracle",)

    def compile(self, value: Any) -> CompiledAtom:
        if isinstance(value, (bool, np.bool_)):
            passage_open = int(value)
        else:
            try:
                passage_open = int(value)
            except (TypeError, ValueError) as exc:
                raise AtomValidationError(
                    "passage_open must be 0 (blocked) or 1 (passable)"
                ) from exc
            try:
                is_integer = float(value) == float(passage_open)
            except (TypeError, ValueError):
                is_integer = False
            if not is_integer:
                raise AtomValidationError(
                    "passage_open must be 0 (blocked) or 1 (passable)"
                )
        if passage_open not in (0, 1):
            raise AtomValidationError(
                "passage_open must be 0 (blocked) or 1 (passable)"
            )
        return CompiledAtom(
            kind=self.kind,
            factor_key=self.factor_key,
            factor_value=passage_open,
            variation_value=passage_open,
        )


class ActionDelayAtom(VariationAtom):
    """Hidden raw-step delay between a command and its physical effect."""

    kind = "action_delay"
    factor_key = "action.delay_steps"
    pixel_effect = "temporal_dynamics"
    required_oracles = ("action_delay_temporal_oracle",)
    minimum = 0
    maximum = 4

    def compile(self, value: Any) -> CompiledAtom:
        if isinstance(value, (bool, np.bool_)):
            raise AtomValidationError(
                "action_delay must be an integer in [0, 4]"
            )
        try:
            delay = int(value)
        except (TypeError, ValueError) as exc:
            raise AtomValidationError(
                "action_delay must be an integer in [0, 4]"
            ) from exc
        try:
            is_integer = float(value) == float(delay)
        except (TypeError, ValueError):
            is_integer = False
        if not is_integer or not self.minimum <= delay <= self.maximum:
            raise AtomValidationError(
                "action_delay must be an integer in [0, 4]"
            )
        return CompiledAtom(
            kind=self.kind,
            factor_key=self.factor_key,
            factor_value=delay,
            variation_value=delay,
        )


def tworoom_atom_registry() -> dict[str, VariationAtom]:
    atoms: tuple[VariationAtom, ...] = (
        AgentSpeedAtom(),
        DoorPositionAtom(),
        PassageOpenAtom(),
        ActionDelayAtom(),
    )
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
