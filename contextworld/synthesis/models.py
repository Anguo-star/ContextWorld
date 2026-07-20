from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class AtomRequest:
    """One named, independently testable environment mutation."""

    kind: str
    value: Any


@dataclass(frozen=True)
class ScenarioRequest:
    """Human-authored composition of one or more atom requests."""

    name: str
    split: str
    atoms: tuple[AtomRequest, ...]
    regime: str | None = None
    episodes: int | None = None
    seed_group: str | None = None
    reset_constraints: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompiledScenario:
    """Fully resolved, deterministic input to a collector."""

    schema_version: int
    experiment: str
    task: str
    env_id: str
    name: str
    split: str
    regime: str | None
    scenario_id: str
    fingerprint: str
    atoms: tuple[AtomRequest, ...]
    factors: dict[str, Any]
    variation: tuple[str, ...]
    variation_values: dict[str, Any]
    env_seed: int
    policy_seed: int
    seed_group: str | None
    episodes: int
    max_episode_steps: int
    image_shape: tuple[int, int]
    pixel_codec: dict[str, Any]
    output_path: Path
    reset_constraints: dict[str, Any] = field(default_factory=dict)

    def to_manifest_record(self, root: Path | None = None) -> dict[str, Any]:
        record = asdict(self)
        record["atoms"] = [asdict(atom) for atom in self.atoms]
        output_path = self.output_path
        if root is not None:
            try:
                output_path = output_path.relative_to(root)
            except ValueError:
                pass
        record["output_path"] = str(output_path)
        return _json_safe(record)
