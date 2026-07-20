from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .atoms import VariationAtom, tworoom_atom_registry
from .lance import normalize_pixel_codec
from .models import AtomRequest, CompiledScenario, ScenarioRequest
from .reset_constraints import normalize_reset_constraints


_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class ScenarioCompiler:
    """Compile declarative atoms into deterministic TwoRoom reset options."""

    schema_version = 1

    def __init__(
        self,
        *,
        experiment: str,
        task: str,
        env_id: str,
        seed: int,
        episodes: int,
        max_episode_steps: int,
        image_shape: tuple[int, int],
        output_root: Path,
        pixel_codec: dict[str, Any] | None = None,
        registry: dict[str, VariationAtom] | None = None,
        base_variation: tuple[str, ...] = (
            "agent.position",
            "target.position",
        ),
        reset_constraints: dict[str, Any] | None = None,
    ) -> None:
        if task != "tworoom":
            raise ValueError(f"The smoke compiler currently supports tworoom, got {task!r}")
        self.experiment = experiment
        self.task = task
        self.env_id = env_id
        self.seed = int(seed)
        self.episodes = int(episodes)
        self.max_episode_steps = int(max_episode_steps)
        self.image_shape = tuple(int(x) for x in image_shape)
        self.output_root = Path(output_root).resolve()
        self.pixel_codec = normalize_pixel_codec(pixel_codec)
        self.registry = registry or tworoom_atom_registry()
        self.base_variation = base_variation
        self.reset_constraints = normalize_reset_constraints(reset_constraints)

    def compile_all(
        self, requests: list[ScenarioRequest]
    ) -> list[CompiledScenario]:
        names = [request.name for request in requests]
        if len(names) != len(set(names)):
            raise ValueError("Scenario names must be unique")
        seed_sequence = np.random.SeedSequence(self.seed)
        children = seed_sequence.spawn(len(requests) * 2)
        compiled: list[CompiledScenario] = []
        for index, request in enumerate(requests):
            if request.seed_group is None:
                env_seed = int(children[index * 2].generate_state(1)[0])
                policy_seed = int(children[index * 2 + 1].generate_state(1)[0])
            else:
                env_seed = self._group_seed(request.seed_group, "environment")
                policy_seed = self._group_seed(request.seed_group, "policy")
            compiled.append(self.compile(request, env_seed, policy_seed))
        return compiled

    def _group_seed(self, seed_group: str, stream: str) -> int:
        payload = f"{self.seed}:{seed_group}:{stream}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")

    def compile(
        self,
        request: ScenarioRequest,
        env_seed: int,
        policy_seed: int,
    ) -> CompiledScenario:
        if not _SAFE_NAME.fullmatch(request.name):
            raise ValueError(
                f"Unsafe scenario name {request.name!r}; use lowercase letters, "
                "digits, underscores, and hyphens"
            )
        if request.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split {request.split!r}")
        if not request.atoms:
            raise ValueError(f"Scenario {request.name!r} has no atoms")
        episodes = self.episodes if request.episodes is None else request.episodes
        if episodes <= 0:
            raise ValueError(f"Scenario {request.name!r} must have episodes > 0")

        factors: dict[str, Any] = {}
        variation_values: dict[str, Any] = {}
        reset_constraints = (
            self.reset_constraints
            if request.reset_constraints is None
            else normalize_reset_constraints(request.reset_constraints)
        )
        for request_atom in request.atoms:
            try:
                adapter = self.registry[request_atom.kind]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown atom {request_atom.kind!r}; available: "
                    f"{sorted(self.registry)}"
                ) from exc
            atom = adapter.compile(request_atom.value)
            if atom.factor_key in factors:
                raise ValueError(
                    f"Scenario {request.name!r} mutates {atom.factor_key!r} twice"
                )
            factors[atom.factor_key] = atom.factor_value
            variation_values[atom.factor_key] = atom.variation_value

        identity = {
            "schema_version": self.schema_version,
            "experiment": self.experiment,
            "task": self.task,
            "env_id": self.env_id,
            "name": request.name,
            "split": request.split,
            "atoms": [
                {"kind": atom.kind, "value": atom.value}
                for atom in request.atoms
            ],
            "factors": factors,
            "episodes": episodes,
            "max_episode_steps": self.max_episode_steps,
            "image_shape": self.image_shape,
            "pixel_codec": self.pixel_codec,
            "env_seed": env_seed,
            "policy_seed": policy_seed,
        }
        if request.regime is not None:
            identity["regime"] = request.regime
        if request.seed_group is not None:
            identity["seed_group"] = request.seed_group
        if reset_constraints:
            identity["reset_constraints"] = reset_constraints
        canonical = json.dumps(
            _jsonable(identity), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        fingerprint = hashlib.sha256(canonical).hexdigest()
        scenario_id = f"{self.task}--{request.name}--{fingerprint[:10]}"
        output_dir = self.output_root / request.split
        if request.regime is not None:
            output_dir = output_dir / request.regime
        output_path = output_dir / f"{scenario_id}.lance"
        variation = tuple(
            dict.fromkeys((*self.base_variation, *variation_values.keys()))
        )
        return CompiledScenario(
            schema_version=self.schema_version,
            experiment=self.experiment,
            task=self.task,
            env_id=self.env_id,
            name=request.name,
            split=request.split,
            regime=request.regime,
            scenario_id=scenario_id,
            fingerprint=fingerprint,
            atoms=request.atoms,
            factors=_jsonable(factors),
            variation=variation,
            variation_values=variation_values,
            env_seed=env_seed,
            policy_seed=policy_seed,
            seed_group=request.seed_group,
            episodes=episodes,
            max_episode_steps=self.max_episode_steps,
            image_shape=self.image_shape,
            pixel_codec=self.pixel_codec,
            output_path=output_path,
            reset_constraints=reset_constraints,
        )
