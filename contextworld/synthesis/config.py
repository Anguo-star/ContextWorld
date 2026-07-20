from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from contextworld.paths import resolve_contextworld_path

from .compiler import ScenarioCompiler
from .models import AtomRequest, ScenarioRequest


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a mapping")
    config["_config_path"] = path
    return config


def scenario_requests(config: dict[str, Any]) -> list[ScenarioRequest]:
    requests: list[ScenarioRequest] = []
    for raw in config.get("scenarios", []):
        atoms = tuple(
            AtomRequest(kind=atom["kind"], value=atom["value"])
            for atom in raw["atoms"]
        )
        requests.append(
            ScenarioRequest(
                name=raw["name"],
                split=raw["split"],
                atoms=atoms,
                regime=raw.get("regime"),
                episodes=raw.get("episodes_per_scenario"),
                seed_group=raw.get("seed_group"),
                reset_constraints=raw.get("reset_constraints"),
            )
        )
    requests.extend(_scenario_set_requests(config))
    if not requests:
        raise ValueError("Config must define scenarios or scenario_sets")
    return requests


def _scenario_set_requests(config: dict[str, Any]) -> list[ScenarioRequest]:
    generated: list[ScenarioRequest] = []
    reserved: dict[str, list[float]] = {}
    master_seed = int(config.get("scenario_generation_seed", config["seed"]))
    value_pools = _value_pools(config, master_seed)

    for set_index, raw in enumerate(config.get("scenario_sets", [])):
        if "combinations" in raw:
            generated.extend(_combination_set_requests(raw))
            continue

        atom_kind = raw["atom"]
        minimum_gap = float(raw.get("minimum_gap", 0.0))
        value_specification = raw["values"]
        if isinstance(value_specification, dict) and "pool" in value_specification:
            pool = value_pools[value_specification["pool"]]
            start = int(value_specification.get("start", 0))
            stop = start + int(value_specification["count"])
            if start < 0 or stop > len(pool):
                raise ValueError(
                    f"Pool slice [{start}:{stop}] is outside "
                    f"{value_specification['pool']!r} with {len(pool)} values"
                )
            value_specification = pool[start:stop]
        value_reservations = (
            []
            if raw.get("allow_value_reuse", False)
            else reserved.setdefault(atom_kind, [])
        )
        values = _scenario_values(
            value_specification,
            master_seed=master_seed,
            set_index=set_index,
            reserved=value_reservations,
            minimum_gap=minimum_gap,
        )
        prefix = raw["name_prefix"]
        seed_groups = _seed_groups(raw)
        assignment = raw.get("assignment", "cartesian")
        if assignment == "cartesian":
            assignments = [
                (value_index, value, group_index, seed_group)
                for value_index, value in enumerate(values)
                for group_index, seed_group in enumerate(seed_groups)
            ]
        elif assignment == "paired_cycle":
            if any(seed_group is None for seed_group in seed_groups):
                raise ValueError("paired_cycle requires explicit seed_groups")
            if len(seed_groups) % len(values):
                raise ValueError(
                    "paired_cycle requires the seed-group count to be a multiple "
                    "of the value count"
                )
            assignments = [
                (
                    group_index % len(values),
                    values[group_index % len(values)],
                    group_index,
                    seed_group,
                )
                for group_index, seed_group in enumerate(seed_groups)
            ]
        else:
            raise ValueError(f"Unsupported scenario-set assignment: {assignment!r}")

        for value_index, value, group_index, seed_group in assignments:
            group_slug = (
                ""
                if len(seed_groups) == 1
                else f"_g{group_index:02d}"
            )
            name = (
                f"{prefix}{group_slug}_{value_index:03d}_"
                f"v{_value_slug(value)}"
            )
            generated.append(
                ScenarioRequest(
                    name=name,
                    split=raw["split"],
                    atoms=(AtomRequest(kind=atom_kind, value=value),),
                    regime=raw.get("regime"),
                    episodes=raw.get("episodes_per_scenario"),
                    seed_group=seed_group,
                    reset_constraints=raw.get("reset_constraints"),
                )
            )
    return generated


def _seed_groups(raw: dict[str, Any]) -> list[str | None]:
    has_single = "seed_group" in raw
    has_multiple = "seed_groups" in raw
    if has_single and has_multiple:
        raise ValueError("Use seed_group or seed_groups, not both")
    if not has_multiple:
        return [raw.get("seed_group")]
    values = raw["seed_groups"]
    if isinstance(values, dict):
        if values.get("sampler") != "indexed":
            raise ValueError(f"Unsupported seed-group sampler: {values}")
        prefix = str(values["prefix"])
        count = int(values["count"])
        start = int(values.get("start", 0))
        width = int(values.get("width", 4))
        if count <= 0 or start < 0 or width <= 0:
            raise ValueError(f"Invalid indexed seed groups: {values}")
        rendered = [
            f"{prefix}_{index:0{width}d}"
            for index in range(start, start + count)
        ]
    elif isinstance(values, list) and values:
        rendered = [str(value) for value in values]
    else:
        raise ValueError("seed_groups must be a non-empty list or indexed sampler")
    if len(rendered) != len(set(rendered)):
        raise ValueError("seed_groups must be unique")
    return rendered


def _combination_set_requests(raw: dict[str, Any]) -> list[ScenarioRequest]:
    """Expand an explicit table of multi-atom combinations.

    Combination tables deliberately preserve authored scalar values (notably
    integer geometry values) and sort atom names so scenario fingerprints do
    not depend on YAML mapping insertion order.
    """

    if "atom" in raw or "values" in raw:
        raise ValueError(
            "A scenario set must use either combinations or atom/values, not both"
        )
    combinations = raw["combinations"]
    if not isinstance(combinations, list) or not combinations:
        raise ValueError("Scenario-set combinations must be a non-empty list")

    prefix = raw["name_prefix"]
    generated: list[ScenarioRequest] = []
    for combination_index, combination in enumerate(combinations):
        if not isinstance(combination, dict) or len(combination) < 2:
            raise ValueError(
                "Each combination must map at least two atom names to values"
            )
        atoms = tuple(
            AtomRequest(kind=kind, value=combination[kind])
            for kind in sorted(combination)
        )
        suffix = "_".join(
            f"{kind}_v{_scalar_value_slug(value)}"
            for kind, value in sorted(combination.items())
        )
        generated.append(
            ScenarioRequest(
                name=f"{prefix}_{combination_index:03d}_{suffix}",
                split=raw["split"],
                atoms=atoms,
                regime=raw.get("regime"),
                episodes=raw.get("episodes_per_scenario"),
                seed_group=raw.get("seed_group"),
                reset_constraints=raw.get("reset_constraints"),
            )
        )
    return generated


def _value_pools(config: dict[str, Any], master_seed: int) -> dict[str, list[float]]:
    pools: dict[str, list[float]] = {}
    for pool_index, (name, raw) in enumerate(
        config.get("scenario_value_pools", {}).items()
    ):
        if raw.get("sampler") != "stratified_unique":
            raise ValueError(f"Unsupported value-pool sampler: {raw}")
        low, high = (float(value) for value in raw["range"])
        count = int(raw["count"])
        if not low < high or count <= 1:
            raise ValueError(f"Invalid stratified_unique pool: {raw}")
        values = np.linspace(low, high, count, dtype=np.float64)
        if raw.get("shuffle", True):
            rng = np.random.default_rng(
                np.random.SeedSequence([master_seed, pool_index, 0x570A71F1])
            )
            rng.shuffle(values)
        pools[name] = [float(value) for value in values]
    return pools


def _scenario_values(
    specification: Any,
    *,
    master_seed: int,
    set_index: int,
    reserved: list[float],
    minimum_gap: float,
) -> list[float]:
    if isinstance(specification, list):
        candidates = [float(value) for value in specification]
    elif specification.get("sampler") == "fixed":
        candidates = [float(value) for value in specification["values"]]
    elif specification.get("sampler") == "uniform_unique":
        low, high = (float(value) for value in specification["range"])
        count = int(specification["count"])
        if not low < high or count <= 0:
            raise ValueError(f"Invalid uniform_unique specification: {specification}")
        rng = np.random.default_rng(
            np.random.SeedSequence([master_seed, set_index, 0xC07E57])
        )
        candidates = []
        for _ in range(count):
            for _attempt in range(100_000):
                candidate = float(rng.uniform(low, high))
                if _far_enough(candidate, (*reserved, *candidates), minimum_gap):
                    candidates.append(candidate)
                    break
            else:
                raise ValueError(
                    f"Could not sample {count} values in [{low}, {high}] "
                    f"with minimum_gap={minimum_gap}"
                )
    else:
        raise ValueError(f"Unsupported scenario value specification: {specification}")

    for candidate in candidates:
        if not _far_enough(candidate, reserved, minimum_gap):
            raise ValueError(
                f"Scenario value {candidate} violates minimum_gap={minimum_gap} "
                "with an earlier scenario set"
            )
        reserved.append(candidate)
    return candidates


def _far_enough(value: float, others: Any, minimum_gap: float) -> bool:
    return all(abs(value - float(other)) >= minimum_gap for other in others)


def _value_slug(value: float) -> str:
    rendered = f"{value:.8f}".rstrip("0").rstrip(".")
    return rendered.replace("-", "m").replace(".", "p")


def _scalar_value_slug(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "Generated combination names currently require numeric scalar values"
        )
    return _value_slug(float(value))


def build_compiler(
    config: dict[str, Any], repo_root: Path
) -> ScenarioCompiler:
    collection = config["collection"]
    output_root = resolve_contextworld_path(
        config["output"]["data_root"], repo_root=repo_root
    )
    return ScenarioCompiler(
        experiment=config["experiment"],
        task=config["task"],
        env_id=config["env_id"],
        seed=config["seed"],
        episodes=collection["episodes_per_scenario"],
        max_episode_steps=collection["max_episode_steps"],
        image_shape=tuple(collection["image_shape"]),
        output_root=output_root,
        pixel_codec=collection.get("pixel_codec"),
        reset_constraints=collection.get("reset_constraints"),
    )
