from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .atoms import PIXEL_EFFECT_CONTRACTS
from .models import CompiledScenario
from .reset_constraints import apply_tworoom_reset_constraints


_MODEL_COLUMNS = ("pixels", "action", "proprio")


def factor_column(factor_key: str) -> str:
    return f"variation_{factor_key.replace('.', '_')}"


def scenario_signature(scenario: CompiledScenario) -> tuple[tuple[str, Any], ...]:
    return tuple(
        sorted((atom.kind, _hashable(atom.value)) for atom in scenario.atoms)
    )


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(val)) for key, val in value.items()))
    return value


def validate_split_isolation(
    scenarios: list[CompiledScenario],
) -> dict[str, Any]:
    signatures: dict[str, set[tuple[tuple[str, Any], ...]]] = {}
    for scenario in scenarios:
        signatures.setdefault(scenario.split, set()).add(
            scenario_signature(scenario)
        )
    train_test_overlap = sorted(
        repr(value)
        for value in signatures.get("train", set())
        & signatures.get("test", set())
    )
    return {
        "passed": not train_test_overlap,
        "train_test_overlap": train_test_overlap,
        "counts": {key: len(value) for key, value in signatures.items()},
    }


def validate_training_unseen_combinations(
    scenarios: list[CompiledScenario],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Prove a combination-level split without withholding atomic values.

    Evaluation combinations must be absent from composed training scenarios,
    while each individual value used by evaluation must occur in at least one
    composed training scenario. This distinguishes combination generalization
    from ordinary single-factor extrapolation.
    """

    atom_kinds = tuple(config["atoms"])
    if len(atom_kinds) < 2 or len(set(atom_kinds)) != len(atom_kinds):
        raise ValueError(
            "training_unseen_combinations requires at least two unique atoms"
        )
    evaluation_splits = tuple(
        config.get("evaluation_splits", ("val", "test"))
    )
    if (
        not evaluation_splits
        or "train" in evaluation_splits
        or len(set(evaluation_splits)) != len(evaluation_splits)
    ):
        raise ValueError(
            "evaluation_splits must be unique, non-empty, and exclude train"
        )

    def serializable(value: Any) -> Any:
        if isinstance(value, tuple):
            return [serializable(item) for item in value]
        return value

    def combination_record(
        combination: tuple[Any, ...],
    ) -> dict[str, Any]:
        return {
            atom_kind: serializable(value)
            for atom_kind, value in zip(atom_kinds, combination)
        }

    combinations_by_split: dict[
        str, list[tuple[str, tuple[Any, ...]]]
    ] = {}
    invalid_scenarios: list[dict[str, Any]] = []
    ignored_partial_scenarios: list[str] = []
    for scenario in scenarios:
        values_by_kind: dict[str, list[Any]] = {
            atom_kind: [] for atom_kind in atom_kinds
        }
        for atom in scenario.atoms:
            if atom.kind in values_by_kind:
                values_by_kind[atom.kind].append(_hashable(atom.value))
        if any(len(values) > 1 for values in values_by_kind.values()):
            invalid_scenarios.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "reason": "duplicate_atom_in_scenario",
                }
            )
            continue
        if not all(len(values) == 1 for values in values_by_kind.values()):
            ignored_partial_scenarios.append(scenario.scenario_id)
            continue
        combination = tuple(
            values_by_kind[atom_kind][0] for atom_kind in atom_kinds
        )
        combinations_by_split.setdefault(scenario.split, []).append(
            (scenario.scenario_id, combination)
        )

    combination_sets = {
        split: {combination for _, combination in entries}
        for split, entries in combinations_by_split.items()
    }
    duplicate_combinations: dict[str, list[dict[str, Any]]] = {}
    for split, entries in combinations_by_split.items():
        scenario_ids_by_combination: dict[
            tuple[Any, ...], list[str]
        ] = {}
        for scenario_id, combination in entries:
            scenario_ids_by_combination.setdefault(combination, []).append(
                scenario_id
            )
        duplicates = [
            {
                "combination": combination_record(combination),
                "scenario_ids": scenario_ids,
            }
            for combination, scenario_ids in scenario_ids_by_combination.items()
            if len(scenario_ids) > 1
        ]
        if duplicates:
            duplicate_combinations[split] = sorted(
                duplicates,
                key=lambda item: repr(item["combination"]),
            )

    train_combinations = combination_sets.get("train", set())
    evaluation_combinations = set().union(
        *(
            combination_sets.get(split, set())
            for split in evaluation_splits
        )
    )
    train_evaluation_overlap = sorted(
        (
            combination_record(combination)
            for combination in train_combinations & evaluation_combinations
        ),
        key=repr,
    )

    evaluation_split_overlap: list[dict[str, Any]] = []
    for left_index, left_split in enumerate(evaluation_splits):
        for right_split in evaluation_splits[left_index + 1 :]:
            for combination in (
                combination_sets.get(left_split, set())
                & combination_sets.get(right_split, set())
            ):
                evaluation_split_overlap.append(
                    {
                        "left_split": left_split,
                        "right_split": right_split,
                        "combination": combination_record(combination),
                    }
                )
    evaluation_split_overlap.sort(
        key=lambda item: (
            item["left_split"],
            item["right_split"],
            repr(item["combination"]),
        )
    )

    train_atomic_support = {
        atom_kind: {
            combination[index] for combination in train_combinations
        }
        for index, atom_kind in enumerate(atom_kinds)
    }
    missing_train_atomic_support: list[dict[str, Any]] = []
    for split in evaluation_splits:
        for scenario_id, combination in combinations_by_split.get(
            split, []
        ):
            for index, atom_kind in enumerate(atom_kinds):
                value = combination[index]
                if value not in train_atomic_support[atom_kind]:
                    missing_train_atomic_support.append(
                        {
                            "scenario_id": scenario_id,
                            "split": split,
                            "atom": atom_kind,
                            "value": serializable(value),
                        }
                    )

    passed = bool(
        train_combinations
        and evaluation_combinations
        and not invalid_scenarios
        and not duplicate_combinations
        and not train_evaluation_overlap
        and not evaluation_split_overlap
        and not missing_train_atomic_support
    )
    return {
        "passed": passed,
        "semantics": (
            "validation/test atom combinations are absent from composed "
            "training scenarios, while every constituent atomic value is "
            "supported by a composed training scenario"
        ),
        "atoms": list(atom_kinds),
        "evaluation_splits": list(evaluation_splits),
        "combination_counts": {
            split: len(combinations)
            for split, combinations in sorted(combination_sets.items())
        },
        "scenario_counts": {
            split: len(entries)
            for split, entries in sorted(combinations_by_split.items())
        },
        "train_atomic_support": {
            atom_kind: sorted(
                (serializable(value) for value in values), key=repr
            )
            for atom_kind, values in train_atomic_support.items()
        },
        "train_evaluation_overlap": train_evaluation_overlap,
        "evaluation_split_overlap": evaluation_split_overlap,
        "missing_train_atomic_support": missing_train_atomic_support,
        "duplicate_combinations": duplicate_combinations,
        "invalid_scenarios": invalid_scenarios,
        "ignored_partial_scenarios": ignored_partial_scenarios,
    }


def validate_numeric_atom_isolation(
    scenarios: list[CompiledScenario],
    *,
    atom_kind: str,
    minimum_cross_split_gap: float,
) -> dict[str, Any]:
    values: list[tuple[str, str, float]] = []
    for scenario in scenarios:
        matching = [atom for atom in scenario.atoms if atom.kind == atom_kind]
        if len(matching) != 1:
            continue
        values.append((scenario.scenario_id, scenario.split, float(matching[0].value)))

    closest: dict[str, Any] | None = None
    for left_index, left in enumerate(values):
        for right in values[left_index + 1 :]:
            if left[1] == right[1]:
                continue
            gap = abs(left[2] - right[2])
            if closest is None or gap < closest["gap"]:
                closest = {
                    "left_scenario_id": left[0],
                    "left_split": left[1],
                    "left_value": left[2],
                    "right_scenario_id": right[0],
                    "right_split": right[1],
                    "right_value": right[2],
                    "gap": gap,
                }

    observed_gap = None if closest is None else float(closest["gap"])
    passed = observed_gap is not None and observed_gap >= minimum_cross_split_gap
    return {
        "passed": passed,
        "atom_kind": atom_kind,
        "minimum_required_gap": minimum_cross_split_gap,
        "minimum_observed_cross_split_gap": observed_gap,
        "closest_pair": closest,
        "values_checked": len(values),
    }


def validate_paired_seed_crossing(
    scenarios: list[CompiledScenario],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Prove every declared seed block is crossed with all split factors."""

    atom_kind = str(config.get("atom", "agent_speed"))
    requirements = dict(config.get("splits", {}))
    failures: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    all_seed_ranges: dict[str, set[int]] = {}

    for split, requirement in requirements.items():
        selected = [
            scenario
            for scenario in scenarios
            if scenario.split == split
            and any(atom.kind == atom_kind for atom in scenario.atoms)
        ]
        groups: dict[str, list[CompiledScenario]] = defaultdict(list)
        for scenario in selected:
            if scenario.seed_group is None:
                failures.append(
                    {
                        "split": split,
                        "scenario_id": scenario.scenario_id,
                        "reason": "missing_seed_group",
                    }
                )
                continue
            groups[scenario.seed_group].append(scenario)

        factor_sets: list[set[float]] = []
        paired_resets = 0
        group_reports = []
        for seed_group, entries in sorted(groups.items()):
            factor_values = {
                float(
                    next(
                        atom.value
                        for atom in scenario.atoms
                        if atom.kind == atom_kind
                    )
                )
                for scenario in entries
            }
            factor_sets.append(factor_values)
            env_seeds = {scenario.env_seed for scenario in entries}
            policy_seeds = {scenario.policy_seed for scenario in entries}
            episode_counts = {scenario.episodes for scenario in entries}
            crossed = bool(
                len(entries) == len(factor_values)
                and len(env_seeds) == 1
                and len(policy_seeds) == 1
                and len(episode_counts) == 1
            )
            if not crossed:
                failures.append(
                    {
                        "split": split,
                        "seed_group": seed_group,
                        "reason": "incomplete_or_unpaired_factor_cross",
                    }
                )
            episodes = min(episode_counts) if episode_counts else 0
            paired_resets += episodes
            reset_seeds = (
                {next(iter(env_seeds)) + index for index in range(episodes)}
                if len(env_seeds) == 1
                else set()
            )
            key = f"{split}:{seed_group}"
            all_seed_ranges[key] = reset_seeds
            group_reports.append(
                {
                    "seed_group": seed_group,
                    "factor_values": sorted(factor_values),
                    "factor_count": len(factor_values),
                    "episodes_per_factor": episodes,
                    "env_seed": next(iter(env_seeds)) if len(env_seeds) == 1 else None,
                    "policy_seed": (
                        next(iter(policy_seeds))
                        if len(policy_seeds) == 1
                        else None
                    ),
                    "crossed": crossed,
                }
            )

        shared_factors = bool(
            factor_sets and all(values == factor_sets[0] for values in factor_sets)
        )
        minimum_groups = int(requirement.get("minimum_seed_groups", 1))
        minimum_factors = int(requirement.get("minimum_factor_values", 1))
        minimum_resets = int(
            requirement.get("minimum_paired_resets_per_factor", 1)
        )
        observed_factors = len(factor_sets[0]) if shared_factors else 0
        gates = {
            "seed_groups": len(groups) >= minimum_groups,
            "shared_factor_set": shared_factors,
            "factor_values": observed_factors >= minimum_factors,
            "paired_resets_per_factor": paired_resets >= minimum_resets,
        }
        if not all(gates.values()):
            failures.append(
                {
                    "split": split,
                    "reason": "split_pairing_gate",
                    "gates": gates,
                }
            )
        split_reports[split] = {
            "passed": all(gates.values()),
            "requirements": {
                "minimum_seed_groups": minimum_groups,
                "minimum_factor_values": minimum_factors,
                "minimum_paired_resets_per_factor": minimum_resets,
            },
            "observed": {
                "seed_groups": len(groups),
                "factor_values": observed_factors,
                "paired_resets_per_factor": paired_resets,
            },
            "gates": gates,
            "groups": group_reports,
        }

    seed_overlaps = []
    keys = sorted(all_seed_ranges)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            overlap = all_seed_ranges[left] & all_seed_ranges[right]
            if overlap:
                seed_overlaps.append(
                    {
                        "left": left,
                        "right": right,
                        "overlap_count": len(overlap),
                    }
                )
    if seed_overlaps:
        failures.append({"reason": "episode_seed_overlap", "pairs": seed_overlaps})
    return {
        "passed": bool(requirements and not failures),
        "atom": atom_kind,
        "semantics": (
            "each deterministic episode-seed block is fully crossed with "
            "all factor values in its split; reset seeds are disjoint across blocks"
        ),
        "splits": split_reports,
        "episode_seed_overlaps": seed_overlaps,
        "failures": failures,
    }


def validate_reset_coverage(
    scenarios: list[CompiledScenario],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Measure reset-state/goal coverage before expensive collection."""

    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    grid_size = float(config.get("grid_size", 14.0))
    split_requirements = dict(config.get("splits", {}))
    split_reports: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    for split, requirements in split_requirements.items():
        representatives: dict[str, CompiledScenario] = {}
        for scenario in scenarios:
            if scenario.split != split or scenario.seed_group is None:
                continue
            current = representatives.get(scenario.seed_group)
            speed = float(scenario.factors.get("agent.speed", 0.0))
            current_speed = (
                float(current.factors.get("agent.speed", 0.0))
                if current is not None
                else float("-inf")
            )
            if current is None or speed > current_speed:
                representatives[scenario.seed_group] = scenario

        starts: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        cross_room_flags: list[bool] = []
        left_to_right_flags: list[bool] = []
        right_to_left_flags: list[bool] = []
        reset_seeds: set[int] = set()
        env = TwoRoomEnv(render_mode="rgb_array")
        try:
            for scenario in representatives.values():
                apply_tworoom_reset_constraints(
                    env, scenario.reset_constraints
                )
                for episode_index in range(scenario.episodes):
                    reset_seed = scenario.env_seed + episode_index
                    try:
                        observation, _ = env.reset(
                            seed=reset_seed,
                            options={
                                "variation": scenario.variation,
                                "variation_values": scenario.variation_values,
                            },
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "Reset coverage sampling failed for "
                            f"{scenario.scenario_id} episode={episode_index} "
                            f"seed={reset_seed} constraints="
                            f"{scenario.reset_constraints}"
                        ) from exc
                    starts.append(np.asarray(observation[:2], dtype=np.float32))
                    goals.append(np.asarray(observation[2:4], dtype=np.float32))
                    wall_axis = int(
                        env.variation_space["wall"]["axis"].value
                    )
                    coordinate = 0 if wall_axis == 1 else 1
                    wall_center = float(env.WALL_CENTER)
                    start_left = float(observation[coordinate]) < wall_center
                    goal_left = float(observation[2 + coordinate]) < wall_center
                    cross_room_flags.append(start_left != goal_left)
                    left_to_right_flags.append(start_left and not goal_left)
                    right_to_left_flags.append(not start_left and goal_left)
                    reset_seeds.add(reset_seed)
        finally:
            env.close()

        start_array = np.asarray(starts, dtype=np.float32)
        goal_array = np.asarray(goals, dtype=np.float32)
        initial_distances = (
            np.linalg.norm(start_array - goal_array, axis=1)
            if starts
            else np.asarray([], dtype=np.float32)
        )

        def bins(values: np.ndarray) -> set[tuple[int, int]]:
            if values.size == 0:
                return set()
            indices = np.floor(values / grid_size).astype(np.int64)
            return set(map(tuple, indices.tolist()))

        start_bins = bins(start_array)
        goal_bins = bins(goal_array)
        pair_bins = (
            set(
                map(
                    tuple,
                    np.concatenate(
                        [
                            np.floor(start_array / grid_size).astype(np.int64),
                            np.floor(goal_array / grid_size).astype(np.int64),
                        ],
                        axis=1,
                    ).tolist(),
                )
            )
            if starts
            else set()
        )
        template_requirements = dict(requirements.get("templates", {}))
        template_observed = {}
        if starts:
            start_grid = np.floor(start_array / grid_size).astype(np.int64)
            goal_grid = np.floor(goal_array / grid_size).astype(np.int64)
            for template, specification in template_requirements.items():
                expected_start = np.asarray(
                    specification["start_grid"], dtype=np.int64
                )
                expected_goal = np.asarray(
                    specification["goal_grid"], dtype=np.int64
                )
                template_observed[template] = int(
                    np.sum(
                        np.all(start_grid == expected_start, axis=1)
                        & np.all(goal_grid == expected_goal, axis=1)
                    )
                )
        observed = {
            "unique_reset_seeds": len(reset_seeds),
            "unique_start_states": (
                len(np.unique(np.round(start_array, 4), axis=0)) if starts else 0
            ),
            "unique_goal_states": (
                len(np.unique(np.round(goal_array, 4), axis=0)) if goals else 0
            ),
            "start_grid_bins": len(start_bins),
            "goal_grid_bins": len(goal_bins),
            "start_goal_grid_pairs": len(pair_bins),
            "cross_room_resets": int(sum(cross_room_flags)),
            "same_room_resets": int(len(cross_room_flags) - sum(cross_room_flags)),
            "cross_room_fraction": (
                float(np.mean(cross_room_flags)) if cross_room_flags else 0.0
            ),
            "left_to_right_resets": int(sum(left_to_right_flags)),
            "right_to_left_resets": int(sum(right_to_left_flags)),
            "minimum_initial_distance": (
                float(initial_distances.min()) if len(initial_distances) else 0.0
            ),
            "mean_initial_distance": (
                float(initial_distances.mean()) if len(initial_distances) else 0.0
            ),
            "initial_distance_p10": (
                float(np.percentile(initial_distances, 10))
                if len(initial_distances)
                else 0.0
            ),
            "initial_distance_p90": (
                float(np.percentile(initial_distances, 90))
                if len(initial_distances)
                else 0.0
            ),
            "templates": template_observed,
        }
        gates = {
            key: float(observed[key]) >= float(required)
            for key, required in requirements.items()
            if key in observed
            and key != "templates"
        }
        gates.update(
            {
                f"template:{template}": (
                    template_observed.get(template, 0)
                    >= int(specification["minimum_resets"])
                )
                for template, specification in template_requirements.items()
            }
        )
        if not gates or not all(gates.values()):
            failures.append(
                {
                    "split": split,
                    "reason": "reset_coverage_gate",
                    "gates": gates,
                }
            )
        split_reports[split] = {
            "passed": bool(gates and all(gates.values())),
            "requirements": requirements,
            "observed": observed,
            "gates": gates,
        }

    return {
        "passed": bool(split_requirements and not failures),
        "grid_size": grid_size,
        "semantics": (
            "coverage is computed from unique deterministic reset seeds, once "
            "per seed block rather than once per crossed speed; room relation "
            "and distance statistics use the same constrained resets"
        ),
        "splits": split_reports,
        "failures": failures,
    }


def validate_independent_seed_assignment(
    scenarios: list[CompiledScenario],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Prove that reset blocks are assigned to one factor value, not crossed."""

    atom_kind = str(config["atom"])
    factor_key = str(config.get("factor_key", atom_kind.replace("_", ".")))
    split_requirements = dict(config.get("splits", {}))
    failures: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}

    for split, requirements in split_requirements.items():
        selected = [scenario for scenario in scenarios if scenario.split == split]
        by_seed_group: dict[str, list[CompiledScenario]] = defaultdict(list)
        by_factor: dict[str, list[CompiledScenario]] = defaultdict(list)
        reset_seeds: set[int] = set()
        reset_seed_duplicates = 0
        for scenario in selected:
            if scenario.seed_group is None:
                failures.append(
                    {"split": split, "reason": "scenario_without_seed_group"}
                )
                continue
            by_seed_group[scenario.seed_group].append(scenario)
            factor = json.dumps(scenario.factors[factor_key], sort_keys=True)
            by_factor[factor].append(scenario)
            for reset_seed in range(
                scenario.env_seed, scenario.env_seed + scenario.episodes
            ):
                if reset_seed in reset_seeds:
                    reset_seed_duplicates += 1
                reset_seeds.add(reset_seed)

        scenario_counts = [len(value) for value in by_factor.values()]
        episode_counts = [
            sum(scenario.episodes for scenario in value)
            for value in by_factor.values()
        ]
        one_scenario_per_seed_group = bool(by_seed_group) and all(
            len(value) == 1 for value in by_seed_group.values()
        )
        factor_balanced = bool(scenario_counts) and len(set(scenario_counts)) == 1
        observed = {
            "seed_groups": len(by_seed_group),
            "factor_values": len(by_factor),
            "minimum_scenarios_per_factor": min(scenario_counts, default=0),
            "maximum_scenarios_per_factor": max(scenario_counts, default=0),
            "minimum_episodes_per_factor": min(episode_counts, default=0),
            "maximum_episodes_per_factor": max(episode_counts, default=0),
            "unique_reset_seeds": len(reset_seeds),
            "reset_seed_duplicates": reset_seed_duplicates,
            "one_scenario_per_seed_group": one_scenario_per_seed_group,
            "factor_balanced": factor_balanced,
        }
        gates = {
            "minimum_seed_groups": observed["seed_groups"]
            >= int(requirements.get("minimum_seed_groups", 1)),
            "minimum_factor_values": observed["factor_values"]
            >= int(requirements.get("minimum_factor_values", 1)),
            "minimum_scenarios_per_factor": observed[
                "minimum_scenarios_per_factor"
            ]
            >= int(requirements.get("minimum_scenarios_per_factor", 1)),
            "minimum_episodes_per_factor": observed[
                "minimum_episodes_per_factor"
            ]
            >= int(requirements.get("minimum_episodes_per_factor", 1)),
            "one_scenario_per_seed_group": (
                one_scenario_per_seed_group
                if requirements.get("require_one_scenario_per_seed_group", True)
                else True
            ),
            "factor_balanced": (
                factor_balanced
                if requirements.get("require_factor_balance", True)
                else True
            ),
            "reset_seeds_disjoint": reset_seed_duplicates == 0,
        }
        passed = bool(selected and all(gates.values()))
        if not passed:
            failures.append(
                {"split": split, "reason": "independent_assignment_gate", "gates": gates}
            )
        split_reports[split] = {
            "passed": passed,
            "requirements": requirements,
            "observed": observed,
            "gates": gates,
        }

    return {
        "passed": bool(split_requirements and not failures),
        "atom": atom_kind,
        "semantics": (
            "each deterministic seed block belongs to exactly one factor value; "
            "factor counts are balanced and episode reset seeds are disjoint"
        ),
        "splits": split_reports,
        "failures": failures,
    }


def validate_minimum_episode_start_oracle(
    scenarios: list[CompiledScenario],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Guarantee that every configured reset produces at least two rows.

    Stable-WM records the first row after the first environment step. For a
    two-frame training clip, the episode therefore must not terminate on that
    first step. This oracle proves that for every reset seed using a
    policy-independent upper bound on one-step displacement.
    """

    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    minimum_rows = int(config.get("minimum_rows", 2))
    if minimum_rows != 2:
        raise ValueError(
            "minimum_episode_start_oracle currently proves exactly two rows"
        )
    termination_radius = float(config.get("termination_radius", 16.0))
    minimum_margin = float(config.get("minimum_guaranteed_margin", 0.0))
    expected_agent_speed = config.get("expected_agent_speed")
    if expected_agent_speed is not None:
        expected_agent_speed = float(expected_agent_speed)
    failures: list[dict[str, Any]] = []
    speed_mismatches: list[dict[str, Any]] = []
    observed_agent_speeds: set[float] = set()
    closest: dict[str, Any] | None = None
    starts_checked = 0

    for scenario in scenarios:
        if scenario.max_episode_steps < minimum_rows:
            failures.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "reason": "max_episode_steps_below_minimum_rows",
                    "max_episode_steps": scenario.max_episode_steps,
                }
            )
            continue

        env = TwoRoomEnv(render_mode="rgb_array")
        try:
            apply_tworoom_reset_constraints(
                env, scenario.reset_constraints
            )
            action_l2_bound = float(np.linalg.norm(env.action_space.high))
            for episode_index in range(scenario.episodes):
                _, info = env.reset(
                    seed=scenario.env_seed + episode_index,
                    options={
                        "variation": scenario.variation,
                        "variation_values": scenario.variation_values,
                    },
                )
                speed = float(
                    env.variation_space["agent"]["speed"].value.item()
                )
                observed_agent_speeds.add(speed)
                if expected_agent_speed is not None and not np.isclose(
                    speed, expected_agent_speed, atol=1e-7
                ):
                    speed_mismatches.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "episode_index": episode_index,
                            "env_seed": scenario.env_seed + episode_index,
                            "expected_agent_speed": expected_agent_speed,
                            "observed_agent_speed": speed,
                        }
                    )
                initial_distance = float(info["distance_to_target"])
                maximum_displacement = speed * action_l2_bound
                guaranteed_margin = (
                    initial_distance
                    - maximum_displacement
                    - termination_radius
                )
                entry = {
                    "scenario_id": scenario.scenario_id,
                    "episode_index": episode_index,
                    "env_seed": scenario.env_seed + episode_index,
                    "speed": speed,
                    "initial_distance": initial_distance,
                    "maximum_step_displacement": maximum_displacement,
                    "termination_radius": termination_radius,
                    "guaranteed_margin": guaranteed_margin,
                }
                starts_checked += 1
                if closest is None or guaranteed_margin < closest[
                    "guaranteed_margin"
                ]:
                    closest = entry
                if guaranteed_margin < minimum_margin:
                    failures.append(entry)
        finally:
            env.close()

    return {
        "passed": bool(
            starts_checked and not failures and not speed_mismatches
        ),
        "semantics": (
            "every reset remains outside the success radius after the "
            "largest possible one-step displacement, guaranteeing at least "
            "two recorded rows"
        ),
        "minimum_rows": minimum_rows,
        "minimum_guaranteed_margin": minimum_margin,
        "expected_agent_speed": expected_agent_speed,
        "observed_agent_speeds": sorted(observed_agent_speeds),
        "agent_speed_readback_passed": bool(
            observed_agent_speeds
            and (
                expected_agent_speed is None
                or not speed_mismatches
            )
        ),
        "agent_speed_mismatches": speed_mismatches,
        "starts_checked": starts_checked,
        "minimum_observed_margin": (
            None if closest is None else closest["guaranteed_margin"]
        ),
        "closest_start": closest,
        "failures": failures,
    }


def validate_atom_oracle_coverage(
    scenarios: list[CompiledScenario],
    registry: dict[str, Any],
    validation_config: dict[str, Any],
    oracle_runners: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if oracle_runners is None:
        oracle_runners = atom_oracle_runners()
    atom_kinds = sorted(
        {atom.kind for scenario in scenarios for atom in scenario.atoms}
    )
    contracts: dict[str, Any] = {}
    missing_config: dict[str, list[str]] = {}
    missing_implementation: dict[str, list[str]] = {}
    for atom_kind in atom_kinds:
        adapter = registry[atom_kind]
        required = list(adapter.required_oracles)
        absent_config = [
            name for name in required if name not in validation_config
        ]
        absent_implementation = [
            name for name in required if name not in oracle_runners
        ]
        contracts[atom_kind] = {
            "pixel_effect": adapter.pixel_effect,
            "required_evidence": list(
                PIXEL_EFFECT_CONTRACTS[adapter.pixel_effect]
            ),
            "required_oracles": required,
            "configured_oracles": [
                name for name in required if name not in absent_config
            ],
            "implemented_oracles": [
                name for name in required if name not in absent_implementation
            ],
        }
        if absent_config:
            missing_config[atom_kind] = absent_config
        if absent_implementation:
            missing_implementation[atom_kind] = absent_implementation
    missing = {
        atom_kind: sorted(
            set(missing_config.get(atom_kind, ()))
            | set(missing_implementation.get(atom_kind, ()))
        )
        for atom_kind in sorted(set(missing_config) | set(missing_implementation))
    }
    return {
        "passed": not missing,
        "contracts": contracts,
        "missing_oracles": missing,
        "missing_config": missing_config,
        "missing_implementation": missing_implementation,
    }


def validate_door_position_pixel_oracle(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate every pixel changed by moving a door in a fixed state.

    The expected mask is derived analytically from the pinned renderer. Border
    lines are rendered after the door, so an opening that reaches a border is
    correctly treated as occluded there rather than as a visible pixel change.
    """

    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    raw_cases = config.get("cases")
    if raw_cases is None:
        raw_cases = [
            {
                "first_position": config["first_position"],
                "second_position": config["second_position"],
            }
        ]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("door_position_pixel_oracle cases must be non-empty")

    door_size = int(config["door_half_size"])
    wall_thickness = int(config.get("wall_thickness", 10))
    state = np.asarray(config["state"], dtype=np.float32)
    target = np.asarray(config["target_state"], dtype=np.float32)
    seed = int(config["seed"])
    border_thickness = int(config.get("border_thickness", 4))
    case_reports: list[dict[str, Any]] = []

    for case_index, raw_case in enumerate(raw_cases):
        first_position = int(raw_case["first_position"])
        second_position = int(raw_case["second_position"])

        def make_options(position: int) -> dict[str, Any]:
            return {
                # Do not randomly sample fields before overwriting them. A
                # temporary sampled horizontal wall can invalidate the
                # default agent position before explicit values are applied.
                "variation": (),
                "variation_values": {
                    "door.position": np.asarray(
                        [position] * 3, dtype=np.int64
                    ),
                    "door.size": np.asarray(
                        [door_size] * 3, dtype=np.int64
                    ),
                    "door.number": 1,
                    "wall.axis": 1,
                    "wall.thickness": wall_thickness,
                },
                "state": state.copy(),
                "target_state": target.copy(),
            }

        first = TwoRoomEnv(render_mode="rgb_array")
        second = TwoRoomEnv(render_mode="rgb_array")
        try:
            first_observation, first_info = first.reset(
                seed=seed, options=make_options(first_position)
            )
            second_observation, second_info = second.reset(
                seed=seed, options=make_options(second_position)
            )
            first_readback = np.asarray(
                first.variation_space["door"]["position"].value
            ).copy()
            second_readback = np.asarray(
                second.variation_space["door"]["position"].value
            ).copy()
            door_color = np.asarray(
                first.variation_space["door"]["color"].value,
                dtype=np.uint8,
            )
            wall_color = np.asarray(
                first.variation_space["wall"]["color"].value,
                dtype=np.uint8,
            )
            first_pixel = first.render()
            second_pixel = second.render()
        finally:
            first.close()
            second.close()

        height, width = first_pixel.shape[:2]
        grid_y, grid_x = np.mgrid[:height, :width]
        half = wall_thickness // 2
        wall_stripe = (
            (grid_x >= TwoRoomEnv.WALL_CENTER - half)
            & (grid_x <= TwoRoomEnv.WALL_CENTER + half)
        )
        first_span = (
            (grid_y >= first_position - door_size)
            & (grid_y <= first_position + door_size)
        )
        second_span = (
            (grid_y >= second_position - door_size)
            & (grid_y <= second_position + door_size)
        )
        border_mask = np.zeros((height, width), dtype=bool)
        border_start = TwoRoomEnv.BORDER_SIZE - border_thickness
        border_end = TwoRoomEnv.BORDER_SIZE
        far_start = width - TwoRoomEnv.BORDER_SIZE
        far_end = far_start + border_thickness
        border_mask[:, border_start:border_end] = True
        border_mask[:, far_start:far_end] = True
        border_mask[border_start:border_end, :] = True
        border_mask[far_start:far_end, :] = True

        raw_change_mask = wall_stripe & (first_span ^ second_span)
        expected_change_mask = raw_change_mask & (~border_mask)
        actual_change_mask = np.any(first_pixel != second_pixel, axis=-1)
        first_only = (
            wall_stripe & first_span & (~second_span) & (~border_mask)
        )
        second_only = (
            wall_stripe & second_span & (~first_span) & (~border_mask)
        )

        state_unchanged = bool(
            np.array_equal(first_info["proprio"], second_info["proprio"])
            and np.array_equal(
                first_observation[:4], second_observation[:4]
            )
        )
        observation_readback = bool(
            first_observation[5] == first_position
            and second_observation[5] == second_position
        )
        simulator_readback = bool(
            np.array_equal(
                first_readback, np.asarray([first_position] * 3)
            )
            and np.array_equal(
                second_readback, np.asarray([second_position] * 3)
            )
        )
        factor_readback = observation_readback and simulator_readback
        change_mask_exact = np.array_equal(
            actual_change_mask, expected_change_mask
        )
        unchanged_pixels_exact = np.array_equal(
            first_pixel[~expected_change_mask],
            second_pixel[~expected_change_mask],
        )
        color_transition_exact = bool(
            np.all(first_pixel[first_only] == door_color)
            and np.all(second_pixel[first_only] == wall_color)
            and np.all(first_pixel[second_only] == wall_color)
            and np.all(second_pixel[second_only] == door_color)
        )
        passed = all(
            (
                state_unchanged,
                factor_readback,
                change_mask_exact,
                unchanged_pixels_exact,
                color_transition_exact,
            )
        )
        case_reports.append(
            {
                "case": case_index,
                "passed": passed,
                "first_position": first_position,
                "second_position": second_position,
                "door_half_size": door_size,
                "state_unchanged": state_unchanged,
                "observation_readback": observation_readback,
                "simulator_readback": simulator_readback,
                "factor_readback": factor_readback,
                "change_mask_exact": change_mask_exact,
                "unchanged_pixels_exact": unchanged_pixels_exact,
                "color_transition_exact": color_transition_exact,
                "raw_door_mask_changed_pixels": int(raw_change_mask.sum()),
                "border_occluded_pixels": int(
                    (raw_change_mask & border_mask).sum()
                ),
                "expected_changed_pixels": int(
                    expected_change_mask.sum()
                ),
                "actual_changed_pixels": int(actual_change_mask.sum()),
            }
        )

    evidence = {
        "factor_readback": all(
            case["factor_readback"] for case in case_reports
        ),
        "state_invariance": all(
            case["state_unchanged"] for case in case_reports
        ),
        "single_frame_pixel_semantics": all(
            case["change_mask_exact"] and case["color_transition_exact"]
            for case in case_reports
        ),
        "unchanged_pixels": all(
            case["unchanged_pixels_exact"] for case in case_reports
        ),
    }
    report: dict[str, Any] = {
        "passed": bool(
            case_reports
            and all(case["passed"] for case in case_reports)
            and all(evidence.values())
        ),
        "semantics": (
            "moving door.position changes exactly the visible old/new "
            "opening mask in one rendered frame; border-overdraw pixels and "
            "every unrelated pixel remain unchanged"
        ),
        "door_half_size": door_size,
        "wall_thickness": wall_thickness,
        "border_thickness": border_thickness,
        "cases": case_reports,
        "total_expected_changed_pixels": sum(
            case["expected_changed_pixels"] for case in case_reports
        ),
        "total_actual_changed_pixels": sum(
            case["actual_changed_pixels"] for case in case_reports
        ),
        "evidence": evidence,
    }
    if len(case_reports) == 1:
        report.update(
            {
                key: case_reports[0][key]
                for key in (
                    "first_position",
                    "second_position",
                    "state_unchanged",
                    "observation_readback",
                    "simulator_readback",
                    "factor_readback",
                    "change_mask_exact",
                    "unchanged_pixels_exact",
                    "color_transition_exact",
                    "expected_changed_pixels",
                    "actual_changed_pixels",
                )
            }
        )
    return report


def validate_door_position_passage_oracle(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Prove that door.position changes collision and its rendered trajectory."""

    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    speed = float(config.get("speed", 5.0))
    door_size = int(config["door_half_size"])
    wall_thickness = int(config.get("wall_thickness", 10))
    agent_radius = float(config.get("agent_radius", 7.0))
    door_margin = float(config.get("door_collision_margin", 1.75))
    steps = int(config.get("steps", 8))
    seed = int(config["seed"])
    if steps < 3:
        raise ValueError("door_position_passage_oracle requires steps >= 3")

    half = wall_thickness // 2
    effective_left = (
        float(TwoRoomEnv.WALL_CENTER) - half - agent_radius
    )
    effective_right = (
        float(TwoRoomEnv.WALL_CENTER) + half + agent_radius
    )
    lower_border = float(TwoRoomEnv.BORDER_SIZE) + agent_radius
    upper_border = (
        float(TwoRoomEnv.IMG_SIZE)
        - float(TwoRoomEnv.BORDER_SIZE)
        - agent_radius
    )

    def make_options(
        door_position: int,
        state: np.ndarray,
        target: np.ndarray,
    ) -> dict[str, Any]:
        return {
            # Controlled oracle: set every relevant field directly instead
            # of first sampling a transient, potentially invalid geometry.
            "variation": (),
            "variation_values": {
                "agent.speed": np.asarray([speed], dtype=np.float32),
                "agent.radius": np.asarray(
                    [agent_radius], dtype=np.float32
                ),
                "door.position": np.asarray(
                    [door_position] * 3, dtype=np.int64
                ),
                "door.size": np.asarray(
                    [door_size] * 3, dtype=np.int64
                ),
                "door.number": 1,
                "wall.axis": 1,
                "wall.thickness": wall_thickness,
            },
            "state": state.copy(),
            "target_state": target.copy(),
        }

    def in_door(y_value: float, door_position: int) -> bool:
        return bool(
            door_position - door_size - door_margin
            <= y_value
            <= door_position + door_size + door_margin
        )

    def analytic_trajectory(
        state: np.ndarray,
        action: np.ndarray,
        door_position: int,
    ) -> np.ndarray:
        position = state.astype(np.float32, copy=True)
        trajectory: list[np.ndarray] = []
        displacement = (
            action.astype(np.float32) * np.float32(speed)
        ).astype(np.float32)
        for _ in range(steps):
            candidate = (position + displacement).astype(np.float32)
            candidate[0] = np.float32(
                min(max(float(candidate[0]), lower_border), upper_border)
            )
            candidate[1] = np.float32(
                min(max(float(candidate[1]), lower_border), upper_border)
            )
            started_left = float(position[0]) < TwoRoomEnv.WALL_CENTER
            if (
                started_left
                and float(candidate[0]) > effective_left
                and not in_door(float(candidate[1]), door_position)
            ):
                candidate[0] = np.float32(effective_left - 0.5)
            elif (
                not started_left
                and float(candidate[0]) < effective_right
                and not in_door(float(candidate[1]), door_position)
            ):
                candidate[0] = np.float32(effective_right + 0.5)
            position = candidate
            trajectory.append(position.copy())
        return np.stack(trajectory)

    case_reports: list[dict[str, Any]] = []
    raw_cases = config.get("cases", [])
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("door_position_passage_oracle cases must be non-empty")

    for case_index, case in enumerate(raw_cases):
        state = np.asarray(case["state"], dtype=np.float32)
        target = np.asarray(case["target_state"], dtype=np.float32)
        action = np.asarray(case["action"], dtype=np.float32)
        open_position = int(case["open_position"])
        blocked_position = int(case["blocked_position"])
        if not (
            action.shape == (2,)
            and abs(float(action[0])) == 1.0
            and float(action[1]) == 0.0
        ):
            raise ValueError(
                "Passage cases require cardinal horizontal action [±1, 0]"
            )
        if not in_door(float(state[1]), open_position):
            raise ValueError("open_position does not cover the passage y value")
        if in_door(float(state[1]), blocked_position):
            raise ValueError(
                "blocked_position unexpectedly covers the passage y value"
            )

        open_expected = analytic_trajectory(
            state, action, open_position
        )
        blocked_expected = analytic_trajectory(
            state, action, blocked_position
        )
        differing_steps = np.flatnonzero(
            np.any(open_expected != blocked_expected, axis=1)
        )
        if not len(differing_steps):
            raise ValueError("Passage case never reaches the central wall")
        contact_index = int(differing_steps[0])

        opened = TwoRoomEnv(render_mode="rgb_array")
        blocked = TwoRoomEnv(render_mode="rgb_array")
        try:
            open_initial_observation, open_initial_info = opened.reset(
                seed=seed,
                options=make_options(open_position, state, target),
            )
            blocked_initial_observation, blocked_initial_info = blocked.reset(
                seed=seed,
                options=make_options(blocked_position, state, target),
            )
            open_speed_readback = float(
                opened.variation_space["agent"]["speed"].value.item()
            )
            blocked_speed_readback = float(
                blocked.variation_space["agent"]["speed"].value.item()
            )
            open_door_readback = np.asarray(
                opened.variation_space["door"]["position"].value
            ).copy()
            blocked_door_readback = np.asarray(
                blocked.variation_space["door"]["position"].value
            ).copy()
            open_pixels = [opened.render()]
            blocked_pixels = [blocked.render()]
            open_states: list[np.ndarray] = []
            blocked_states: list[np.ndarray] = []
            ended = False
            for _ in range(steps):
                open_observation, _, terminated, truncated, _ = opened.step(
                    action
                )
                blocked_observation, _, blocked_terminated, blocked_truncated, _ = (
                    blocked.step(action)
                )
                open_states.append(
                    np.asarray(
                        open_observation[:2], dtype=np.float32
                    ).copy()
                )
                blocked_states.append(
                    np.asarray(
                        blocked_observation[:2], dtype=np.float32
                    ).copy()
                )
                open_pixels.append(opened.render())
                blocked_pixels.append(blocked.render())
                ended = ended or any(
                    (
                        terminated,
                        truncated,
                        blocked_terminated,
                        blocked_truncated,
                    )
                )
        finally:
            opened.close()
            blocked.close()

        open_actual = np.stack(open_states)
        blocked_actual = np.stack(blocked_states)
        initial_state_equal = bool(
            np.array_equal(
                open_initial_info["proprio"],
                blocked_initial_info["proprio"],
            )
            and np.array_equal(
                open_initial_observation[:4],
                blocked_initial_observation[:4],
            )
        )
        factor_readback = bool(
            np.isclose(open_speed_readback, speed, atol=1e-7)
            and np.isclose(blocked_speed_readback, speed, atol=1e-7)
            and np.array_equal(
                open_door_readback,
                np.asarray([open_position] * 3),
            )
            and np.array_equal(
                blocked_door_readback,
                np.asarray([blocked_position] * 3),
            )
            and open_initial_observation[5] == open_position
            and blocked_initial_observation[5] == blocked_position
        )
        state_trajectory_exact = bool(
            np.array_equal(open_actual, open_expected)
            and np.array_equal(blocked_actual, blocked_expected)
        )
        same_until_contact = bool(
            contact_index == 0
            or np.array_equal(
                open_actual[:contact_index],
                blocked_actual[:contact_index],
            )
        )
        diverges_at_contact = bool(
            not np.array_equal(
                open_actual[contact_index],
                blocked_actual[contact_index],
            )
        )
        direction = float(action[0])
        if direction > 0:
            open_crossed_wall = bool(
                open_actual[-1, 0] > effective_right
            )
            blocked_at_wall = bool(
                blocked_actual[-1, 0]
                == np.float32(effective_left - 0.5)
            )
        else:
            open_crossed_wall = bool(
                open_actual[-1, 0] < effective_left
            )
            blocked_at_wall = bool(
                blocked_actual[-1, 0]
                == np.float32(effective_right + 0.5)
            )

        height, width = open_pixels[0].shape[:2]
        _, grid_x = np.mgrid[:height, :width]
        wall_stripe = (
            grid_x >= TwoRoomEnv.WALL_CENTER - half
        ) & (grid_x <= TwoRoomEnv.WALL_CENTER + half)
        outside_wall = ~wall_stripe
        outside_pixels_equal_before_contact = all(
            np.array_equal(
                open_pixels[index + 1][outside_wall],
                blocked_pixels[index + 1][outside_wall],
            )
            for index in range(contact_index)
        )
        outside_pixels_diverge_at_contact = bool(
            not np.array_equal(
                open_pixels[contact_index + 1][outside_wall],
                blocked_pixels[contact_index + 1][outside_wall],
            )
        )

        def state_pixel_alignment(
            initial: np.ndarray,
            states: np.ndarray,
            pixels: list[np.ndarray],
        ) -> bool:
            all_states = np.concatenate((initial[None, :], states), axis=0)
            return all(
                bool(np.array_equal(all_states[index], all_states[index + 1]))
                == bool(np.array_equal(pixels[index], pixels[index + 1]))
                for index in range(steps)
            )

        state_pixel_transition_aligned = bool(
            state_pixel_alignment(state, open_actual, open_pixels)
            and state_pixel_alignment(state, blocked_actual, blocked_pixels)
        )

        final_pixels_reproducible = True
        for position, door_position, actual_pixel in (
            (open_expected[-1], open_position, open_pixels[-1]),
            (blocked_expected[-1], blocked_position, blocked_pixels[-1]),
        ):
            reference = TwoRoomEnv(render_mode="rgb_array")
            try:
                reference.reset(
                    seed=seed,
                    options=make_options(
                        door_position, position, target
                    ),
                )
                final_pixels_reproducible = bool(
                    final_pixels_reproducible
                    and np.array_equal(reference.render(), actual_pixel)
                )
            finally:
                reference.close()

        no_early_end = not ended
        passed = all(
            (
                initial_state_equal,
                factor_readback,
                state_trajectory_exact,
                same_until_contact,
                diverges_at_contact,
                open_crossed_wall,
                blocked_at_wall,
                outside_pixels_equal_before_contact,
                outside_pixels_diverge_at_contact,
                state_pixel_transition_aligned,
                final_pixels_reproducible,
                no_early_end,
            )
        )
        case_reports.append(
            {
                "case": case_index,
                "passed": passed,
                "open_position": open_position,
                "blocked_position": blocked_position,
                "state": state.tolist(),
                "target_state": target.tolist(),
                "action": action.tolist(),
                "steps": steps,
                "contact_step": contact_index + 1,
                "initial_state_equal": initial_state_equal,
                "factor_readback": factor_readback,
                "state_trajectory_exact": state_trajectory_exact,
                "same_until_contact": same_until_contact,
                "diverges_at_contact": diverges_at_contact,
                "open_crossed_wall": open_crossed_wall,
                "blocked_at_wall": blocked_at_wall,
                "outside_wall_pixels_equal_before_contact": (
                    outside_pixels_equal_before_contact
                ),
                "outside_wall_pixels_diverge_at_contact": (
                    outside_pixels_diverge_at_contact
                ),
                "state_pixel_transition_aligned": (
                    state_pixel_transition_aligned
                ),
                "final_pixels_reproducible": final_pixels_reproducible,
                "no_early_termination": no_early_end,
                "expected_open_trajectory": open_expected.tolist(),
                "actual_open_trajectory": open_actual.tolist(),
                "expected_blocked_trajectory": blocked_expected.tolist(),
                "actual_blocked_trajectory": blocked_actual.tolist(),
            }
        )

    evidence = {
        "factor_readback": all(
            case["factor_readback"] for case in case_reports
        ),
        "passage_state_transition": all(
            case["state_trajectory_exact"]
            and case["open_crossed_wall"]
            and case["blocked_at_wall"]
            for case in case_reports
        ),
        "passage_pixel_transition": all(
            case["state_pixel_transition_aligned"]
            and case["outside_wall_pixels_diverge_at_contact"]
            and case["final_pixels_reproducible"]
            for case in case_reports
        ),
        "contact_semantics": all(
            case["same_until_contact"]
            and case["diverges_at_contact"]
            and case["no_early_termination"]
            for case in case_reports
        ),
    }
    return {
        "passed": bool(
            case_reports
            and all(case["passed"] for case in case_reports)
            and all(evidence.values())
        ),
        "semantics": (
            "at identical initial state and actions, an aligned door permits "
            "the agent to cross while a displaced door clamps it at the "
            "analytically expected wall boundary; rendered motion follows "
            "the state trajectory exactly"
        ),
        "speed": speed,
        "door_half_size": door_size,
        "wall_thickness": wall_thickness,
        "agent_radius": agent_radius,
        "cases": case_reports,
        "evidence": evidence,
    }


def validate_speed_frame_skip_oracle(config: dict[str, Any]) -> dict[str, Any]:
    """Prove the integer speed-ratio/frame-skip equivalence in raw TwoRoom.

    With a collision-free reset and one open-loop action repeated ``m`` times,
    ``speed=s`` for ``m`` steps must end at exactly the same rendered frame as
    ``speed=m*s`` for one step. This is deliberately separate from expert
    trajectories, whose closed-loop action can change at every intermediate
    frame.
    """

    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    slow_speed = float(config["slow_speed"])
    multiplier = int(config["multiplier"])
    if multiplier < 2 or slow_speed * multiplier > 10.5:
        raise ValueError(
            "speed_frame_skip_oracle requires integer multiplier >= 2 and "
            "fast speed <= 10.5"
        )
    fast_speed = slow_speed * multiplier
    case_reports: list[dict[str, Any]] = []

    for case_index, case in enumerate(config["cases"]):
        state = np.asarray(case["state"], dtype=np.float32)
        target = np.asarray(case["target_state"], dtype=np.float32)
        action = np.clip(
            np.asarray(case["action"], dtype=np.float32), -1.0, 1.0
        )

        slow = TwoRoomEnv(render_mode="rgb_array")
        fast = TwoRoomEnv(render_mode="rgb_array")

        def options(speed: float) -> dict[str, Any]:
            return {
                "variation": ("agent.speed",),
                "variation_values": {
                    "agent.speed": np.asarray([speed], dtype=np.float32)
                },
                "state": state.copy(),
                "target_state": target.copy(),
            }

        try:
            slow.reset(seed=int(config["seed"]), options=options(slow_speed))
            fast.reset(seed=int(config["seed"]), options=options(fast_speed))
            slow_speed_readback = float(
                slow.variation_space["agent"]["speed"].value.item()
            )
            fast_speed_readback = float(
                fast.variation_space["agent"]["speed"].value.item()
            )
            slow_initial_pixel = slow.render()
            fast_initial_pixel = fast.render()

            slow_intermediate_pixels: list[np.ndarray] = []
            slow_terminated = False
            for _ in range(multiplier):
                slow_observation, _, terminated, truncated, _ = slow.step(action)
                slow_terminated = slow_terminated or terminated or truncated
                slow_intermediate_pixels.append(slow.render())

            fast_observation, _, fast_terminated, fast_truncated, _ = fast.step(
                action
            )
            fast_final_pixel = fast.render()
        finally:
            slow.close()
            fast.close()

        expected_position = state + slow_speed * multiplier * action
        initial_pixels_equal = np.array_equal(
            slow_initial_pixel, fast_initial_pixel
        )
        final_states_equal = np.array_equal(
            slow_observation[:2], fast_observation[:2]
        )
        expected_state_equal = np.allclose(
            slow_observation[:2], expected_position, atol=1e-6
        ) and np.allclose(fast_observation[:2], expected_position, atol=1e-6)
        final_pixels_equal = np.array_equal(
            slow_intermediate_pixels[-1], fast_final_pixel
        )
        middle_pixel_differs = any(
            not np.array_equal(pixel, fast_final_pixel)
            for pixel in slow_intermediate_pixels[:-1]
        )
        no_early_end = not (
            slow_terminated or fast_terminated or fast_truncated
        )
        factor_readback = bool(
            np.isclose(slow_speed_readback, slow_speed, atol=1e-7)
            and np.isclose(fast_speed_readback, fast_speed, atol=1e-7)
        )
        passed = all(
            (
                factor_readback,
                initial_pixels_equal,
                final_states_equal,
                expected_state_equal,
                final_pixels_equal,
                middle_pixel_differs,
                no_early_end,
            )
        )
        case_reports.append(
            {
                "case": case_index,
                "passed": passed,
                "state": state.tolist(),
                "target_state": target.tolist(),
                "action": action.tolist(),
                "expected_final_position": expected_position.tolist(),
                "slow_final_position": slow_observation[:2].tolist(),
                "fast_final_position": fast_observation[:2].tolist(),
                "slow_speed_readback": slow_speed_readback,
                "fast_speed_readback": fast_speed_readback,
                "factor_readback": factor_readback,
                "initial_pixels_equal": initial_pixels_equal,
                "final_states_equal": final_states_equal,
                "expected_state_equal": bool(expected_state_equal),
                "final_pixels_equal": final_pixels_equal,
                "middle_pixel_differs": middle_pixel_differs,
                "middle_to_final_mean_abs_pixel_difference": [
                    float(
                        np.mean(
                            np.abs(
                                pixel.astype(np.float32)
                                - fast_final_pixel.astype(np.float32)
                            )
                        )
                    )
                    for pixel in slow_intermediate_pixels[:-1]
                ],
                "no_collision_or_early_termination": no_early_end,
            }
        )

    evidence = {
        "factor_readback": all(
            case["factor_readback"] for case in case_reports
        ),
        "state_transition": all(
            case["final_states_equal"]
            and case["expected_state_equal"]
            and case["no_collision_or_early_termination"]
            for case in case_reports
        ),
        "pixel_transition": all(
            case["initial_pixels_equal"]
            and case["final_pixels_equal"]
            and case["middle_pixel_differs"]
            for case in case_reports
        ),
        "temporal_alignment": all(
            case["final_states_equal"]
            and case["final_pixels_equal"]
            and case["no_collision_or_early_termination"]
            for case in case_reports
        ),
    }
    return {
        "passed": bool(
            case_reports
            and all(case["passed"] for case in case_reports)
            and all(evidence.values())
        ),
        "semantics": (
            "fast one-step frame equals slow repeated-action frame with "
            "intermediate slow frames removed"
        ),
        "slow_speed": slow_speed,
        "fast_speed": fast_speed,
        "multiplier": multiplier,
        "cases": case_reports,
        "evidence": evidence,
    }


def validate_action_delay_temporal_oracle(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Prove the hidden action-delay semantics used by the History=3 task.

    Every delay receives the same reset, five probe commands, five zero
    commands, and five query commands.  The probe result exposes the delay.
    The zero block then flushes the command queue so that every delay reaches
    the exact same query state and pixels with an all-zero pending queue.
    Repeating the probe command from that common query produces one distinct
    true future for each delay.
    """

    from contextworld.evaluation.action_delay_env import (
        ACTION_DELAY_FACTOR,
        make_action_delay_env,
    )

    delays = tuple(int(value) for value in config.get("delays", range(5)))
    if (
        not delays
        or len(set(delays)) != len(delays)
        or any(value < 0 or value > 4 for value in delays)
    ):
        raise ValueError(
            "action_delay_temporal_oracle delays must be unique integers "
            "within [0, 4]"
        )
    block_steps = int(config.get("raw_steps_per_action_block", 5))
    if block_steps <= max(delays):
        raise ValueError(
            "raw_steps_per_action_block must be larger than every delay"
        )
    speed = float(config.get("agent_speed", 7.0))
    seed = int(config.get("seed", 20260726))
    raw_cases = config.get("cases")
    if raw_cases is None:
        raw_cases = (
            {
                "name": "up",
                "state": [50.0, 30.0],
                "target_state": [205.0, 205.0],
                "action": [0.0, 1.0],
            },
            {
                "name": "down",
                "state": [75.0, 194.0],
                "target_state": [205.0, 205.0],
                "action": [0.0, -1.0],
            },
        )

    def as_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value).copy()

    def all_equal(values: list[np.ndarray]) -> bool:
        return bool(
            values
            and all(np.array_equal(values[0], value) for value in values[1:])
        )

    def all_pairwise_distinct(values: list[np.ndarray]) -> bool:
        return all(
            not np.array_equal(left, right)
            for left_index, left in enumerate(values)
            for right in values[left_index + 1 :]
        )

    def reset_options(
        delay: int,
        state: np.ndarray,
        target: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "variation": (),
            "variation_values": {
                "agent.speed": np.asarray([speed], dtype=np.float32),
                ACTION_DELAY_FACTOR: delay,
            },
            "state": state.copy(),
            "target_state": target.copy(),
        }

    def reference_pixel(
        state: np.ndarray,
        target: np.ndarray,
        case_seed: int,
    ) -> np.ndarray:
        reference = make_action_delay_env(render_mode="rgb_array")
        try:
            reference.reset(
                seed=case_seed,
                options=reset_options(0, state, target),
            )
            return reference.render().copy()
        finally:
            reference.close()

    case_reports: list[dict[str, Any]] = []
    for case_index, raw_case in enumerate(raw_cases):
        case_seed = seed + case_index
        state = np.asarray(raw_case["state"], dtype=np.float32)
        target = np.asarray(raw_case["target_state"], dtype=np.float32)
        action = np.clip(
            np.asarray(raw_case["action"], dtype=np.float32),
            -1.0,
            1.0,
        )
        zero = np.zeros_like(action)
        command_trace = (
            [action] * block_steps
            + [zero] * block_steps
            + [action] * block_steps
        )
        rollouts: list[dict[str, Any]] = []

        for delay in delays:
            env = make_action_delay_env(render_mode="rgb_array")
            try:
                initial_observation, _ = env.reset(
                    seed=case_seed,
                    options=reset_options(delay, state, target),
                )
                initial_pixel = env.render().copy()
                factor_readback = bool(
                    env.action_delay_steps == delay
                    and env._contextworld_action_delay_readback == delay
                )
                observations: list[np.ndarray] = []
                pixels: list[np.ndarray] = []
                executed_actions: list[np.ndarray] = []
                pending_at_query: np.ndarray | None = None
                ended = False
                query_index = 2 * block_steps - 1
                for command_index, command in enumerate(command_trace):
                    observation, _, terminated, truncated, info = env.step(
                        command
                    )
                    observations.append(as_numpy(observation))
                    pixels.append(env.render().copy())
                    executed_actions.append(
                        np.asarray(
                            info["contextworld.executed_action"],
                            dtype=np.float32,
                        ).copy()
                    )
                    ended = ended or terminated or truncated
                    if command_index == query_index:
                        pending_at_query = env.pending_actions()

                middle_index = block_steps - 1
                future_index = 3 * block_steps - 1
                expected_executed = [
                    (
                        zero
                        if command_index < delay
                        else command_trace[command_index - delay]
                    )
                    for command_index in range(len(command_trace))
                ]
                executed_trace_exact = all(
                    np.array_equal(observed, expected)
                    for observed, expected in zip(
                        executed_actions, expected_executed
                    )
                )
                expected_middle = (
                    state + speed * (block_steps - delay) * action
                )
                expected_query = state + speed * block_steps * action
                expected_future = (
                    expected_query
                    + speed * (block_steps - delay) * action
                )
                phase_states = (
                    state,
                    observations[middle_index][:2],
                    observations[query_index][:2],
                    observations[future_index][:2],
                )
                phase_pixels = (
                    initial_pixel,
                    pixels[middle_index],
                    pixels[query_index],
                    pixels[future_index],
                )
                pixel_state_alignment = all(
                    np.array_equal(
                        pixel,
                        reference_pixel(
                            np.asarray(phase_state, dtype=np.float32),
                            target,
                            case_seed,
                        ),
                    )
                    for phase_state, pixel in zip(
                        phase_states, phase_pixels
                    )
                )
                pending_queue_zero = bool(
                    pending_at_query is not None
                    and pending_at_query.shape == (delay, 2)
                    and np.array_equal(
                        pending_at_query,
                        np.zeros((delay, 2), dtype=np.float32),
                    )
                )
            finally:
                env.close()

            state_trajectory_exact = bool(
                np.allclose(
                    observations[middle_index][:2],
                    expected_middle,
                    atol=1e-6,
                )
                and np.allclose(
                    observations[query_index][:2],
                    expected_query,
                    atol=1e-6,
                )
                and np.allclose(
                    observations[future_index][:2],
                    expected_future,
                    atol=1e-6,
                )
            )
            rollouts.append(
                {
                    "delay_steps": delay,
                    "factor_readback": factor_readback,
                    "initial_observation": as_numpy(
                        initial_observation
                    ),
                    "initial_pixel": initial_pixel,
                    "middle_state": observations[middle_index][:2],
                    "middle_pixel": pixels[middle_index],
                    "query_observation": observations[query_index],
                    "query_pixel": pixels[query_index],
                    "future_state": observations[future_index][:2],
                    "future_pixel": pixels[future_index],
                    "expected_middle_state": expected_middle,
                    "expected_query_state": expected_query,
                    "expected_future_state": expected_future,
                    "state_trajectory_exact": state_trajectory_exact,
                    "executed_trace_exact": executed_trace_exact,
                    "pending_queue_zero_at_query": pending_queue_zero,
                    "pixel_state_alignment": pixel_state_alignment,
                    "no_collision_or_early_termination": not ended,
                }
            )

        initial_observations = [
            rollout["initial_observation"] for rollout in rollouts
        ]
        initial_pixels = [
            rollout["initial_pixel"] for rollout in rollouts
        ]
        middle_states = [
            rollout["middle_state"] for rollout in rollouts
        ]
        middle_pixels = [
            rollout["middle_pixel"] for rollout in rollouts
        ]
        query_observations = [
            rollout["query_observation"] for rollout in rollouts
        ]
        query_pixels = [
            rollout["query_pixel"] for rollout in rollouts
        ]
        future_states = [
            rollout["future_state"] for rollout in rollouts
        ]
        future_pixels = [
            rollout["future_pixel"] for rollout in rollouts
        ]

        reset_hidden = bool(
            all_equal(initial_observations) and all_equal(initial_pixels)
        )
        history_distinguishes_delays = bool(
            all_pairwise_distinct(middle_states)
            and all_pairwise_distinct(middle_pixels)
        )
        query_exactly_matched = bool(
            all_equal(query_observations) and all_equal(query_pixels)
        )
        futures_distinguish_delays = bool(
            all_pairwise_distinct(future_states)
            and all_pairwise_distinct(future_pixels)
        )
        rollout_checks_passed = all(
            rollout["factor_readback"]
            and rollout["state_trajectory_exact"]
            and rollout["executed_trace_exact"]
            and rollout["pending_queue_zero_at_query"]
            and rollout["pixel_state_alignment"]
            and rollout["no_collision_or_early_termination"]
            for rollout in rollouts
        )
        passed = bool(
            rollout_checks_passed
            and reset_hidden
            and history_distinguishes_delays
            and query_exactly_matched
            and futures_distinguish_delays
        )
        case_reports.append(
            {
                "case": case_index,
                "name": raw_case.get("name", f"case_{case_index}"),
                "passed": passed,
                "state": state.tolist(),
                "target_state": target.tolist(),
                "action": action.tolist(),
                "reset_observation_and_pixels_equal": reset_hidden,
                "history_midpoints_distinguish_delays": (
                    history_distinguishes_delays
                ),
                "query_observation_and_pixels_exactly_equal": (
                    query_exactly_matched
                ),
                "true_futures_distinguish_delays": (
                    futures_distinguish_delays
                ),
                "rollouts": [
                    {
                        key: (
                            value.tolist()
                            if isinstance(value, np.ndarray)
                            else value
                        )
                        for key, value in rollout.items()
                        if key
                        not in {
                            "initial_observation",
                            "initial_pixel",
                            "middle_pixel",
                            "query_observation",
                            "query_pixel",
                            "future_pixel",
                        }
                    }
                    for rollout in rollouts
                ],
            }
        )

    evidence = {
        "factor_readback": all(
            rollout["factor_readback"]
            for case in case_reports
            for rollout in case["rollouts"]
        ),
        "state_transition": all(
            case["history_midpoints_distinguish_delays"]
            and case["query_observation_and_pixels_exactly_equal"]
            and case["true_futures_distinguish_delays"]
            and all(
                rollout["state_trajectory_exact"]
                and rollout["no_collision_or_early_termination"]
                for rollout in case["rollouts"]
            )
            for case in case_reports
        ),
        "pixel_transition": all(
            case["reset_observation_and_pixels_equal"]
            and case["history_midpoints_distinguish_delays"]
            and case["query_observation_and_pixels_exactly_equal"]
            and case["true_futures_distinguish_delays"]
            and all(
                rollout["pixel_state_alignment"]
                for rollout in case["rollouts"]
            )
            for case in case_reports
        ),
        "temporal_alignment": all(
            all(
                rollout["executed_trace_exact"]
                and rollout["pending_queue_zero_at_query"]
                for rollout in case["rollouts"]
            )
            for case in case_reports
        ),
    }
    return {
        "passed": bool(
            case_reports
            and all(case["passed"] for case in case_reports)
            and all(evidence.values())
        ),
        "semantics": (
            "the history midpoint identifies the hidden command delay; a "
            "zero-action flush makes the query identical across delays; the "
            "same query action then yields one distinct true future per delay"
        ),
        "delays": list(delays),
        "agent_speed": speed,
        "raw_steps_per_action_block": block_steps,
        "cases": case_reports,
        "evidence": evidence,
    }


def atom_oracle_runners() -> dict[str, Any]:
    """Return every executable atom oracle known to this generator."""

    return {
        "action_delay_temporal_oracle": (
            validate_action_delay_temporal_oracle
        ),
        "speed_frame_skip_oracle": validate_speed_frame_skip_oracle,
        "door_position_pixel_oracle": validate_door_position_pixel_oracle,
        "door_position_passage_oracle": (
            validate_door_position_passage_oracle
        ),
    }


def run_required_atom_oracles(
    scenarios: list[CompiledScenario],
    registry: dict[str, Any],
    validation_config: dict[str, Any],
    oracle_runners: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless every used atom has executable, sufficient evidence."""

    if oracle_runners is None:
        oracle_runners = atom_oracle_runners()
    coverage = validate_atom_oracle_coverage(
        scenarios,
        registry,
        validation_config,
        oracle_runners=oracle_runners,
    )
    atom_kinds = sorted(
        {atom.kind for scenario in scenarios for atom in scenario.atoms}
    )
    required_oracles = sorted(
        {
            oracle
            for atom_kind in atom_kinds
            for oracle in registry[atom_kind].required_oracles
        }
    )
    checks: dict[str, Any] = {}
    for oracle_name in required_oracles:
        if (
            oracle_name not in validation_config
            or oracle_name not in oracle_runners
        ):
            continue
        try:
            result = oracle_runners[oracle_name](
                validation_config[oracle_name]
            )
            if not isinstance(result, dict):
                raise TypeError("oracle must return a dictionary report")
            checks[oracle_name] = result
        except Exception as exc:
            checks[oracle_name] = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "evidence": {},
            }

    atom_contracts: dict[str, Any] = {}
    for atom_kind in atom_kinds:
        adapter = registry[atom_kind]
        required_evidence = set(
            PIXEL_EFFECT_CONTRACTS[adapter.pixel_effect]
        )
        observed_evidence: dict[str, bool] = {}
        oracle_results: dict[str, bool] = {}
        for oracle_name in adapter.required_oracles:
            oracle_result = checks.get(oracle_name)
            oracle_results[oracle_name] = bool(
                oracle_result and oracle_result.get("passed", False)
            )
            if not oracle_result:
                continue
            for evidence_name, evidence_passed in oracle_result.get(
                "evidence", {}
            ).items():
                observed_evidence[evidence_name] = bool(evidence_passed) and (
                    observed_evidence.get(evidence_name, True)
                )

        missing_evidence = sorted(required_evidence - observed_evidence.keys())
        failed_evidence = sorted(
            name
            for name in required_evidence
            if name in observed_evidence and not observed_evidence[name]
        )
        atom_contracts[atom_kind] = {
            "passed": bool(
                all(oracle_results.values())
                and not missing_evidence
                and not failed_evidence
            ),
            "pixel_effect": adapter.pixel_effect,
            "required_evidence": sorted(required_evidence),
            "observed_evidence": dict(sorted(observed_evidence.items())),
            "missing_evidence": missing_evidence,
            "failed_evidence": failed_evidence,
            "oracle_results": oracle_results,
        }

    return {
        "passed": bool(
            coverage["passed"]
            and atom_contracts
            and all(item["passed"] for item in atom_contracts.values())
        ),
        "coverage": coverage,
        "checks": checks,
        "atom_contracts": atom_contracts,
    }


def validate_exact_tworoom_replay(
    scenario: CompiledScenario,
    dataset: Any,
) -> dict[str, Any]:
    """Replay every transition and reproduce the declared pixel encoding."""

    import io
    from PIL import Image
    from .environments import make_raw_contextworld_environment
    from .lance import encode_frame

    required = {
        "pixels",
        "proprio",
        "action",
        "goal_state",
        "terminated",
        "truncated",
    }
    missing = sorted(required - set(dataset.column_names))
    if missing:
        return {
            "passed": False,
            "errors": [f"Missing exact-replay columns: {missing}"],
        }

    pixels = dataset.get_col_data("pixels")
    proprio = dataset.get_col_data("proprio")
    action = dataset.get_col_data("action")
    goal_state = dataset.get_col_data("goal_state")
    stored_terminated = dataset.get_col_data("terminated").reshape(-1)
    stored_truncated = dataset.get_col_data("truncated").reshape(-1)

    rows_checked = 0
    transitions_checked = 0
    pixel_mismatches = 0
    decoded_pixel_mismatches = 0
    state_mismatches = 0
    goal_mismatches = 0
    termination_mismatches = 0
    maximum_state_error = 0.0
    failure_examples: list[dict[str, Any]] = []

    def add_failure(kind: str, episode_index: int, row_index: int) -> None:
        if len(failure_examples) < 20:
            failure_examples.append(
                {
                    "kind": kind,
                    "episode_index": episode_index,
                    "row_index": row_index,
                }
            )

    def rendered_frame(env: Any) -> np.ndarray:
        frame = env.render()
        if tuple(frame.shape[:2]) != tuple(scenario.image_shape):
            height, width = scenario.image_shape
            frame = np.asarray(
                Image.fromarray(frame).resize(
                    (width, height), resample=Image.BILINEAR
                )
            )
        return np.asarray(frame, dtype=np.uint8)

    def pixel_checks(blob: bytes, env: Any) -> tuple[bool, bool]:
        frame = rendered_frame(env)
        encoded_match = encode_frame(frame, scenario.pixel_codec) == blob
        if not scenario.pixel_codec.get("lossless", False):
            return encoded_match, True
        with Image.open(io.BytesIO(blob)) as image:
            decoded = np.asarray(image.convert("RGB"))
        return encoded_match, bool(np.array_equal(decoded, frame))

    env = make_raw_contextworld_environment(
        scenario.env_id,
        render_mode="rgb_array",
    )
    try:
        apply_tworoom_reset_constraints(env, scenario.reset_constraints)
        for episode_index, (offset, length) in enumerate(
            zip(dataset.offsets, dataset.lengths)
        ):
            start = int(offset)
            stop = start + int(length)
            if stop <= start:
                continue
            episode_goal = goal_state[start]
            if not np.all(goal_state[start:stop] == episode_goal):
                goal_mismatches += 1
                add_failure("goal_state_not_constant", episode_index, 0)

            env.reset(
                seed=scenario.env_seed + episode_index,
                options={
                    "variation": scenario.variation,
                    "variation_values": scenario.variation_values,
                    "state": proprio[start].copy(),
                    "target_state": episode_goal.copy(),
                },
            )
            rows_checked += 1
            encoded_match, decoded_match = pixel_checks(pixels[start], env)
            if not encoded_match:
                pixel_mismatches += 1
                add_failure("pixel_bytes", episode_index, 0)
            if not decoded_match:
                decoded_pixel_mismatches += 1
                add_failure("decoded_lossless_pixels", episode_index, 0)

            for global_index in range(start, stop - 1):
                local_next = global_index - start + 1
                observation, _, terminated, truncated, _ = env.step(
                    action[global_index]
                )
                observed_state = np.asarray(observation[:2])
                expected_state = proprio[global_index + 1]
                state_error = float(
                    np.max(np.abs(observed_state - expected_state))
                )
                maximum_state_error = max(maximum_state_error, state_error)
                transitions_checked += 1
                if not np.array_equal(observed_state, expected_state):
                    state_mismatches += 1
                    add_failure("state_transition", episode_index, local_next)

                rows_checked += 1
                encoded_match, decoded_match = pixel_checks(
                    pixels[global_index + 1], env
                )
                if not encoded_match:
                    pixel_mismatches += 1
                    add_failure("pixel_bytes", episode_index, local_next)
                if not decoded_match:
                    decoded_pixel_mismatches += 1
                    add_failure(
                        "decoded_lossless_pixels", episode_index, local_next
                    )

                expected_truncated = bool(
                    truncated
                    or local_next + 1 >= scenario.max_episode_steps
                )
                flags_match = bool(
                    bool(stored_terminated[global_index + 1])
                    == bool(terminated)
                    and bool(stored_truncated[global_index + 1])
                    == expected_truncated
                )
                if not flags_match:
                    termination_mismatches += 1
                    add_failure(
                        "termination_or_truncation", episode_index, local_next
                    )
    finally:
        env.close()

    passed = not any(
        (
            pixel_mismatches,
            decoded_pixel_mismatches,
            state_mismatches,
            goal_mismatches,
            termination_mismatches,
        )
    )
    return {
        "passed": passed,
        "semantics": (
            "every stored proprio/action transition is replayed in raw "
            "TwoRoom and every resulting frame reproduces the declared "
            "stored pixel bytes exactly"
        ),
        "pixel_codec": scenario.pixel_codec,
        "rows_checked": rows_checked,
        "transitions_checked": transitions_checked,
        "pixel_byte_mismatches": pixel_mismatches,
        "decoded_lossless_pixel_mismatches": decoded_pixel_mismatches,
        "state_transition_mismatches": state_mismatches,
        "goal_invariance_mismatches": goal_mismatches,
        "termination_flag_mismatches": termination_mismatches,
        "maximum_state_absolute_error": maximum_state_error,
        "failure_examples": failure_examples,
    }


def validate_scenario(swm: ModuleType, scenario: CompiledScenario) -> dict[str, Any]:
    if not scenario.output_path.exists():
        return {
            "scenario_id": scenario.scenario_id,
            "passed": False,
            "errors": [f"Missing output {scenario.output_path}"],
        }

    errors: list[str] = []
    dataset = swm.data.LanceDataset(path=scenario.output_path)
    columns = dataset.column_names
    missing = sorted(set(_MODEL_COLUMNS) - set(columns))
    if missing:
        errors.append(f"Missing model columns: {missing}")

    episode_lengths = [int(value) for value in dataset.lengths]
    if len(episode_lengths) != scenario.episodes:
        errors.append(
            f"Expected {scenario.episodes} episodes, found {len(episode_lengths)}"
        )
    if not episode_lengths or min(episode_lengths) < 2:
        errors.append(f"Episode lengths are too short: {episode_lengths}")

    factor_checks: dict[str, Any] = {}
    for key, expected in scenario.factors.items():
        column = factor_column(key)
        if column not in columns:
            errors.append(f"Missing factor column {column}")
            continue
        actual = dataset.get_col_data(column)
        expected_array = np.asarray(expected).reshape(-1)
        if key == "door.position":
            expected_array = np.repeat(expected_array, 3)
        flattened = actual.reshape(actual.shape[0], -1)
        matches = bool(
            flattened.shape[1] == expected_array.size
            and np.allclose(flattened, expected_array[None, :], atol=1e-6)
        )
        factor_checks[key] = {
            "column": column,
            "expected": expected_array.tolist(),
            "observed_first": flattened[0].tolist(),
            "constant_and_equal": matches,
        }
        if not matches:
            errors.append(f"Factor {key} was not applied exactly")

    try:
        exact_replay = validate_exact_tworoom_replay(scenario, dataset)
    except Exception as exc:
        exact_replay = {
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    if not exact_replay["passed"]:
        errors.append("Exact state/pixel trajectory replay failed")

    action_alignment: dict[str, Any] = {}
    if {"action", "proprio"}.issubset(columns) and episode_lengths:
        action = dataset.get_col_data("action")
        proprio = dataset.get_col_data("proprio")
        speed = (
            dataset.get_col_data(factor_column("agent.speed"))
            if factor_column("agent.speed") in columns
            else np.full((len(action), 1), 5.0, dtype=np.float32)
        )
        current_residuals: list[np.ndarray] = []
        shifted_residuals: list[np.ndarray] = []
        for offset, length in zip(dataset.offsets, dataset.lengths):
            start = int(offset)
            stop = start + int(length)
            if stop - start < 2:
                continue
            delta = proprio[start + 1 : stop] - proprio[start : stop - 1]
            scale = speed[start : stop - 1]
            current_prediction = scale * action[start : stop - 1]
            shifted_prediction = scale * action[start + 1 : stop]
            current_residuals.append(
                np.linalg.norm(delta - current_prediction, axis=1)
            )
            shifted_residuals.append(
                np.linalg.norm(delta - shifted_prediction, axis=1)
            )
        if current_residuals:
            current = np.concatenate(current_residuals)
            shifted = np.concatenate(shifted_residuals)
            current_median = float(np.median(current))
            shifted_median = float(np.median(shifted))
            aligned = bool(
                current_median < shifted_median
                and np.count_nonzero(current <= 1e-4) > 0
            )
            action_alignment = {
                "passed": aligned,
                "current_action_median_residual": current_median,
                "next_action_median_residual": shifted_median,
                "exact_transition_fraction": float(np.mean(current <= 1e-4)),
                "transitions": int(current.size),
            }
            if not aligned:
                errors.append(
                    "Stored action is not aligned with the current-state transition"
                )

    sample_shapes: dict[str, list[int]] = {}
    if not missing and episode_lengths and min(episode_lengths) >= 2:
        sequence_dataset = swm.data.LanceDataset(
            path=scenario.output_path,
            keys_to_load=list(_MODEL_COLUMNS),
            frameskip=1,
            num_steps=2,
        )
        sample = sequence_dataset[0]
        sample_shapes = {
            key: list(sample[key].shape) for key in _MODEL_COLUMNS
        }
        expected_shapes = {
            "pixels": [2, 3, *scenario.image_shape],
            "action": [2, 2],
            "proprio": [2, 2],
        }
        for key, expected_shape in expected_shapes.items():
            if sample_shapes[key] != expected_shape:
                errors.append(
                    f"Unexpected {key} shape {sample_shapes[key]}, "
                    f"expected {expected_shape}"
                )

    return {
        "scenario_id": scenario.scenario_id,
        "name": scenario.name,
        "split": scenario.split,
        "passed": not errors,
        "errors": errors,
        "columns": columns,
        "episode_lengths": episode_lengths,
        "rows": sum(episode_lengths),
        "pixel_codec": scenario.pixel_codec,
        "factor_checks": factor_checks,
        "exact_replay": exact_replay,
        "action_alignment": action_alignment,
        "sample_shapes": sample_shapes,
    }


def validate_loader_mix(
    swm: ModuleType,
    original_dataset: Path,
    synthetic_dataset: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    """Read one original H5 and one synthetic Lance sequence, then concat."""

    kwargs = {
        "keys_to_load": list(_MODEL_COLUMNS),
        "frameskip": 1,
        "num_steps": 2,
    }
    original = swm.data.load_dataset(
        str(original_dataset.resolve()), cache_dir=str(cache_dir), **kwargs
    )
    synthetic = swm.data.load_dataset(
        str(synthetic_dataset.resolve()), cache_dir=str(cache_dir), **kwargs
    )
    mixed = swm.data.ConcatDataset([original, synthetic])
    original_sample = mixed[0]
    synthetic_sample = mixed[len(original)]

    shapes = {
        "original": {key: list(original_sample[key].shape) for key in _MODEL_COLUMNS},
        "synthetic": {
            key: list(synthetic_sample[key].shape) for key in _MODEL_COLUMNS
        },
    }
    matching = {
        key: shapes["original"][key] == shapes["synthetic"][key]
        for key in _MODEL_COLUMNS
    }
    return {
        "passed": all(matching.values()),
        "original_format": type(original).__name__,
        "synthetic_format": type(synthetic).__name__,
        "mixed_format": type(mixed).__name__,
        "original_samples": len(original),
        "synthetic_samples": len(synthetic),
        "mixed_samples": len(mixed),
        "sample_shapes": shapes,
        "shape_match": matching,
    }
