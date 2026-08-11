from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)

from .speed_door_rule_composition import (
    MODEL_INPUT_KEYS,
    RULE_NAMES,
    SpeedDoorRuleTemplate,
    model_input_projection,
    replay_is_exact,
    simulate_template,
    validate_factor_grid,
)


FROZEN_CONFIG_CANONICAL_SHA256 = (
    "b40f1e6efff21ed080a7d76def9ce7ea7b209370bcc3f630385de793d29be0bf"
)
DIRECTIONS = ("left_to_right", "right_to_left")
HISTORY_TOKENS = 3
ACTION_BLOCK = 5


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_config_sha256(config: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            key: value
            for key, value in config.items()
            if key != "_config_path"
        }
    )


def validate_frozen_config(config: dict[str, Any]) -> str:
    observed = frozen_config_sha256(config)
    if observed != FROZEN_CONFIG_CANONICAL_SHA256:
        raise ValueError(
            "Speed-door-rule Validation config differs from its frozen "
            f"contract: expected {FROZEN_CONFIG_CANONICAL_SHA256}, "
            f"observed {observed}"
        )
    return observed


def _factor_key(speed: float, rule: str) -> str:
    return f"s{float(speed):04.1f}_{rule}".replace(".", "p")


def factor_conditions(
    config: dict[str, Any],
) -> tuple[tuple[float, str], ...]:
    speeds = tuple(map(float, config["protocol"]["eval_speeds"]))
    if tuple(sorted(speeds)) != speeds or len(set(speeds)) != len(speeds):
        raise ValueError("Eval speeds must be unique and increasing")
    return tuple(
        (speed, rule) for speed in speeds for rule in RULE_NAMES
    )


def _integer_range(specification: dict[str, Any]) -> list[int]:
    start = int(specification["start"])
    stop = int(specification["stop_inclusive"])
    step = int(specification.get("step", 1))
    if step <= 0 or stop < start:
        raise ValueError(f"Invalid integer range: {specification}")
    return list(range(start, stop + 1, step))


def training_and_eval_doors(
    config: dict[str, Any],
) -> dict[str, tuple[int, ...]]:
    selection = config["training_isolation"]["door_selection"]
    safe = set(_integer_range(selection["safe_integer_range"]))
    eval_only = set(_integer_range(selection["eval_only_range"]))
    excluded_training = set(
        _integer_range(selection["excluded_training_range"])
    )
    eligible = np.asarray(
        sorted(safe - eval_only - excluded_training),
        dtype=np.int64,
    )
    rng = np.random.default_rng(int(selection["split_seed"]))
    shuffled = list(map(int, rng.permutation(eligible)))
    train_count = int(selection["train_count"])
    val_count = int(selection["loader_val_count"])
    guard_count = int(selection["guard_count"])
    if train_count + val_count + guard_count != len(shuffled):
        raise ValueError(
            "Training door split does not exhaust the eligible set"
        )
    result = {
        "train": tuple(sorted(shuffled[:train_count])),
        "loader_val": tuple(
            sorted(shuffled[train_count : train_count + val_count])
        ),
        "guard": tuple(sorted(shuffled[train_count + val_count :])),
        "eval_only": tuple(sorted(eval_only)),
    }
    named = {name: set(values) for name, values in result.items()}
    for index, (left_name, left) in enumerate(named.items()):
        for right_name, right in list(named.items())[index + 1 :]:
            overlap = left & right
            if overlap:
                raise ValueError(
                    f"Door splits overlap: {left_name}/{right_name}="
                    f"{sorted(overlap)}"
                )
    return result


def _vertical_sign(door_position: int) -> float:
    return 1.0 if int(door_position) <= 112 else -1.0


def candidate_templates(
    config: dict[str, Any],
) -> list[SpeedDoorRuleTemplate]:
    geometry = config["candidate_geometry"]
    wall = geometry["wall_geometry"]
    splits = training_and_eval_doors(config)
    excluded_eval = set(
        map(int, geometry.get("excluded_eval_door_positions", ()))
    )
    unknown_exclusions = excluded_eval - set(splits["eval_only"])
    if unknown_exclusions:
        raise ValueError(
            "Candidate exclusions are not Eval-only doors: "
            f"{sorted(unknown_exclusions)}"
        )
    templates: list[SpeedDoorRuleTemplate] = []
    for door_index, door in enumerate(splits["eval_only"]):
        if door in excluded_eval:
            continue
        vertical_sign = _vertical_sign(door)
        for direction_index, direction in enumerate(
            geometry["directions"]
        ):
            if direction not in DIRECTIONS:
                raise ValueError(f"Unknown direction {direction!r}")
            left_to_right = direction == "left_to_right"
            direction_slug = "ltr" if left_to_right else "rtl"
            for distance_index, distance in enumerate(
                map(float, geometry["wall_distances_px"])
            ):
                reset_x = (
                    float(wall["left_contact_x"]) - distance
                    if left_to_right
                    else float(wall["right_contact_x"]) + distance
                )
                goal_x = (
                    float(wall["left_to_right_goal_x"])
                    if left_to_right
                    else float(wall["right_to_left_goal_x"])
                )
                for offset_index, offset in enumerate(
                    map(float, geometry["doorway_offsets_px"])
                ):
                    reset_y = float(door) + vertical_sign * offset
                    seed = int(
                        np.random.SeedSequence(
                            [
                                int(config["catalog_seed"]),
                                door_index,
                                direction_index,
                                distance_index,
                                offset_index,
                            ]
                        ).generate_state(1)[0]
                    )
                    templates.append(
                        SpeedDoorRuleTemplate(
                            template_id=(
                                f"sdrv-d{door:03d}-{direction_slug}-"
                                f"w{distance_index:02d}-o{offset_index:02d}"
                            ),
                            door_position=int(door),
                            direction=direction,
                            doorway_offset_px=offset,
                            reset_state=(reset_x, reset_y),
                            goal_state=(goal_x, reset_y),
                            simulator_seed=seed,
                        )
                    )
    if len({template.template_id for template in templates}) != len(
        templates
    ):
        raise RuntimeError("Validation candidate IDs are not unique")
    return templates


def _query_hash(
    template: SpeedDoorRuleTemplate,
    *,
    config: dict[str, Any],
) -> str:
    speed, rule = factor_conditions(config)[0]
    rollout = simulate_template(
        template,
        speed=speed,
        rule=rule,
        protocol=config["protocol"],
    )
    return array_sha256(rollout["query_pixels"])


def select_validation_assignments(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluation = config["evaluation"]
    eval_seeds = tuple(map(int, evaluation["eval_seeds"]))
    per_seed = int(evaluation["unique_queries_per_seed"])
    per_direction = int(evaluation["queries_per_direction_per_seed"])
    if per_seed != per_direction * len(DIRECTIONS):
        raise ValueError("Per-seed direction balance is inconsistent")

    unique_by_direction: dict[
        str, dict[str, SpeedDoorRuleTemplate]
    ] = {direction: {} for direction in DIRECTIONS}
    collision_count = 0
    for template in candidate_templates(config):
        query_hash = _query_hash(template, config=config)
        bucket = unique_by_direction[template.direction]
        if query_hash in bucket:
            collision_count += 1
            continue
        bucket[query_hash] = template

    available = {
        direction: list(values.values())
        for direction, values in unique_by_direction.items()
    }
    assignments: list[dict[str, Any]] = []
    for eval_seed in eval_seeds:
        selected_for_seed: list[SpeedDoorRuleTemplate] = []
        for direction_index, direction in enumerate(DIRECTIONS):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        int(config["catalog_seed"]),
                        int(eval_seed),
                        direction_index,
                    ]
                )
            )
            pool = available[direction]
            if len(pool) < per_direction:
                raise RuntimeError(
                    f"Not enough unique {direction} queries: {len(pool)}"
                )
            indices = sorted(
                map(
                    int,
                    rng.choice(
                        len(pool),
                        size=per_direction,
                        replace=False,
                    ),
                ),
                reverse=True,
            )
            chosen = [pool[index] for index in reversed(indices)]
            for index in indices:
                pool.pop(index)
            selected_for_seed.extend(chosen)
        order_rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(config["catalog_seed"]), int(eval_seed), 0xA551]
            )
        )
        order = order_rng.permutation(len(selected_for_seed))
        for evaluation_index, selected_index in enumerate(order):
            template = selected_for_seed[int(selected_index)]
            assignments.append(
                {
                    "eval_seed": int(eval_seed),
                    "evaluation_index": int(evaluation_index),
                    "template": template,
                }
            )

    expected = len(eval_seeds) * per_seed
    if len(assignments) != expected:
        raise RuntimeError(
            f"Expected {expected} Validation assignments, got "
            f"{len(assignments)}"
        )
    return assignments, {
        "candidate_templates": sum(
            len(bucket) for bucket in unique_by_direction.values()
        )
        + collision_count,
        "unique_query_candidates": sum(
            len(bucket) for bucket in unique_by_direction.values()
        ),
        "deduplicated_query_pixel_collisions": collision_count,
        "remaining_candidates_after_selection": {
            direction: len(pool)
            for direction, pool in available.items()
        },
    }


def _payload_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    return canonical_sha256(
        {
            name: array_sha256(value)
            for name, value in sorted(arrays.items())
        }
    )


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _payload_arrays(
    rollouts: dict[tuple[float, str], dict[str, Any]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    representative = next(iter(rollouts.values()))
    arrays["query_pixels"] = np.asarray(
        representative["query_pixels"], dtype=np.uint8
    )
    arrays["query_state"] = np.asarray(
        representative["query_state"], dtype=np.float32
    )
    arrays["goal_pixels"] = np.asarray(
        representative["goal_pixels"], dtype=np.uint8
    )
    arrays["goal_state"] = np.asarray(
        representative["goal_state"], dtype=np.float32
    )
    for (speed, rule), rollout in rollouts.items():
        key = _factor_key(speed, rule)
        projection = model_input_projection(rollout)
        arrays[f"{key}_history_pixels"] = np.asarray(
            projection["pixels"], dtype=np.uint8
        )
        arrays[f"{key}_action_blocks"] = np.asarray(
            projection["action"], dtype=np.float32
        )
        arrays[f"{key}_target_pixels"] = np.asarray(
            rollout["target_pixels"], dtype=np.uint8
        )
        arrays[f"{key}_target_state"] = np.asarray(
            rollout["target_state"], dtype=np.float32
        )
    return arrays


def _serialized_payload_audit(
    payload_path: Path,
    arrays: dict[str, np.ndarray],
) -> dict[str, bool]:
    with np.load(payload_path, allow_pickle=False) as payload:
        keys_exact = set(payload.files) == set(arrays)
        arrays_exact = keys_exact and all(
            np.array_equal(payload[name], value)
            for name, value in arrays.items()
        )
    return {
        "serialized_keys_exact": keys_exact,
        "serialized_arrays_roundtrip_exact": arrays_exact,
    }


def _content_manifest_projection(
    catalog: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": int(catalog["schema_version"]),
        "benchmark": str(catalog["benchmark"]),
        "status": str(catalog["status"]),
        "protocol": catalog["protocol"],
        "summary": catalog["summary"],
        "bundles": [
            {
                key: bundle[key]
                for key in (
                    "query_id",
                    "eval_seed",
                    "evaluation_index",
                    "template",
                    "factor_conditions",
                    "payload_sha256",
                    "payload_content_sha256",
                    "query_pixels_sha256",
                    "history_pixels_sha256",
                    "action_blocks_sha256",
                    "target_pixels_sha256",
                    "target_state_sha256",
                )
            }
            for bundle in catalog["bundles"]
        ],
    }


def build_validation_catalog(
    *,
    config: dict[str, Any],
    repo_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_hash = validate_frozen_config(config)
    assignments, candidate_audit = select_validation_assignments(config)
    factors = factor_conditions(config)
    speeds = tuple(sorted({speed for speed, _ in factors}))
    payload_root = output_root / "payloads"
    payload_root.mkdir(parents=True, exist_ok=True)
    bundles: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    replay_passed = 0
    physical_minimums = {
        "middle_rule_centroid_gap_px": float("inf"),
        "middle_adjacent_speed_centroid_gap_px": float("inf"),
        "future_rule_state_gap_px": float("inf"),
        "future_adjacent_speed_centroid_gap_px": float("inf"),
    }

    for assignment in assignments:
        template = assignment["template"]
        rollouts = {
            factor: simulate_template(
                template,
                speed=factor[0],
                rule=factor[1],
                protocol=config["protocol"],
            )
            for factor in factors
        }
        replays = {
            factor: simulate_template(
                template,
                speed=factor[0],
                rule=factor[1],
                protocol=config["protocol"],
            )
            for factor in factors
        }
        exact_replay = all(
            replay_is_exact(rollout, replays[factor])
            for factor, rollout in rollouts.items()
        )
        replay_passed += int(exact_replay)
        validation = validate_factor_grid(
            template,
            rollouts,
            speeds=speeds,
            thresholds=config["physical_gates"],
        )
        validation["checks"]["exact_replay"] = exact_replay
        arrays = _payload_arrays(rollouts)
        query_id = (
            f"s{assignment['eval_seed']}-"
            f"e{assignment['evaluation_index']:03d}-"
            f"{template.template_id}"
        )
        payload_path = payload_root / f"{query_id}.npz"
        _atomic_savez(payload_path, arrays)
        serialized = _serialized_payload_audit(payload_path, arrays)
        validation["checks"].update(serialized)
        validation["passed"] = all(validation["checks"].values())

        history_hashes = {
            _factor_key(*factor): array_sha256(
                arrays[f"{_factor_key(*factor)}_history_pixels"]
            )
            for factor in factors
        }
        action_hashes = {
            _factor_key(*factor): array_sha256(
                arrays[f"{_factor_key(*factor)}_action_blocks"]
            )
            for factor in factors
        }
        target_hashes = {
            _factor_key(*factor): array_sha256(
                arrays[f"{_factor_key(*factor)}_target_pixels"]
            )
            for factor in factors
        }
        target_state_hashes = {
            _factor_key(*factor): array_sha256(
                arrays[f"{_factor_key(*factor)}_target_state"]
            )
            for factor in factors
        }
        bundle = {
            "query_id": query_id,
            "eval_seed": int(assignment["eval_seed"]),
            "evaluation_index": int(assignment["evaluation_index"]),
            "direction": template.direction,
            "door_position": int(template.door_position),
            "template": asdict(template),
            "factor_conditions": [
                {
                    "key": _factor_key(speed, rule),
                    "speed": speed,
                    "rule": rule,
                }
                for speed, rule in factors
            ],
            "payload": portable_contextworld_path(
                payload_path, repo_root=repo_root
            ),
            "payload_sha256": file_sha256(payload_path),
            "payload_content_sha256": _payload_content_sha256(arrays),
            "query_pixels_sha256": array_sha256(
                arrays["query_pixels"]
            ),
            "history_pixels_sha256": history_hashes,
            "action_blocks_sha256": action_hashes,
            "target_pixels_sha256": target_hashes,
            "target_state_sha256": target_state_hashes,
            "validation": validation,
        }
        bundles.append(bundle)
        if not validation["passed"]:
            failures.append(
                {
                    "query_id": query_id,
                    "failed_checks": sorted(
                        name
                        for name, passed in validation["checks"].items()
                        if not passed
                    ),
                }
            )
        observed = {
            "middle_rule_centroid_gap_px": validation[
                "minimum_middle_rule_centroid_gap_px"
            ],
            "middle_adjacent_speed_centroid_gap_px": validation[
                "minimum_middle_adjacent_speed_centroid_gap_px"
            ],
            "future_rule_state_gap_px": validation[
                "minimum_future_rule_state_gap_px"
            ],
            "future_adjacent_speed_centroid_gap_px": validation[
                "minimum_future_adjacent_speed_centroid_gap_px"
            ],
        }
        physical_minimums = {
            name: min(physical_minimums[name], value)
            for name, value in observed.items()
        }

    eval_seeds = tuple(map(int, config["evaluation"]["eval_seeds"]))
    per_seed = int(config["evaluation"]["unique_queries_per_seed"])
    by_seed = Counter(bundle["eval_seed"] for bundle in bundles)
    by_seed_direction = Counter(
        (bundle["eval_seed"], bundle["direction"]) for bundle in bundles
    )
    query_hashes = {
        bundle["query_pixels_sha256"] for bundle in bundles
    }
    action_signatures: dict[str, Counter[str]] = defaultdict(Counter)
    query_signatures: dict[str, Counter[str]] = defaultdict(Counter)
    for bundle in bundles:
        for factor in bundle["factor_conditions"]:
            key = factor["key"]
            action_signatures[
                bundle["action_blocks_sha256"][key]
            ][key] += 1
            query_signatures[bundle["query_pixels_sha256"]][key] += 1

    def best_signature_accuracy(
        counts: dict[str, Counter[str]],
    ) -> float:
        total = sum(sum(row.values()) for row in counts.values())
        return (
            float(sum(max(row.values()) for row in counts.values()) / total)
            if total
            else 1.0
        )

    training_doors = training_and_eval_doors(config)
    training_speeds = tuple(
        map(float, config["training_isolation"]["training_speeds"])
    )
    eval_speed_values = tuple(
        map(float, config["protocol"]["eval_speeds"])
    )
    checks = {
        "exact_query_count": len(bundles) == len(eval_seeds) * per_seed,
        "all_queries_pass_physical_and_serialization_gates": not failures,
        "exact_replay_all_queries": replay_passed == len(bundles),
        "query_ids_unique": (
            len({bundle["query_id"] for bundle in bundles}) == len(bundles)
        ),
        "query_pixels_unique": len(query_hashes) == len(bundles),
        "per_seed_counts_exact": all(
            by_seed[seed] == per_seed for seed in eval_seeds
        ),
        "directions_balanced_per_seed": all(
            by_seed_direction[(seed, direction)]
            == int(
                config["evaluation"][
                    "queries_per_direction_per_seed"
                ]
            )
            for seed in eval_seeds
            for direction in DIRECTIONS
        ),
        "query_only_joint_accuracy_is_chance": bool(
            np.isclose(
                best_signature_accuracy(query_signatures),
                1.0 / len(factors),
            )
        ),
        "action_only_joint_accuracy_is_chance": bool(
            np.isclose(
                best_signature_accuracy(action_signatures),
                1.0 / len(factors),
            )
        ),
        "train_eval_speeds_have_no_exact_overlap": not (
            set(training_speeds) & set(eval_speed_values)
        ),
        "eval_speeds_inside_training_range": all(
            min(training_speeds) < speed < max(training_speeds)
            for speed in eval_speed_values
        ),
        "train_doors_exclude_eval_doors": not (
            set(training_doors["train"])
            & set(training_doors["eval_only"])
        ),
        "loader_val_doors_exclude_eval_doors": not (
            set(training_doors["loader_val"])
            & set(training_doors["eval_only"])
        ),
        "model_visible_fields_exact": (
            tuple(config["protocol"]["model_visible_fields"])
            == MODEL_INPUT_KEYS
        ),
        "frozen_config_exact_match": (
            config_hash == FROZEN_CONFIG_CANONICAL_SHA256
        ),
    }
    summary = {
        "eval_seeds": list(eval_seeds),
        "unique_queries_per_eval_seed": per_seed,
        "unique_queries": len(bundles),
        "factor_conditions": len(factors),
        "samples_per_model_per_true_condition": len(bundles),
        "model_predictions_per_checkpoint": len(bundles) * len(factors),
        "target_encodings_per_checkpoint": len(bundles) * len(factors),
        "loss_comparisons_per_checkpoint": (
            len(bundles) * len(factors) * len(factors)
        ),
        "by_eval_seed": {
            str(seed): by_seed[seed] for seed in eval_seeds
        },
        "by_eval_seed_and_direction": {
            f"{seed}/{direction}": by_seed_direction[(seed, direction)]
            for seed in eval_seeds
            for direction in DIRECTIONS
        },
        "eval_speeds": list(eval_speed_values),
        "training_speeds": list(training_speeds),
        "rules": list(RULE_NAMES),
        "eval_only_doors": list(training_doors["eval_only"]),
        "evaluated_doors": sorted(
            {int(bundle["door_position"]) for bundle in bundles}
        ),
        "excluded_eval_doors": sorted(
            set(training_doors["eval_only"])
            - {int(bundle["door_position"]) for bundle in bundles}
        ),
    }
    catalog = {
        "schema_version": 1,
        "benchmark": str(config["benchmark"]),
        "status": "frozen_before_model_scoring",
        "protocol": {
            "history_tokens": HISTORY_TOKENS,
            "raw_steps_per_action_block": ACTION_BLOCK,
            "eval_speeds": list(eval_speed_values),
            "rules": list(RULE_NAMES),
            "factor_conditions": [
                {
                    "key": _factor_key(speed, rule),
                    "speed": speed,
                    "rule": rule,
                }
                for speed, rule in factors
            ],
            "model_visible_fields": list(MODEL_INPUT_KEYS),
            "online_environment_calls_during_scoring": 0,
        },
        "summary": summary,
        "bundles": bundles,
    }
    catalog["content_manifest_sha256"] = canonical_sha256(
        _content_manifest_projection(catalog)
    )
    report = {
        "schema_version": 1,
        "benchmark": str(config["benchmark"]),
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "candidate_selection": candidate_audit,
        "observed_physical_minimums": physical_minimums,
        "physical_thresholds": config["physical_gates"],
        "query_only_best_joint_accuracy": best_signature_accuracy(
            query_signatures
        ),
        "action_only_best_joint_accuracy": best_signature_accuracy(
            action_signatures
        ),
        "exact_replay_queries": replay_passed,
        "failed_queries": failures,
        "content_manifest_sha256": catalog[
            "content_manifest_sha256"
        ],
        "frozen_config_canonical_sha256": config_hash,
        "claim_limit": (
            "This build freezes offline data and proves physical "
            "identifiability. It is not a model ICL result."
        ),
    }
    return catalog, report


def load_validation_assets(
    catalog_path: Path,
    *,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog_path = Path(catalog_path).resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("status") != "frozen_before_model_scoring":
        raise ValueError("Validation catalog is not frozen before scoring")
    observed_content = canonical_sha256(
        _content_manifest_projection(catalog)
    )
    if observed_content != catalog.get("content_manifest_sha256"):
        raise RuntimeError("Validation content manifest hash mismatch")
    factors = tuple(
        (float(item["speed"]), str(item["rule"]))
        for item in catalog["protocol"]["factor_conditions"]
    )
    assets: list[dict[str, Any]] = []
    for bundle in sorted(
        catalog["bundles"],
        key=lambda row: (
            int(row["eval_seed"]),
            int(row["evaluation_index"]),
        ),
    ):
        payload_path = resolve_contextworld_path(
            bundle["payload"], repo_root=repo_root
        )
        if file_sha256(payload_path) != bundle["payload_sha256"]:
            raise RuntimeError(f"Payload hash mismatch: {payload_path}")
        with np.load(payload_path, allow_pickle=False) as payload:
            arrays = {
                name: np.asarray(payload[name]).copy()
                for name in payload.files
            }
        if _payload_content_sha256(arrays) != bundle[
            "payload_content_sha256"
        ]:
            raise RuntimeError(
                f"Payload content hash mismatch: {payload_path}"
            )
        query_pixels = np.asarray(arrays["query_pixels"], dtype=np.uint8)
        histories: dict[tuple[float, str], np.ndarray] = {}
        actions: dict[tuple[float, str], np.ndarray] = {}
        targets: dict[tuple[float, str], np.ndarray] = {}
        target_states: dict[tuple[float, str], np.ndarray] = {}
        for factor in factors:
            key = _factor_key(*factor)
            history = np.asarray(
                arrays[f"{key}_history_pixels"], dtype=np.uint8
            )
            action = np.asarray(
                arrays[f"{key}_action_blocks"], dtype=np.float32
            )
            target = np.asarray(
                arrays[f"{key}_target_pixels"], dtype=np.uint8
            )
            if history.shape != (HISTORY_TOKENS, 224, 224, 3):
                raise RuntimeError(f"Invalid history shape: {history.shape}")
            if action.shape != (HISTORY_TOKENS, ACTION_BLOCK, 2):
                raise RuntimeError(f"Invalid action shape: {action.shape}")
            if not np.array_equal(history[-1], query_pixels):
                raise RuntimeError("History does not end at frozen query")
            if array_sha256(history) != bundle[
                "history_pixels_sha256"
            ][key]:
                raise RuntimeError("History hash mismatch")
            if array_sha256(action) != bundle[
                "action_blocks_sha256"
            ][key]:
                raise RuntimeError("Action hash mismatch")
            if array_sha256(target) != bundle[
                "target_pixels_sha256"
            ][key]:
                raise RuntimeError("Target hash mismatch")
            histories[factor] = history
            actions[factor] = action
            targets[factor] = target
            target_states[factor] = np.asarray(
                arrays[f"{key}_target_state"], dtype=np.float32
            )
        if len({array_sha256(value) for value in actions.values()}) != 1:
            raise RuntimeError("Factor histories use different actions")
        if len({array_sha256(value) for value in targets.values()}) != len(
            factors
        ):
            raise RuntimeError("True future pixels are not all distinct")
        assets.append(
            {
                "query_id": str(bundle["query_id"]),
                "eval_seed": int(bundle["eval_seed"]),
                "evaluation_index": int(bundle["evaluation_index"]),
                "direction": str(bundle["direction"]),
                "door_position": int(bundle["door_position"]),
                "template_id": str(bundle["template"]["template_id"]),
                "query_pixels": query_pixels,
                "histories": histories,
                "actions": actions,
                "targets": targets,
                "target_states": target_states,
            }
        )
    summary = catalog["summary"]
    expected = int(summary["unique_queries"])
    by_seed = Counter(asset["eval_seed"] for asset in assets)
    audit = {
        "passed": (
            len(assets) == expected
            and len({asset["query_id"] for asset in assets}) == expected
            and all(
                by_seed[int(seed)]
                == int(summary["unique_queries_per_eval_seed"])
                for seed in summary["eval_seeds"]
            )
        ),
        "catalog": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "content_manifest_sha256": observed_content,
        "unique_queries": len(assets),
        "factor_conditions": len(factors),
        "online_environment_calls": 0,
    }
    if not audit["passed"]:
        raise RuntimeError(f"Loaded Validation asset audit failed: {audit}")
    return assets, audit


__all__ = [
    "ACTION_BLOCK",
    "DIRECTIONS",
    "FROZEN_CONFIG_CANONICAL_SHA256",
    "HISTORY_TOKENS",
    "array_sha256",
    "build_validation_catalog",
    "candidate_templates",
    "canonical_sha256",
    "factor_conditions",
    "file_sha256",
    "frozen_config_sha256",
    "load_validation_assets",
    "select_validation_assignments",
    "training_and_eval_doors",
    "validate_frozen_config",
]
