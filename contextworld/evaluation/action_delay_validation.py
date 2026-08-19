from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.manifest import write_json

from .action_delay import (
    ACTION_BLOCK,
    DELAY_VALUES,
    ActionDelayTemplate,
    array_sha256,
    canonical_sha256,
    model_input_projection,
    simulate_template,
    validate_delay_family,
)


EVAL_SEEDS = (42, 43, 44, 45, 46, 47)
QUERIES_PER_SEED = 50
QUERY_COUNT = len(EVAL_SEEDS) * QUERIES_PER_SEED
HISTORY_DELAYS = DELAY_VALUES
TARGET_DELAYS = DELAY_VALUES
MODEL_PREDICTIONS_PER_CHECKPOINT = QUERY_COUNT * len(HISTORY_DELAYS)
LOSS_RECORDS_PER_CHECKPOINT = (
    MODEL_PREDICTIONS_PER_CHECKPOINT * len(TARGET_DELAYS)
)
SEEN_DELAYS = (0, 2, 4)
INTERPOLATION_DELAYS = (1, 3)


@dataclass(frozen=True)
class ActionDelayValidationAssignment:
    query_id: str
    eval_seed: int
    evaluation_index: int
    template: ActionDelayTemplate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _direction_action(direction: str) -> np.ndarray:
    if direction == "up":
        return np.asarray([0.0, 1.0], dtype=np.float32)
    if direction == "down":
        return np.asarray([0.0, -1.0], dtype=np.float32)
    raise ValueError(f"Unknown direction {direction!r}")


def select_validation_assignments(
    config: dict[str, Any],
) -> list[ActionDelayValidationAssignment]:
    evaluation = config["evaluation"]
    eval_seeds = tuple(map(int, evaluation["eval_seeds"]))
    queries_per_seed = int(evaluation["queries_per_seed"])
    if eval_seeds != EVAL_SEEDS or queries_per_seed != QUERIES_PER_SEED:
        raise ValueError(
            "Action-delay Validation requires six seeds 42..47 and "
            "50 unique queries per seed"
        )
    catalog_seed = int(evaluation["catalog_seed"])
    left_x = list(range(25, 94, 3))
    right_x = list(range(133, 201, 3))
    # Keeping the query within [60, 165] leaves 35 px of clearance for
    # both the pre-query probe and the fastest post-query future.  States
    # at y=205 touch TwoRoom's collision boundary and would break the
    # otherwise exact queue-flush identity for the downward direction.
    query_y = list(range(60, 166, 5))
    candidates = [
        (room, float(x), float(y))
        for room, x_values in (("left", left_x), ("right", right_x))
        for x in x_values
        for y in query_y
    ]
    rng = np.random.default_rng(catalog_seed)
    candidates = [
        candidates[index] for index in rng.permutation(len(candidates))
    ]

    selected: list[ActionDelayValidationAssignment] = []
    used_coordinates: set[tuple[float, float]] = set()
    cursor = 0
    for eval_seed in eval_seeds:
        directions = ["up"] * 25 + ["down"] * 25
        direction_rng = np.random.default_rng(
            np.random.SeedSequence([catalog_seed, eval_seed, 0xD31A])
        )
        direction_rng.shuffle(directions)
        for evaluation_index, direction in enumerate(directions):
            preferred_room = (
                "left" if evaluation_index % 2 == 0 else "right"
            )
            while True:
                if cursor >= len(candidates):
                    raise RuntimeError(
                        "Exhausted action-delay Validation candidate grid"
                    )
                room, x_position, y_position = candidates[cursor]
                cursor += 1
                coordinate = (x_position, y_position)
                if (
                    room == preferred_room
                    and coordinate not in used_coordinates
                ):
                    break
            used_coordinates.add(coordinate)
            action = _direction_action(direction)
            reset = (
                np.asarray([x_position, y_position], dtype=np.float32)
                - 7.0 * ACTION_BLOCK * action
            )
            goal = (
                (200.0, 205.0 if y_position < 115.0 else 20.0)
                if room == "left"
                else (25.0, 205.0 if y_position < 115.0 else 20.0)
            )
            simulator_seed = int(
                np.random.SeedSequence(
                    [
                        catalog_seed,
                        eval_seed,
                        evaluation_index,
                    ]
                ).generate_state(1)[0]
            )
            query_id = (
                f"action-delay-val-s{eval_seed}-q{evaluation_index:02d}"
            )
            selected.append(
                ActionDelayValidationAssignment(
                    query_id=query_id,
                    eval_seed=eval_seed,
                    evaluation_index=evaluation_index,
                    template=ActionDelayTemplate(
                        template_id=query_id,
                        direction=direction,
                        reset_state=tuple(map(float, reset)),
                        goal_state=tuple(map(float, goal)),
                        simulator_seed=simulator_seed,
                    ),
                )
            )
    return selected


def build_validation_asset(
    assignment: ActionDelayValidationAssignment,
    *,
    agent_speed: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rollouts = {
        delay: simulate_template(
            assignment.template,
            delay_steps=delay,
            agent_speed=agent_speed,
        )
        for delay in DELAY_VALUES
    }
    family = validate_delay_family(
        assignment.template,
        rollouts,
        agent_speed=agent_speed,
    )
    if not family["passed"]:
        raise RuntimeError(
            f"Validation family failed: {assignment.query_id}"
        )
    history_pixels = np.stack(
        [
            rollouts[delay]["history_pixels"]
            for delay in HISTORY_DELAYS
        ]
    ).astype(np.uint8)
    action_blocks = np.stack(
        [
            model_input_projection(rollouts[delay])["action"]
            for delay in HISTORY_DELAYS
        ]
    ).astype(np.float32)
    target_pixels = np.stack(
        [
            rollouts[delay]["target_pixels"]
            for delay in TARGET_DELAYS
        ]
    ).astype(np.uint8)
    arrays = {
        "history_pixels": history_pixels,
        "action_blocks": action_blocks,
        "target_pixels": target_pixels,
        "query_pixels": rollouts[0]["query_pixels"].astype(np.uint8),
        "history_states": np.stack(
            [
                rollouts[delay]["history_states"]
                for delay in HISTORY_DELAYS
            ]
        ).astype(np.float32),
        "target_states": np.stack(
            [
                rollouts[delay]["target_state"]
                for delay in TARGET_DELAYS
            ]
        ).astype(np.float32),
        "goal_state": rollouts[0]["goal_state"].astype(np.float32),
        "history_delays": np.asarray(HISTORY_DELAYS, dtype=np.int64),
        "target_delays": np.asarray(TARGET_DELAYS, dtype=np.int64),
    }
    payload_sha256 = canonical_sha256(
        {
            name: array_sha256(value)
            for name, value in sorted(arrays.items())
        }
    )
    audit = {
        "query_id": assignment.query_id,
        "eval_seed": assignment.eval_seed,
        "evaluation_index": assignment.evaluation_index,
        "direction": assignment.template.direction,
        "template": asdict(assignment.template),
        "family_passed": family["passed"],
        "family_checks": family["checks"],
        "payload_sha256": payload_sha256,
        "query_pixels_sha256": array_sha256(arrays["query_pixels"]),
        "history_pixels_sha256": array_sha256(history_pixels),
        "action_blocks_sha256": array_sha256(action_blocks),
        "target_pixels_sha256": array_sha256(target_pixels),
    }
    return arrays, audit


def audit_validation_catalog(
    rows: list[dict[str, Any]],
    *,
    eval_seeds: Iterable[int] = EVAL_SEEDS,
    queries_per_seed: int = QUERIES_PER_SEED,
) -> dict[str, Any]:
    eval_seeds = tuple(map(int, eval_seeds))
    counts = Counter(int(row["eval_seed"]) for row in rows)
    direction_counts = {
        seed: Counter(
            row["direction"]
            for row in rows
            if int(row["eval_seed"]) == seed
        )
        for seed in eval_seeds
    }
    checks = {
        "exact_query_count": len(rows)
        == len(eval_seeds) * queries_per_seed,
        "exact_50_queries_per_seed": counts
        == Counter({seed: queries_per_seed for seed in eval_seeds}),
        "directions_balanced_per_seed": all(
            direction_counts[seed]
            == Counter({"up": 25, "down": 25})
            for seed in eval_seeds
        ),
        "query_ids_unique": len(
            {row["query_id"] for row in rows}
        )
        == len(rows),
        "template_ids_unique": len(
            {row["template"]["template_id"] for row in rows}
        )
        == len(rows),
        "query_pixels_unique": len(
            {row["query_pixels_sha256"] for row in rows}
        )
        == len(rows),
        "payloads_unique": len(
            {row["payload_sha256"] for row in rows}
        )
        == len(rows),
        "every_family_passed": all(
            row["family_passed"]
            and all(row["family_checks"].values())
            for row in rows
        ),
        "every_asset_reopens": all(
            bool(row.get("asset_reopens", False)) for row in rows
        ),
        "every_asset_hash_matches": all(
            bool(row.get("asset_hash_matches", False)) for row in rows
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "query_count": len(rows),
        "queries_by_eval_seed": dict(sorted(counts.items())),
        "directions_by_eval_seed": {
            str(seed): dict(sorted(direction_counts[seed].items()))
            for seed in eval_seeds
        },
    }


def build_validation_release(
    *,
    config: dict[str, Any],
    repo_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if tuple(map(int, config["protocol"]["delay_values"])) != DELAY_VALUES:
        raise ValueError("Validation must cover delays 0,1,2,3,4")
    if tuple(map(int, config["protocol"]["training_delay_values"])) != (
        0,
        2,
        4,
    ):
        raise ValueError("Training delays must remain 0,2,4")
    output_root.mkdir(parents=True, exist_ok=False)
    assets_root = output_root / "assets"
    assets_root.mkdir()
    assignments = select_validation_assignments(config)
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        arrays, audit = build_validation_asset(
            assignment,
            agent_speed=float(config["protocol"]["agent_speed"]),
        )
        asset_path = assets_root / f"{assignment.query_id}.npz"
        np.savez_compressed(asset_path, **arrays)
        reopened = dict(np.load(asset_path, allow_pickle=False))
        reopened_payload = canonical_sha256(
            {
                name: array_sha256(value)
                for name, value in sorted(reopened.items())
            }
        )
        row = {
            **audit,
            "asset": portable_contextworld_path(
                asset_path,
                repo_root=repo_root,
            ),
            "asset_sha256": file_sha256(asset_path),
            "asset_reopens": set(reopened) == set(arrays),
            "asset_hash_matches": reopened_payload
            == audit["payload_sha256"],
        }
        rows.append(row)

    audit = audit_validation_catalog(rows)
    if not audit["passed"]:
        raise RuntimeError(f"Validation asset audit failed: {audit}")
    content_projection = {
        "benchmark": config["benchmark"],
        "protocol": {
            "history_tokens": 3,
            "raw_steps_per_action_block": ACTION_BLOCK,
            "agent_speed": float(config["protocol"]["agent_speed"]),
            "delay_values": list(DELAY_VALUES),
            "training_delay_values": list(SEEN_DELAYS),
        },
        "queries": [
            {
                key: row[key]
                for key in (
                    "query_id",
                    "eval_seed",
                    "evaluation_index",
                    "direction",
                    "template",
                    "asset",
                    "asset_sha256",
                    "payload_sha256",
                    "query_pixels_sha256",
                    "history_pixels_sha256",
                    "action_blocks_sha256",
                    "target_pixels_sha256",
                )
            }
            for row in rows
        ],
    }
    content_sha256 = canonical_sha256(content_projection)
    catalog = {
        "schema_version": 1,
        **content_projection,
        "content_manifest_sha256": content_sha256,
        "counts": {
            "queries": QUERY_COUNT,
            "history_conditions_per_query": len(HISTORY_DELAYS),
            "true_targets_per_query": len(TARGET_DELAYS),
            "model_predictions_per_checkpoint": (
                MODEL_PREDICTIONS_PER_CHECKPOINT
            ),
            "loss_records_per_checkpoint": LOSS_RECORDS_PER_CHECKPOINT,
        },
    }
    catalog_path = output_root / "catalog.json"
    write_json(catalog_path, catalog)
    exclusion = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "content_manifest_sha256": content_sha256,
        "query_count": QUERY_COUNT,
        "query_records": [
            {
                "query_id": row["query_id"],
                "template_id": row["template"]["template_id"],
                "query_pixels_sha256": row["query_pixels_sha256"],
                "payload_sha256": row["payload_sha256"],
            }
            for row in rows
        ],
    }
    exclusion_path = output_root / "training_exclusion_manifest.json"
    write_json(exclusion_path, exclusion)
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "checks": {
            **audit["checks"],
            "catalog_counts_exact": catalog["counts"]
            == {
                "queries": 300,
                "history_conditions_per_query": 5,
                "true_targets_per_query": 5,
                "model_predictions_per_checkpoint": 1500,
                "loss_records_per_checkpoint": 7500,
            },
            "training_seen_and_interpolation_are_separate": (
                set(SEEN_DELAYS).isdisjoint(INTERPOLATION_DELAYS)
                and set(SEEN_DELAYS) | set(INTERPOLATION_DELAYS)
                == set(DELAY_VALUES)
            ),
        },
        "audit": audit,
        "content_manifest_sha256": content_sha256,
        "catalog": portable_contextworld_path(
            catalog_path,
            repo_root=repo_root,
        ),
        "catalog_sha256": file_sha256(catalog_path),
        "training_exclusion_manifest": portable_contextworld_path(
            exclusion_path,
            repo_root=repo_root,
        ),
        "training_exclusion_manifest_sha256": file_sha256(
            exclusion_path
        ),
        "counts": catalog["counts"],
    }
    report["status"] = (
        "passed" if all(report["checks"].values()) else "failed"
    )
    write_json(output_root / "build_report.json", report)
    return report


def load_validation_assets(
    catalog_path: Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog_path = Path(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assets: list[dict[str, Any]] = []
    for row in catalog["queries"]:
        path = resolve_contextworld_path(
            row["asset"],
            repo_root=repo_root,
        )
        if file_sha256(path) != row["asset_sha256"]:
            raise ValueError(f"Validation asset hash mismatch: {path}")
        arrays = dict(np.load(path, allow_pickle=False))
        payload_sha256 = canonical_sha256(
            {
                name: array_sha256(value)
                for name, value in sorted(arrays.items())
            }
        )
        if payload_sha256 != row["payload_sha256"]:
            raise ValueError(
                f"Validation payload hash mismatch: {path}"
            )
        assets.append({**row, **arrays})
    return catalog, assets


def score_validation_assets(
    adapter: Any,
    assets: list[dict[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    input_pixels = np.concatenate(
        [asset["history_pixels"] for asset in assets],
        axis=0,
    )
    action_blocks = np.concatenate(
        [asset["action_blocks"] for asset in assets],
        axis=0,
    )
    predicted = np.asarray(
        adapter.rollout_latents(
            input_pixels,
            action_blocks,
            batch_size=batch_size,
        )
    )
    if predicted.ndim < 3 or predicted.shape[1] != 1:
        raise ValueError(
            f"Expected one future latent per input, got {predicted.shape}"
        )
    predicted = predicted[:, -1]
    target_pixels = np.concatenate(
        [asset["target_pixels"] for asset in assets],
        axis=0,
    )
    encoded_targets = np.asarray(
        adapter.encode_pixels(
            target_pixels,
            batch_size=batch_size,
        )
    )
    predicted = predicted.reshape(len(predicted), -1)
    encoded_targets = encoded_targets.reshape(len(encoded_targets), -1)
    records: list[dict[str, Any]] = []
    for query_index, asset in enumerate(assets):
        prediction_start = query_index * len(HISTORY_DELAYS)
        target_start = query_index * len(TARGET_DELAYS)
        for history_index, history_delay in enumerate(HISTORY_DELAYS):
            prediction = predicted[prediction_start + history_index]
            for target_index, target_delay in enumerate(TARGET_DELAYS):
                target = encoded_targets[target_start + target_index]
                loss = float(np.mean((prediction - target) ** 2))
                records.append(
                    {
                        "query_id": asset["query_id"],
                        "eval_seed": int(asset["eval_seed"]),
                        "evaluation_index": int(
                            asset["evaluation_index"]
                        ),
                        "direction": asset["direction"],
                        "history_delay": int(history_delay),
                        "target_delay": int(target_delay),
                        "target_track": (
                            "training_seen"
                            if target_delay in SEEN_DELAYS
                            else "interpolation"
                        ),
                        "latent_mse": loss,
                    }
                )
    return {
        "records": records,
        "score_audit": {
            "queries": len(assets),
            "model_predictions": len(predicted),
            "target_encodings": len(encoded_targets),
            "loss_records": len(records),
            "online_environment_calls": 0,
            "privileged_fields_passed_to_adapter": [],
        },
    }


def summarize_validation_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_query[str(record["query_id"])].append(record)
    if len(by_query) != QUERY_COUNT:
        raise ValueError(
            f"Expected {QUERY_COUNT} independent queries, got {len(by_query)}"
        )

    query_metrics: list[dict[str, Any]] = []
    for query_id, rows in sorted(by_query.items()):
        loss = {
            (int(row["history_delay"]), int(row["target_delay"])): float(
                row["latent_mse"]
            )
            for row in rows
        }
        if len(loss) != len(HISTORY_DELAYS) * len(TARGET_DELAYS):
            raise ValueError(f"Incomplete 5x5 matrix for {query_id}")
        exemplar = rows[0]
        for target_delay in TARGET_DELAYS:
            matching = loss[(target_delay, target_delay)]
            other = [
                loss[(history_delay, target_delay)]
                for history_delay in HISTORY_DELAYS
                if history_delay != target_delay
            ]
            selected_history = min(
                HISTORY_DELAYS,
                key=lambda value: (
                    loss[(value, target_delay)],
                    value,
                ),
            )
            selected_target = min(
                TARGET_DELAYS,
                key=lambda value: (
                    loss[(target_delay, value)],
                    value,
                ),
            )
            query_metrics.append(
                {
                    "query_id": query_id,
                    "eval_seed": int(exemplar["eval_seed"]),
                    "direction": exemplar["direction"],
                    "target_delay": target_delay,
                    "target_track": (
                        "training_seen"
                        if target_delay in SEEN_DELAYS
                        else "interpolation"
                    ),
                    "matching_history_loss": matching,
                    "other_history_mean_loss": float(np.mean(other)),
                    "history_margin": float(np.mean(other) - matching),
                    "history_loss_ratio": float(
                        matching / max(float(np.mean(other)), 1e-12)
                    ),
                    "matching_history_strict_win": bool(
                        matching < min(other)
                    ),
                    "selected_history": int(selected_history),
                    "history_selection_correct": bool(
                        selected_history == target_delay
                    ),
                    "selected_target": int(selected_target),
                    "target_selection_correct": bool(
                        selected_target == target_delay
                    ),
                }
            )

    def aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "queries": len(values),
            "mean_matching_history_loss": float(
                np.mean(
                    [value["matching_history_loss"] for value in values]
                )
            ),
            "mean_other_history_loss": float(
                np.mean(
                    [value["other_history_mean_loss"] for value in values]
                )
            ),
            "mean_history_margin": float(
                np.mean([value["history_margin"] for value in values])
            ),
            "mean_history_loss_ratio": float(
                np.mean([value["history_loss_ratio"] for value in values])
            ),
            "matching_history_strict_win_rate": float(
                np.mean(
                    [
                        value["matching_history_strict_win"]
                        for value in values
                    ]
                )
            ),
            "history_selection_accuracy": float(
                np.mean(
                    [
                        value["history_selection_correct"]
                        for value in values
                    ]
                )
            ),
            "target_selection_accuracy": float(
                np.mean(
                    [
                        value["target_selection_correct"]
                        for value in values
                    ]
                )
            ),
        }

    by_delay = {
        str(delay): aggregate(
            [
                value
                for value in query_metrics
                if value["target_delay"] == delay
            ]
        )
        for delay in TARGET_DELAYS
    }
    by_track = {
        track: aggregate(
            [
                value
                for value in query_metrics
                if value["target_track"] == track
            ]
        )
        for track in ("training_seen", "interpolation")
    }
    by_delay_seed = {
        str(delay): {
            str(seed): aggregate(
                [
                    value
                    for value in query_metrics
                    if value["target_delay"] == delay
                    and value["eval_seed"] == seed
                ]
            )
            for seed in EVAL_SEEDS
        }
        for delay in TARGET_DELAYS
    }
    by_delay_direction = {
        str(delay): {
            direction: aggregate(
                [
                    value
                    for value in query_metrics
                    if value["target_delay"] == delay
                    and value["direction"] == direction
                ]
            )
            for direction in ("up", "down")
        }
        for delay in TARGET_DELAYS
    }
    return {
        "overall": aggregate(query_metrics),
        "by_target_delay": by_delay,
        "by_track": by_track,
        "by_target_delay_and_eval_seed": by_delay_seed,
        "by_target_delay_and_direction": by_delay_direction,
        "query_metrics": query_metrics,
    }


__all__ = [
    "EVAL_SEEDS",
    "HISTORY_DELAYS",
    "INTERPOLATION_DELAYS",
    "LOSS_RECORDS_PER_CHECKPOINT",
    "MODEL_PREDICTIONS_PER_CHECKPOINT",
    "QUERIES_PER_SEED",
    "QUERY_COUNT",
    "SEEN_DELAYS",
    "TARGET_DELAYS",
    "ActionDelayValidationAssignment",
    "audit_validation_catalog",
    "build_validation_asset",
    "build_validation_release",
    "file_sha256",
    "load_validation_assets",
    "score_validation_assets",
    "select_validation_assignments",
    "summarize_validation_records",
]
