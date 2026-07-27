from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.benchmarks.adapters import SpeedICLModelAdapter
from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)

from .hidden_passage import (
    ACTION_BLOCK,
    DIRECTIONS,
    HiddenPassageTemplate,
    make_templates,
    simulate_template,
    validate_pair,
)


TRUE_RULES = ("passable", "blocked")
HISTORY_CONDITIONS = (
    "observed_passable",
    "observed_blocked",
    "did_not_attempt_crossing",
)
SAME_HISTORY = {
    "passable": "observed_passable",
    "blocked": "observed_blocked",
}
OTHER_HISTORY = {
    "passable": "observed_blocked",
    "blocked": "observed_passable",
}
NO_EVIDENCE_HISTORY = "did_not_attempt_crossing"
MODEL_INPUT_KEYS = ("pixels", "action")
MODEL_HISTORY_TOKENS = 3
RAW_ACTION_BLOCKS = 3
DEFAULT_EPSILON = 1e-12
DEFAULT_MINIMUM_TARGET_ACCURACY = 0.5
DEFAULT_MINIMUM_STRICT_WIN_RATE = 0.5
DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_SEED = 20260725
DEFAULT_BOOTSTRAP_LOWER_BOUND = 0.0
ALL_HISTORIES_STRICT_V1 = "all_histories_strict_v1"
INFORMATIVE_HISTORY_RULE_SWITCH_V2 = (
    "informative_history_rule_switch_v2"
)
ALL_HISTORIES_BOOTSTRAP_METRICS = (
    "passable/same_vs_other_rule_history",
    "passable/same_vs_no_crossing_attempt",
    "blocked/same_vs_other_rule_history",
    "blocked/same_vs_no_crossing_attempt",
    "passable/matching_history_two_target_margin",
    "blocked/matching_history_two_target_margin",
)
INFORMATIVE_HISTORY_BOOTSTRAP_METRICS = (
    "passable/same_vs_other_rule_history",
    "blocked/same_vs_other_rule_history",
    "passable/matching_history_two_target_margin",
    "blocked/matching_history_two_target_margin",
)
# Backward-compatible name used by the v1 renderer and stored result tests.
PAIRED_BOOTSTRAP_METRICS = ALL_HISTORIES_BOOTSTRAP_METRICS

_INPUT_INVARIANT_KEYS = (
    "initial_observation",
    "history_pixels",
    "history_states",
    "history_raw_states",
    "history_actions",
    "query_pixels",
    "query_state",
    "query_action",
    "goal_pixels",
    "goal_state",
)


@dataclass(frozen=True)
class ValidationAssignment:
    template: HiddenPassageTemplate
    eval_seed: int
    evaluation_index: int


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
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _catalog_content_manifest_projection(
    catalog: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": catalog["protocol"],
        "bundles": [
            {
                key: row[key]
                for key in (
                    "query_id",
                    "static_query_id",
                    "eval_seed",
                    "evaluation_index",
                    "direction",
                    "template",
                    "payload_sha256",
                    "query_pixels_sha256",
                    "history_pixels_sha256",
                    "action_blocks_sha256",
                    "target_pixels_sha256",
                )
            }
            for row in catalog["bundles"]
        ],
    }


def _direction_sign(direction: str) -> float:
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction {direction!r}")
    return 1.0 if direction == "left_to_right" else -1.0


def _goal_for_template(template: HiddenPassageTemplate) -> tuple[float, float]:
    x = 190.0 if template.direction == "left_to_right" else 30.0
    return (x, float(template.reset_state[1]))


def candidate_templates(
    config: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> list[HiddenPassageTemplate]:
    generation = config["data"]["generation"]
    candidate_positions = [
        int(value) for value in generation["candidate_door_positions"]
    ]
    eval_only_positions = [
        int(value) for value in generation["eval_only_door_positions"]
    ]
    template_source = str(
        generation.get("template_source", "validation_geometry")
    )
    query_domain = str(generation.get("query_domain", "eval_only"))
    if len(set(eval_only_positions)) != len(eval_only_positions):
        raise ValueError("Eval-only door positions contain duplicates")
    if len(set(candidate_positions)) != len(candidate_positions):
        raise ValueError("Candidate door positions contain duplicates")

    if template_source == "validation_geometry":
        if query_domain != "eval_only":
            raise ValueError(
                "Validation geometry is reserved for the eval-only domain"
            )
        if candidate_positions != eval_only_positions:
            raise ValueError(
                "Every hidden-passage Validation door position must be "
                "eval-only"
            )
        if eval_only_positions != list(range(30, 195, 4)):
            raise ValueError(
                "The frozen 42-position eval-only door list changed"
            )
        if tuple(generation["directions"]) != DIRECTIONS:
            raise ValueError("The two frozen passage directions changed")
        templates = make_templates(
            door_positions=candidate_positions,
            directions=DIRECTIONS,
            doorway_offsets_px=generation["doorway_offsets_px"],
            catalog_seed=int(generation["catalog_seed"]),
        )
        result = [
            replace(template, goal_state=_goal_for_template(template))
            for template in templates
        ]
    elif template_source == "frozen_training_geometry":
        if query_domain != "training_seen":
            raise ValueError(
                "Frozen training geometry requires query_domain=training_seen"
            )
        if repo_root is None:
            raise ValueError(
                "repo_root is required for frozen training geometry"
            )
        from contextworld.evaluation.hidden_passage_h3_data import (
            door_splits_for_scale,
            templates_for_door,
        )
        from contextworld.synthesis.config import load_config

        training_config_path = resolve_contextworld_path(
            generation["training_data_config"],
            repo_root=repo_root,
        )
        expected_training_config_sha256 = str(
            generation["training_data_config_sha256"]
        )
        observed_training_config_sha256 = file_sha256(training_config_path)
        if (
            observed_training_config_sha256
            != expected_training_config_sha256
        ):
            raise ValueError(
                "Frozen hidden-passage training-data config hash mismatch: "
                f"expected={expected_training_config_sha256}, "
                f"observed={observed_training_config_sha256}"
            )
        training_config = load_config(training_config_path)
        training_scale = str(generation["training_scale"])
        source_split = str(generation["training_split"])
        splits = door_splits_for_scale(training_config, training_scale)
        split_positions = {
            "train": tuple(map(int, splits.train)),
            "loader_val": tuple(map(int, splits.val)),
        }
        if source_split not in split_positions:
            raise ValueError(
                "training_split must be train or loader_val for the "
                "training-seen diagnostic"
            )
        allowed_positions = set(split_positions[source_split])
        candidate_set = set(candidate_positions)
        if not candidate_set or not candidate_set <= allowed_positions:
            raise ValueError(
                "Training-seen candidate doors are not a non-empty subset "
                f"of the frozen {source_split} split"
            )
        if not bool(generation.get("allow_training_split_subset", False)):
            if candidate_set != allowed_positions:
                raise ValueError(
                    "Training-seen candidate doors must equal the complete "
                    f"frozen {source_split} split"
                )
        if candidate_set & set(eval_only_positions):
            raise ValueError(
                "Training-seen diagnostic doors overlap eval-only doors"
            )
        result = [
            template
            for door_position in candidate_positions
            for template in templates_for_door(
                training_config,
                scale=training_scale,
                door_position=door_position,
            )
        ]
        geometry_filter = generation.get("training_geometry_filter")
        if geometry_filter is not None:
            allowed_wall = {
                int(value)
                for value in geometry_filter["wall_distance_indices"]
            }
            allowed_offset = {
                int(value)
                for value in geometry_filter["doorway_offset_indices"]
            }

            def selected_geometry(template: HiddenPassageTemplate) -> bool:
                parts = template.template_id.rsplit("-", 2)
                if len(parts) != 3:
                    raise ValueError(
                        "Unexpected frozen training template ID: "
                        f"{template.template_id}"
                    )
                wall_slug, offset_slug = parts[-2:]
                if not (
                    wall_slug.startswith("w")
                    and offset_slug.startswith("o")
                ):
                    raise ValueError(
                        "Unexpected frozen training template geometry ID: "
                        f"{template.template_id}"
                    )
                return (
                    int(wall_slug[1:]) in allowed_wall
                    and int(offset_slug[1:]) in allowed_offset
                )

            result = [
                template for template in result if selected_geometry(template)
            ]
    else:
        raise ValueError(
            f"Unsupported hidden-passage template_source {template_source!r}"
        )

    if len(result) != int(generation["candidate_templates"]):
        raise ValueError(
            "Candidate template count differs from the frozen config"
        )
    if len({template.template_id for template in result}) != len(result):
        raise ValueError("Candidate template IDs are not unique")
    return result


def select_validation_assignments(
    config: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> list[ValidationAssignment]:
    """Select 300 queries as six disjoint 50-query groups.

    V2 uses each eval seed in its own without-replacement selection from the
    still-available template pool.  V1's historical one-shuffle partition is
    retained only so the already versioned V1 protocol remains loadable.
    """

    evaluation = config["evaluation"]
    eval_seeds = [int(value) for value in evaluation["eval_seeds"]]
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("eval_seeds must be unique")
    per_seed = int(evaluation["unique_queries_per_seed"])
    if per_seed % 2:
        raise ValueError("unique_queries_per_seed must be even")
    per_direction_per_seed = per_seed // 2
    needed_per_direction = len(eval_seeds) * per_direction_per_seed
    selection_seed = int(config["data"]["generation"]["assignment_seed"])
    selection_semantics = str(evaluation["eval_seed_semantics"])
    templates = candidate_templates(config, repo_root=repo_root)
    generation = config["data"]["generation"]
    if bool(
        generation.get(
            "deduplicate_query_pixels_before_selection",
            False,
        )
    ):
        unique_by_query_hash: dict[str, HiddenPassageTemplate] = {}
        for template in templates:
            rollout = simulate_template(template, rule="passable")
            query_hash = array_sha256(
                np.asarray(rollout["query_pixels"], dtype=np.uint8)
            )
            unique_by_query_hash.setdefault(query_hash, template)
        templates = sorted(
            unique_by_query_hash.values(),
            key=lambda item: item.template_id,
        )
        minimum_unique = int(
            generation["minimum_unique_candidate_queries"]
        )
        if len(templates) < minimum_unique:
            raise ValueError(
                "Frozen training geometry does not provide enough unique "
                f"query pixels: observed={len(templates)}, "
                f"required={minimum_unique}"
            )

    by_direction: dict[str, list[HiddenPassageTemplate]] = defaultdict(list)
    for template in templates:
        by_direction[template.direction].append(template)

    candidates_by_direction = {
        direction: sorted(
            by_direction[direction],
            key=lambda item: item.template_id,
        )
        for direction in DIRECTIONS
    }
    for direction, candidates in candidates_by_direction.items():
        if len(candidates) < needed_per_direction:
            raise ValueError(
                f"{direction} has {len(candidates)} candidates; "
                f"{needed_per_direction} are required"
            )

    assignments: list[ValidationAssignment] = []
    if selection_semantics == "disjoint_deterministic_query_partitions":
        for direction_index, direction in enumerate(DIRECTIONS):
            candidates = candidates_by_direction[direction]
            rng = np.random.default_rng(
                np.random.SeedSequence([selection_seed, direction_index])
            )
            selected = [
                candidates[int(index)]
                for index in rng.permutation(len(candidates))[
                    :needed_per_direction
                ]
            ]
            for seed_index, eval_seed in enumerate(eval_seeds):
                rows = selected[
                    seed_index
                    * per_direction_per_seed : (seed_index + 1)
                    * per_direction_per_seed
                ]
                index_offset = (
                    0
                    if direction == DIRECTIONS[0]
                    else per_direction_per_seed
                )
                assignments.extend(
                    ValidationAssignment(
                        template=template,
                        eval_seed=eval_seed,
                        evaluation_index=index_offset + local_index,
                    )
                    for local_index, template in enumerate(rows)
                )
    elif (
        selection_semantics
        == "seeded_disjoint_without_replacement_query_sets"
    ):
        remaining = {
            direction: list(candidates)
            for direction, candidates in candidates_by_direction.items()
        }
        for eval_seed in eval_seeds:
            for direction_index, direction in enumerate(DIRECTIONS):
                candidates = remaining[direction]
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [selection_seed, eval_seed, direction_index]
                    )
                )
                selected_indices = sorted(
                    map(
                        int,
                        rng.choice(
                            len(candidates),
                            size=per_direction_per_seed,
                            replace=False,
                        ),
                    ),
                    reverse=True,
                )
                rows = [candidates[index] for index in selected_indices]
                for index in selected_indices:
                    del candidates[index]
                rows.sort(key=lambda item: item.template_id)
                index_offset = (
                    0
                    if direction == DIRECTIONS[0]
                    else per_direction_per_seed
                )
                assignments.extend(
                    ValidationAssignment(
                        template=template,
                        eval_seed=eval_seed,
                        evaluation_index=index_offset + local_index,
                    )
                    for local_index, template in enumerate(rows)
                )
    else:
        raise ValueError(
            f"Unsupported eval_seed_semantics {selection_semantics!r}"
        )

    assignments.sort(
        key=lambda row: (
            row.eval_seed,
            row.evaluation_index,
            row.template.template_id,
        )
    )
    expected = len(eval_seeds) * per_seed
    configured_expected = int(
        config["data"]["generation"]["selected_unique_queries"]
    )
    if expected != configured_expected or expected != int(
        evaluation["unique_queries"]
    ):
        raise ValueError("Frozen Validation query counts disagree")
    if len(assignments) != expected:
        raise RuntimeError(
            f"Expected {expected} assignments, got {len(assignments)}"
        )
    template_ids = {row.template.template_id for row in assignments}
    if len(template_ids) != expected:
        raise RuntimeError("Validation template selection is not unique")
    for eval_seed in eval_seeds:
        rows = [row for row in assignments if row.eval_seed == eval_seed]
        counts = Counter(row.template.direction for row in rows)
        if counts != Counter(
            {direction: per_direction_per_seed for direction in DIRECTIONS}
        ):
            raise RuntimeError(
                f"Direction imbalance for eval seed {eval_seed}: {counts}"
            )
        if {row.evaluation_index for row in rows} != set(range(per_seed)):
            raise RuntimeError(
                f"Evaluation indices are incomplete for seed {eval_seed}"
            )
    return assignments


def make_no_attempt_template(
    template: HiddenPassageTemplate,
    query_state: np.ndarray,
) -> HiddenPassageTemplate:
    """Return a continuous history that reaches, but never probes past, the wall."""

    query = np.asarray(query_state, dtype=np.float32)
    if query.shape != (2,):
        raise ValueError(f"Expected a 2-D query state, got {query.shape}")
    sign = _direction_sign(template.direction)
    reset = (float(query[0] - 10.0 * sign), float(query[1]))
    return replace(template, reset_state=reset)


def _same_arrays(
    left: dict[str, Any],
    right: dict[str, Any],
    keys: Iterable[str],
) -> bool:
    return all(
        np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
        for key in keys
    )


def _no_attempt_stays_on_approach_side(
    template: HiddenPassageTemplate,
    rollout: dict[str, Any],
) -> bool:
    sign = _direction_sign(template.direction)
    boundary = 99.5 if sign > 0 else 124.5
    states = np.concatenate(
        [
            np.asarray(rollout["history_states"][0], dtype=np.float32)[None],
            np.asarray(rollout["history_raw_states"], dtype=np.float32),
        ],
        axis=0,
    )
    signed_overrun = sign * (states[:, 0] - boundary)
    return bool(np.max(signed_overrun) <= 1e-6)


def build_rollout_matrix(
    template: HiddenPassageTemplate,
    *,
    minimum_middle_state_gap_px: float = 8.0,
    minimum_future_state_gap_px: float = 20.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build one exact 2-future x 3-history offline evaluation payload."""

    passable = simulate_template(template, rule="passable")
    blocked = simulate_template(template, rule="blocked")
    pair_audit = validate_pair(
        template,
        passable,
        blocked,
        minimum_middle_state_gap_px=minimum_middle_state_gap_px,
        minimum_future_state_gap_px=minimum_future_state_gap_px,
    )
    no_attempt_template = make_no_attempt_template(
        template,
        passable["query_state"],
    )
    no_attempt_passable = simulate_template(
        no_attempt_template,
        rule="passable",
    )
    no_attempt_blocked = simulate_template(
        no_attempt_template,
        rule="blocked",
    )

    actions = {
        "observed_passable": np.concatenate(
            [
                np.asarray(passable["history_actions"], dtype=np.float32),
                np.asarray(passable["query_action"], dtype=np.float32)[None],
            ],
            axis=0,
        ),
        "observed_blocked": np.concatenate(
            [
                np.asarray(blocked["history_actions"], dtype=np.float32),
                np.asarray(blocked["query_action"], dtype=np.float32)[None],
            ],
            axis=0,
        ),
        "did_not_attempt_crossing": np.concatenate(
            [
                np.asarray(
                    no_attempt_passable["history_actions"],
                    dtype=np.float32,
                ),
                np.asarray(
                    no_attempt_passable["query_action"],
                    dtype=np.float32,
                )[None],
            ],
            axis=0,
        ),
    }
    histories = {
        "observed_passable": np.asarray(
            passable["history_pixels"],
            dtype=np.uint8,
        ),
        "observed_blocked": np.asarray(
            blocked["history_pixels"],
            dtype=np.uint8,
        ),
        "did_not_attempt_crossing": np.asarray(
            no_attempt_passable["history_pixels"],
            dtype=np.uint8,
        ),
    }
    history_states = {
        "observed_passable": np.asarray(
            passable["history_states"],
            dtype=np.float32,
        ),
        "observed_blocked": np.asarray(
            blocked["history_states"],
            dtype=np.float32,
        ),
        "did_not_attempt_crossing": np.asarray(
            no_attempt_passable["history_states"],
            dtype=np.float32,
        ),
    }
    checks = {
        "evidence_pair_passed": bool(pair_audit["passed"]),
        "no_attempt_rule_invariant": _same_arrays(
            no_attempt_passable,
            no_attempt_blocked,
            _INPUT_INVARIANT_KEYS,
        ),
        "all_histories_end_at_same_query_pixels": all(
            np.array_equal(value[-1], passable["query_pixels"])
            for value in histories.values()
        ),
        "all_histories_end_at_same_query_state": all(
            np.array_equal(value[-1], passable["query_state"])
            for value in history_states.values()
        ),
        "all_action_blocks_identical": (
            len({array_sha256(value) for value in actions.values()}) == 1
        ),
        "no_attempt_stays_on_approach_side": (
            _no_attempt_stays_on_approach_side(
                no_attempt_template,
                no_attempt_passable,
            )
            and _no_attempt_stays_on_approach_side(
                no_attempt_template,
                no_attempt_blocked,
            )
        ),
        "target_pixels_differ": not np.array_equal(
            passable["target_pixels"],
            blocked["target_pixels"],
        ),
        "blocked_target_is_unchanged_query": np.array_equal(
            blocked["target_pixels"],
            blocked["query_pixels"],
        ),
        "passable_target_is_not_query": not np.array_equal(
            passable["target_pixels"],
            passable["query_pixels"],
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            f"Hidden-passage matrix failed for {template.template_id}: {failed}"
        )

    arrays: dict[str, np.ndarray] = {
        "query_pixels": np.asarray(passable["query_pixels"], dtype=np.uint8),
        "query_state": np.asarray(passable["query_state"], dtype=np.float32),
        "query_action": np.asarray(passable["query_action"], dtype=np.float32),
        "goal_pixels": np.asarray(passable["goal_pixels"], dtype=np.uint8),
        "goal_state": np.asarray(passable["goal_state"], dtype=np.float32),
        "target_passable_pixels": np.asarray(
            passable["target_pixels"],
            dtype=np.uint8,
        ),
        "target_passable_state": np.asarray(
            passable["target_state"],
            dtype=np.float32,
        ),
        "target_blocked_pixels": np.asarray(
            blocked["target_pixels"],
            dtype=np.uint8,
        ),
        "target_blocked_state": np.asarray(
            blocked["target_state"],
            dtype=np.float32,
        ),
    }
    for condition in HISTORY_CONDITIONS:
        arrays[f"{condition}_history_pixels"] = histories[condition]
        arrays[f"{condition}_history_states"] = history_states[condition]
        arrays[f"{condition}_action_blocks"] = actions[condition]
    arrays["did_not_attempt_crossing_history_raw_states"] = np.asarray(
        no_attempt_passable["history_raw_states"],
        dtype=np.float32,
    )
    arrays["observed_passable_history_raw_states"] = np.asarray(
        passable["history_raw_states"],
        dtype=np.float32,
    )
    arrays["observed_blocked_history_raw_states"] = np.asarray(
        blocked["history_raw_states"],
        dtype=np.float32,
    )
    audit = {
        "passed": True,
        "checks": checks,
        "pair": pair_audit,
        "no_attempt_reset_state": list(no_attempt_template.reset_state),
        "query_state": arrays["query_state"].tolist(),
        "target_states": {
            rule: arrays[f"target_{rule}_state"].tolist()
            for rule in TRUE_RULES
        },
        "history_action_sha256": {
            condition: array_sha256(actions[condition])
            for condition in HISTORY_CONDITIONS
        },
    }
    return arrays, audit


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _payload_hashes(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "query_pixels_sha256": array_sha256(arrays["query_pixels"]),
        "history_pixels_sha256": {
            condition: array_sha256(
                arrays[f"{condition}_history_pixels"]
            )
            for condition in HISTORY_CONDITIONS
        },
        "action_blocks_sha256": {
            condition: array_sha256(
                arrays[f"{condition}_action_blocks"]
            )
            for condition in HISTORY_CONDITIONS
        },
        "target_pixels_sha256": {
            rule: array_sha256(arrays[f"target_{rule}_pixels"])
            for rule in TRUE_RULES
        },
    }


def build_validation_catalog(
    *,
    config: dict[str, Any],
    repo_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate and freeze the 300-query offline Validation catalog."""

    assignments = select_validation_assignments(
        config,
        repo_root=repo_root,
    )
    payload_root = output_root / "payloads"
    payload_root.mkdir(parents=True, exist_ok=False)
    gates = config["gates"]
    bundles = []
    query_hashes = set()
    manifest_rows = []
    all_physics_passed = True
    for assignment in assignments:
        template = assignment.template
        arrays, physical_audit = build_rollout_matrix(
            template,
            minimum_middle_state_gap_px=float(
                gates["minimum_middle_state_gap_px"]
            ),
            minimum_future_state_gap_px=float(
                gates["minimum_future_state_gap_px"]
            ),
        )
        query_hash = array_sha256(arrays["query_pixels"])
        if query_hash in query_hashes:
            raise RuntimeError(
                f"Duplicate validation query pixels: {template.template_id}"
            )
        query_hashes.add(query_hash)
        static_query_id = f"h3-hidden-{template.template_id}"
        query_id = (
            f"s{assignment.eval_seed}-"
            f"e{assignment.evaluation_index:03d}-{static_query_id}"
        )
        payload_path = payload_root / f"{query_id}.npz"
        _atomic_savez(payload_path, arrays)
        hashes = _payload_hashes(arrays)
        payload_hash = file_sha256(payload_path)
        bundle = {
            "query_id": query_id,
            "static_query_id": static_query_id,
            "eval_seed": assignment.eval_seed,
            "evaluation_index": assignment.evaluation_index,
            "direction": template.direction,
            "template": asdict(template),
            "payload": portable_contextworld_path(
                payload_path,
                repo_root=repo_root,
            ),
            "payload_sha256": payload_hash,
            **hashes,
            "physical_audit": physical_audit,
        }
        bundles.append(bundle)
        all_physics_passed = (
            all_physics_passed and bool(physical_audit["passed"])
        )
        manifest_rows.append(
            {
                "query_id": query_id,
                "static_query_id": static_query_id,
                "template_id": template.template_id,
                "query_pixels_sha256": query_hash,
                "query_state": arrays["query_state"].tolist(),
                "target_pixels_sha256": hashes["target_pixels_sha256"],
            }
        )

    eval_seeds = [int(value) for value in config["evaluation"]["eval_seeds"]]
    per_seed = int(config["evaluation"]["unique_queries_per_seed"])
    generation = config["data"]["generation"]
    eval_only_door_positions = [
        int(value)
        for value in generation["eval_only_door_positions"]
    ]
    candidate_door_positions = [
        int(value) for value in generation["candidate_door_positions"]
    ]
    query_domain = str(generation.get("query_domain", "eval_only"))
    expected_query_count = int(config["evaluation"]["unique_queries"])
    selected_doors = {
        int(row["template"]["door_position"]) for row in bundles
    }
    by_seed = Counter(row["eval_seed"] for row in bundles)
    by_seed_direction = Counter(
        (row["eval_seed"], row["direction"]) for row in bundles
    )
    expected_per_direction = per_seed // 2
    checks = {
        "exact_unique_query_count": (
            len(query_hashes) == expected_query_count
        ),
        "exact_queries_per_eval_seed": (
            by_seed == Counter({seed: per_seed for seed in eval_seeds})
        ),
        "direction_balance_per_eval_seed": all(
            by_seed_direction[(seed, direction)]
            == expected_per_direction
            for seed in eval_seeds
            for direction in DIRECTIONS
        ),
        "all_physics_checks_passed": all_physics_passed,
        "all_payloads_have_distinct_paths": (
            len({row["payload"] for row in bundles}) == len(bundles)
        ),
        "all_eval_seed_partitions_are_disjoint": (
            len({row["static_query_id"] for row in bundles}) == len(bundles)
        ),
        "all_selected_doors_match_declared_domain": (
            selected_doors <= set(candidate_door_positions)
        ),
        "declared_door_positions_are_unique": (
            len(candidate_door_positions)
            == len(set(candidate_door_positions))
        ),
        "query_domain_contract": (
            (
                query_domain == "eval_only"
                and candidate_door_positions == eval_only_door_positions
                and eval_only_door_positions == list(range(30, 195, 4))
            )
            or (
                query_domain == "training_seen"
                and not (
                    set(candidate_door_positions)
                    & set(eval_only_door_positions)
                )
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Validation build failed: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )

    protocol = {
        "history_tokens": MODEL_HISTORY_TOKENS,
        "raw_steps_per_action_block": ACTION_BLOCK,
        "model_action_blocks": RAW_ACTION_BLOCKS,
        "true_future_rules": list(TRUE_RULES),
        "history_conditions": list(HISTORY_CONDITIONS),
        "matrix": "2 true futures x 3 histories",
        "queries_are_paired_across_all_six_cells": True,
        "eval_seed_selection_semantics": str(
            config["evaluation"]["eval_seed_semantics"]
        ),
        "eval_seed_values_used_in_selection": (
            str(config["evaluation"]["eval_seed_semantics"])
            == "seeded_disjoint_without_replacement_query_sets"
        ),
        "eval_seed_labels_are_disjoint_query_partitions": True,
        "environment_calls_during_scoring": 0,
        "primary_metric": "frozen_true_next_frame_native_latent_mse",
        "unchanged_query_relative_loss_is_forbidden": True,
    }
    catalog = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "frozen_before_model_scoring",
        "protocol": protocol,
        "bundles": bundles,
        "summary": {
            "unique_queries": len(bundles),
            "eval_seeds": eval_seeds,
            "unique_queries_per_eval_seed": per_seed,
            "directions_per_eval_seed": {
                direction: expected_per_direction
                for direction in DIRECTIONS
            },
            "matrix_cells": len(TRUE_RULES) * len(HISTORY_CONDITIONS),
            "records_per_checkpoint": (
                len(bundles) * len(TRUE_RULES) * len(HISTORY_CONDITIONS)
            ),
            "model_predictions_per_checkpoint": (
                len(bundles) * len(HISTORY_CONDITIONS)
            ),
            "selection_sha256_by_eval_seed": {
                str(seed): canonical_sha256(
                    [
                        row["static_query_id"]
                        for row in bundles
                        if int(row["eval_seed"]) == seed
                    ]
                )
                for seed in eval_seeds
            },
        },
        "training_exclusion": {
            "policy": (
                "fail_closed"
                if query_domain == "eval_only"
                else "not_applicable_training_seen_diagnostic"
            ),
            "eval_only_door_positions": eval_only_door_positions,
            "exclude_all_listed_door_positions": (
                query_domain == "eval_only"
            ),
            "exclude_selected_template_ids": query_domain == "eval_only",
            "exclude_selected_query_pixel_hashes": (
                query_domain == "eval_only"
            ),
            "query_domain": query_domain,
            "candidate_door_positions": candidate_door_positions,
            "query_records": manifest_rows,
        },
        "training_exclusion_manifest": manifest_rows,
    }
    catalog["content_manifest_sha256"] = canonical_sha256(
        _catalog_content_manifest_projection(catalog)
    )
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "checks": checks,
        "summary": catalog["summary"],
        "content_manifest_sha256": catalog["content_manifest_sha256"],
    }
    return catalog, report


def _load_payload(
    bundle: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    payload_path = resolve_contextworld_path(
        bundle["payload"],
        repo_root=repo_root,
    )
    if file_sha256(payload_path) != bundle["payload_sha256"]:
        raise RuntimeError(f"Payload hash mismatch: {payload_path}")
    with np.load(payload_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}

    query_pixels = np.asarray(arrays["query_pixels"], dtype=np.uint8)
    if array_sha256(query_pixels) != bundle["query_pixels_sha256"]:
        raise RuntimeError(f"Query hash mismatch: {payload_path}")
    histories = {}
    actions = {}
    for condition in HISTORY_CONDITIONS:
        history = np.asarray(
            arrays[f"{condition}_history_pixels"],
            dtype=np.uint8,
        )
        action = np.asarray(
            arrays[f"{condition}_action_blocks"],
            dtype=np.float32,
        )
        if history.ndim != 4 or history.shape[0] != MODEL_HISTORY_TOKENS:
            raise RuntimeError(
                f"Invalid History-3 pixels for {condition}: {history.shape}"
            )
        if action.shape != (RAW_ACTION_BLOCKS, ACTION_BLOCK, 2):
            raise RuntimeError(
                f"Invalid action blocks for {condition}: {action.shape}"
            )
        if not np.array_equal(history[-1], query_pixels):
            raise RuntimeError(
                f"{condition} does not end at the frozen query"
            )
        if (
            array_sha256(history)
            != bundle["history_pixels_sha256"][condition]
        ):
            raise RuntimeError(f"History hash mismatch: {condition}")
        if (
            array_sha256(action)
            != bundle["action_blocks_sha256"][condition]
        ):
            raise RuntimeError(f"Action hash mismatch: {condition}")
        histories[condition] = history
        actions[condition] = action
    if len({array_sha256(value) for value in actions.values()}) != 1:
        raise RuntimeError("The three history conditions use different actions")

    targets = {}
    for rule in TRUE_RULES:
        target = np.asarray(
            arrays[f"target_{rule}_pixels"],
            dtype=np.uint8,
        )
        if target.shape != query_pixels.shape:
            raise RuntimeError(f"Target shape mismatch for {rule}")
        if (
            array_sha256(target)
            != bundle["target_pixels_sha256"][rule]
        ):
            raise RuntimeError(f"Target hash mismatch for {rule}")
        targets[rule] = target
    if np.array_equal(targets["passable"], targets["blocked"]):
        raise RuntimeError("The two frozen true futures are identical")
    if not np.array_equal(targets["blocked"], query_pixels):
        raise RuntimeError(
            "Blocked true future must be the unchanged query frame"
        )
    return {
        "query_id": str(bundle["query_id"]),
        "static_query_id": str(bundle["static_query_id"]),
        "eval_seed": int(bundle["eval_seed"]),
        "evaluation_index": int(bundle["evaluation_index"]),
        "direction": str(bundle["direction"]),
        "template_id": str(bundle["template"]["template_id"]),
        "query_pixels": query_pixels,
        "histories": histories,
        "actions": actions,
        "targets": targets,
    }


def audit_validation_assets(
    assets: list[dict[str, Any]],
    *,
    eval_seeds: Iterable[int],
    unique_queries_per_seed: int,
) -> dict[str, Any]:
    seeds = tuple(map(int, eval_seeds))
    expected = len(seeds) * int(unique_queries_per_seed)
    query_ids = [str(row["query_id"]) for row in assets]
    static_ids = [str(row["static_query_id"]) for row in assets]
    query_hashes = [
        array_sha256(np.asarray(row["query_pixels"], dtype=np.uint8))
        for row in assets
    ]
    by_seed = Counter(int(row["eval_seed"]) for row in assets)
    by_seed_direction = Counter(
        (int(row["eval_seed"]), str(row["direction"])) for row in assets
    )
    expected_direction = int(unique_queries_per_seed) // 2
    checks = {
        "exact_query_count": len(assets) == expected,
        "query_ids_unique": len(set(query_ids)) == expected,
        "static_queries_unique": len(set(static_ids)) == expected,
        "query_pixels_unique": len(set(query_hashes)) == expected,
        "eval_seed_partitions_exact": (
            by_seed
            == Counter(
                {seed: int(unique_queries_per_seed) for seed in seeds}
            )
        ),
        "directions_balanced_in_each_partition": all(
            by_seed_direction[(seed, direction)] == expected_direction
            for seed in seeds
            for direction in DIRECTIONS
        ),
        "three_histories_present": all(
            set(row["histories"]) == set(HISTORY_CONDITIONS)
            for row in assets
        ),
        "two_targets_present": all(
            set(row["targets"]) == set(TRUE_RULES) for row in assets
        ),
        "all_histories_share_query": all(
            all(
                np.array_equal(
                    np.asarray(row["histories"][condition])[-1],
                    row["query_pixels"],
                )
                for condition in HISTORY_CONDITIONS
            )
            for row in assets
        ),
        "all_history_actions_identical": all(
            len(
                {
                    array_sha256(np.asarray(row["actions"][condition]))
                    for condition in HISTORY_CONDITIONS
                }
            )
            == 1
            for row in assets
        ),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "unique_queries": len(set(static_ids)),
        "eval_seeds": list(seeds),
        "queries_by_eval_seed": dict(sorted(by_seed.items())),
        "queries_by_eval_seed_and_direction": {
            f"s{seed}/{direction}": by_seed_direction[(seed, direction)]
            for seed in seeds
            for direction in DIRECTIONS
        },
    }


def load_validation_assets(
    catalog_path: Path,
    *,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog_path = Path(catalog_path).resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("status") != "frozen_before_model_scoring":
        raise ValueError("Validation catalog is not frozen before scoring")
    protocol = catalog["protocol"]
    if tuple(protocol["true_future_rules"]) != TRUE_RULES:
        raise ValueError("True-future rules changed")
    if tuple(protocol["history_conditions"]) != HISTORY_CONDITIONS:
        raise ValueError("History conditions changed")
    if int(protocol["history_tokens"]) != MODEL_HISTORY_TOKENS:
        raise ValueError("History token count changed")
    if int(protocol["raw_steps_per_action_block"]) != ACTION_BLOCK:
        raise ValueError("Action block changed")
    if not protocol.get("unchanged_query_relative_loss_is_forbidden"):
        raise ValueError("Unsafe unchanged-query normalization was enabled")
    observed_content_hash = canonical_sha256(
        _catalog_content_manifest_projection(catalog)
    )
    expected_content_hash = str(catalog["content_manifest_sha256"])
    if observed_content_hash != expected_content_hash:
        raise RuntimeError(
            "Validation catalog content manifest mismatch: "
            f"expected={expected_content_hash}, "
            f"observed={observed_content_hash}"
        )

    assets = [
        _load_payload(bundle, repo_root=repo_root)
        for bundle in sorted(
            catalog["bundles"],
            key=lambda row: (
                int(row["eval_seed"]),
                int(row["evaluation_index"]),
            ),
        )
    ]
    summary = catalog["summary"]
    audit = audit_validation_assets(
        assets,
        eval_seeds=summary["eval_seeds"],
        unique_queries_per_seed=int(
            summary["unique_queries_per_eval_seed"]
        ),
    )
    if not audit["passed"]:
        failed = [
            name for name, passed in audit["checks"].items() if not passed
        ]
        raise RuntimeError(f"Validation catalog audit failed: {failed}")
    audit.update(
        {
            "catalog": str(catalog_path),
            "catalog_sha256": file_sha256(catalog_path),
            "content_manifest_sha256": catalog[
                "content_manifest_sha256"
            ],
            "content_manifest_recomputed_sha256": observed_content_hash,
            "online_environment_calls": 0,
        }
    )
    return assets, audit


def _adapter_protocol_audit(adapter: SpeedICLModelAdapter) -> dict[str, Any]:
    protocol = adapter.protocol
    checks = {
        "history_tokens": int(protocol.history_tokens) == MODEL_HISTORY_TOKENS,
        "action_block_raw_steps": (
            int(protocol.action_block_raw_steps) == ACTION_BLOCK
        ),
        "action_dim": int(protocol.action_dim) == 2,
        "at_least_one_future": int(protocol.future_action_blocks) >= 1,
        "native_target_encoder": bool(protocol.native_target_encoder),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Adapter protocol mismatch: {checks}")
    return {"passed": True, "checks": checks}


def score_validation_assets(
    adapter: SpeedICLModelAdapter,
    assets: list[dict[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Score only frozen arrays; this function never constructs an environment."""

    protocol_audit = _adapter_protocol_audit(adapter)
    state_before = adapter.frozen_state_hash()
    samples: list[tuple[dict[str, Any], str]] = [
        (asset, condition)
        for asset in assets
        for condition in HISTORY_CONDITIONS
    ]
    input_pixels = np.stack(
        [asset["histories"][condition] for asset, condition in samples]
    ).astype(np.uint8)
    raw_actions = np.stack(
        [asset["actions"][condition] for asset, condition in samples]
    ).astype(np.float32)
    predicted = np.asarray(
        adapter.rollout_latents(
            input_pixels,
            raw_actions,
            batch_size=int(batch_size),
        )
    )
    if predicted.ndim < 3 or predicted.shape[:2] != (len(samples), 1):
        raise RuntimeError(
            "Adapter must return one next latent per History-3 sample, "
            f"got {predicted.shape}"
        )
    predicted = predicted[:, 0].reshape(len(samples), -1).astype(np.float64)

    target_pixels = np.stack(
        [
            asset["targets"][rule]
            for asset in assets
            for rule in TRUE_RULES
        ]
    ).astype(np.uint8)
    encoded_targets = np.asarray(
        adapter.encode_pixels(
            target_pixels,
            batch_size=int(batch_size),
        )
    ).reshape(len(assets), len(TRUE_RULES), -1)
    encoded_targets = encoded_targets.astype(np.float64)
    if encoded_targets.shape[-1] != predicted.shape[-1]:
        raise RuntimeError(
            "Predicted and target latent dimensions differ: "
            f"{predicted.shape[-1]} != {encoded_targets.shape[-1]}"
        )
    if not np.isfinite(predicted).all() or not np.isfinite(
        encoded_targets
    ).all():
        raise RuntimeError("Predicted or target latents contain non-finite values")
    target_pair_latent_mse = np.mean(
        np.square(
            encoded_targets[:, 0] - encoded_targets[:, 1]
        ),
        axis=-1,
    )

    target_by_query = {
        str(asset["query_id"]): {
            rule: encoded_targets[asset_index, rule_index]
            for rule_index, rule in enumerate(TRUE_RULES)
        }
        for asset_index, asset in enumerate(assets)
    }
    asset_index_by_query = {
        str(asset["query_id"]): asset_index
        for asset_index, asset in enumerate(assets)
    }
    rows = []
    prediction_ties = 0
    for prediction, (asset, condition) in zip(predicted, samples):
        targets = target_by_query[str(asset["query_id"])]
        losses = {
            rule: float(np.mean(np.square(prediction - targets[rule])))
            for rule in TRUE_RULES
        }
        if losses["passable"] < losses["blocked"]:
            predicted_rule = "passable"
        elif losses["blocked"] < losses["passable"]:
            predicted_rule = "blocked"
        else:
            predicted_rule = "tie"
            prediction_ties += 1
        asset_index = asset_index_by_query[str(asset["query_id"])]
        for true_rule in TRUE_RULES:
            other_rule = (
                "blocked" if true_rule == "passable" else "passable"
            )
            rows.append(
                {
                    "evaluation_id": (
                        f"{asset['query_id']}/{true_rule}/{condition}"
                    ),
                    "query_id": str(asset["query_id"]),
                    "static_query_id": str(asset["static_query_id"]),
                    "template_id": str(asset["template_id"]),
                    "eval_seed": int(asset["eval_seed"]),
                    "evaluation_index": int(asset["evaluation_index"]),
                    "direction": str(asset["direction"]),
                    "true_rule": true_rule,
                    "history_condition": condition,
                    "true_next_frame_latent_mse": losses[true_rule],
                    "other_target_latent_mse": losses[other_rule],
                    "two_target_margin": (
                        losses[other_rule] - losses[true_rule]
                    ),
                    "predicted_rule": predicted_rule,
                    "true_target_closer": predicted_rule == true_rule,
                    "target_pair_latent_mse": float(
                        target_pair_latent_mse[asset_index]
                    ),
                }
            )
    state_after = adapter.frozen_state_hash()
    expected_rows = len(assets) * len(TRUE_RULES) * len(HISTORY_CONDITIONS)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} score rows, got {len(rows)}"
        )
    if state_before != state_after:
        raise RuntimeError("Adapter state changed during frozen scoring")
    return {
        "records": rows,
        "score_audit": {
            "passed": True,
            "unique_queries": len(assets),
            "model_predictions": len(samples),
            "target_encodings": len(target_pixels),
            "target_latent_separation": {
                "minimum_mse": float(np.min(target_pair_latent_mse)),
                "mean_mse": float(np.mean(target_pair_latent_mse)),
                "median_mse": float(np.median(target_pair_latent_mse)),
                "maximum_mse": float(np.max(target_pair_latent_mse)),
                "collapsed_queries_at_default_epsilon": int(
                    np.sum(target_pair_latent_mse <= DEFAULT_EPSILON)
                ),
                "passed_at_default_epsilon": bool(
                    np.all(target_pair_latent_mse > DEFAULT_EPSILON)
                ),
            },
            "two_target_ties": {
                "unique_predictions": len(samples),
                "ties": prediction_ties,
                "tie_rate": float(prediction_ties / len(samples)),
            },
            "records": len(rows),
            "expected_records": expected_rows,
            "model_input_keys": list(MODEL_INPUT_KEYS),
            "privileged_fields_passed_to_adapter": [],
            "online_environment_calls": 0,
            "adapter_protocol": protocol_audit,
            "frozen_state_hash_before": state_before,
            "frozen_state_hash_after": state_after,
        },
    }


def paired_effect_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(
        dict
    )
    for row in records:
        key = (str(row["query_id"]), str(row["true_rule"]))
        condition = str(row["history_condition"])
        if condition in grouped[key]:
            raise RuntimeError(f"Duplicate score row: {key}/{condition}")
        grouped[key][condition] = row

    paired = []
    for (query_id, true_rule), values in sorted(grouped.items()):
        if set(values) != set(HISTORY_CONDITIONS):
            raise RuntimeError(
                f"Incomplete 2x3 matrix for {query_id}/{true_rule}"
            )
        same = float(
            values[SAME_HISTORY[true_rule]][
                "true_next_frame_latent_mse"
            ]
        )
        other = float(
            values[OTHER_HISTORY[true_rule]][
                "true_next_frame_latent_mse"
            ]
        )
        no_evidence = float(
            values[NO_EVIDENCE_HISTORY][
                "true_next_frame_latent_mse"
            ]
        )
        identity = next(iter(values.values()))
        paired.append(
            {
                "query_id": query_id,
                "static_query_id": str(identity["static_query_id"]),
                "eval_seed": int(identity["eval_seed"]),
                "evaluation_index": int(identity["evaluation_index"]),
                "direction": str(identity["direction"]),
                "true_rule": true_rule,
                "same_history_loss": same,
                "other_history_loss": other,
                "no_evidence_history_loss": no_evidence,
                "same_vs_other_advantage": other - same,
                "same_vs_no_evidence_advantage": no_evidence - same,
                "same_vs_other_symmetric_contrast": (
                    (other - same) / max(other + same, DEFAULT_EPSILON)
                ),
                "same_vs_no_evidence_symmetric_contrast": (
                    (no_evidence - same)
                    / max(no_evidence + same, DEFAULT_EPSILON)
                ),
                "strict_win": bool(same < other and same < no_evidence),
                "same_history_true_target_closer": bool(
                    values[SAME_HISTORY[true_rule]][
                        "true_target_closer"
                    ]
                ),
                "same_history_two_target_margin": float(
                    values[SAME_HISTORY[true_rule]]["two_target_margin"]
                ),
            }
        )
    return paired


def _mean_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize zero paired rows")
    return {
        "pairs": len(rows),
        "native_latent_mse": {
            "same_rule_history": float(
                np.mean([row["same_history_loss"] for row in rows])
            ),
            "other_rule_history": float(
                np.mean([row["other_history_loss"] for row in rows])
            ),
            "no_crossing_attempt_history": float(
                np.mean(
                    [row["no_evidence_history_loss"] for row in rows]
                )
            ),
        },
        "paired_advantage": {
            "same_vs_other_rule_history": float(
                np.mean(
                    [row["same_vs_other_advantage"] for row in rows]
                )
            ),
            "same_vs_no_crossing_attempt": float(
                np.mean(
                    [
                        row["same_vs_no_evidence_advantage"]
                        for row in rows
                    ]
                )
            ),
        },
        "symmetric_contrast": {
            "same_vs_other_rule_history": float(
                np.mean(
                    [
                        row["same_vs_other_symmetric_contrast"]
                        for row in rows
                    ]
                )
            ),
            "same_vs_no_crossing_attempt": float(
                np.mean(
                    [
                        row["same_vs_no_evidence_symmetric_contrast"]
                        for row in rows
                    ]
                )
            ),
        },
        "strict_win_rate": float(
            np.mean([row["strict_win"] for row in rows])
        ),
        "matching_vs_opposite_history_win_rate": float(
            np.mean(
                [
                    row["same_vs_other_advantage"] > 0.0
                    for row in rows
                ]
            )
        ),
        "same_history_two_target_accuracy": float(
            np.mean(
                [row["same_history_true_target_closer"] for row in rows]
            )
        ),
}


def _required_bootstrap_metrics(
    decision_contract: str,
) -> tuple[str, ...]:
    if decision_contract == ALL_HISTORIES_STRICT_V1:
        return ALL_HISTORIES_BOOTSTRAP_METRICS
    if decision_contract == INFORMATIVE_HISTORY_RULE_SWITCH_V2:
        return INFORMATIVE_HISTORY_BOOTSTRAP_METRICS
    raise ValueError(
        f"Unsupported hidden-passage decision contract "
        f"{decision_contract!r}"
    )


def _positive_effect(
    summary: dict[str, Any],
    *,
    decision_contract: str,
) -> bool:
    effects = summary["paired_advantage"]
    if decision_contract == ALL_HISTORIES_STRICT_V1:
        return bool(
            effects["same_vs_other_rule_history"] > 0.0
            and effects["same_vs_no_crossing_attempt"] > 0.0
        )
    if decision_contract == INFORMATIVE_HISTORY_RULE_SWITCH_V2:
        return bool(effects["same_vs_other_rule_history"] > 0.0)
    raise ValueError(
        f"Unsupported hidden-passage decision contract "
        f"{decision_contract!r}"
    )


def _decision_gate_config(gates: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve the preregistered checkpoint-decision thresholds."""

    source = dict(gates or {})
    decision_contract = str(
        source.get("decision_contract", ALL_HISTORIES_STRICT_V1)
    )
    required_bootstrap_metrics = _required_bootstrap_metrics(
        decision_contract
    )
    bootstrap = dict(source.get("paired_bootstrap", {}))
    if bootstrap:
        expected_metadata = {
            "unit": "static_query_within_eval_seed_direction",
            "strata": "eval_seed_x_direction",
            "method": "percentile",
        }
        mismatched = {
            key: bootstrap.get(key)
            for key, expected in expected_metadata.items()
            if bootstrap.get(key) != expected
        }
        if mismatched:
            raise ValueError(
                "Paired bootstrap protocol metadata changed: "
                f"{mismatched}"
            )
        if (
            tuple(bootstrap.get("required_metrics", ()))
            != required_bootstrap_metrics
        ):
            raise ValueError("Paired bootstrap required metrics changed")
    resolved = {
        "decision_contract": decision_contract,
        "required_bootstrap_metrics": required_bootstrap_metrics,
        "minimum_same_history_two_target_accuracy_exclusive": float(
            source.get(
                "minimum_same_history_two_target_accuracy_exclusive",
                DEFAULT_MINIMUM_TARGET_ACCURACY,
            )
        ),
        "minimum_strict_win_rate_exclusive": float(
            source.get(
                "minimum_strict_win_rate_exclusive",
                DEFAULT_MINIMUM_STRICT_WIN_RATE,
            )
        ),
        "minimum_matching_vs_opposite_history_win_rate_exclusive": float(
            source.get(
                "minimum_matching_vs_opposite_history_win_rate_exclusive",
                DEFAULT_MINIMUM_STRICT_WIN_RATE,
            )
        ),
        "minimum_target_pair_latent_mse_exclusive": float(
            source.get(
                "minimum_target_pair_latent_mse_exclusive",
                DEFAULT_EPSILON,
            )
        ),
        "bootstrap_resamples": int(
            bootstrap.get("resamples", DEFAULT_BOOTSTRAP_RESAMPLES)
        ),
        "bootstrap_confidence": float(
            bootstrap.get("confidence", DEFAULT_BOOTSTRAP_CONFIDENCE)
        ),
        "bootstrap_seed": int(
            bootstrap.get("seed", DEFAULT_BOOTSTRAP_SEED)
        ),
        "minimum_bootstrap_lower_bound_exclusive": float(
            bootstrap.get(
                "minimum_lower_bound_exclusive",
                DEFAULT_BOOTSTRAP_LOWER_BOUND,
            )
        ),
    }
    accuracy = resolved[
        "minimum_same_history_two_target_accuracy_exclusive"
    ]
    strict_win = resolved["minimum_strict_win_rate_exclusive"]
    matching_win = resolved[
        "minimum_matching_vs_opposite_history_win_rate_exclusive"
    ]
    confidence = resolved["bootstrap_confidence"]
    if not 0.0 <= accuracy < 1.0:
        raise ValueError(
            "Two-target accuracy threshold must be in [0, 1)"
        )
    if not 0.0 <= strict_win < 1.0:
        raise ValueError("Strict-win threshold must be in [0, 1)")
    if not 0.0 <= matching_win < 1.0:
        raise ValueError(
            "Matching-vs-opposite win threshold must be in [0, 1)"
        )
    if resolved["bootstrap_resamples"] < 100:
        raise ValueError("Paired bootstrap requires at least 100 resamples")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap confidence must be in (0, 1)")
    if resolved["minimum_target_pair_latent_mse_exclusive"] < 0.0:
        raise ValueError("Target latent separation threshold must be nonnegative")
    return resolved


def _paired_bootstrap_summary(
    rows: list[dict[str, Any]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    """Bootstrap paired queries within each eval-seed x direction stratum."""

    by_static_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        static_query_id = str(row["static_query_id"])
        true_rule = str(row["true_rule"])
        if true_rule in by_static_query[static_query_id]:
            raise RuntimeError(
                "Duplicate paired row for "
                f"{static_query_id}/{true_rule}"
            )
        by_static_query[static_query_id][true_rule] = row
    incomplete = {
        query_id: sorted(values)
        for query_id, values in by_static_query.items()
        if set(values) != set(TRUE_RULES)
    }
    if incomplete:
        raise RuntimeError(
            "Static-query bootstrap pairing is incomplete: "
            f"{list(incomplete.items())[:10]}"
        )

    static_query_ids = sorted(by_static_query)
    values = np.empty(
        (len(static_query_ids), len(metric_names)),
        dtype=np.float64,
    )
    for index, static_query_id in enumerate(static_query_ids):
        pair = by_static_query[static_query_id]
        passable = pair["passable"]
        blocked = pair["blocked"]
        if (
            int(passable["eval_seed"]) != int(blocked["eval_seed"])
            or str(passable["direction"]) != str(blocked["direction"])
        ):
            raise RuntimeError(
                "Both rules of one static query must share seed/direction: "
                f"{static_query_id}"
            )
        metric_values = {
            "passable/same_vs_other_rule_history": float(
                passable["same_vs_other_advantage"]
            ),
            "passable/same_vs_no_crossing_attempt": float(
                passable["same_vs_no_evidence_advantage"]
            ),
            "blocked/same_vs_other_rule_history": float(
                blocked["same_vs_other_advantage"]
            ),
            "blocked/same_vs_no_crossing_attempt": float(
                blocked["same_vs_no_evidence_advantage"]
            ),
            "passable/matching_history_two_target_margin": float(
                passable["same_history_two_target_margin"]
            ),
            "blocked/matching_history_two_target_margin": float(
                blocked["same_history_two_target_margin"]
            ),
        }
        values[index] = tuple(
            metric_values[name] for name in metric_names
        )
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite paired effects cannot be bootstrapped")

    strata: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, static_query_id in enumerate(static_query_ids):
        identity = by_static_query[static_query_id]["passable"]
        strata[
            (int(identity["eval_seed"]), str(identity["direction"]))
        ].append(index)
    stratum_sizes = {key: len(indices) for key, indices in strata.items()}
    if len(strata) != 12 or set(stratum_sizes.values()) != {25}:
        raise RuntimeError(
            "Paired bootstrap requires 12 seed-direction strata of 25 "
            f"static queries, got {stratum_sizes}"
        )

    rng = np.random.default_rng(int(seed))
    bootstrap_means = np.empty(
        (int(resamples), len(metric_names)),
        dtype=np.float64,
    )
    chunk_size = 512
    for start in range(0, int(resamples), chunk_size):
        stop = min(start + chunk_size, int(resamples))
        stratum_means = []
        for stratum in sorted(strata):
            source_indices = np.asarray(strata[stratum], dtype=np.int64)
            sampled_offsets = rng.integers(
                0,
                len(source_indices),
                size=(stop - start, len(source_indices)),
            )
            sampled_indices = source_indices[sampled_offsets]
            stratum_means.append(
                values[sampled_indices].mean(axis=1)
            )
        bootstrap_means[start:stop] = np.stack(
            stratum_means,
            axis=1,
        ).mean(axis=1)

    tail = (1.0 - float(confidence)) / 2.0
    lower = np.quantile(bootstrap_means, tail, axis=0)
    upper = np.quantile(bootstrap_means, 1.0 - tail, axis=0)
    estimates = values.mean(axis=0)
    return {
        "method": "paired_percentile_bootstrap",
        "unit": "static_query_within_eval_seed_direction",
        "unique_static_queries": len(static_query_ids),
        "strata": {
            f"s{eval_seed}/{direction}": len(indices)
            for (eval_seed, direction), indices in sorted(strata.items())
        },
        "resamples": int(resamples),
        "confidence": float(confidence),
        "seed": int(seed),
        "metrics": {
            name: {
                "mean": float(estimates[index]),
                "lower": float(lower[index]),
                "upper": float(upper[index]),
            }
            for index, name in enumerate(metric_names)
        },
    }


def summarize_validation_records(
    records: list[dict[str, Any]],
    *,
    eval_seeds: Iterable[int],
    unique_queries_per_seed: int,
    gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seeds = tuple(map(int, eval_seeds))
    paired = paired_effect_rows(records)
    expected_queries = len(seeds) * int(unique_queries_per_seed)
    expected_records = expected_queries * len(TRUE_RULES) * len(
        HISTORY_CONDITIONS
    )
    cell_counts = Counter(
        (str(row["true_rule"]), str(row["history_condition"]))
        for row in records
    )
    seed_cell_counts = Counter(
        (
            int(row["eval_seed"]),
            str(row["true_rule"]),
            str(row["history_condition"]),
        )
        for row in records
    )
    seed_direction_cell_counts = Counter(
        (
            int(row["eval_seed"]),
            str(row["direction"]),
            str(row["true_rule"]),
            str(row["history_condition"]),
        )
        for row in records
    )
    expected_direction = int(unique_queries_per_seed) // 2
    count_checks = {
        "exact_record_count": len(records) == expected_records,
        "six_matrix_cells": (
            set(cell_counts)
            == {
                (rule, condition)
                for rule in TRUE_RULES
                for condition in HISTORY_CONDITIONS
            }
        ),
        "three_hundred_records_per_matrix_cell": (
            set(cell_counts.values()) == {expected_queries}
        ),
        "fifty_records_per_seed_matrix_cell": (
            set(seed_cell_counts.values()) == {int(unique_queries_per_seed)}
        ),
        "twenty_five_per_seed_direction_matrix_cell": (
            set(seed_direction_cell_counts.values()) == {expected_direction}
        ),
        "exact_paired_rows": len(paired) == expected_queries * len(TRUE_RULES),
    }
    if not all(count_checks.values()):
        failed = [
            name for name, passed in count_checks.items() if not passed
        ]
        raise RuntimeError(f"Validation result count audit failed: {failed}")

    gate_config = _decision_gate_config(gates)
    decision_contract = gate_config["decision_contract"]
    by_rule: dict[str, Any] = {}
    all_seed_direction_effects_positive = True
    for rule in TRUE_RULES:
        rule_rows = [row for row in paired if row["true_rule"] == rule]
        by_seed = {}
        by_direction = {}
        by_seed_direction = {}
        for seed in seeds:
            selected = [
                row for row in rule_rows if row["eval_seed"] == seed
            ]
            by_seed[str(seed)] = _mean_effect(selected)
        for direction in DIRECTIONS:
            selected = [
                row for row in rule_rows if row["direction"] == direction
            ]
            by_direction[direction] = _mean_effect(selected)
        for seed in seeds:
            for direction in DIRECTIONS:
                selected = [
                    row
                    for row in rule_rows
                    if row["eval_seed"] == seed
                    and row["direction"] == direction
                ]
                cell = _mean_effect(selected)
                by_seed_direction[f"s{seed}/{direction}"] = cell
                all_seed_direction_effects_positive = (
                    all_seed_direction_effects_positive
                    and _positive_effect(
                        cell,
                        decision_contract=decision_contract,
                    )
                )
        overall = _mean_effect(rule_rows)
        by_rule[rule] = {
            "overall": overall,
            "by_eval_seed": by_seed,
            "by_direction": by_direction,
            "by_eval_seed_and_direction": by_seed_direction,
            "overall_both_paired_effects_positive": _positive_effect(
                overall,
                decision_contract=decision_contract,
            ),
            "overall_required_paired_effects_positive": _positive_effect(
                overall,
                decision_contract=decision_contract,
            ),
        }

    classification = {}
    for rule in TRUE_RULES:
        classification[rule] = {}
        for condition in HISTORY_CONDITIONS:
            selected = [
                row
                for row in records
                if row["true_rule"] == rule
                and row["history_condition"] == condition
            ]
            classification[rule][condition] = {
                "records": len(selected),
                "true_target_closer_rate": float(
                    np.mean([row["true_target_closer"] for row in selected])
                ),
                "mean_two_target_margin": float(
                    np.mean([row["two_target_margin"] for row in selected])
                ),
            }
    target_separation_by_query: dict[str, set[float]] = defaultdict(set)
    prediction_label_by_input: dict[tuple[str, str], set[str]] = defaultdict(
        set
    )
    for row in records:
        target_separation_by_query[str(row["query_id"])].add(
            float(row["target_pair_latent_mse"])
        )
        prediction_label_by_input[
            (str(row["query_id"]), str(row["history_condition"]))
        ].add(str(row["predicted_rule"]))
    if any(
        len(values) != 1 for values in target_separation_by_query.values()
    ):
        raise RuntimeError("Target separation differs within one query")
    if any(len(values) != 1 for values in prediction_label_by_input.values()):
        raise RuntimeError("Predicted rule differs across duplicated target rows")
    target_separations = np.asarray(
        [
            next(iter(values))
            for _, values in sorted(target_separation_by_query.items())
        ],
        dtype=np.float64,
    )
    prediction_labels = [
        next(iter(values))
        for _, values in sorted(prediction_label_by_input.items())
    ]
    target_separation = {
        "queries": len(target_separations),
        "minimum_mse": float(np.min(target_separations)),
        "mean_mse": float(np.mean(target_separations)),
        "median_mse": float(np.median(target_separations)),
        "maximum_mse": float(np.max(target_separations)),
    }
    two_target_ties = {
        "unique_predictions": len(prediction_labels),
        "ties": sum(label == "tie" for label in prediction_labels),
    }
    two_target_ties["tie_rate"] = float(
        two_target_ties["ties"] / two_target_ties["unique_predictions"]
    )
    all_rules_positive = all(
        row["overall_both_paired_effects_positive"]
        for row in by_rule.values()
    )
    accuracy_threshold = gate_config[
        "minimum_same_history_two_target_accuracy_exclusive"
    ]
    strict_win_threshold = gate_config[
        "minimum_strict_win_rate_exclusive"
    ]
    accuracy_above_threshold_for_each_rule = all(
        row["overall"]["same_history_two_target_accuracy"]
        > accuracy_threshold
        for row in by_rule.values()
    )
    accuracy_above_threshold_in_every_seed_direction = all(
        cell["same_history_two_target_accuracy"] > accuracy_threshold
        for row in by_rule.values()
        for cell in row["by_eval_seed_and_direction"].values()
    )
    strict_win_above_threshold_for_each_rule = all(
        row["overall"]["strict_win_rate"] > strict_win_threshold
        for row in by_rule.values()
    )
    matching_win_threshold = gate_config[
        "minimum_matching_vs_opposite_history_win_rate_exclusive"
    ]
    matching_win_above_threshold_for_each_rule = all(
        row["overall"]["matching_vs_opposite_history_win_rate"]
        > matching_win_threshold
        for row in by_rule.values()
    )
    paired_bootstrap = _paired_bootstrap_summary(
        paired,
        resamples=gate_config["bootstrap_resamples"],
        confidence=gate_config["bootstrap_confidence"],
        seed=gate_config["bootstrap_seed"],
        metric_names=gate_config["required_bootstrap_metrics"],
    )
    bootstrap_lower_threshold = gate_config[
        "minimum_bootstrap_lower_bound_exclusive"
    ]
    bootstrap_lower_bounds_above_threshold = all(
        metric["lower"] > bootstrap_lower_threshold
        for metric in paired_bootstrap["metrics"].values()
    )
    target_separation_above_threshold = bool(
        np.all(
            target_separations
            > gate_config["minimum_target_pair_latent_mse_exclusive"]
        )
    )
    if decision_contract == ALL_HISTORIES_STRICT_V1:
        decision_checks = {
            "paired_latent_advantages_positive_for_each_true_rule": (
                all_rules_positive
            ),
            "paired_latent_advantages_positive_in_every_seed_direction_cell": (
                all_seed_direction_effects_positive
            ),
            "evidence_history_target_accuracy_above_threshold_for_each_rule": (
                accuracy_above_threshold_for_each_rule
            ),
            "evidence_history_target_accuracy_above_threshold_in_every_seed_direction_cell": (
                accuracy_above_threshold_in_every_seed_direction
            ),
            "strict_win_rate_above_threshold_for_each_rule": (
                strict_win_above_threshold_for_each_rule
            ),
            "paired_static_query_bootstrap_lower_bounds_above_threshold": (
                bootstrap_lower_bounds_above_threshold
            ),
            "target_latents_are_separated_for_every_query": (
                target_separation_above_threshold
            ),
        }
    elif decision_contract == INFORMATIVE_HISTORY_RULE_SWITCH_V2:
        decision_checks = {
            "matching_history_beats_opposite_history_for_each_true_rule": (
                all_rules_positive
            ),
            "matching_history_beats_opposite_history_in_every_seed_direction_cell": (
                all_seed_direction_effects_positive
            ),
            "matching_history_target_accuracy_above_threshold_for_each_rule": (
                accuracy_above_threshold_for_each_rule
            ),
            "matching_history_target_accuracy_above_threshold_in_every_seed_direction_cell": (
                accuracy_above_threshold_in_every_seed_direction
            ),
            "matching_history_beats_opposite_history_on_majority_queries_for_each_rule": (
                matching_win_above_threshold_for_each_rule
            ),
            "required_bootstrap_lower_bounds_above_threshold": (
                bootstrap_lower_bounds_above_threshold
            ),
            "target_latents_are_separated_for_every_query": (
                target_separation_above_threshold
            ),
        }
    else:  # pragma: no cover - resolved fail-closed above
        raise AssertionError(decision_contract)
    decision = {
        "decision_contract": decision_contract,
        "both_effects_positive_for_each_true_rule": all_rules_positive,
        "both_effects_positive_in_every_seed_direction_cell": (
            all_seed_direction_effects_positive
        ),
        "checks": decision_checks,
        "failed_checks": [
            name for name, passed in decision_checks.items() if not passed
        ],
        "thresholds": gate_config,
        "passed": bool(all(decision_checks.values())),
        "scope": (
            "checkpoint-level Validation only; training attribution still "
            "requires three paired training seeds and single-rule controls"
        ),
    }
    return {
        "overall": _mean_effect(paired),
        "by_true_rule": by_rule,
        "two_target_discrimination": classification,
        "target_latent_separation": target_separation,
        "two_target_ties": two_target_ties,
        "paired_static_query_bootstrap": paired_bootstrap,
        "decision": decision,
        "count_audit": {
            "passed": True,
            "checks": count_checks,
            "records": len(records),
            "paired_rows": len(paired),
            "matrix_cells": len(cell_counts),
            "records_per_matrix_cell": {
                f"{rule}/{condition}": count
                for (rule, condition), count in sorted(cell_counts.items())
            },
        },
        "metric_contract": {
            "primary": "frozen_true_next_frame_native_latent_mse",
            "decision_contract": decision_contract,
            "no_crossing_attempt_role": (
                "mandatory_comparator"
                if decision_contract == ALL_HISTORIES_STRICT_V1
                else "auxiliary_default_tendency_only"
            ),
            "raw_latent_mse_cross_checkpoint_comparison_allowed": False,
            "cross_checkpoint_quantities": [
                "paired symmetric contrast",
                "strict win rate",
                "two-target accuracy",
                "seed and direction sign consistency",
            ],
            "unchanged_query_relative_loss_used": False,
        },
    }


__all__ = [
    "HISTORY_CONDITIONS",
    "MODEL_INPUT_KEYS",
    "NO_EVIDENCE_HISTORY",
    "OTHER_HISTORY",
    "SAME_HISTORY",
    "TRUE_RULES",
    "ValidationAssignment",
    "array_sha256",
    "audit_validation_assets",
    "build_rollout_matrix",
    "build_validation_catalog",
    "candidate_templates",
    "file_sha256",
    "load_validation_assets",
    "make_no_attempt_template",
    "paired_effect_rows",
    "score_validation_assets",
    "select_validation_assignments",
    "summarize_validation_records",
]
