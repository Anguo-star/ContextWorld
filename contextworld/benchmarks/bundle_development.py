"""Development-only ICL evaluation over the public ``ContextWorld-v1`` bundle.

The public dataset package deliberately contains Training and Development
payloads, but not the held-out Public Test payloads.  This module is the
matching evaluation boundary: it reads only the release-bound bundle, rebuilds
the documented Development comparisons, and labels every result accordingly.

It is intentionally separate from the historical task scorers.  Those files
are part of frozen release provenance and, in several cases, open protected
Public-Test artifacts.  Reusing their small, model-independent metric kernels
is useful; reusing their data readers is not.  A result from this module is a
useful training/development diagnostic, never an official scoreboard row or a
formal pass decision.

Since dataset_version 1.0.1-rc1 the speed and door Development payloads are
the ``development_structural_parity_v1`` Lance collections: replicas of the
Public Test protocols whose only difference from Test is the data rows.  The
two readers below therefore rebuild the Test comparison structure (speed:
four tracks, one history condition per reference speed, 1..5-block true
futures; door: three history conditions, two true futures, eval-seed x
direction stratification) and score it with the same model-independent
kernels the Public Test scorers use.  Every threshold-bearing decision field
is stripped: this module emits the gate *inputs* only, and any pass judgment
is computed outside it against the published thresholds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from contextworld.benchmarks.adapters import (
    LatentWorldModelAdapter,
    validate_adapter_protocol,
)
from contextworld.benchmarks.paired_latent_response import (
    paired_latent_response_metrics,
)
from contextworld.evaluation.action_delay_h7_core import (
    physical_group,
    summarize_action_delay_h1_physical,
)
from contextworld.evaluation.action_delay_h7_score import physical_future_group
from contextworld.training import stablewm_bundle


DEVELOPMENT_RESULT_KIND = "development_only_not_public_test"
DEVELOPMENT_PROTOCOL_VERSION = "contextworld_bundle_development_icl_v1"

_SINGLE_TABLE_TASKS = {
    "action_strength",
    "contact_friction",
    "motion_damping",
    "robot_arm_mass",
    "portal_exit",
    "cube_gripper_carry",
}
# Structural-parity Development payloads (dataset_version 1.0.1-rc1).  One
# Lance table per (track, history condition) for speed and per (door
# position, history condition) for door; the stratification keys ride as
# per-episode constant ``dev_*`` columns because the pinned Lance schema has
# no episode side table.
_SPEED_STRUCTURAL_MEMBER_PATTERN = re.compile(
    r"^twmsdev-(?P<track>[a-z_]+)-(?P<condition>history_[a-z_]+)-"
    r"(?P<fingerprint>[0-9a-f]+)\.lance$"
)
_DOOR_STRUCTURAL_MEMBER_PATTERN = re.compile(
    r"^hpdev-d(?P<door>\d+)-(?P<condition>[a-z_]+)-"
    r"(?P<fingerprint>[0-9a-f]+)\.lance$"
)
_SPEED_DEV_EVAL_SEEDS = (42, 43, 44, 45, 46, 47)
_SPEED_DEV_QUERIES_PER_REFERENCE_SPEED_PER_SEED = 50
_SPEED_TOKEN_ROWS = (0, 5, 10, 15, 20, 25, 30, 35)
_SPEED_EPISODE_ROWS = 40
_DOOR_TOKEN_ROWS = (0, 5, 10, 15)
_DOOR_EPISODE_ROWS = 20
_DOOR_DEV_EVAL_SEEDS = (42, 43, 44, 45, 46, 47)
_DOOR_DEV_QUERIES_PER_SEED = 50
# The formal Public Test fixes its door decision contract in
# ``tworoom_hidden_passage_h3_validation_v2.yaml`` (gates) and its additive
# gate inputs in ``contextworld_test_gate_completion_v1.yaml``.  The
# Development bundle ships neither, so the same numbers are restated here and
# fed to the identical kernels; only the resulting decision fields are
# stripped from the emitted metrics.
_DOOR_DEV_GATES = {
    "minimum_same_history_two_target_accuracy_exclusive": 0.5,
    "minimum_strict_win_rate_exclusive": 0.5,
    "minimum_target_pair_latent_mse_exclusive": 1.0e-12,
    "paired_bootstrap": {
        "unit": "static_query_within_eval_seed_direction",
        "strata": "eval_seed_x_direction",
        "method": "percentile",
        "resamples": 10_000,
        "confidence": 0.95,
        "seed": 20_260_725,
        "minimum_lower_bound_exclusive": 0.0,
        "required_metrics": [
            "passable/same_vs_other_rule_history",
            "passable/same_vs_no_crossing_attempt",
            "blocked/same_vs_other_rule_history",
            "blocked/same_vs_no_crossing_attempt",
            "passable/matching_history_two_target_margin",
            "blocked/matching_history_two_target_margin",
        ],
    },
}
# Keys that carry pass judgments; a Development result must never contain
# them, so everything the shared Test kernels return is filtered through
# this set before it reaches the emitted metrics.
_DECISION_FREE_KEYS = frozenset(
    {
        "gate",
        "gates",
        "decision",
        "decision_contract",
        "passed",
        "checks",
        "failed_checks",
        "verdict",
        "thresholds",
        "count_audit",
        "diagnostic_within_sample_pass",
        "formal_protocol_eligible",
        "formal_within_checkpoint_pass",
        "gate_horizon",
        "additive_only",
        "full_protocol",
        "full_protocol_track",
        "threshold_source",
        "threshold_provenance",
    }
)
_ACTION_DELAY_MEMBER_PATTERN = re.compile(
    r"^ad-h7-paired-val-(?P<profile>p\d+)-d(?P<delay>\d+)-"
)
_ACTION_DELAY_FRAME_STEPS = tuple(range(0, 50, 5))
_ACTION_DELAY_METADATA_COLUMNS = (
    "dev_eval_seed",
    "dev_room",
    "dev_direction",
    "dev_query_id",
    "dev_delay",
)
# The formal Public Test fixes its bootstrap uncertainty contract in
# ``tworoom_action_delay_h7_core_icl_v2.yaml`` (scoring.uncertainty).  The
# Development bundle ships no uncertainty section, so these fixed diagnostic
# values reuse the formal resample count and seed when feeding the same
# model-independent kernel.
_ACTION_DELAY_BOOTSTRAP_RESAMPLES = 10_000
_ACTION_DELAY_BOOTSTRAP_RANDOM_SEED = 2_026_073_002


@dataclass(frozen=True)
class DevelopmentPayload:
    """A verified, Development-only payload selected from ``ContextWorld-v1``."""

    root: Path
    task: str
    component: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    payload: Mapping[str, Any]
    members: tuple[Path, ...]
    manifest_sha256: str
    task_registry_sha256: str
    normalizer_path: Path | None

    @property
    def history_length(self) -> int:
        return int(self.component["history_length"])

    @property
    def action_dimension(self) -> int:
        return int(self.component["action_dimension"])

    @property
    def frameskip(self) -> int:
        return int(self.component["frameskip"])


@dataclass(frozen=True)
class _PairedArrays:
    pair_ids: tuple[str, ...]
    first_pixels: np.ndarray
    second_pixels: np.ndarray
    raw_action_blocks: np.ndarray
    first_label: str
    second_label: str
    selection: Mapping[str, Any]


@dataclass(frozen=True)
class _DelayFamilyArrays:
    """One matched query per profile/episode with every delay condition.

    Action Delay is scored as six-group recognition, so the reader keeps the
    full delay family together: ``member_pixels`` is indexed
    ``(query, delay_condition, history_length + 1, ...)`` and the final frame
    of every condition is that condition's one-step future target.
    """

    query_ids: tuple[str, ...]
    query_metadata: tuple[Mapping[str, Any], ...]
    delay_values: tuple[int, ...]
    member_pixels: np.ndarray
    raw_action_blocks: np.ndarray
    selection: Mapping[str, Any]


def _absolute_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(
            "ContextWorld Development evaluation requires an absolute "
            f"--benchmark-root: {path}"
        )
    return Path(str(path))


def resolve_development_payload(
    bundle_root: str | Path,
    *,
    task: str,
) -> DevelopmentPayload:
    """Resolve one verified public Development payload.

    ``stablewm_bundle`` verifies the manifest receipt and the registry digest
    before returning paths.  Its member resolver rejects traversal and
    unregistered Lance directories, so this evaluation path has no fallback
    to ``CONTEXTWORLD_ARTIFACT_ROOT`` or any historical private tree.
    """

    root = _absolute_root(bundle_root)
    resolved = stablewm_bundle.resolve_contextworld_development_payload(
        root, component=task
    )
    evaluation = resolved.get("development_evaluation")
    if not isinstance(evaluation, Mapping):  # pragma: no cover - resolver contract
        raise ValueError(f"Development resolver returned no contract for {task!r}")
    member_paths = resolved.get("member_paths")
    relative_members = resolved.get("relative_members")
    if (
        not isinstance(member_paths, tuple)
        or not member_paths
        or not isinstance(relative_members, tuple)
    ):  # pragma: no cover - resolver contract
        raise ValueError(f"Development resolver returned no members for {task!r}")
    members = tuple(Path(value) for value in member_paths)
    component = {
        "component_id": resolved["component_id"],
        "dataset_id": resolved["dataset_id"],
        "environment": resolved["environment"],
        "history_length": resolved["history_length"],
        "action_dimension": resolved["action_dimension"],
        "frameskip": resolved["frameskip"],
    }
    payload = {
        "payload_id": resolved["payload_id"],
        "payload_kind": resolved["payload_kind"],
        "members": list(relative_members),
    }
    normalizer_value = resolved.get("normalizer_path")
    normalizer = Path(normalizer_value) if isinstance(normalizer_value, str) else None
    return DevelopmentPayload(
        root=Path(str(resolved["bundle_root"])),
        task=task,
        component=component,
        evaluation=evaluation,
        payload=payload,
        members=members,
        manifest_sha256=str(resolved["manifest_sha256"]),
        task_registry_sha256=str(resolved["task_registry_sha256"]),
        normalizer_path=normalizer,
    )


def development_action_normalization(
    payload: DevelopmentPayload,
    *,
    preferred_std_key: str | None = None,
) -> tuple[list[float], list[float]]:
    """Return inline action statistics from the public registry contract.

    Legacy TwoRoom release files point to a private normalizer JSON.  The
    clean bundle instead carries the numerical values inline, which makes the
    public route self-contained and prevents an accidental private fallback.
    """

    value = payload.evaluation.get("action_normalization")
    if not isinstance(value, Mapping):
        raise ValueError(
            f"ContextWorld-v1 {payload.task!r} Development contract lacks "
            "inline action_normalization"
        )
    mean = value.get("mean")
    if not isinstance(mean, Sequence) or isinstance(mean, (str, bytes)):
        raise ValueError("Development action_normalization.mean must be a list")
    requested = (
        str(preferred_std_key)
        if preferred_std_key
        else str(value.get("std_key", "std_population"))
    )
    candidates = (requested, "std_population", "std_unbiased", "std")
    std: Any = next((value.get(key) for key in candidates if key in value), None)
    if not isinstance(std, Sequence) or isinstance(std, (str, bytes)):
        raise ValueError(
            "Development action_normalization must provide a finite action "
            "standard deviation"
        )
    mean_values = [float(item) for item in mean]
    std_values = [float(item) for item in std]
    if (
        len(mean_values) != payload.action_dimension
        or len(std_values) != payload.action_dimension
        or not np.isfinite(np.asarray(mean_values, dtype=np.float64)).all()
        or not np.isfinite(np.asarray(std_values, dtype=np.float64)).all()
        or np.any(np.asarray(std_values, dtype=np.float64) <= 0)
    ):
        raise ValueError(
            f"Invalid Development action normalization for {payload.task!r}"
        )
    return mean_values, std_values


def development_action_normalizer_path(payload: DevelopmentPayload) -> Path:
    """Resolve a manifest-bound normalizer bundled with a Development task.

    StableWM's historical LeWM/PLDM adapters accept a normalizer file rather
    than inline statistics for the Three TwoRoom components.  The clean export
    ships that small JSON beside the public metadata.  This helper verifies it
    is manifest-bound before handing it to an adapter; it never looks up the
    old artifact-root normalizer.
    """

    path = payload.normalizer_path
    if path is None:
        raise ValueError(
            f"ContextWorld-v1 {payload.task!r} Development contract lacks "
            "normalizer_path required by the legacy StableWM adapter"
        )
    return path


def _decode_rgb(value: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("ContextWorld Development ICL needs Pillow") from exc
    with Image.open(BytesIO(bytes(value))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _lance_table(path: Path, *, columns: Sequence[str]):
    try:
        import lance
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("ContextWorld Development ICL needs the lance package") from exc
    return lance.dataset(path).to_table(columns=list(columns))


def _episode_rows(
    table: Any,
    *,
    expected_steps: int,
    selected_episode_ids: Sequence[int] | None = None,
    allow_prefix_clip: bool = False,
) -> tuple[tuple[int, ...], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Read exact clips, or (only when requested) valid leading windows.

    The Speed diagnostic deliberately samples a 20-step prefix from variable
    length rollouts.  Its caller opts into ``allow_prefix_clip``; all paired
    rule readers use the default exact-clip check so their episode contracts
    remain strict.
    """
    episode_indices = np.asarray(table["episode_idx"].to_numpy(), dtype=np.int64)
    step_indices = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
    available = tuple(sorted(int(value) for value in np.unique(episode_indices)))
    requested = available if selected_episode_ids is None else tuple(
        int(value) for value in selected_episode_ids
    )
    if len(requested) != len(set(requested)) or not set(requested).issubset(available):
        raise RuntimeError("Development payload has inconsistent episode ids")
    rows_by_episode: dict[int, np.ndarray] = {}
    accepted_ids: list[int] = []
    for episode in requested:
        rows = np.flatnonzero(episode_indices == episode)
        rows = rows[np.argsort(step_indices[rows])]
        if allow_prefix_clip:
            valid = (
                len(rows) >= expected_steps
                and np.array_equal(
                    step_indices[rows[:expected_steps]], np.arange(expected_steps)
                )
            )
            selected_rows = rows[:expected_steps]
        else:
            valid = np.array_equal(step_indices[rows], np.arange(expected_steps))
            selected_rows = rows
        if not valid:
            if selected_episode_ids is None and allow_prefix_clip:
                continue
            raise RuntimeError(
                f"Development episode {episode} does not provide a valid "
                f"{expected_steps}-step {'prefix window' if allow_prefix_clip else 'clip'}"
            )
        rows_by_episode[episode] = selected_rows
        accepted_ids.append(episode)
    return (
        tuple(accepted_ids) if allow_prefix_clip else available,
        rows_by_episode,
        episode_indices,
        step_indices,
    )


def _read_tworoom_episodes(
    path: Path,
    *,
    expected_steps: int,
    frame_steps: Sequence[int],
    selected_episode_ids: Sequence[int] | None = None,
    include_speed: bool = False,
    allow_prefix_clip: bool = False,
    metadata_columns: Sequence[str] = (),
) -> tuple[tuple[int, ...], dict[int, tuple[Any, ...]]]:
    columns = ["episode_idx", "step_idx", "pixels", "action"]
    if include_speed:
        columns.append("variation_agent_speed")
    columns.extend(str(name) for name in metadata_columns)
    table = _lance_table(path, columns=columns)
    available, rows_by_episode, _, _ = _episode_rows(
        table,
        expected_steps=expected_steps,
        selected_episode_ids=selected_episode_ids,
        allow_prefix_clip=allow_prefix_clip,
    )
    pixels = table["pixels"].to_pylist()
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    speeds = (
        np.asarray(table["variation_agent_speed"].to_pylist(), dtype=np.float32)
        .reshape(-1)
        if include_speed
        else None
    )
    metadata_values = {
        str(name): table[str(name)].to_pylist()
        for name in metadata_columns
    }
    if expected_steps % 5:
        raise RuntimeError("Development TwoRoom clips must be divisible into 5-step actions")
    result: dict[int, tuple[Any, ...]] = {}
    for episode, rows in rows_by_episode.items():
        speed: float | None = None
        if speeds is not None:
            speed_values = speeds[rows]
            if not np.allclose(speed_values, speed_values[0], atol=0.0, rtol=0.0):
                raise RuntimeError(
                    f"Development speed changes within episode {episode}"
                )
            speed = float(speed_values[0])
        values: list[Any] = [
            np.stack([_decode_rgb(pixels[int(rows[index])]) for index in frame_steps]),
            actions[rows].reshape(expected_steps // 5, 5, actions.shape[-1]),
            speed,
        ]
        if metadata_columns:
            metadata: dict[str, Any] = {}
            for name in metadata_columns:
                raw_values = [metadata_values[str(name)][int(row)] for row in rows]
                normalized = {
                    _scalar(value)
                    if str(name) in {"dev_eval_seed", "dev_delay"}
                    else str(value)
                    for value in raw_values
                }
                if len(normalized) != 1:
                    raise RuntimeError(
                        f"Development column {name} is not constant in episode "
                        f"{episode} of {path.name}"
                    )
                metadata[str(name)] = normalized.pop()
            values.append(metadata)
        result[episode] = tuple(values)
    return available, result


def _scalar(value: Any) -> float:
    """Normalize a per-row scalar column value (scalar or length-1 list)."""

    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
        if len(items) != 1:
            raise ValueError(
                f"Development scalar column must hold one value, got {items}"
            )
        value = items[0]
    return float(value)


def _read_dev_episodes(
    path: Path,
    *,
    expected_steps: int,
    frame_steps: Sequence[int],
    string_columns: Sequence[str] = (),
    scalar_columns: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Read one structural-parity Development table with its ``dev_*`` keys.

    Returns one row per episode with the decoded frames at ``frame_steps``,
    the full raw-step action vector, and the episode-constant stratification
    columns.  Every column must be constant within an episode; the exact-clip
    contract is the same one the paired readers enforce.
    """

    columns = [
        "episode_idx",
        "step_idx",
        "pixels",
        "action",
        *string_columns,
        *scalar_columns,
    ]
    table = _lance_table(path, columns=columns)
    episode_indices = np.asarray(table["episode_idx"].to_numpy(), dtype=np.int64)
    step_indices = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
    string_values = {
        name: table[name].to_pylist() for name in string_columns
    }
    scalar_values = {
        name: table[name].to_pylist() for name in scalar_columns
    }
    pixels = table["pixels"].to_pylist()
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    episodes: list[dict[str, Any]] = []
    for episode in sorted(int(value) for value in np.unique(episode_indices)):
        rows = np.flatnonzero(episode_indices == episode)
        rows = rows[np.argsort(step_indices[rows])]
        if not np.array_equal(step_indices[rows], np.arange(expected_steps)):
            raise RuntimeError(
                f"Development episode {episode} of {path.name} is not a valid "
                f"{expected_steps}-step clip"
            )
        record: dict[str, Any] = {
            "episode": episode,
            "frames": np.stack(
                [
                    _decode_rgb(pixels[int(rows[step])])
                    for step in frame_steps
                ]
            ),
            "actions": actions[rows].reshape(expected_steps, -1),
        }
        for name in string_columns:
            values = {str(string_values[name][int(row)]) for row in rows}
            if len(values) != 1:
                raise RuntimeError(
                    f"Development column {name} is not constant in episode "
                    f"{episode} of {path.name}"
                )
            record[name] = values.pop()
        for name in scalar_columns:
            values = {_scalar(scalar_values[name][int(row)]) for row in rows}
            if len(values) != 1:
                raise RuntimeError(
                    f"Development column {name} is not constant in episode "
                    f"{episode} of {path.name}"
                )
            record[name] = values.pop()
        episodes.append(record)
    if not episodes:
        raise RuntimeError(f"Development table {path.name} has no episodes")
    return episodes


def _decision_free(value: Any) -> Any:
    """Recursively drop every decision-bearing field from kernel output."""

    if isinstance(value, dict):
        return {
            key: _decision_free(item)
            for key, item in value.items()
            if key not in _DECISION_FREE_KEYS
        }
    if isinstance(value, list):
        return [_decision_free(item) for item in value]
    return value


def _ensure_paired_example(
    *,
    pair_id: str,
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    first_actions: np.ndarray,
    second_actions: np.ndarray,
    history_length: int,
) -> None:
    if not np.array_equal(first_pixels[0], second_pixels[0]):
        raise RuntimeError(f"Development pair {pair_id} has different initial frames")
    if not np.array_equal(first_pixels[history_length - 1], second_pixels[history_length - 1]):
        raise RuntimeError(f"Development pair {pair_id} has different query frames")
    if not np.array_equal(first_actions, second_actions):
        raise RuntimeError(f"Development pair {pair_id} has different actions")
    if np.array_equal(
        first_pixels[:history_length], second_pixels[:history_length]
    ):
        raise RuntimeError(f"Development pair {pair_id} does not reveal its context")
    if np.array_equal(
        first_pixels[history_length], second_pixels[history_length]
    ):
        raise RuntimeError(f"Development pair {pair_id} has no divergent future")


def _selection_value(
    payload: DevelopmentPayload, name: str, default: int
) -> int:
    selection = _selection_mapping(payload)
    value = selection.get(name, default)
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Development selection value {name!r} must be an integer"
        ) from exc
    return resolved


def _selection_mapping(payload: DevelopmentPayload) -> Mapping[str, Any]:
    value = payload.evaluation.get("selection")
    if not isinstance(value, Mapping):
        raise ValueError(
            f"ContextWorld-v1 {payload.task!r} Development contract lacks selection"
        )
    return value


def _single_table_arrays(payload: DevelopmentPayload) -> _PairedArrays:
    if len(payload.members) != 1:
        raise ValueError(
            f"Development component {payload.task!r} requires exactly one "
            f"Lance table, found {len(payload.members)}"
        )
    expected_pairs = _selection_value(payload, "expected_pair_count", 256)
    path = payload.members[0]
    task = payload.task
    if task == "action_strength":
        from contextworld.benchmarks.action_strength_icl_data import _read_lance_pairs

        arrays = _read_lance_pairs(path, expected_pairs=expected_pairs)
        first, second, labels = arrays.low_pixels, arrays.high_pixels, ("low_gain", "high_gain")
    elif task == "contact_friction":
        from contextworld.benchmarks.contact_friction_icl_data import _read_lance_pairs

        arrays = _read_lance_pairs(
            path,
            expected_pairs=expected_pairs,
            expected_split=str(payload.evaluation.get("expected_split", "loader_validation")),
        )
        first, second, labels = arrays.low_pixels, arrays.high_pixels, ("low_friction", "high_friction")
    elif task == "motion_damping":
        from contextworld.benchmarks.motion_damping_icl_data import _read_lance_pairs

        arrays = _read_lance_pairs(
            path,
            expected_pairs=expected_pairs,
            expected_split=str(payload.evaluation.get("expected_split", "loader_validation")),
        )
        first, second, labels = (
            arrays.faster_decay_pixels,
            arrays.no_extra_decay_pixels,
            ("faster_decay", "no_extra_decay"),
        )
    elif task == "robot_arm_mass":
        from contextworld.benchmarks.reacher_arm_mass_icl_data import _read_lance_pairs

        arrays = _read_lance_pairs(
            path,
            expected_pairs=expected_pairs,
            expected_split=str(payload.evaluation.get("expected_split", "loader_validation")),
        )
        first, second, labels = arrays.lighter_pixels, arrays.heavier_pixels, ("lighter", "heavier")
    elif task == "portal_exit":
        from contextworld.benchmarks.portal_exit_icl_data import _read_lance_pairs

        arrays = _read_lance_pairs(
            path,
            expected_pairs=expected_pairs,
            expected_split=str(payload.evaluation.get("expected_split", "loader_validation")),
        )
        first, second, labels = (
            arrays.near_border_pixels,
            arrays.farther_from_border_pixels,
            ("near_border", "farther_from_border"),
        )
    elif task == "cube_gripper_carry":
        from contextworld.benchmarks.cube_grasp_rule_icl_data import _read_lance_pairs

        arrays = _read_lance_pairs(
            path,
            expected_pairs=expected_pairs,
            expected_split=str(payload.evaluation.get("expected_split", "loader_validation")),
        )
        first, second, labels = (
            arrays.cannot_hold_pixels,
            arrays.can_hold_pixels,
            ("cannot_hold", "can_hold"),
        )
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unsupported single-table Development task: {task}")
    selected_pairs = _selection_value(payload, "selected_pair_count", expected_pairs)
    if arrays.pair_count != expected_pairs or arrays.pair_count != selected_pairs:
        raise RuntimeError(
            f"{task} Development pair count disagrees with its public contract: "
            f"observed={arrays.pair_count} expected={expected_pairs} "
            f"selected={selected_pairs}"
        )
    return _PairedArrays(
        pair_ids=tuple(str(value) for value in arrays.pair_ids),
        first_pixels=np.asarray(first),
        second_pixels=np.asarray(second),
        raw_action_blocks=np.asarray(arrays.raw_action_blocks, dtype=np.float32),
        first_label=labels[0],
        second_label=labels[1],
        selection={
            "kind": "complete_registered_development_table",
            "pair_count": int(arrays.pair_count),
            "member": str(path.relative_to(payload.root)),
        },
    )


def _door_structural_assets(
    payload: DevelopmentPayload,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build Public-Test-shaped scoring assets from the door payload.

    Each static query contributes one asset with its three history
    conditions, the shared history action blocks, and the two rule targets
    (frame 15 of the ``observed_passable`` / ``observed_blocked`` episodes;
    the no-attempt episode repeats the passable target, which is audited
    here).  This is the asset shape ``score_validation_assets`` scores for
    the Public Test, so the Development scoring chain is kernel-identical.
    """

    from contextworld.evaluation.hidden_passage_validation import (
        HISTORY_CONDITIONS,
    )

    grouped: dict[str, dict[str, Path]] = {}
    for path in payload.members:
        match = _DOOR_STRUCTURAL_MEMBER_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected Door Development member: {path.name}")
        condition = str(match["condition"])
        if condition not in HISTORY_CONDITIONS:
            raise ValueError(f"Unknown Door Development condition: {condition}")
        slot = grouped.setdefault(str(match["door"]), {})
        if condition in slot:
            raise ValueError(f"Duplicate Door Development member: {path.name}")
        slot[condition] = path
    if not grouped or any(
        set(value) != set(HISTORY_CONDITIONS) for value in grouped.values()
    ):
        raise ValueError(
            "Door Development payload does not cover all three history conditions"
        )

    episodes_by_query: dict[str, dict[str, dict[str, Any]]] = {}
    table_episodes: list[int] = []
    for door in sorted(grouped):
        for condition in HISTORY_CONDITIONS:
            records = _read_dev_episodes(
                grouped[door][condition],
                expected_steps=_DOOR_EPISODE_ROWS,
                frame_steps=_DOOR_TOKEN_ROWS,
                string_columns=(
                    "dev_query_id",
                    "dev_static_query_id",
                    "dev_direction",
                    "dev_env_rule",
                    "dev_template_id",
                ),
                scalar_columns=(
                    "dev_eval_seed",
                    "dev_evaluation_index",
                    "dev_door_position",
                ),
            )
            table_episodes.append(len(records))
            for record in records:
                static_id = record["dev_static_query_id"]
                slot = episodes_by_query.setdefault(static_id, {})
                if condition in slot:
                    raise RuntimeError(
                        f"Door Development query {static_id} repeats {condition}"
                    )
                slot[condition] = record

    assets: list[dict[str, Any]] = []
    for static_id in sorted(episodes_by_query):
        conditions = episodes_by_query[static_id]
        if set(conditions) != set(HISTORY_CONDITIONS):
            raise RuntimeError(
                f"Door Development query {static_id} lacks a history condition"
            )
        reference = conditions["observed_passable"]
        targets = {
            "passable": conditions["observed_passable"]["frames"][3],
            "blocked": conditions["observed_blocked"]["frames"][3],
        }
        if not np.array_equal(
            conditions["did_not_attempt_crossing"]["frames"][3],
            targets["passable"],
        ):
            raise RuntimeError(
                f"Door Development {static_id}: the no-attempt future is not "
                "the passable target"
            )
        shared_actions = conditions["observed_passable"]["actions"][:15]
        for condition in HISTORY_CONDITIONS:
            if not np.array_equal(
                conditions[condition]["actions"][:15], shared_actions
            ):
                raise RuntimeError(
                    f"Door Development {static_id}: history actions differ "
                    f"across conditions"
                )
        assets.append(
            {
                "query_id": reference["dev_query_id"],
                "static_query_id": static_id,
                "template_id": reference["dev_template_id"],
                "eval_seed": int(reference["dev_eval_seed"]),
                "evaluation_index": int(reference["dev_evaluation_index"]),
                "direction": reference["dev_direction"],
                "histories": {
                    condition: conditions[condition]["frames"][:3]
                    for condition in HISTORY_CONDITIONS
                },
                "actions": {
                    condition: shared_actions.reshape(3, 5, shared_actions.shape[-1])
                    for condition in HISTORY_CONDITIONS
                },
                "targets": targets,
            }
        )

    by_seed: dict[int, int] = {}
    by_seed_direction: dict[tuple[int, str], int] = {}
    directions = set()
    for asset in assets:
        seed = int(asset["eval_seed"])
        direction = str(asset["direction"])
        by_seed[seed] = by_seed.get(seed, 0) + 1
        by_seed_direction[(seed, direction)] = (
            by_seed_direction.get((seed, direction), 0) + 1
        )
        directions.add(direction)
    expected_strata = {
        (seed, direction)
        for seed in _DOOR_DEV_EVAL_SEEDS
        for direction in sorted(directions)
    }
    if (
        len(assets) != 300
        or set(by_seed_direction) != expected_strata
        or set(by_seed_direction.values()) != {25}
        or set(by_seed.values()) != {_DOOR_DEV_QUERIES_PER_SEED}
    ):
        raise RuntimeError(
            "Door Development stratification disagrees with the Public Test "
            f"structure: queries={len(assets)} "
            f"strata={sorted(by_seed_direction.items())[:4]}..."
        )
    selection = {
        "kind": "structural_parity_static_queries",
        "tables": len(payload.members),
        "episodes_per_table": sorted(set(table_episodes)),
        "unique_queries": len(assets),
        "eval_seeds": list(_DOOR_DEV_EVAL_SEEDS),
        "unique_queries_per_eval_seed": _DOOR_DEV_QUERIES_PER_SEED,
        "per_direction_per_eval_seed": 25,
        "history_conditions": list(HISTORY_CONDITIONS),
        "rule": "one Lance table per (door position, history condition)",
    }
    return assets, selection


def _door_structural_metrics(
    *,
    payload: DevelopmentPayload,
    adapter: LatentWorldModelAdapter,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str], Mapping[str, Any]]:
    """Score the door Development payload with the Public Test kernels."""

    from contextworld.benchmarks.door_icl_score import door_gate_completion_metrics
    from contextworld.benchmarks.test_gate_completion import (
        load_test_gate_completion_config,
    )
    from contextworld.evaluation.hidden_passage_validation import (
        HISTORY_CONDITIONS,
        TRUE_RULES,
        score_validation_assets,
        summarize_validation_records,
    )

    validate_adapter_protocol(
        adapter,
        history_tokens=payload.history_length,
        action_block_raw_steps=payload.frameskip,
        action_dim=payload.action_dimension,
        minimum_future_action_blocks=1,
        task_name="door Development",
    )
    assets, selection = _door_structural_assets(payload)
    scored = score_validation_assets(
        adapter,
        assets,
        batch_size=int(batch_size),
        return_latents=True,
    )
    summary = summarize_validation_records(
        scored["records"],
        eval_seeds=_DOOR_DEV_EVAL_SEEDS,
        unique_queries_per_seed=_DOOR_DEV_QUERIES_PER_SEED,
        gates=_DOOR_DEV_GATES,
    )
    latents = scored["latents"]
    gate_inputs = door_gate_completion_metrics(
        predicted=latents["predicted"],
        encoded_targets=latents["encoded_targets"],
        query_ids=list(latents["query_ids"]),
        static_query_ids=list(latents["static_query_ids"]),
        config=load_test_gate_completion_config(),
        full_protocol=True,
    )
    by_rule = summary["by_true_rule"]
    primary = float(
        np.mean(
            [
                by_rule[rule]["overall"]["same_history_two_target_accuracy"]
                for rule in TRUE_RULES
            ]
        )
    )
    # Legacy paired diagnostics keep the historical field names (the
    # blocked/passable evidence pair against the two rule futures).
    predicted = np.asarray(latents["predicted"], dtype=np.float64)
    encoded_targets = np.asarray(latents["encoded_targets"], dtype=np.float64)
    condition_index = {
        condition: index for index, condition in enumerate(latents["history_conditions"])
    }
    legacy, _legacy_records = _paired_prediction_metrics(
        pair_ids=tuple(str(value) for value in latents["query_ids"]),
        predicted_first=predicted[:, condition_index["observed_blocked"]],
        predicted_second=predicted[:, condition_index["observed_passable"]],
        target_first=encoded_targets[:, list(TRUE_RULES).index("blocked")],
        target_second=encoded_targets[:, list(TRUE_RULES).index("passable")],
        first_label="blocked",
        second_label="passable",
    )
    metrics = {
        "diagnostic": "door_structural_parity_development_v1",
        "primary_metric": "same_history_two_target_accuracy",
        "same_history_two_target_accuracy": primary,
        "history_conditions": list(HISTORY_CONDITIONS),
        "true_future_rules": list(TRUE_RULES),
        "queries": len(assets),
        "loss_records": len(scored["records"]),
        "by_true_rule": _decision_free(by_rule),
        "two_target_discrimination": summary["two_target_discrimination"],
        "target_latent_separation": summary["target_latent_separation"],
        "two_target_ties": summary["two_target_ties"],
        "paired_static_query_bootstrap": summary["paired_static_query_bootstrap"],
        "gate_completion_inputs": _decision_free(
            {
                "metrics": gate_inputs["metrics"],
                "uncertainty": gate_inputs["uncertainty"],
            }
        ),
        "legacy_paired_diagnostics": _decision_free(legacy),
    }
    state = {
        "before": str(
            scored["score_audit"]["frozen_state_hash_before"]
        ),
        "after": str(scored["score_audit"]["frozen_state_hash_after"]),
    }
    return metrics, scored["records"], state, selection


def _delay_episode_parts(
    value: tuple[Any, ...],
    *,
    profile: str,
    episode: int,
    fallback_delay: int,
) -> tuple[np.ndarray, np.ndarray, float | None, dict[str, Any]]:
    """Normalize the paired reader's optional metadata extension.

    The first three values are the long-standing ``(pixels, actions, speed)``
    contract.  The fourth value is present for the 1.0.3-rc1 Action Delay
    payload and carries the episode-constant ``dev_*`` identity columns.  The
    small fallback keeps synthetic/unit readers that implement the old tuple
    shape useful while ensuring every emitted query still has a stable local
    identity.
    """

    if len(value) < 3:
        raise RuntimeError(
            f"Action Delay Development {profile}/episode_{episode:04d} "
            "returned an incomplete episode tuple"
        )
    pixels = np.asarray(value[0])
    actions = np.asarray(value[1], dtype=np.float32)
    speed = value[2]
    metadata = dict(value[3]) if len(value) >= 4 and isinstance(value[3], Mapping) else {}
    metadata.setdefault(
        "dev_query_id",
        f"{profile}/episode_{episode:04d}",
    )
    metadata.setdefault("dev_eval_seed", 0.0)
    metadata.setdefault("dev_delay", float(fallback_delay))
    metadata.setdefault("dev_room", None)
    metadata.setdefault("dev_direction", None)
    return pixels, actions, speed, metadata


def _action_delay_arrays(payload: DevelopmentPayload) -> _DelayFamilyArrays:
    """Read complete per-episode delay families for six-group scoring.

    Every profile registers one member per delay value 0-10 that all share
    their episode ids, so one matched query keeps all eleven conditions of
    one episode together instead of collapsing them into d0-vs-delayed
    pairs.
    """

    grouped: dict[str, dict[int, Path]] = {}
    for path in payload.members:
        match = _ACTION_DELAY_MEMBER_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected Action Delay Development member: {path.name}")
        profile = match["profile"]
        delay = int(match["delay"])
        if delay in grouped.setdefault(profile, {}):
            raise ValueError(f"Duplicate Action Delay Development member: {path.name}")
        grouped[profile][delay] = path
    selection_contract = _selection_mapping(payload)
    reference_delay = int(selection_contract.get("reference_condition", 0))
    contrast_values = selection_contract.get("contrasts", tuple(range(1, 11)))
    if (
        not isinstance(contrast_values, Sequence)
        or isinstance(contrast_values, (str, bytes))
    ):
        raise ValueError("Action Delay Development contrasts must be a list")
    contrasts = tuple(int(value) for value in contrast_values)
    expected_delays = (reference_delay, *contrasts)
    if (
        reference_delay != 0
        or not contrasts
        or len(set(expected_delays)) != len(expected_delays)
        or tuple(sorted(expected_delays)) != expected_delays
    ):
        raise ValueError("Action Delay Development contract must include baseline delay 0")
    if any(tuple(sorted(values)) != expected_delays for values in grouped.values()):
        raise ValueError("Action Delay Development profiles have inconsistent delay members")
    per_contrast = _selection_value(
        payload, "pairs_per_contrast_per_profile", 5
    )
    if per_contrast <= 0:
        raise ValueError("Action Delay Development pairs_per_contrast must be positive")
    expected_profiles = _selection_value(payload, "profiles", 6)
    expected_selected = _selection_value(
        payload,
        "selected_pair_count",
        expected_profiles * len(contrasts) * per_contrast,
    )
    query_ids: list[str] = []
    query_metadata: list[dict[str, Any]] = []
    member_pixels: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    candidate_pairs = 0
    for profile in sorted(grouped):
        reference_ids, reference = _read_tworoom_episodes(
            grouped[profile][reference_delay],
            expected_steps=50,
            frame_steps=_ACTION_DELAY_FRAME_STEPS,
            metadata_columns=_ACTION_DELAY_METADATA_COLUMNS,
        )
        if len(reference_ids) < per_contrast:
            raise RuntimeError(
                f"Action Delay Development {profile} has fewer than "
                f"{per_contrast} episodes"
            )
        selected_ids = reference_ids[:per_contrast]
        conditions: dict[int, dict[int, tuple[np.ndarray, np.ndarray, float | None]]] = {
            reference_delay: reference
        }
        for delay in contrasts:
            delayed_ids, delayed = _read_tworoom_episodes(
                grouped[profile][delay],
                expected_steps=50,
                frame_steps=_ACTION_DELAY_FRAME_STEPS,
                selected_episode_ids=selected_ids,
                metadata_columns=_ACTION_DELAY_METADATA_COLUMNS,
            )
            # ``delayed_ids`` is the complete id set even when frame decoding
            # is limited to selected ids, so it also checks file pairing.
            if delayed_ids != reference_ids:
                raise RuntimeError(
                    f"Action Delay Development episode ids differ for "
                    f"{profile}/d{delay}"
                )
            candidate_pairs += len(reference_ids)
            conditions[delay] = delayed
        for episode in selected_ids:
            base_pixels, base_actions, _, base_metadata = _delay_episode_parts(
                reference[episode],
                profile=profile,
                episode=episode,
                fallback_delay=reference_delay,
            )
            query_id = str(base_metadata["dev_query_id"])
            if not query_id:
                raise RuntimeError(
                    f"Action Delay Development {profile}/episode_{episode:04d} "
                    "has an empty dev_query_id"
                )
            for delay in contrasts:
                delayed_pixels, delayed_actions, _, delayed_metadata = (
                    _delay_episode_parts(
                        conditions[delay][episode],
                        profile=profile,
                        episode=episode,
                        fallback_delay=delay,
                    )
                )
                for field in (
                    "dev_query_id",
                    "dev_eval_seed",
                    "dev_room",
                    "dev_direction",
                ):
                    if delayed_metadata.get(field) != base_metadata.get(field):
                        raise RuntimeError(
                            f"Action Delay Development metadata {field} differs "
                            f"for {profile}/episode_{episode:04d}/d{delay}"
                        )
                observed_delay = int(float(delayed_metadata["dev_delay"]))
                if observed_delay != delay:
                    raise RuntimeError(
                        f"Action Delay Development metadata dev_delay disagrees "
                        f"with filename for {profile}/episode_{episode:04d}: "
                        f"{observed_delay} != {delay}"
                    )
                _ensure_paired_example(
                    pair_id=(
                        f"{profile}/d{reference_delay}_vs_d{delay}/"
                        f"episode_{episode:04d}"
                    ),
                    first_pixels=base_pixels,
                    second_pixels=delayed_pixels,
                    first_actions=base_actions,
                    second_actions=delayed_actions,
                    history_length=7,
                )
            query_ids.append(query_id)
            query_index_match = re.search(r"(?:^|-)q(?P<index>\d+)$", query_id)
            query_metadata.append(
                {
                    "profile": profile,
                    "query_id": query_id,
                    "eval_seed": int(float(base_metadata["dev_eval_seed"])),
                    "evaluation_index": (
                        int(query_index_match["index"])
                        if query_index_match is not None
                        else int(episode)
                    ),
                    "room": base_metadata.get("dev_room"),
                    "direction": base_metadata.get("dev_direction"),
                }
            )
            member_pixels.append(
                np.stack(
                    [
                        _delay_episode_parts(
                            conditions[delay][episode],
                            profile=profile,
                            episode=episode,
                            fallback_delay=delay,
                        )[0]
                        for delay in expected_delays
                    ]
                )
            )
            actions.append(base_actions)
    if len(grouped) != expected_profiles:
        raise RuntimeError(
            "Action Delay Development profile count disagrees with public contract: "
            f"observed={len(grouped)} expected={expected_profiles}"
        )
    if len(query_ids) * len(contrasts) != expected_selected:
        raise RuntimeError(
            "Action Delay Development selected pair count disagrees with public contract: "
            f"observed={len(query_ids) * len(contrasts)} expected={expected_selected}"
        )
    if len(set(query_ids)) != len(query_ids):
        raise RuntimeError("Action Delay Development query ids must be unique")

    # The 1.0.3-rc1 contract carries six real evaluation seeds and balanced
    # room/direction strata.  Validate those values when the payload provides
    # them; the compatibility fallback above uses ``0``/``None`` only for
    # synthetic readers that predate the metadata columns.
    observed_seeds = {int(row["eval_seed"]) for row in query_metadata}
    observed_rooms = {row["room"] for row in query_metadata}
    observed_directions = {row["direction"] for row in query_metadata}
    expected_seeds = {
        int(value)
        for value in selection_contract.get("eval_seeds", ())
    }
    profile_seed_mapping = selection_contract.get("profile_eval_seed_mapping", {})
    if isinstance(profile_seed_mapping, Mapping):
        for profile, expected_seed in profile_seed_mapping.items():
            profile_query_seeds = {
                int(row["eval_seed"])
                for row in query_metadata
                if str(row.get("profile")) == str(profile)
            }
            if profile_query_seeds and profile_query_seeds != {int(expected_seed)}:
                raise RuntimeError(
                    "Action Delay Development profile/eval_seed metadata disagrees "
                    f"for {profile}: {sorted(profile_query_seeds)} != "
                    f"{int(expected_seed)}"
                )
    if expected_seeds and observed_seeds != expected_seeds:
        if not (observed_seeds == {0} and all(value is None for value in observed_rooms)):
            raise RuntimeError(
                "Action Delay Development eval_seed metadata disagrees with "
                f"the public contract: observed={sorted(observed_seeds)} "
                f"expected={sorted(expected_seeds)}"
            )
    seed_counts: dict[int, int] = {}
    seed_room_counts: dict[tuple[int, str], int] = {}
    seed_direction_counts: dict[tuple[int, str], int] = {}
    for row in query_metadata:
        seed = int(row["eval_seed"])
        seed_counts[seed] = seed_counts.get(seed, 0) + 1
        if row["room"] is not None:
            key = (seed, str(row["room"]))
            seed_room_counts[key] = seed_room_counts.get(key, 0) + 1
        if row["direction"] is not None:
            key = (seed, str(row["direction"]))
            seed_direction_counts[key] = seed_direction_counts.get(key, 0) + 1
    if expected_seeds and observed_seeds == expected_seeds:
        if set(seed_counts.values()) != {
            int(selection_contract.get("unique_queries_per_eval_seed", 50))
        }:
            raise RuntimeError(
                "Action Delay Development queries are not balanced by eval seed"
            )
        expected_rooms = selection_contract.get("rooms_per_seed", {})
        expected_directions = selection_contract.get("directions_per_seed", {})
        if expected_rooms and {
            (seed, room): count for (seed, room), count in seed_room_counts.items()
        } != {
            (seed, str(room)): int(count)
            for seed in expected_seeds
            for room, count in expected_rooms.items()
        }:
            raise RuntimeError(
                "Action Delay Development room strata disagree with the public contract"
            )
        if expected_directions and {
            (seed, direction): count
            for (seed, direction), count in seed_direction_counts.items()
        } != {
            (seed, str(direction)): int(count)
            for seed in expected_seeds
            for direction, count in expected_directions.items()
        }:
            raise RuntimeError(
                "Action Delay Development direction strata disagree with the public contract"
            )
    return _DelayFamilyArrays(
        query_ids=tuple(query_ids),
        query_metadata=tuple(query_metadata),
        delay_values=tuple(expected_delays),
        member_pixels=np.stack(member_pixels),
        raw_action_blocks=np.stack(actions),
        selection={
            "kind": "matched_filename_profile_delay_family_and_episode_id",
            "profiles": len(grouped),
            "delay_values": list(expected_delays),
            "candidate_pairs": candidate_pairs,
            "selected_queries": len(query_ids),
            "queries_per_profile": per_contrast,
            "selected_contrast_pairs": len(query_ids) * len(contrasts),
            "eval_seeds": sorted(observed_seeds),
            "eval_seed_query_counts": {
                str(seed): int(count)
                for seed, count in sorted(seed_counts.items())
            },
            "query_metadata_fields": [
                "query_id",
                "eval_seed",
                "room",
                "direction",
            ],
            "rule": (
                "all registered delay members per profile; first sorted "
                "shared episode ids"
            ),
        },
    )


def _paired_prediction_metrics(
    *,
    pair_ids: tuple[str, ...],
    predicted_first: np.ndarray,
    predicted_second: np.ndarray,
    target_first: np.ndarray,
    target_second: np.ndarray,
    first_label: str,
    second_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arrays = [
        np.asarray(value, dtype=np.float32)
        for value in (
            predicted_first,
            predicted_second,
            target_first,
            target_second,
        )
    ]
    if not pair_ids or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Development pair ids must be non-empty and unique")
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("Development latent arrays must be rank two")
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("Development predicted and target latent shapes differ")
    if arrays[0].shape[0] != len(pair_ids) or not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Development latent arrays are malformed")
    predicted_first, predicted_second, target_first, target_second = arrays
    first_first = np.square(predicted_first - target_first).mean(axis=-1)
    first_second = np.square(predicted_first - target_second).mean(axis=-1)
    second_first = np.square(predicted_second - target_first).mean(axis=-1)
    second_second = np.square(predicted_second - target_second).mean(axis=-1)
    first_future = first_first < first_second
    second_future = second_second < second_first
    first_history = first_first < second_first
    second_history = second_second < first_second
    switch = np.sum(
        (predicted_second - predicted_first) * (target_second - target_first), axis=-1
    ) > 0
    correct_future = np.concatenate([first_future, second_future])
    correct_history = np.concatenate([first_history, second_history])
    correct_losses = np.concatenate([first_first, second_second])
    other_losses = np.concatenate([first_second, second_first])
    latent_response, latent_records = paired_latent_response_metrics(
        pair_ids=pair_ids,
        predicted_first=predicted_first,
        predicted_second=predicted_second,
        target_first=target_first,
        target_second=target_second,
    )
    calibrated = np.asarray(
        [row["calibrated_response_success"] for row in latent_records], dtype=bool
    )
    joint = (
        first_future
        & second_future
        & first_history
        & second_history
        & calibrated
    )
    metrics = {
        "pair_count": len(pair_ids),
        "decision_count": 2 * len(pair_ids),
        "correct_future_rate": float(correct_future.mean()),
        "correct_history_rate": float(correct_history.mean()),
        "context_switch_rate": float(switch.mean()),
        f"{first_label}_correct_future_rate": float(first_future.mean()),
        f"{second_label}_correct_future_rate": float(second_future.mean()),
        "worst_condition_correct_future_rate": float(
            min(first_future.mean(), second_future.mean())
        ),
        "correct_future_mse_mean": float(correct_losses.mean()),
        "other_future_mse_mean": float(other_losses.mean()),
        "other_minus_correct_mse_margin_mean": float(
            (other_losses - correct_losses).mean()
        ),
        "current_frame_only_accuracy_bound": 0.5,
        "latent_response": latent_response,
        "joint_icl_pair_success_rate": float(joint.mean()),
    }
    records = [
        {
            "pair_id": pair_id,
            first_label: {
                "correct_future_mse": float(first_first[index]),
                "other_future_mse": float(first_second[index]),
                "correct_future": bool(first_future[index]),
                "correct_history": bool(first_history[index]),
            },
            second_label: {
                "correct_future_mse": float(second_second[index]),
                "other_future_mse": float(second_first[index]),
                "correct_future": bool(second_future[index]),
                "correct_history": bool(second_history[index]),
            },
            "context_switch_correct": bool(switch[index]),
            "joint_icl_pair_success": bool(joint[index]),
            "latent_response": {
                key: value for key, value in latent_records[index].items() if key != "pair_id"
            },
        }
        for index, pair_id in enumerate(pair_ids)
    ]
    return metrics, records


def _task_prediction_metrics(
    *,
    task: str,
    arrays: _PairedArrays,
    predicted_first: np.ndarray,
    predicted_second: np.ndarray,
    target_first: np.ndarray,
    target_second: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if task == "action_strength":
        from contextworld.benchmarks.action_strength_icl_score import _prediction_metrics

        return _prediction_metrics(
            pair_ids=arrays.pair_ids,
            predicted_low=predicted_first,
            predicted_high=predicted_second,
            target_low=target_first,
            target_high=target_second,
        )
    if task == "contact_friction":
        from contextworld.benchmarks.contact_friction_icl_score import contact_friction_prediction_metrics

        return contact_friction_prediction_metrics(
            pair_ids=arrays.pair_ids,
            predicted_low=predicted_first,
            predicted_high=predicted_second,
            target_low=target_first,
            target_high=target_second,
        )
    if task == "motion_damping":
        from contextworld.benchmarks.motion_damping_icl_score import motion_damping_prediction_metrics

        return motion_damping_prediction_metrics(
            pair_ids=arrays.pair_ids,
            predicted_faster_decay=predicted_first,
            predicted_no_extra_decay=predicted_second,
            target_faster_decay=target_first,
            target_no_extra_decay=target_second,
        )
    if task == "robot_arm_mass":
        from contextworld.benchmarks.reacher_arm_mass_icl_score import reacher_arm_mass_prediction_metrics

        return reacher_arm_mass_prediction_metrics(
            pair_ids=arrays.pair_ids,
            predicted_lighter=predicted_first,
            predicted_heavier=predicted_second,
            target_lighter=target_first,
            target_heavier=target_second,
        )
    if task == "portal_exit":
        from contextworld.benchmarks.portal_exit_icl_score import portal_exit_prediction_metrics

        return portal_exit_prediction_metrics(
            pair_ids=arrays.pair_ids,
            predicted_near=predicted_first,
            predicted_farther=predicted_second,
            target_near=target_first,
            target_farther=target_second,
        )
    if task == "cube_gripper_carry":
        from contextworld.benchmarks.cube_grasp_rule_icl_score import cube_grasp_rule_prediction_metrics

        return cube_grasp_rule_prediction_metrics(
            pair_ids=arrays.pair_ids,
            predicted_cannot_hold=predicted_first,
            predicted_can_hold=predicted_second,
            target_cannot_hold=target_first,
            target_can_hold=target_second,
        )
    return _paired_prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_first=predicted_first,
        predicted_second=predicted_second,
        target_first=target_first,
        target_second=target_second,
        first_label=arrays.first_label,
        second_label=arrays.second_label,
    )


def _evaluate_paired(
    *,
    payload: DevelopmentPayload,
    arrays: _PairedArrays,
    adapter: LatentWorldModelAdapter,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    history_length = payload.history_length
    validate_adapter_protocol(
        adapter,
        history_tokens=history_length,
        action_block_raw_steps=payload.frameskip,
        action_dim=payload.action_dimension,
        minimum_future_action_blocks=1,
        task_name=f"{payload.task} Development",
    )
    if (
        arrays.first_pixels.ndim != 5
        or arrays.second_pixels.shape != arrays.first_pixels.shape
        or arrays.first_pixels.shape[0] != len(arrays.pair_ids)
        or arrays.first_pixels.shape[1] != history_length + 1
        or arrays.raw_action_blocks.shape[:2]
        != (len(arrays.pair_ids), arrays.raw_action_blocks.shape[1])
        or arrays.raw_action_blocks.shape[1] < history_length
        or arrays.raw_action_blocks.shape[-2:]
        != (payload.frameskip, payload.action_dimension)
    ):
        raise RuntimeError(f"Malformed paired Development arrays for {payload.task}")
    histories = np.concatenate(
        [arrays.first_pixels[:, :history_length], arrays.second_pixels[:, :history_length]],
        axis=0,
    )
    actions = np.concatenate(
        [
            arrays.raw_action_blocks[:, :history_length],
            arrays.raw_action_blocks[:, :history_length],
        ],
        axis=0,
    )
    before = adapter.frozen_state_hash()
    predicted = np.asarray(
        adapter.rollout_latents(histories, actions, batch_size=int(batch_size))
    )
    count = len(arrays.pair_ids)
    if (
        predicted.ndim != 3
        or predicted.shape[:2] != (2 * count, 1)
        or not np.isfinite(predicted).all()
    ):
        raise RuntimeError(
            f"{payload.task} Development adapter must return finite "
            "(2 * pair_count, 1, latent_dim) futures"
        )
    targets = np.concatenate(
        [arrays.first_pixels[:, history_length], arrays.second_pixels[:, history_length]],
        axis=0,
    )
    encoded = np.asarray(adapter.encode_pixels(targets, batch_size=int(batch_size)))
    if (
        encoded.ndim != 2
        or encoded.shape != (2 * count, predicted.shape[-1])
        or not np.isfinite(encoded).all()
    ):
        raise RuntimeError(
            f"{payload.task} Development target encodings do not match predicted latents"
        )
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError(f"Model state changed during {payload.task} Development scoring")
    metrics, records = _task_prediction_metrics(
        task=payload.task,
        arrays=arrays,
        predicted_first=predicted[:count, 0],
        predicted_second=predicted[count:, 0],
        target_first=encoded[:count],
        target_second=encoded[count:],
    )
    return metrics, records, {"before": before, "after": after}


def _summarize_development_action_delay_horizons(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the Test horizon matrix kernel with Development strata.

    ``summarize_h7_validation_records`` is frozen around the Public-Test
    seed constants (42--47).  Development deliberately uses seeds 52--57,
    so this wrapper reuses its matrix selector and metric aggregation while
    constructing the two seed/direction breakdowns from the actual rows.
    """

    from collections import defaultdict

    from contextworld.evaluation.action_delay_h7_score import (
        _aggregate_metrics,
        _summarize_query_matrices,
    )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["query_id"]), int(row["horizon"]))].append(row)
    horizon_matrices: list[tuple[dict[str, Any], int, dict[tuple[int, int], float]]] = []
    trajectory_groups: dict[str, dict[tuple[int, int], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    trajectory_exemplars: dict[str, dict[str, Any]] = {}
    for (query_id, horizon), rows in sorted(grouped.items()):
        losses = {
            (int(row["history_delay"]), int(row["target_delay"])): float(
                row["latent_mse"]
            )
            for row in rows
        }
        horizon_matrices.append((rows[0], horizon, losses))
        trajectory_exemplars[query_id] = rows[0]
        for pair, loss in losses.items():
            trajectory_groups[query_id][pair].append(loss)

    horizon_metrics = _summarize_query_matrices(horizon_matrices)
    trajectory_matrices: list[
        tuple[dict[str, Any], str, dict[tuple[int, int], float]]
    ] = []
    for query_id in sorted(trajectory_groups):
        values = trajectory_groups[query_id]
        if not all(len(losses) == 3 for losses in values.values()):
            raise ValueError(f"Incomplete Development three-step trajectory: {query_id}")
        trajectory_matrices.append(
            (
                trajectory_exemplars[query_id],
                "trajectory",
                {pair: float(np.mean(losses)) for pair, losses in values.items()},
            )
        )
    trajectory_metrics = _summarize_query_matrices(trajectory_matrices)

    def breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
        seeds = sorted({int(row["eval_seed"]) for row in rows})
        directions = sorted({str(row["direction"]) for row in rows})
        return {
            "overall": _aggregate_metrics(rows),
            "by_target_delay": {
                str(delay): _aggregate_metrics(
                    [row for row in rows if int(row["target_delay"]) == delay]
                )
                for delay in range(11)
            },
            "by_track": {
                track: _aggregate_metrics(
                    [row for row in rows if row["target_track"] == track]
                )
                for track in (
                    "training_seen",
                    "within_range_unseen",
                    "above_range_unseen",
                )
            },
            "by_target_delay_and_eval_seed": {
                str(delay): {
                    str(seed): _aggregate_metrics(
                        [
                            row
                            for row in rows
                            if int(row["target_delay"]) == delay
                            and int(row["eval_seed"]) == seed
                        ]
                    )
                    for seed in seeds
                }
                for delay in range(11)
            },
            "by_target_delay_and_direction": {
                str(delay): {
                    direction: _aggregate_metrics(
                        [
                            row
                            for row in rows
                            if int(row["target_delay"]) == delay
                            and str(row["direction"]) == direction
                        ]
                    )
                    for direction in directions
                }
                for delay in range(11)
            },
        }

    return {
        "trajectory": {
            **breakdown(trajectory_metrics),
            "query_metrics": trajectory_metrics,
        },
        "by_horizon": {
            str(horizon): {
                **breakdown(
                    [row for row in horizon_metrics if int(row["horizon"]) == horizon]
                ),
                "query_metrics": [
                    row
                    for row in horizon_metrics
                    if int(row["horizon"]) == horizon
                ],
            }
            for horizon in (1, 2, 3)
        },
    }


def _action_delay_physical_group_metrics(
    *,
    payload: DevelopmentPayload,
    arrays: _DelayFamilyArrays,
    adapter: LatentWorldModelAdapter,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Score the six physical response groups (0/1/2/3/4/5-10) per query.

    This mirrors the frozen Public-Test protocol behind ``core_h1``: for each
    matched query the adapter predicts one future from every delay
    condition's history, each prediction is compared against every
    condition's encoded future (an 11x11 latent-MSE matrix per query, the
    Development equivalent of the records built in
    ``action_delay_h7_score.score_h7_validation_assets`` /
    ``_summarize_query_matrices``), and a condition is credited when the
    nearest target lands in its own physical group.  The aggregation reuses
    the model-independent kernel ``action_delay_h7_core.
    summarize_action_delay_h1_physical`` (the same function that produces
    ``core_h1``); only the Development reader above is bundle-specific.
    """

    history_length = payload.history_length
    validate_adapter_protocol(
        adapter,
        history_tokens=history_length,
        action_block_raw_steps=payload.frameskip,
        action_dim=payload.action_dimension,
        minimum_future_action_blocks=1,
        task_name="action_delay Development",
    )
    queries = len(arrays.query_ids)
    delays = len(arrays.delay_values)
    if (
        not queries
        or len(set(arrays.query_ids)) != queries
        or arrays.member_pixels.ndim != 6
        or arrays.member_pixels.shape[:2] != (queries, delays)
        or arrays.member_pixels.shape[2] < history_length + 1
        or arrays.raw_action_blocks.ndim != 4
        or arrays.raw_action_blocks.shape[0] != queries
        or arrays.raw_action_blocks.shape[1] < history_length
        or arrays.raw_action_blocks.shape[-2:]
        != (payload.frameskip, payload.action_dimension)
    ):
        raise RuntimeError("Malformed Action Delay Development arrays")
    if len(arrays.query_metadata) != queries:
        raise RuntimeError("Action Delay Development query metadata is incomplete")

    # The 1.0.3-rc1 Development payload carries h1/h2/h3 target frames at
    # token offsets 7/8/9.  A compatible adapter can request all three with a
    # single nine-block call, letting the same Test-side horizon kernel score
    # the two auxiliary outputs.  Older synthetic readers and one-future
    # adapters retain the original h1-only route.
    available_horizons = min(
        3,
        int(arrays.member_pixels.shape[2] - history_length),
    )
    protocol_horizons = int(adapter.protocol.future_action_blocks)
    auxiliary_horizons = (
        (2, 3)
        if available_horizons >= 3 and protocol_horizons >= 3
        else ()
    )
    requested_horizons = 3 if auxiliary_horizons else 1
    histories = arrays.member_pixels[:, :, :history_length].reshape(
        queries * delays,
        history_length,
        *arrays.member_pixels.shape[3:],
    )
    # ``_ensure_paired_example`` verified every family member shares the
    # episode's action blocks, so the reference member's blocks describe all
    # eleven conditions.
    actions = np.repeat(
        arrays.raw_action_blocks[
            :, None, : history_length - 1 + requested_horizons
        ],
        repeats=delays,
        axis=1,
    ).reshape(
        queries * delays,
        history_length - 1 + requested_horizons,
        payload.frameskip,
        payload.action_dimension,
    )
    before = adapter.frozen_state_hash()
    predicted = np.asarray(
        adapter.rollout_latents(histories, actions, batch_size=int(batch_size))
    )
    if (
        predicted.ndim != 3
        or predicted.shape[:2] != (queries * delays, requested_horizons)
        or not np.isfinite(predicted).all()
    ):
        raise RuntimeError(
            "Action Delay Development adapter must return finite "
            f"(query_count * delay_count, {requested_horizons}, latent_dim) futures"
        )
    futures = arrays.member_pixels[
        :, :, history_length : history_length + requested_horizons
    ].reshape(
        queries * delays * requested_horizons,
        *arrays.member_pixels.shape[3:],
    )
    encoded = np.asarray(adapter.encode_pixels(futures, batch_size=int(batch_size)))
    if (
        encoded.ndim != 2
        or encoded.shape
        != (queries * delays * requested_horizons, predicted.shape[-1])
        or not np.isfinite(encoded).all()
    ):
        raise RuntimeError(
            "Action Delay Development target encodings do not match predicted latents"
        )
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during action_delay Development scoring")
    predictions = predicted.reshape(queries, delays, requested_horizons, -1)
    targets = encoded.reshape(queries, delays, requested_horizons, -1)
    predictions_h1 = predictions[:, :, 0]
    targets_h1 = targets[:, :, 0]
    losses = np.square(
        predictions_h1[:, :, None, :] - targets_h1[:, None, :, :]
    ).mean(axis=-1)
    query_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for query_index, query_id in enumerate(arrays.query_ids):
        metadata = arrays.query_metadata[query_index]
        for condition_index, delay in enumerate(arrays.delay_values):
            # Same nearest-target decision and lowest-delay tie-break as the
            # frozen scorer's ``selected_target``
            # (action_delay_h7_score._summarize_query_matrices).
            selected_index = min(
                range(delays),
                key=lambda target_index: (
                    losses[query_index, condition_index, target_index],
                    arrays.delay_values[target_index],
                ),
            )
            selected_delay = int(arrays.delay_values[selected_index])
            query_rows.append(
                {
                    "query_id": query_id,
                    "eval_seed": int(metadata.get("eval_seed", 0)),
                    "evaluation_index": int(
                        metadata.get("evaluation_index", query_index % 50)
                    ),
                    "room": metadata.get("room"),
                    "direction": metadata.get("direction"),
                    "horizon": 1,
                    "target_delay": int(delay),
                    "selected_target": selected_delay,
                }
            )
            true_group = physical_group(int(delay))
            selected_group = physical_group(selected_delay)
            records.append(
                {
                    "query_id": query_id,
                    "eval_seed": int(metadata.get("eval_seed", 0)),
                    "evaluation_index": int(
                        metadata.get("evaluation_index", query_index % 50)
                    ),
                    "room": metadata.get("room"),
                    "direction": metadata.get("direction"),
                    "target_delay": int(delay),
                    "target_physical_group": true_group,
                    "selected_target": selected_delay,
                    "selected_physical_group": selected_group,
                    "physical_group_correct": bool(selected_group == true_group),
                    "matching_target_mse": float(
                        losses[query_index, condition_index, condition_index]
                    ),
                    "selected_target_mse": float(
                        losses[query_index, condition_index, selected_index]
                    ),
                }
            )
    metrics = summarize_action_delay_h1_physical(
        query_rows,
        bootstrap_resamples=_ACTION_DELAY_BOOTSTRAP_RESAMPLES,
        bootstrap_seed=_ACTION_DELAY_BOOTSTRAP_RANDOM_SEED,
    )
    metadata_by_query = {
        str(row["query_id"]): row for row in arrays.query_metadata
    }
    for row in metrics.get("query_metrics", []):
        metadata = metadata_by_query.get(str(row["query_id"]), {})
        row["room"] = metadata.get("room")
        row["direction"] = metadata.get("direction")
    metrics["query_metadata"] = {
        "fields": ["query_id", "eval_seed", "room", "direction"],
        "eval_seed_query_counts": metrics.get("eval_seed_query_counts", {}),
        "room_counts": {
            str(room): sum(
                1 for row in arrays.query_metadata if row.get("room") == room
            )
            for room in sorted(
                {
                    row.get("room")
                    for row in arrays.query_metadata
                    if row.get("room") is not None
                }
            )
        },
        "direction_counts": {
            str(direction): sum(
                1
                for row in arrays.query_metadata
                if row.get("direction") == direction
            )
            for direction in sorted(
                {
                    row.get("direction")
                    for row in arrays.query_metadata
                    if row.get("direction") is not None
                }
            )
        },
    }

    if auxiliary_horizons:
        # Build the full Test-shaped horizon loss records from the one h3
        # rollout.  This is retained as Development diagnostics only; no gate
        # or pass decision is copied into the result.
        def _target_track(delay: int) -> str:
            if delay in (0, 4, 8):
                return "training_seen"
            if delay in (1, 2, 3, 5, 6, 7):
                return "within_range_unseen"
            return "above_range_unseen"

        horizon_records: list[dict[str, Any]] = []
        horizon_losses = np.square(
            predictions[:, :, None, :, :]
            - targets[:, None, :, :, :]
        ).mean(axis=-1)
        for query_index, query_id in enumerate(arrays.query_ids):
            metadata = arrays.query_metadata[query_index]
            for horizon_index in range(requested_horizons):
                horizon = horizon_index + 1
                for history_index, history_delay in enumerate(arrays.delay_values):
                    for target_index, target_delay in enumerate(arrays.delay_values):
                        horizon_records.append(
                            {
                                "query_id": str(query_id),
                                "eval_seed": int(metadata.get("eval_seed", 0)),
                                "evaluation_index": int(
                                    metadata.get("evaluation_index", query_index % 50)
                                ),
                                "room": metadata.get("room"),
                                "direction": metadata.get("direction"),
                                "history_delay": int(history_delay),
                                "target_delay": int(target_delay),
                                "target_track": _target_track(int(target_delay)),
                                "horizon": horizon,
                                "target_physical_group": physical_future_group(
                                    int(target_delay), horizon
                                ),
                                "latent_mse": float(
                                    horizon_losses[
                                        query_index,
                                        history_index,
                                        target_index,
                                        horizon_index,
                                    ]
                                ),
                            }
                        )
        horizon_summary = _summarize_development_action_delay_horizons(
            horizon_records
        )
        metrics["by_horizon"] = _decision_free(horizon_summary["by_horizon"])
        metrics["trajectory"] = _decision_free(horizon_summary["trajectory"])
        metrics["auxiliary_horizons"] = {
            "available": [2, 3],
            "source": "single_h3_rollout_reused_for_h2_h3",
            "record_count": len(horizon_records),
        }
    # Structural parity with Public Test: emit the same six anti-shortcut gate
    # INPUT metrics through the Test kernel itself.  ``losses`` is indexed
    # [query, history condition, target], so the matching-history entry for a
    # target is the diagonal and the alternatives vary the history condition.
    # Alternatives sharing the target's physical group are excluded -- at
    # horizon 1 delays 5..10 are all stationary, so requiring the matching
    # history to beat them would score a distinction the task does not contain
    # (the same exclusion the Test kernel applies to its delay pairs).
    from contextworld.benchmarks.action_delay_icl_score import (
        action_delay_gate_completion_metrics,
    )
    from contextworld.benchmarks.test_gate_completion import (
        load_test_gate_completion_config,
    )

    strict_wins = np.zeros((queries, delays), dtype=bool)
    for query_index in range(queries):
        for target_index, target_delay in enumerate(arrays.delay_values):
            alternatives = [
                losses[query_index, history_index, target_index]
                for history_index, history_delay in enumerate(arrays.delay_values)
                if physical_group(int(history_delay))
                != physical_group(int(target_delay))
            ]
            if not alternatives:
                raise RuntimeError(
                    "No physically distinguishable alternative for delay"
                    f" {target_delay}"
                )
            strict_wins[query_index, target_index] = bool(
                losses[query_index, target_index, target_index] < min(alternatives)
            )
    gate_inputs = action_delay_gate_completion_metrics(
        predicted_h1=predictions_h1,
        encoded_h1=targets_h1,
        query_ids=[str(value) for value in arrays.query_ids],
        history_strict_wins=strict_wins,
        config=load_test_gate_completion_config(),
    )
    metrics["gate_completion_inputs"] = _decision_free(
        {
            "metrics": gate_inputs["metrics"],
            "uncertainty": gate_inputs["uncertainty"],
        }
    )
    return metrics, records, {"before": before, "after": after}


def _speed_structural_tracks(
    payload: DevelopmentPayload,
) -> tuple[dict[str, dict[tuple[str, float], dict[str, dict[str, Any]]]], dict[str, Any]]:
    """Group the speed payload into per-track speed families.

    Every track table holds one episode per (static query, reference speed,
    history condition).  The returned mapping keeps one family entry per
    ``(static_query_id, reference_speed)`` with all of the track's history
    conditions, which is exactly the grouping the Public Test speed scorer
    builds from its frozen catalogs.
    """

    grouped: dict[tuple[str, str], Path] = {}
    for path in payload.members:
        match = _SPEED_STRUCTURAL_MEMBER_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected Speed Development member: {path.name}")
        key = (str(match["track"]), str(match["condition"]))
        if key in grouped:
            raise ValueError(f"Duplicate Speed Development member: {path.name}")
        grouped[key] = path

    tracks: dict[str, dict[tuple[str, float], dict[str, dict[str, Any]]]] = {}
    table_sizes: dict[str, int] = {}
    for (track, condition), path in sorted(grouped.items()):
        records = _read_dev_episodes(
            path,
            expected_steps=_SPEED_EPISODE_ROWS,
            frame_steps=_SPEED_TOKEN_ROWS,
            string_columns=(
                "dev_query_id",
                "dev_static_query_id",
                "dev_track",
                "dev_condition",
                "dev_template_id",
            ),
            scalar_columns=(
                "dev_eval_seed",
                "dev_evaluation_index",
                "dev_reference_speed",
                "dev_condition_speed",
            ),
        )
        table_sizes[f"{track}/{condition}"] = len(records)
        for record in records:
            if record["dev_track"] != track or record["dev_condition"] != condition:
                raise RuntimeError(
                    f"Speed Development episode columns disagree with its "
                    f"table: {path.name}"
                )
            # The per-episode scalar columns are float32; the Public Test
            # catalogs key speeds as float64.  Round to four decimals (well
            # below the 0.05 speed spacing) so both sides produce identical
            # reference-speed keys.
            record["dev_reference_speed"] = round(
                float(record["dev_reference_speed"]), 4
            )
            record["dev_condition_speed"] = round(
                float(record["dev_condition_speed"]), 4
            )
            family = tracks.setdefault(track, {})
            key = (
                record["dev_static_query_id"],
                record["dev_reference_speed"],
            )
            slot = family.setdefault(key, {})
            if condition in slot:
                raise RuntimeError(
                    f"Speed Development query {key} repeats {condition}"
                )
            slot[condition] = record

    selection_facts: dict[str, Any] = {"tracks": {}}
    for track, families in sorted(tracks.items()):
        conditions = sorted(
            {condition for slot in families.values() for condition in slot}
        )
        speeds = sorted({key[1] for key in families})
        seeds_by_static: dict[str, int] = {}
        for (static_id, speed), slot in families.items():
            if set(slot) != set(conditions):
                raise RuntimeError(
                    f"Speed Development {track}/{static_id}/{speed} lacks a "
                    "history condition"
                )
            seeds = {record["dev_eval_seed"] for record in slot.values()}
            if len(seeds) != 1:
                raise RuntimeError(
                    f"Speed Development {track}/{static_id} mixes eval seeds"
                )
            seed = int(seeds.pop())
            previous = seeds_by_static.setdefault(static_id, seed)
            if previous != seed:
                raise RuntimeError(
                    f"Speed Development {static_id} mixes eval seeds across "
                    "reference speeds"
                )
        # One history condition per reference speed: the condition simulated
        # at the reference speed is the matching history for that speed.
        condition_speeds: dict[str, float] = {}
        for condition in conditions:
            values = {
                record["dev_condition_speed"]
                for slot in families.values()
                for name, record in slot.items()
                if name == condition
            }
            if len(values) != 1:
                raise RuntimeError(
                    f"Speed Development condition {track}/{condition} has "
                    "inconsistent condition speeds"
                )
            condition_speeds[condition] = float(values.pop())
        if sorted(condition_speeds.values()) != speeds:
            raise RuntimeError(
                f"Speed Development track {track} conditions do not cover its "
                f"reference speeds one-to-one: {condition_speeds} vs {speeds}"
            )
        by_seed: dict[int, int] = {}
        for static_id, seed in seeds_by_static.items():
            by_seed[seed] = by_seed.get(seed, 0) + 1
        if (
            len(families) != len(speeds) * 300
            or set(by_seed.values())
            != {_SPEED_DEV_QUERIES_PER_REFERENCE_SPEED_PER_SEED}
            or sorted(by_seed) != list(_SPEED_DEV_EVAL_SEEDS)
        ):
            raise RuntimeError(
                f"Speed Development track {track} disagrees with the Public "
                f"Test stratification: families={len(families)} "
                f"speeds={speeds} by_seed={by_seed}"
            )
        selection_facts["tracks"][track] = {
            "reference_speeds": [float(value) for value in speeds],
            "history_conditions": conditions,
            "condition_speeds": condition_speeds,
            "static_queries": len(seeds_by_static),
            "episodes": len(families) * len(conditions),
        }
    selection = {
        "kind": "structural_parity_speed_families",
        "tables": len(payload.members),
        "episodes_per_table": sorted(set(table_sizes.values())),
        "eval_seeds": list(_SPEED_DEV_EVAL_SEEDS),
        "unique_queries_per_reference_speed_per_seed": (
            _SPEED_DEV_QUERIES_PER_REFERENCE_SPEED_PER_SEED
        ),
        **selection_facts,
    }
    return tracks, selection


def _speed_structural_metrics(
    *,
    payload: DevelopmentPayload,
    adapter: LatentWorldModelAdapter,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str], Mapping[str, Any]]:
    """Score the speed Development payload with the Public Test kernels.

    Mirrors ``speed_icl_score._score_track`` per track: one rollout per
    (static query, reference speed, history condition) over the three
    history tokens and the shared 3+5 action blocks, horizon losses against
    the five encoded true futures, per-horizon summaries from the frozen
    ``_loss_summary`` kernel, and additive gate inputs from
    ``speed_gate_completion_metrics``.  Tracks and horizons are never pooled.
    """

    from contextworld.benchmarks.speed_icl_data import HORIZONS
    from contextworld.benchmarks.speed_icl_score import (
        _loss_summary,
        speed_gate_completion_metrics,
    )
    from contextworld.benchmarks.test_gate_completion import (
        load_test_gate_completion_config,
    )

    validate_adapter_protocol(
        adapter,
        history_tokens=payload.history_length,
        action_block_raw_steps=payload.frameskip,
        action_dim=payload.action_dimension,
        minimum_future_action_blocks=max(HORIZONS),
        task_name="speed Development",
    )
    tracks, selection = _speed_structural_tracks(payload)
    bundle_chunk = max(1, int(batch_size))
    before = adapter.frozen_state_hash()
    track_outputs: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for track in sorted(tracks):
        families = tracks[track]
        conditions = selection["tracks"][track]["history_conditions"]
        condition_speeds = selection["tracks"][track]["condition_speeds"]
        speeds = selection["tracks"][track]["reference_speeds"]
        matching_by_speed = {
            speed: condition for condition, speed in condition_speeds.items()
        }
        keys = sorted(families)
        records: list[dict[str, Any]] = []
        gate_members: dict[str, dict[float, dict[str, Any]]] = {}
        for start in range(0, len(keys), bundle_chunk):
            chunk = keys[start : start + bundle_chunk]
            matching_condition = matching_by_speed[chunk[0][1]]
            target_pixels = np.stack(
                [
                    families[key][matching_condition]["frames"][3:8]
                    for key in chunk
                ]
            )
            for key in chunk:
                for condition in conditions:
                    if condition == matching_by_speed[key[1]]:
                        continue
                    if not np.array_equal(
                        families[key][condition]["frames"][3],
                        families[key][matching_by_speed[key[1]]]["frames"][3],
                    ):
                        raise RuntimeError(
                            f"Speed Development {key}: true futures differ "
                            "across history conditions"
                        )
            encoded = np.asarray(
                adapter.encode_pixels(
                    target_pixels.reshape(-1, *target_pixels.shape[2:]),
                    batch_size=int(batch_size),
                )
            ).reshape(len(chunk), target_pixels.shape[1], -1)
            if not np.isfinite(encoded).all():
                raise RuntimeError(
                    "Speed Development target encodings are not finite"
                )
            samples = [
                (index, families[key][condition], condition)
                for index, key in enumerate(chunk)
                for condition in conditions
            ]
            pixels = np.stack([record["frames"][:3] for _, record, _ in samples])
            # The episode carries 3+5+1 blocks (the trailing zero block only
            # gives the last future frame a row).  A rollout request is
            # (history_tokens - 1) context blocks plus the 5 future blocks:
            # 7 blocks = 5 predicted futures, the Public Test request shape.
            actions = np.stack(
                [
                    record["actions"].reshape(
                        _SPEED_EPISODE_ROWS // 5,
                        5,
                        record["actions"].shape[-1],
                    )[: _SPEED_EPISODE_ROWS // 5 - 1]
                    for _, record, _ in samples
                ]
            )
            predictions = np.asarray(
                adapter.rollout_latents(pixels, actions, batch_size=int(batch_size))
            )
            if (
                predictions.ndim != 3
                or predictions.shape[0] != len(samples)
                or predictions.shape[1] < max(HORIZONS)
                or not np.isfinite(predictions).all()
            ):
                raise RuntimeError(
                    "Speed Development adapter must return finite "
                    f"(sample, >= {max(HORIZONS)}, latent_dim) futures"
                )
            for (index, record, condition), prediction in zip(samples, predictions):
                static_id, reference_speed = chunk[index]
                losses = np.square(
                    prediction.astype(np.float64) - encoded[index].astype(np.float64)
                ).mean(axis=-1)
                records.append(
                    {
                        "query_id": record["dev_query_id"],
                        "static_query_id": static_id,
                        "track": track,
                        "reference_speed": float(reference_speed),
                        "matching_condition": matching_by_speed[reference_speed],
                        "eval_seed": int(record["dev_eval_seed"]),
                        "evaluation_index": int(record["dev_evaluation_index"]),
                        "condition": condition,
                        "history_speed": float(record["dev_condition_speed"]),
                        "latent_mse_by_horizon": {
                            str(horizon): float(losses[horizon - 1])
                            for horizon in HORIZONS
                        },
                    }
                )
                member = gate_members.setdefault(static_id, {}).get(reference_speed)
                if member is None:
                    member = {
                        "query_id": record["dev_query_id"],
                        "eval_seed": int(record["dev_eval_seed"]),
                        "matching_condition": matching_by_speed[reference_speed],
                        "condition_speeds": {},
                        "condition_predictions": {},
                        "target": encoded[index][0],
                    }
                    gate_members[static_id][reference_speed] = member
                member["condition_speeds"][condition] = float(
                    record["dev_condition_speed"]
                )
                member["condition_predictions"][condition] = np.asarray(
                    prediction[0]
                )
        horizons = {
            str(horizon): _decision_free(_loss_summary(records, horizon))
            for horizon in HORIZONS
        }
        gate_inputs = speed_gate_completion_metrics(
            groups={key: dict(value) for key, value in gate_members.items()},
            records=records,
            config=load_test_gate_completion_config(),
            full_protocol=True,
        )
        if gate_inputs.get("metrics") is None:
            raise RuntimeError(
                f"Speed Development track {track} produced no complete speed "
                "family for the paired gate inputs"
            )
        track_outputs[track] = {
            "reference_speeds": speeds,
            "history_conditions": conditions,
            "condition_speeds": condition_speeds,
            "episodes": len(records),
            "horizons": horizons,
            "gate_completion_inputs": _decision_free(
                {
                    "metrics": gate_inputs["metrics"],
                    "uncertainty": gate_inputs["uncertainty"],
                }
            ),
        }
        all_records.extend(records)
    legacy = _speed_legacy_history_utility(
        payload=payload,
        adapter=adapter,
        tracks=tracks,
        selection=selection,
        batch_size=int(batch_size),
    )
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during speed Development scoring")
    primary = {
        track: track_outputs[track]["horizons"]["1"][
            "reference_speed_balanced_strict_query_win_rate_vs_every_other"
        ]
        for track in sorted(track_outputs)
    }
    metrics = {
        "diagnostic": "speed_structural_parity_development_v1",
        "primary_metric": (
            "reference_speed_balanced_strict_query_win_rate_vs_every_other"
        ),
        "primary_metric_horizon": 1,
        "reference_speed_balanced_strict_query_win_rate_vs_every_other": primary,
        "tracks_are_never_pooled": True,
        "horizons_are_never_averaged": True,
        "tracks": track_outputs,
        "condition_trajectories": len(all_records),
        "legacy_history_utility_diagnostics": legacy,
    }
    return metrics, all_records, {"before": before, "after": after}, selection


def _speed_legacy_history_utility(
    *,
    payload: DevelopmentPayload,
    adapter: LatentWorldModelAdapter,
    tracks: dict[str, dict[tuple[str, float], dict[str, dict[str, Any]]]],
    selection: Mapping[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """The historical history-utility diagnostic, computed on the new data.

    Keeps the pre-1.0.1 Development fields (``history_better_rate`` and the
    per-speed context-free ablation) as additional diagnostics; they are not
    the primary metric of this protocol.
    """

    histories: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    speeds: list[float] = []
    for track in sorted(tracks):
        matching_by_speed = {
            speed: condition
            for condition, speed in selection["tracks"][track][
                "condition_speeds"
            ].items()
        }
        for (static_id, reference_speed), slot in sorted(tracks[track].items()):
            record = slot[matching_by_speed[reference_speed]]
            histories.append(record["frames"][:3])
            actions.append(
                record["actions"]
                .reshape(
                    _SPEED_EPISODE_ROWS // 5, 5, record["actions"].shape[-1]
                )[: _SPEED_EPISODE_ROWS // 5 - 1]
            )
            futures.append(record["frames"][3])
            speeds.append(float(reference_speed))
    history_pixels = np.stack(histories)
    action_blocks = np.stack(actions).astype(np.float32)
    context_free_histories = np.repeat(history_pixels[:, 2:3], repeats=3, axis=1)
    context_free_actions = np.zeros_like(action_blocks)
    context_free_actions[:, 2] = action_blocks[:, 2]
    predicted_history = np.asarray(
        adapter.rollout_latents(
            history_pixels, action_blocks, batch_size=int(batch_size)
        )
    )
    predicted_context_free = np.asarray(
        adapter.rollout_latents(
            context_free_histories,
            context_free_actions,
            batch_size=int(batch_size),
        )
    )
    target = np.asarray(
        adapter.encode_pixels(np.stack(futures), batch_size=int(batch_size))
    )
    if (
        predicted_history.ndim != 3
        or predicted_history.shape[0] != len(speeds)
        or predicted_context_free.shape != predicted_history.shape
        or target.shape != (len(speeds), predicted_history.shape[-1])
        or not np.isfinite(predicted_history).all()
        or not np.isfinite(predicted_context_free).all()
        or not np.isfinite(target).all()
    ):
        raise RuntimeError(
            "Speed Development legacy diagnostic received malformed latents"
        )
    history_mse = np.square(predicted_history[:, 0] - target).mean(axis=-1)
    context_free_mse = np.square(predicted_context_free[:, 0] - target).mean(axis=-1)
    improvement = context_free_mse - history_mse
    by_speed: dict[str, dict[str, Any]] = {}
    for speed in sorted(set(speeds)):
        indices = np.asarray(
            [index for index, value in enumerate(speeds) if value == speed],
            dtype=np.int64,
        )
        by_speed[f"{speed:g}"] = {
            "agent_speed": speed,
            "case_count": int(len(indices)),
            "history_mse_mean": float(history_mse[indices].mean()),
            "context_free_mse_mean": float(context_free_mse[indices].mean()),
            "context_free_minus_history_mse_mean": float(
                improvement[indices].mean()
            ),
            "history_better_rate": float((improvement[indices] > 0).mean()),
        }
    return {
        "diagnostic": "speed_history_utility_development_v1",
        "case_count": len(speeds),
        "history_mse_mean": float(history_mse.mean()),
        "context_free_mse_mean": float(context_free_mse.mean()),
        "context_free_minus_history_mse_mean": float(improvement.mean()),
        "history_better_rate": float((improvement > 0).mean()),
        "by_agent_speed": by_speed,
    }


def _bundle_identity(payload: DevelopmentPayload) -> dict[str, Any]:
    return {
        "bundle_schema_version": payload.component.get("schema_version", "ContextWorld-v1"),
        "manifest_sha256": payload.manifest_sha256,
        "task_registry_sha256": payload.task_registry_sha256,
        "component_id": payload.task,
        "dataset_id": payload.component.get("dataset_id"),
        "development_payload_id": payload.payload.get("payload_id"),
        "member_count": len(payload.members),
        "members": [str(path.relative_to(payload.root)) for path in payload.members],
    }


def evaluate_bundle_development_model(
    *,
    task: str,
    adapter: LatentWorldModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    benchmark_root: str | Path,
    batch_size: int = 64,
    include_records: bool = False,
) -> dict[str, Any]:
    """Evaluate a checkpoint on a public Development protocol only.

    The return value intentionally contains no gate or pass field.  It is
    development evidence for model selection/debugging, not a replacement for
    a held-out Public Test result.
    """

    if task not in {
        "speed",
        "door",
        "action_delay",
        *_SINGLE_TABLE_TASKS,
    }:
        raise ValueError(f"Unknown ContextWorld Development task: {task!r}")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    payload = resolve_development_payload(benchmark_root, task=task)
    if task == "speed":
        metrics, records, state, selection = _speed_structural_metrics(
            payload=payload, adapter=adapter, batch_size=int(batch_size)
        )
        protocol_kind = "matched_development_counterfactual"
        match_status = "matched_development_only"
    elif task == "door":
        metrics, records, state, selection = _door_structural_metrics(
            payload=payload, adapter=adapter, batch_size=int(batch_size)
        )
        protocol_kind = "matched_development_counterfactual"
        match_status = "matched_development_only"
    else:
        if task == "action_delay":
            arrays = _action_delay_arrays(payload)
            metrics, records, state = _action_delay_physical_group_metrics(
                payload=payload,
                arrays=arrays,
                adapter=adapter,
                batch_size=int(batch_size),
            )
        else:
            arrays = _single_table_arrays(payload)
            metrics, records, state = _evaluate_paired(
                payload=payload, arrays=arrays, adapter=adapter, batch_size=int(batch_size)
            )
        selection = arrays.selection
        protocol_kind = "matched_development_counterfactual"
        match_status = "matched_development_only"
    result: dict[str, Any] = {
        "schema_version": 1,
        "result_kind": DEVELOPMENT_RESULT_KIND,
        "status": "completed",
        "protocol": {
            "id": DEVELOPMENT_PROTOCOL_VERSION,
            "kind": protocol_kind,
            "match_status": match_status,
            "evaluation_split": "development",
            "public_test_accessed": False,
            "official_scoreboard_row": False,
            "formal_pass_available": False,
            "claim_boundary": (
                "This result uses only public ContextWorld-v1 Development "
                "data. It is not a held-out Public Test score and must not "
                "be reported as a formal pass or official scoreboard row."
            ),
        },
        "bundle": _bundle_identity(payload),
        "model": {
            "name": str(model_name),
            "training_recipe": str(training_recipe),
            "training_seed": None if training_seed is None else int(training_seed),
            "adapter": dict(adapter.metadata),
            "state_sha256_before": state["before"],
            "state_sha256_after": state["after"],
        },
        "selection": dict(selection),
        "metrics": metrics,
    }
    if include_records:
        result["records"] = records
    else:
        result["record_count"] = len(records)
    return result


__all__ = [
    "DEVELOPMENT_PROTOCOL_VERSION",
    "DEVELOPMENT_RESULT_KIND",
    "DevelopmentPayload",
    "development_action_normalization",
    "development_action_normalizer_path",
    "evaluate_bundle_development_model",
    "resolve_development_payload",
]
