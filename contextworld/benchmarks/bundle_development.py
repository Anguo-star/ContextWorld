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
_DOOR_MEMBER_PATTERN = re.compile(
    r"^hp-val-(?P<door>d\d+)-(?P<mode>blocked|passable)-"
)
_ACTION_DELAY_MEMBER_PATTERN = re.compile(
    r"^ad-h7-paired-val-(?P<profile>p\d+)-d(?P<delay>\d+)-"
)


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
) -> tuple[tuple[int, ...], dict[int, tuple[np.ndarray, np.ndarray, float | None]]]:
    columns = ["episode_idx", "step_idx", "pixels", "action"]
    if include_speed:
        columns.append("variation_agent_speed")
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
    if expected_steps % 5:
        raise RuntimeError("Development TwoRoom clips must be divisible into 5-step actions")
    result: dict[int, tuple[np.ndarray, np.ndarray, float | None]] = {}
    for episode, rows in rows_by_episode.items():
        speed: float | None = None
        if speeds is not None:
            speed_values = speeds[rows]
            if not np.allclose(speed_values, speed_values[0], atol=0.0, rtol=0.0):
                raise RuntimeError(
                    f"Development speed changes within episode {episode}"
                )
            speed = float(speed_values[0])
        result[episode] = (
            np.stack([_decode_rgb(pixels[int(rows[index])]) for index in frame_steps]),
            actions[rows].reshape(expected_steps // 5, 5, actions.shape[-1]),
            speed,
        )
    return available, result


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


def _door_arrays(payload: DevelopmentPayload) -> _PairedArrays:
    grouped: dict[str, dict[str, Path]] = {}
    for path in payload.members:
        match = _DOOR_MEMBER_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected Door Development member: {path.name}")
        group = grouped.setdefault(match["door"], {})
        mode = match["mode"]
        if mode in group:
            raise ValueError(f"Duplicate Door Development member for {match['door']}/{mode}")
        group[mode] = path
    if not grouped or any(set(value) != {"blocked", "passable"} for value in grouped.values()):
        raise ValueError("Door Development payload does not contain blocked/passable pairs")
    per_group = _selection_value(
        payload, "complete_episodes_per_position", 18
    )
    expected_groups = _selection_value(payload, "door_positions", 16)
    expected_selected = _selection_value(
        payload, "selected_pair_count", expected_groups * per_group
    )
    if per_group <= 0:
        raise ValueError("Door Development pairs_per_group must be positive")
    pair_ids: list[str] = []
    first_pixels: list[np.ndarray] = []
    second_pixels: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    candidate_counts: list[int] = []
    for door in sorted(grouped):
        blocked_ids, blocked = _read_tworoom_episodes(
            grouped[door]["blocked"],
            expected_steps=20,
            frame_steps=(0, 5, 10, 15),
        )
        passable_ids, passable = _read_tworoom_episodes(
            grouped[door]["passable"],
            expected_steps=20,
            frame_steps=(0, 5, 10, 15),
        )
        if blocked_ids != passable_ids:
            raise RuntimeError(f"Door Development episode ids differ for {door}")
        candidate_counts.append(len(blocked_ids))
        if len(blocked_ids) < per_group:
            raise RuntimeError(f"Door Development {door} has fewer than {per_group} episodes")
        for episode in blocked_ids[:per_group]:
            blocked_pixels, blocked_actions, _ = blocked[episode]
            passable_pixels, passable_actions, _ = passable[episode]
            pair_id = f"{door}/episode_{episode:04d}"
            _ensure_paired_example(
                pair_id=pair_id,
                first_pixels=blocked_pixels,
                second_pixels=passable_pixels,
                first_actions=blocked_actions,
                second_actions=passable_actions,
                history_length=3,
            )
            pair_ids.append(pair_id)
            first_pixels.append(blocked_pixels)
            second_pixels.append(passable_pixels)
            actions.append(blocked_actions)
    if len(grouped) != expected_groups:
        raise RuntimeError(
            "Door Development filename groups disagree with public contract: "
            f"observed={len(grouped)} expected={expected_groups}"
        )
    if len(pair_ids) != expected_selected:
        raise RuntimeError(
            "Door Development selected pair count disagrees with public contract: "
            f"observed={len(pair_ids)} expected={expected_selected}"
        )
    return _PairedArrays(
        pair_ids=tuple(pair_ids),
        first_pixels=np.stack(first_pixels),
        second_pixels=np.stack(second_pixels),
        raw_action_blocks=np.stack(actions),
        first_label="blocked",
        second_label="passable",
        selection={
            "kind": "matched_filename_group_and_episode_id",
            "groups": len(grouped),
            "candidate_pairs": int(sum(candidate_counts)),
            "selected_pairs": len(pair_ids),
            "pairs_per_group": per_group,
            "rule": "sorted door id; first sorted shared episode ids",
        },
    )


def _action_delay_arrays(payload: DevelopmentPayload) -> _PairedArrays:
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
    pair_ids: list[str] = []
    first_pixels: list[np.ndarray] = []
    second_pixels: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    candidate_pairs = 0
    for profile in sorted(grouped):
        baseline_ids, baseline = _read_tworoom_episodes(
            grouped[profile][reference_delay],
            expected_steps=50,
            frame_steps=(0, 5, 10, 15, 20, 25, 30, 35),
        )
        if len(baseline_ids) < per_contrast:
            raise RuntimeError(
                f"Action Delay Development {profile} has fewer than "
                f"{per_contrast} episodes"
            )
        selected_ids = baseline_ids[:per_contrast]
        for delay in contrasts:
            delayed_ids, delayed = _read_tworoom_episodes(
                grouped[profile][delay],
                expected_steps=50,
                frame_steps=(0, 5, 10, 15, 20, 25, 30, 35),
                selected_episode_ids=selected_ids,
            )
            # ``delayed_ids`` is the complete id set even when frame decoding
            # is limited to selected ids, so it also checks file pairing.
            if delayed_ids != baseline_ids:
                raise RuntimeError(
                    f"Action Delay Development episode ids differ for "
                    f"{profile}/d{delay}"
                )
            candidate_pairs += len(baseline_ids)
            for episode in selected_ids:
                base_pixels, base_actions, _ = baseline[episode]
                delayed_pixels, delayed_actions, _ = delayed[episode]
                pair_id = (
                    f"{profile}/d{reference_delay}_vs_d{delay}/"
                    f"episode_{episode:04d}"
                )
                _ensure_paired_example(
                    pair_id=pair_id,
                    first_pixels=base_pixels,
                    second_pixels=delayed_pixels,
                    first_actions=base_actions,
                    second_actions=delayed_actions,
                    history_length=7,
                )
                pair_ids.append(pair_id)
                first_pixels.append(base_pixels)
                second_pixels.append(delayed_pixels)
                actions.append(base_actions)
    if len(grouped) != expected_profiles:
        raise RuntimeError(
            "Action Delay Development profile count disagrees with public contract: "
            f"observed={len(grouped)} expected={expected_profiles}"
        )
    if len(pair_ids) != expected_selected:
        raise RuntimeError(
            "Action Delay Development selected pair count disagrees with public contract: "
            f"observed={len(pair_ids)} expected={expected_selected}"
        )
    return _PairedArrays(
        pair_ids=tuple(pair_ids),
        first_pixels=np.stack(first_pixels),
        second_pixels=np.stack(second_pixels),
        raw_action_blocks=np.stack(actions),
        first_label="delay_0",
        second_label="delayed",
        selection={
            "kind": "matched_filename_profile_delay_and_episode_id",
            "profiles": len(grouped),
            "delay_values": list(expected_delays),
            "candidate_pairs": candidate_pairs,
            "selected_pairs": len(pair_ids),
            "pairs_per_profile_delay_contrast": per_contrast,
            "rule": (
                f"d{reference_delay} vs each listed contrast; first sorted "
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


def _speed_cases(
    payload: DevelopmentPayload) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    per_member = _selection_value(payload, "complete_windows_per_member", 3)
    expected_members = _selection_value(payload, "member_count", 96)
    expected_selected = _selection_value(
        payload, "selected_case_count", expected_members * per_member
    )
    if per_member <= 0:
        raise ValueError("Speed Development windows_per_member must be positive")
    histories: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    candidate_cases = 0
    for path in payload.members:
        available, episodes = _read_tworoom_episodes(
            path,
            expected_steps=20,
            frame_steps=(0, 5, 10, 15),
            include_speed=True,
            allow_prefix_clip=True,
        )
        candidate_cases += len(available)
        if len(available) < per_member:
            raise RuntimeError(
                f"Speed Development member {path.name} has fewer than "
                f"{per_member} complete 20-step windows"
            )
        for episode in available[:per_member]:
            pixels, action_blocks, speed = episodes[episode]
            if speed is None:
                raise RuntimeError("Speed Development member lacks agent speed metadata")
            histories.append(pixels[:3])
            actions.append(action_blocks[:3])
            futures.append(pixels[3])
            metadata.append(
                {
                    "member": str(path.relative_to(payload.root)),
                    "episode_id": int(episode),
                    "agent_speed": float(speed),
                }
            )
    if len(payload.members) != expected_members:
        raise RuntimeError(
            "Speed Development member count disagrees with public contract: "
            f"observed={len(payload.members)} expected={expected_members}"
        )
    if len(metadata) != expected_selected:
        raise RuntimeError(
            "Speed Development selected case count disagrees with public contract: "
            f"observed={len(metadata)} expected={expected_selected}"
        )
    return (
        np.stack(histories),
        np.stack(actions).astype(np.float32),
        np.stack(futures),
        metadata,
        {
            "kind": "per_member_history_utility_sample",
            "members": len(payload.members),
            "candidate_cases": candidate_cases,
            "selected_cases": len(metadata),
            "windows_per_member": per_member,
            "rule": "first sorted complete 20-step prefix windows from each registered member",
        },
    )


def _speed_history_utility(
    *,
    payload: DevelopmentPayload,
    adapter: LatentWorldModelAdapter,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str], Mapping[str, Any]]:
    validate_adapter_protocol(
        adapter,
        history_tokens=3,
        action_block_raw_steps=payload.frameskip,
        action_dim=payload.action_dimension,
        minimum_future_action_blocks=1,
        task_name="Speed Development history utility",
    )
    histories, actions, futures, cases, selection = _speed_cases(payload)
    context_free_histories = np.repeat(histories[:, 2:3], repeats=3, axis=1)
    context_free_actions = np.zeros_like(actions)
    context_free_actions[:, 2] = actions[:, 2]
    before = adapter.frozen_state_hash()
    predicted_history = np.asarray(
        adapter.rollout_latents(histories, actions, batch_size=int(batch_size))
    )
    predicted_context_free = np.asarray(
        adapter.rollout_latents(
            context_free_histories, context_free_actions, batch_size=int(batch_size)
        )
    )
    count = len(cases)
    if (
        predicted_history.shape[:2] != (count, 1)
        or predicted_context_free.shape != predicted_history.shape
        or not np.isfinite(predicted_history).all()
        or not np.isfinite(predicted_context_free).all()
    ):
        raise RuntimeError(
            "Speed Development adapter must return finite one-step predictions "
            "for both context conditions"
        )
    target = np.asarray(adapter.encode_pixels(futures, batch_size=int(batch_size)))
    if target.shape != (count, predicted_history.shape[-1]) or not np.isfinite(target).all():
        raise RuntimeError("Speed Development target encodings do not match predicted latents")
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during Speed Development scoring")
    history_mse = np.square(predicted_history[:, 0] - target).mean(axis=-1)
    context_free_mse = np.square(predicted_context_free[:, 0] - target).mean(axis=-1)
    improvement = context_free_mse - history_mse
    by_speed: dict[str, dict[str, Any]] = {}
    for speed in sorted({float(row["agent_speed"]) for row in cases}):
        indices = np.asarray(
            [index for index, row in enumerate(cases) if float(row["agent_speed"]) == speed],
            dtype=np.int64,
        )
        by_speed[f"{speed:g}"] = {
            "agent_speed": speed,
            "case_count": int(len(indices)),
            "history_mse_mean": float(history_mse[indices].mean()),
            "context_free_mse_mean": float(context_free_mse[indices].mean()),
            "context_free_minus_history_mse_mean": float(improvement[indices].mean()),
            "history_better_rate": float((improvement[indices] > 0).mean()),
        }
    metrics = {
        "diagnostic": "speed_history_utility_development_v1",
        "case_count": count,
        "history_mse_mean": float(history_mse.mean()),
        "context_free_mse_mean": float(context_free_mse.mean()),
        "context_free_minus_history_mse_mean": float(improvement.mean()),
        "history_better_rate": float((improvement > 0).mean()),
        "by_agent_speed": by_speed,
    }
    records = [
        {
            **row,
            "history_mse": float(history_mse[index]),
            "context_free_mse": float(context_free_mse[index]),
            "context_free_minus_history_mse": float(improvement[index]),
            "history_better": bool(improvement[index] > 0),
        }
        for index, row in enumerate(cases)
    ]
    return metrics, records, {"before": before, "after": after}, selection


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
        metrics, records, state, selection = _speed_history_utility(
            payload=payload, adapter=adapter, batch_size=int(batch_size)
        )
        protocol_kind = "history_utility_diagnostic"
        match_status = "not_matched_counterfactual"
    else:
        arrays = (
            _single_table_arrays(payload)
            if task in _SINGLE_TABLE_TASKS
            else _door_arrays(payload)
            if task == "door"
            else _action_delay_arrays(payload)
        )
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
