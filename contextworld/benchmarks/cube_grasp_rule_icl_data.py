from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from contextworld.paths import (
    artifact_root,
    repository_root,
    resolve_contextworld_path,
)


CUBE_GRASP_RULE_RELEASE_ID = (
    "contextworld_cube_gripper_carry_icl_history3_v1"
)
DEFAULT_CUBE_GRASP_RULE_RELEASE_CONFIG = (
    repository_root()
    / "configs/benchmark/cube_gripper_carry_icl_release_v1.yaml"
)
CUBE_GRASP_RULE_MODES = ("cannot_hold", "can_hold")
CUBE_CAUSAL_STATE_TOLERANCE = 1e-12
CUBE_GRASP_RULE_PROTOCOL = "cube_gripper_carry_rule_history3_release_v1"
CUBE_GRASP_RULE_PAIR_COUNTS = {
    "train": 2048,
    "loader_validation": 256,
    "validation": 256,
}


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Cube release field {field} must be a mapping")
    return value


def _require_keys(
    value: dict[str, Any], *, field: str, keys: tuple[str, ...]
) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(
            f"Cube release field {field} is missing: {', '.join(missing)}"
        )


def _validate_path_specification(value: Any, *, field: str) -> None:
    specification = _mapping(value, field=field)
    if not any(
        specification.get(name)
        for name in (
            "environment_variable",
            "bundled_artifact_path",
            "local_source",
            "path",
        )
    ):
        raise ValueError(f"Cube release field {field} has no path source")


def _validate_cube_release_contract(payload: dict[str, Any]) -> None:
    runtime = _mapping(payload.get("runtime"), field="runtime")
    stable = _mapping(
        runtime.get("stable_worldmodel"), field="runtime.stable_worldmodel"
    )
    _require_keys(
        stable,
        field="runtime.stable_worldmodel",
        keys=("repo", "expected_ref"),
    )

    identity = _mapping(payload.get("identity"), field="identity")
    if not identity:
        raise ValueError("Cube release identity must not be empty")
    for name, value in identity.items():
        specification = _mapping(value, field=f"identity.{name}")
        _require_keys(
            specification,
            field=f"identity.{name}",
            keys=("path", "sha256"),
        )

    data = _mapping(payload.get("data"), field="data")
    _require_keys(
        data,
        field="data",
        keys=(
            "artifact_tree",
            "artifacts",
            "manifest_sha256",
            "pair_counts",
            "lance_tables",
        ),
    )
    tree = _mapping(data["artifact_tree"], field="data.artifact_tree")
    _require_keys(
        tree,
        field="data.artifact_tree",
        keys=("root", "files", "bytes", "sha256"),
    )
    artifacts = _mapping(data["artifacts"], field="data.artifacts")
    if not artifacts:
        raise ValueError("Cube release data.artifacts must not be empty")
    for name, value in artifacts.items():
        specification = _mapping(value, field=f"data.artifacts.{name}")
        _require_keys(
            specification,
            field=f"data.artifacts.{name}",
            keys=("path", "sha256"),
        )
    pair_counts = _mapping(data["pair_counts"], field="data.pair_counts")
    lance_tables = _mapping(data["lance_tables"], field="data.lance_tables")
    split_names = ("train", "loader_validation", "validation")
    _require_keys(pair_counts, field="data.pair_counts", keys=split_names)
    _require_keys(lance_tables, field="data.lance_tables", keys=split_names)
    if any(int(pair_counts[name]) <= 0 for name in split_names):
        raise ValueError("Cube release pair counts must be positive")
    if {
        name: int(pair_counts[name]) for name in split_names
    } != CUBE_GRASP_RULE_PAIR_COUNTS:
        raise ValueError("Cube release requires 2048/256/256 frozen pairs")
    if data.get("protocol") != CUBE_GRASP_RULE_PROTOCOL:
        raise ValueError("Cube release has an unexpected causal data protocol")

    training = _mapping(payload.get("training"), field="training")
    upstream = _mapping(training.get("upstream"), field="training.upstream")
    _require_keys(
        upstream,
        field="training.upstream",
        keys=("original_h5", "original_lance"),
    )
    for name in ("original_h5", "original_lance"):
        _validate_path_specification(
            upstream[name], field=f"training.upstream.{name}"
        )
    matrix = _mapping(
        training.get("reference_matrix"), field="training.reference_matrix"
    )
    _require_keys(
        matrix,
        field="training.reference_matrix",
        keys=("status", "training_seeds", "initial_checkpoints", "common"),
    )
    checkpoints = _mapping(
        matrix["initial_checkpoints"],
        field="training.reference_matrix.initial_checkpoints",
    )
    _require_keys(
        checkpoints,
        field="training.reference_matrix.initial_checkpoints",
        keys=("lewm", "pldm"),
    )
    for name in ("lewm", "pldm"):
        _validate_path_specification(
            checkpoints[name],
            field=f"training.reference_matrix.initial_checkpoints.{name}",
        )
    common = _mapping(
        matrix["common"], field="training.reference_matrix.common"
    )
    _require_keys(
        common,
        field="training.reference_matrix.common",
        keys=(
            "optimizer_steps",
            "fixed_checkpoint_step",
            "loader_validation_monitor_steps",
            "batch_size",
            "original_cube_samples_per_batch",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
        ),
    )
    optimizer_steps = int(common["optimizer_steps"])
    monitors = [int(value) for value in common["loader_validation_monitor_steps"]]
    if (
        optimizer_steps <= 0
        or int(common["fixed_checkpoint_step"]) != optimizer_steps
        or not monitors
        or monitors != sorted(set(monitors))
        or monitors[-1] != optimizer_steps
    ):
        raise ValueError("Cube release has an invalid fixed-step training plan")

    evaluation = _mapping(payload.get("evaluation"), field="evaluation")
    _require_keys(
        evaluation,
        field="evaluation",
        keys=("pair_count", "lance_table", "action_normalization"),
    )
    if int(evaluation["pair_count"]) != int(pair_counts["validation"]):
        raise ValueError("Cube Public Test pair count does not match validation")
    normalization = _mapping(
        evaluation["action_normalization"],
        field="evaluation.action_normalization",
    )
    _require_keys(
        normalization,
        field="evaluation.action_normalization",
        keys=("mean", "std_population"),
    )
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    std = np.asarray(normalization["std_population"], dtype=np.float64)
    if (
        mean.shape != (5,)
        or std.shape != (5,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("Cube action normalization must contain five finite axes")

    scoring = _mapping(payload.get("scoring"), field="scoring")
    prediction = _mapping(
        scoring.get("hidden_future_prediction"),
        field="scoring.hidden_future_prediction",
    )
    gates = _mapping(
        prediction.get("gates"),
        field="scoring.hidden_future_prediction.gates",
    )
    _require_keys(
        gates,
        field="scoring.hidden_future_prediction.gates",
        keys=(
            "correct_future_rate_minimum",
            "correct_history_rate_minimum",
            "context_switch_rate_minimum",
            "worst_rule_correct_future_rate_minimum",
            "target_latent_separation_required",
            "response_gain_minimum",
            "normalized_response_error_strict_maximum",
        ),
    )
    method = _mapping(scoring.get("method_level"), field="scoring.method_level")
    retention = _mapping(
        scoring.get("original_task_retention"),
        field="scoring.original_task_retention",
    )
    _require_keys(
        method,
        field="scoring.method_level",
        keys=("training_seeds_required",),
    )
    _require_keys(
        retention,
        field="scoring.original_task_retention",
        keys=("status",),
    )
    seeds = matrix["training_seeds"]
    required_seeds = int(method["training_seeds_required"])
    if (
        not isinstance(seeds, list)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != required_seeds
        or len(set(seeds)) != required_seeds
    ):
        raise ValueError("Cube release requires distinct integer training seeds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_sha256(child).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def load_cube_grasp_rule_icl_release(
    path: Path | str = DEFAULT_CUBE_GRASP_RULE_RELEASE_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported Cube Grasp Rule release: {config_path}")
    if payload.get("release_id") != CUBE_GRASP_RULE_RELEASE_ID:
        raise ValueError(f"Unexpected Cube Grasp Rule release id: {config_path}")
    if payload.get("release_status") not in {
        "data_ready_training_in_progress",
        "data_ready_reference_failed_development",
        "public_test_release_candidate",
        "public_test_release",
    }:
        raise ValueError("Unsupported Cube Grasp Rule release status")
    scope = payload.get("scope", {})
    if scope.get("history_tokens") != 3:
        raise ValueError("Cube Grasp Rule v1 requires History=3")
    if scope.get("public_test_included") is not True:
        raise ValueError("Cube Grasp Rule v1 must include Public Test")
    if scope.get("sealed_test_included") is not False:
        raise ValueError("Cube Grasp Rule v1 must not include sealed Test")
    if tuple(scope.get("grasp_modes", ())) != CUBE_GRASP_RULE_MODES:
        raise ValueError("Cube Grasp Rule v1 has unexpected rule modes")
    _validate_cube_release_contract(payload)
    return {**payload, "_config_path": str(config_path)}


def _resolve_upstream(
    specification: dict[str, Any], *, repo_root: Path
) -> Path:
    environment = str(specification.get("environment_variable", ""))
    configured = os.environ.get(environment) if environment else None
    if configured:
        return Path(configured).expanduser().resolve()
    bundled = specification.get("bundled_artifact_path")
    if bundled:
        candidate = artifact_root(repo_root) / str(bundled)
        if candidate.exists():
            return candidate.resolve()
    source = specification.get("local_source") or specification.get("path")
    if not source:
        raise ValueError("Cube upstream input has no resolvable source")
    return Path(source).expanduser().resolve()


def resolve_cube_original_h5(
    release: dict[str, Any], *, repo_root: Path
) -> Path:
    return _resolve_upstream(
        release["training"]["upstream"]["original_h5"], repo_root=repo_root
    )


def resolve_cube_original_lance(
    release: dict[str, Any], *, repo_root: Path
) -> Path:
    return _resolve_upstream(
        release["training"]["upstream"]["original_lance"], repo_root=repo_root
    )


def resolve_cube_initial_checkpoint(
    release: dict[str, Any], family: str, *, repo_root: Path
) -> Path:
    if family not in {"lewm", "pldm"}:
        raise ValueError("Cube checkpoint family must be 'lewm' or 'pldm'")
    return _resolve_upstream(
        release["training"]["reference_matrix"]["initial_checkpoints"][family],
        repo_root=repo_root,
    )


@dataclass(frozen=True)
class CubeGraspRuleEvalArrays:
    pair_ids: tuple[str, ...]
    cannot_hold_pixels: np.ndarray
    can_hold_pixels: np.ndarray
    raw_action_blocks: np.ndarray
    cannot_hold_states: np.ndarray
    can_hold_states: np.ndarray

    @property
    def pair_count(self) -> int:
        return len(self.pair_ids)

    @property
    def low_pixels(self) -> np.ndarray:
        return self.cannot_hold_pixels

    @property
    def high_pixels(self) -> np.ndarray:
        return self.can_hold_pixels

    @property
    def low_states(self) -> np.ndarray:
        return self.cannot_hold_states

    @property
    def high_states(self) -> np.ndarray:
        return self.can_hold_states


def _decode(value: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(BytesIO(value)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_lance_pairs(
    path: Path, *, expected_pairs: int, expected_split: str
) -> CubeGraspRuleEvalArrays:
    import lance

    table = lance.dataset(path).to_table(
        columns=[
            "episode_idx",
            "model_step_idx",
            "pixels",
            "action_block",
            "physical_state",
            "hidden_grasp_enabled",
            "pair_id",
            "hidden_mode",
            "split",
        ]
    )
    episode_indices = np.asarray(table["episode_idx"].to_numpy(), dtype=np.int64)
    step_indices = np.asarray(table["model_step_idx"].to_numpy(), dtype=np.int64)
    pixel_bytes = table["pixels"].to_pylist()
    actions = np.asarray(table["action_block"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["physical_state"].to_pylist(), dtype=np.float32)
    hidden = np.asarray(
        table["hidden_grasp_enabled"].to_pylist(), dtype=np.float32
    ).reshape(-1)
    pair_ids = table["pair_id"].to_pylist()
    modes = table["hidden_mode"].to_pylist()
    splits = table["split"].to_pylist()

    episodes: dict[str, dict[str, tuple[np.ndarray, ...]]] = {}
    unique_episodes = np.unique(episode_indices)
    if len(unique_episodes) != 2 * expected_pairs:
        raise RuntimeError(
            f"Expected {2 * expected_pairs} Cube episodes, got {len(unique_episodes)}"
        )
    expected_hidden = {"cannot_hold": 0.0, "can_hold": 1.0}
    for episode in unique_episodes:
        rows = np.flatnonzero(episode_indices == episode)
        rows = rows[np.argsort(step_indices[rows])]
        if not np.array_equal(step_indices[rows], np.arange(4)):
            raise RuntimeError(f"Cube episode {episode} is not a four-frame clip")
        pair_values = {str(pair_ids[index]) for index in rows}
        mode_values = {str(modes[index]) for index in rows}
        if len(pair_values) != 1 or len(mode_values) != 1:
            raise RuntimeError(f"Cube episode {episode} changes pair or mode")
        pair_id = pair_values.pop()
        mode = mode_values.pop()
        if mode not in CUBE_GRASP_RULE_MODES:
            raise RuntimeError(f"Unexpected Cube grasp mode {mode!r}")
        if {str(splits[index]) for index in rows} != {expected_split}:
            raise RuntimeError(f"Unexpected split for {pair_id}")
        if {float(hidden[index]) for index in rows} != {expected_hidden[mode]}:
            raise RuntimeError(f"Unexpected hidden value for {pair_id}/{mode}")
        if mode in episodes.setdefault(pair_id, {}):
            raise RuntimeError(f"Duplicate {mode} episode for {pair_id}")
        episodes[pair_id][mode] = (
            np.stack([_decode(pixel_bytes[index]) for index in rows]),
            actions[rows].reshape(4, 5, 5),
            states[rows],
        )

    if len(episodes) != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} Cube pairs, got {len(episodes)}")
    ordered_ids = tuple(sorted(episodes))
    low_pixels, high_pixels, action_blocks = [], [], []
    low_states, high_states = [], []
    for pair_id in ordered_ids:
        pair = episodes[pair_id]
        if set(pair) != set(CUBE_GRASP_RULE_MODES):
            raise RuntimeError(f"Incomplete Cube grasp pair {pair_id}")
        low, high = (pair[mode] for mode in CUBE_GRASP_RULE_MODES)
        if not np.array_equal(low[0][0], high[0][0]):
            raise RuntimeError(f"Initial frame differs for {pair_id}")
        if not np.array_equal(low[0][2], high[0][2]):
            raise RuntimeError(f"Current query frame differs for {pair_id}")
        if not np.array_equal(low[1], high[1]):
            raise RuntimeError(f"Actions differ for {pair_id}")
        if np.array_equal(low[0][1], high[0][1]):
            raise RuntimeError(f"History does not reveal grasp rule for {pair_id}")
        if np.array_equal(low[0][3], high[0][3]):
            raise RuntimeError(f"True futures do not differ for {pair_id}")
        if not np.array_equal(low[2][2], high[2][2]):
            raise RuntimeError(f"Query physical state differs for {pair_id}")
        low_pixels.append(low[0])
        high_pixels.append(high[0])
        action_blocks.append(low[1])
        low_states.append(low[2])
        high_states.append(high[2])
    return CubeGraspRuleEvalArrays(
        pair_ids=ordered_ids,
        cannot_hold_pixels=np.stack(low_pixels),
        can_hold_pixels=np.stack(high_pixels),
        raw_action_blocks=np.stack(action_blocks),
        cannot_hold_states=np.stack(low_states),
        can_hold_states=np.stack(high_states),
    )


class CubeGraspRuleICLEvalDataset:
    """Frozen 256-pair Public Test for the Cube grasp rule."""

    def __init__(
        self,
        *,
        release: dict[str, Any] | None = None,
        release_config: Path | str = DEFAULT_CUBE_GRASP_RULE_RELEASE_CONFIG,
        repo_root: Path | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.release = release or load_cube_grasp_rule_icl_release(release_config)
        self.root = resolve_contextworld_path(
            self.release["data"]["artifact_tree"]["root"],
            repo_root=self.repo_root,
        )
        self._arrays: CubeGraspRuleEvalArrays | None = None

    @property
    def arrays(self) -> CubeGraspRuleEvalArrays:
        if self._arrays is None:
            evaluation = self.release["evaluation"]
            self._arrays = _read_lance_pairs(
                self.root / evaluation["lance_table"],
                expected_pairs=int(evaluation["pair_count"]),
                expected_split="validation",
            )
        return self._arrays

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "pair_count": self.arrays.pair_count,
            "condition_count": 2 * self.arrays.pair_count,
            "history_tokens": 3,
            "grasp_modes": list(CUBE_GRASP_RULE_MODES),
            "online_environment_calls": 0,
        }


def audit_cube_grasp_rule_icl_release(
    *,
    release_config: Path | str = DEFAULT_CUBE_GRASP_RULE_RELEASE_CONFIG,
    repo_root: Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_cube_grasp_rule_icl_release(release_config)
    data_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"], repo_root=root
    )
    files: dict[str, Any] = {}
    for name, entry in release.get("identity", {}).items():
        path = resolve_contextworld_path(entry["path"], repo_root=root)
        observed = file_sha256(path) if path.is_file() else None
        files[f"identity.{name}"] = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": entry["sha256"],
            "observed_sha256": observed,
            "passed": path.is_file() and observed == entry["sha256"],
        }
    for name, entry in release["data"].get("artifacts", {}).items():
        path = resolve_contextworld_path(entry["path"], repo_root=root)
        observed = file_sha256(path) if path.is_file() else None
        files[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": entry["sha256"],
            "observed_sha256": observed,
            "passed": path.is_file() and observed == entry["sha256"],
        }
    for name, entry in release.get("reference_results", {}).items():
        path = resolve_contextworld_path(entry["path"], repo_root=root)
        observed = file_sha256(path) if path.is_file() else None
        files[f"reference_results.{name}"] = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": entry["sha256"],
            "observed_sha256": observed,
            "passed": path.is_file() and observed == entry["sha256"],
        }
    tree = release["data"]["artifact_tree"]
    observed_files = [value for value in data_root.rglob("*") if value.is_file()]
    tree_result = {
        "path": str(data_root),
        "expected_files": int(tree["files"]),
        "observed_files": len(observed_files),
        "expected_bytes": int(tree["bytes"]),
        "observed_bytes": sum(path.stat().st_size for path in observed_files),
        "expected_sha256": tree["sha256"],
        "observed_sha256": directory_sha256(data_root) if full else None,
        "full_hash_checked": bool(full),
    }
    tree_result["passed"] = (
        tree_result["expected_files"] == tree_result["observed_files"]
        and tree_result["expected_bytes"] == tree_result["observed_bytes"]
        and (
            not full
            or tree_result["expected_sha256"] == tree_result["observed_sha256"]
        )
    )
    try:
        public = CubeGraspRuleICLEvalDataset(release=release, repo_root=root)
        public_result = public.describe()
        public_result["passed"] = public.arrays.pair_count == int(
            release["evaluation"]["pair_count"]
        )
    except Exception as error:  # report incomplete local releases structurally
        public_result = {
            "root": str(data_root),
            "pair_count": None,
            "condition_count": None,
            "history_tokens": 3,
            "grasp_modes": list(CUBE_GRASP_RULE_MODES),
            "online_environment_calls": 0,
            "error": f"{type(error).__name__}: {error}",
            "passed": False,
        }
    build_path = data_root / "build_report.json"
    build_error = None
    try:
        build_payload = json.loads(build_path.read_text(encoding="utf-8"))
        if not isinstance(build_payload, dict):
            raise ValueError("Cube build report must be a mapping")
    except Exception as error:  # missing/partial builds are an audit result
        build_payload = {}
        build_error = f"{type(error).__name__}: {error}"
    splits = build_payload.get("splits", {})
    expected_splits = {"train", "loader_validation", "validation"}
    complete_splits = isinstance(splits, dict) and set(splits) == expected_splits
    overlaps = build_payload.get("cross_split_overlap", {})
    data_checks = {
        "build_passed": build_payload.get("passed") is True,
        "all_three_splits_present": complete_splits,
        "cross_split_overlap_zero": bool(overlaps)
        and not any(overlaps.values()),
        "shared_query_reached_without_state_installation": complete_splits
        and all(
            split.get("maximum_query_simulator_state_gap", float("inf"))
            <= CUBE_CAUSAL_STATE_TOLERANCE
            and split.get("maximum_prequery_object_state_residual", float("inf"))
            <= CUBE_CAUSAL_STATE_TOLERANCE
            and split.get("maximum_state_installations_after_x0") == 0
            and split.get("all_causal_checks_passed") is True
            for split in splits.values()
        ),
        "common_causal_contract_passed": build_payload.get(
            "causal_data_contract", {}
        ).get("passed")
        is True,
    }
    matrix_status = release["training"]["reference_matrix"]["status"]
    retention_status = release["scoring"]["original_task_retention"]["status"]
    development_stopped = matrix_status == "failed_development"
    terminal_negative = (
        release["release_status"] == "data_ready_reference_failed_development"
        and development_stopped
    )
    release_checks = {
        "release_stage_has_terminal_decision": terminal_negative
        or release["release_status"]
        in {"public_test_release_candidate", "public_test_release"},
        "reference_matrix_has_terminal_decision": matrix_status
        in {
            "completed",
            "completed_failed_prediction_gate",
            "failed_development",
        },
        "original_task_retention_respects_decision": retention_status
        == "completed"
        or (
            development_stopped
            and retention_status == "not_run_after_failed_development"
        ),
    }
    data_ready = (
        all(row["passed"] for row in files.values())
        and tree_result["passed"]
        and public_result["passed"]
        and all(data_checks.values())
    )
    passed = data_ready and all(release_checks.values())
    if passed:
        status = "passed"
    elif data_ready and release["release_status"] == "data_ready_training_in_progress":
        status = "data_ready_training_in_progress"
    else:
        status = "failed"
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "full": bool(full),
        "files": files,
        "artifact_tree": tree_result,
        "causal_data_contract": build_payload.get(
            "causal_data_contract", {}
        ),
        "build_report": {
            "path": str(build_path),
            "exists": build_path.is_file(),
            "error": build_error,
        },
        "data_checks": data_checks,
        "release_checks": release_checks,
        "public_test": public_result,
        "status": status,
        "data_ready": data_ready,
        "passed": passed,
    }


__all__ = [
    "CUBE_GRASP_RULE_MODES",
    "CUBE_GRASP_RULE_PAIR_COUNTS",
    "CUBE_GRASP_RULE_PROTOCOL",
    "CUBE_GRASP_RULE_RELEASE_ID",
    "DEFAULT_CUBE_GRASP_RULE_RELEASE_CONFIG",
    "CubeGraspRuleEvalArrays",
    "CubeGraspRuleICLEvalDataset",
    "_read_lance_pairs",
    "audit_cube_grasp_rule_icl_release",
    "directory_sha256",
    "file_sha256",
    "load_cube_grasp_rule_icl_release",
    "resolve_cube_initial_checkpoint",
    "resolve_cube_original_h5",
    "resolve_cube_original_lance",
]
