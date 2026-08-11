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

from contextworld.benchmarks.causal_data_contract import (
    audit_causal_data_contract,
)
from contextworld.paths import (
    artifact_root,
    repository_root,
    resolve_contextworld_path,
)


REACHER_ARM_MASS_RELEASE_ID = (
    "contextworld_reacher_arm_mass_icl_history3_v1"
)
DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG = (
    repository_root()
    / "configs/benchmark/reacher_arm_mass_icl_release_v1.yaml"
)
REACHER_ARM_MASS_MODES = ("lighter", "heavier")


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


def load_reacher_arm_mass_icl_release(
    path: Path | str = DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported Reacher Arm Mass release: {config_path}"
        )
    if payload.get("release_id") != REACHER_ARM_MASS_RELEASE_ID:
        raise ValueError(
            f"Unexpected Reacher Arm Mass release id: {config_path}"
        )
    if payload.get("release_status") not in {
            "data_ready_training_in_progress",
            "public_test_release_candidate",
            "public_test_release",
            "data_ready_reference_requires_latent_response_rescore",
        }:
        raise ValueError("Unsupported Reacher Arm Mass release status")
    scope = payload.get("scope", {})
    if scope.get("history_tokens") != 3:
        raise ValueError("Reacher Arm Mass v1 requires History=3")
    if scope.get("public_test_included") is not True:
        raise ValueError("Reacher Arm Mass v1 must include Public Test")
    if scope.get("arm_density_values") != [500.0, 1500.0]:
        raise ValueError(
            "Reacher Arm Mass v1 requires densities [500.0, 1500.0]"
        )
    return {**payload, "_config_path": str(config_path)}


def _resolve_upstream(
    specification: dict[str, Any],
    *,
    repo_root: Path,
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
    symbol = specification.get("source_symbol", "unspecified_upstream")
    raise ValueError(
        f"Upstream input {symbol!r} is not installed; set "
        f"{environment!r} or provide the bundled artifact"
    )


def resolve_reacher_original_h5(
    release: dict[str, Any],
    *,
    repo_root: Path,
) -> Path:
    return _resolve_upstream(
        release["training"]["upstream"]["original_h5"],
        repo_root=repo_root,
    )


def resolve_reacher_original_lance(
    release: dict[str, Any],
    *,
    repo_root: Path,
) -> Path:
    return _resolve_upstream(
        release["training"]["upstream"]["original_lance"],
        repo_root=repo_root,
    )


def resolve_reacher_initial_checkpoint(
    release: dict[str, Any],
    family: str,
    *,
    repo_root: Path,
) -> Path:
    if family not in {"lewm", "pldm"}:
        raise ValueError("Reacher checkpoint family must be 'lewm' or 'pldm'")
    return _resolve_upstream(
        release["training"]["reference_matrix"]["initial_checkpoints"][family],
        repo_root=repo_root,
    )


def resolve_reacher_initial_checkpoint_config(
    release: dict[str, Any],
    family: str,
    *,
    repo_root: Path,
) -> Path:
    if family not in {"lewm", "pldm"}:
        raise ValueError("Reacher checkpoint family must be 'lewm' or 'pldm'")
    specification = release["training"]["reference_matrix"][
        "initial_checkpoints"
    ][family]
    bundled = artifact_root(repo_root) / specification[
        "config_bundled_artifact_path"
    ]
    if bundled.is_file():
        return bundled.resolve()
    portable_source = specification.get("config_portable_source")
    if portable_source:
        candidate = resolve_contextworld_path(
            portable_source,
            repo_root=repo_root,
        )
        if candidate.is_file():
            return candidate.resolve()
    checkpoint = resolve_reacher_initial_checkpoint(
        release,
        family,
        repo_root=repo_root,
    )
    relative = specification.get("config_relative_to_checkpoint")
    if relative:
        candidate = checkpoint.parent / str(relative)
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(
        f"Configuration for Reacher {family!r} initialization is not "
        "installed beside the checkpoint or in the bundled artifact root"
    )


@dataclass(frozen=True)
class ReacherArmMassEvalArrays:
    pair_ids: tuple[str, ...]
    lighter_pixels: np.ndarray
    heavier_pixels: np.ndarray
    raw_action_blocks: np.ndarray
    lighter_states: np.ndarray
    heavier_states: np.ndarray
    lighter_finger_positions: np.ndarray
    heavier_finger_positions: np.ndarray

    @property
    def pair_count(self) -> int:
        return len(self.pair_ids)

    # The generic paired-training runner consumes these compatibility names.
    @property
    def low_pixels(self) -> np.ndarray:
        return self.lighter_pixels

    @property
    def high_pixels(self) -> np.ndarray:
        return self.heavier_pixels

    @property
    def low_states(self) -> np.ndarray:
        return self.lighter_states

    @property
    def high_states(self) -> np.ndarray:
        return self.heavier_states


def _decode(value: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Reading Reacher Arm Mass pixels requires Pillow"
        ) from exc
    with Image.open(BytesIO(value)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_lance_pairs(
    path: Path,
    *,
    expected_pairs: int,
    expected_split: str,
) -> ReacherArmMassEvalArrays:
    try:
        import lance
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Reading Reacher Arm Mass data requires the Python lance package"
        ) from exc
    table = lance.dataset(path).to_table(
        columns=[
            "episode_idx",
            "step_idx",
            "pixels",
            "action",
            "proprio",
            "finger_pos",
            "hidden_arm_density",
            "pair_id",
            "hidden_mode",
            "split",
        ]
    )
    episode_indices = np.asarray(
        table["episode_idx"].to_numpy(), dtype=np.int64
    )
    step_indices = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
    pixel_bytes = table["pixels"].to_pylist()
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["proprio"].to_pylist(), dtype=np.float32)
    fingers = np.asarray(table["finger_pos"].to_pylist(), dtype=np.float32)
    densities = np.asarray(
        table["hidden_arm_density"].to_pylist(), dtype=np.float32
    ).reshape(-1)
    pair_ids = table["pair_id"].to_pylist()
    modes = table["hidden_mode"].to_pylist()
    splits = table["split"].to_pylist()

    episodes: dict[str, dict[str, tuple[np.ndarray, ...]]] = {}
    unique_episodes = np.unique(episode_indices)
    if len(unique_episodes) != 2 * expected_pairs:
        raise RuntimeError(
            "Unexpected Reacher Arm Mass episode count: "
            f"expected {2 * expected_pairs}, got {len(unique_episodes)}"
        )
    expected_density = {"lighter": 500.0, "heavier": 1500.0}
    for episode in unique_episodes:
        rows = np.flatnonzero(episode_indices == episode)
        rows = rows[np.argsort(step_indices[rows])]
        if not np.array_equal(step_indices[rows], np.arange(20)):
            raise RuntimeError(f"Episode {episode} is not a complete 20-row clip")
        pair_values = {str(pair_ids[index]) for index in rows}
        mode_values = {str(modes[index]) for index in rows}
        split_values = {str(splits[index]) for index in rows}
        density_values = {float(densities[index]) for index in rows}
        if len(pair_values) != 1 or len(mode_values) != 1:
            raise RuntimeError(f"Episode {episode} changes pair or mass mode")
        pair_id = pair_values.pop()
        mode = mode_values.pop()
        if mode not in REACHER_ARM_MASS_MODES:
            raise RuntimeError(f"Unexpected arm-mass mode {mode!r}")
        if split_values != {expected_split}:
            raise RuntimeError(f"Unexpected split for {pair_id}")
        if density_values != {expected_density[mode]}:
            raise RuntimeError(f"Unexpected arm density for {pair_id}/{mode}")
        if mode in episodes.setdefault(pair_id, {}):
            raise RuntimeError(f"Duplicate {mode} episode for {pair_id}")
        frame_rows = rows[[0, 5, 10, 15]]
        episodes[pair_id][mode] = (
            np.stack([_decode(pixel_bytes[index]) for index in frame_rows]),
            actions[rows].reshape(4, 5, 2),
            states[frame_rows],
            fingers[frame_rows],
        )

    if len(episodes) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} arm-mass pairs, got {len(episodes)}"
        )
    ordered_ids = tuple(sorted(episodes))
    lighter_pixels: list[np.ndarray] = []
    heavier_pixels: list[np.ndarray] = []
    action_blocks: list[np.ndarray] = []
    lighter_states: list[np.ndarray] = []
    heavier_states: list[np.ndarray] = []
    lighter_fingers: list[np.ndarray] = []
    heavier_fingers: list[np.ndarray] = []
    for pair_id in ordered_ids:
        pair = episodes[pair_id]
        if set(pair) != set(REACHER_ARM_MASS_MODES):
            raise RuntimeError(f"Incomplete arm-mass pair {pair_id}")
        lighter, heavier = (pair[mode] for mode in REACHER_ARM_MASS_MODES)
        if not np.array_equal(lighter[0][0], heavier[0][0]):
            raise RuntimeError(f"Initial frame differs for {pair_id}")
        if not np.array_equal(lighter[0][2], heavier[0][2]):
            raise RuntimeError(f"Current query frame differs for {pair_id}")
        if not np.array_equal(lighter[1], heavier[1]):
            raise RuntimeError(f"Actions differ for {pair_id}")
        if np.array_equal(lighter[0][1], heavier[0][1]):
            raise RuntimeError(f"History does not reveal arm mass for {pair_id}")
        if np.array_equal(lighter[0][3], heavier[0][3]):
            raise RuntimeError(f"True futures do not differ for {pair_id}")
        if not np.allclose(lighter[2][2], heavier[2][2], atol=1e-6, rtol=0):
            raise RuntimeError(f"Query physical state differs for {pair_id}")
        lighter_pixels.append(lighter[0])
        heavier_pixels.append(heavier[0])
        action_blocks.append(lighter[1])
        lighter_states.append(lighter[2])
        heavier_states.append(heavier[2])
        lighter_fingers.append(lighter[3])
        heavier_fingers.append(heavier[3])

    return ReacherArmMassEvalArrays(
        pair_ids=ordered_ids,
        lighter_pixels=np.stack(lighter_pixels),
        heavier_pixels=np.stack(heavier_pixels),
        raw_action_blocks=np.stack(action_blocks),
        lighter_states=np.stack(lighter_states),
        heavier_states=np.stack(heavier_states),
        lighter_finger_positions=np.stack(lighter_fingers),
        heavier_finger_positions=np.stack(heavier_fingers),
    )


class ReacherArmMassICLEvalDataset:
    """Frozen 256-pair Public Test for Reacher arm-mass ICL."""

    def __init__(
        self,
        *,
        release: dict[str, Any] | None = None,
        release_config: Path | str = DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG,
        repo_root: Path | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.release = release or load_reacher_arm_mass_icl_release(
            release_config
        )
        self.root = resolve_contextworld_path(
            self.release["data"]["artifact_tree"]["root"],
            repo_root=self.repo_root,
        )
        self._arrays: ReacherArmMassEvalArrays | None = None

    @property
    def arrays(self) -> ReacherArmMassEvalArrays:
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
            "arm_mass_modes": list(REACHER_ARM_MASS_MODES),
            "online_environment_calls": 0,
        }


def audit_reacher_arm_mass_icl_release(
    *,
    release_config: Path | str = DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG,
    repo_root: Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_reacher_arm_mass_icl_release(release_config)
    data_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"], repo_root=root
    )
    files: dict[str, Any] = {}
    for name, entry in release["identity"].items():
        path = resolve_contextworld_path(entry["path"], repo_root=root)
        observed = file_sha256(path) if path.is_file() else None
        files[f"identity.{name}"] = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": entry["sha256"],
            "observed_sha256": observed,
            "passed": path.is_file() and observed == entry["sha256"],
        }
    for name, entry in release["data"]["artifacts"].items():
        path = resolve_contextworld_path(entry["path"], repo_root=root)
        observed = file_sha256(path) if path.is_file() else None
        files[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": entry.get("sha256"),
            "observed_sha256": observed,
            "passed": path.is_file() and observed == entry.get("sha256"),
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
    public = ReacherArmMassICLEvalDataset(release=release, repo_root=root)
    public_result = public.describe()
    public_result["passed"] = public.arrays.pair_count == int(
        release["evaluation"]["pair_count"]
    )
    build_report = (data_root / "build_report.json")
    build_payload = yaml.safe_load(build_report.read_text(encoding="utf-8"))
    data_checks = {
        "build_passed": build_payload.get("passed") is True,
        "cross_split_overlap_zero": not any(
            build_payload.get("cross_split_overlap", {}).values()
        ),
    }
    summary_path = Path(
        files["reference_results.final_release_summary"]["path"]
    )
    reference_summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    response_summary_path = Path(
        files["reference_results.latent_response_summary"]["path"]
    )
    response_summary = json.loads(
        response_summary_path.read_text(encoding="utf-8")
    )
    reference_result = {
        "status": reference_summary.get("status"),
        "passed_training_seeds": sum(
            bool(row.get("passed"))
            for row in reference_summary.get("public_test", {}).get(
                "lewm", []
            )
        ),
        "original_task_retention_passed_checkpoints": reference_summary.get(
            "original_task_retention", {}
        ).get("passed_checkpoints"),
    }
    lewm_rows = reference_summary.get("public_test", {}).get("lewm", [])
    response_methods = response_summary.get("methods", {})
    response_rows = response_methods.get("LeWM", [])
    checkpoint_identity_matches = all(
        {
            int(row["training_seed"]): row["checkpoint_sha256"]
            for row in response_methods.get(response_family, [])
        }
        == {
            int(row["seed"]): row["checkpoint_sha256"]
            for row in reference_summary.get("public_test", {}).get(
                reference_family, []
            )
        }
        and all(
            len(str(row.get("checkpoint_sha256", ""))) == 64
            for row in response_methods.get(response_family, [])
        )
        for response_family, reference_family in (
            ("LeWM", "lewm"),
            ("PLDM", "pldm"),
        )
    )
    reference_result["latent_response_checkpoint_identity_matches"] = (
        checkpoint_identity_matches
    )
    legacy_by_seed = {int(row["seed"]): row for row in lewm_rows}
    current_response_passes = 0
    for response in response_rows:
        seed = int(response.get("training_seed", -1))
        legacy = legacy_by_seed.get(seed, {})
        if bool(
            response.get("checkpoint_sha256")
            == legacy.get("checkpoint_sha256")
            and float(response.get("minimum_target_response_mse", 0.0))
            > 0.0
            and float(response.get("response_gain", -np.inf)) >= 0.5
            and float(
                response.get("normalized_response_error", np.inf)
            )
            < 1.0
            and response.get("response_gate_passed") is True
        ):
            current_response_passes += 1
    reference_result["latent_response_gate_passed_training_seeds"] = (
        current_response_passes
    )
    reference_result["requires_latent_response_rescore"] = bool(
        current_response_passes != 3
    )
    reference_result["passed"] = bool(
        reference_summary.get("release_id") == release["release_id"]
        and reference_result["status"] == "passed_public_test_3_of_3"
        and reference_result["passed_training_seeds"] == 3
        and response_summary.get("public_test_manifest_sha256")
        == release["data"]["manifest_sha256"]
        and response_summary.get("gate")
        == {
            "response_gain_minimum": 0.5,
            "normalized_response_error_exclusive_maximum": 1.0,
            "target_latent_separation_required": True,
        }
        and checkpoint_identity_matches
        and current_response_passes == 3
        and reference_result[
            "original_task_retention_passed_checkpoints"
        ]
        == 3
    )
    published_splits = tuple(build_payload["splits"].values())
    published_pairs = tuple(
        pair
        for split in published_splits
        for pair in split["pairs"]
    )
    causal_data = audit_causal_data_contract(
        component_id="robot_arm_mass",
        evidence_scope="all 2,560 published causal pairs",
        continuous_environment_trajectory=True,
        state_installations_after_x0=0,
        query_simulator_recreated=False,
        maximum_query_state_gap=max(
            float(split["maximum_query_state_gap"])
            for split in published_splits
        ),
        query_state_tolerance=1e-6,
        query_pixels_exact=all(
            pair["audit"]["query_pixels_equal"]
            for pair in published_pairs
        ),
        query_actions_exact=all(
            pair["audit"]["actions_equal"] for pair in published_pairs
        ),
        history_effect_present=(
            min(
                float(split["minimum_history_qpos_gap"])
                for split in published_splits
            )
            > 0.0
            and min(
                int(split["minimum_history_changed_rgb_values"])
                for split in published_splits
            )
            > 0
        ),
        true_future_effect_present=(
            min(
                float(split["minimum_true_future_qpos_gap"])
                for split in published_splits
            )
            > 0.0
            and min(
                int(split["minimum_future_changed_rgb_values"])
                for split in published_splits
            )
            > 0
        ),
        x0_policy="shared_visible_start",
        x0_static_leakage_check_passed=all(
            pair["audit"]["initial_pixels_equal"]
            for pair in published_pairs
        ),
        evidence=(
            files["identity.causal_physics_contract"]["path"],
            str(build_report),
        ),
    )
    upstream_checks: dict[str, Any] = {}
    for name, path, specification in (
        (
            "original_h5",
            resolve_reacher_original_h5(release, repo_root=root),
            release["training"]["upstream"]["original_h5"],
        ),
        (
            "original_lance",
            resolve_reacher_original_lance(release, repo_root=root),
            release["training"]["upstream"]["original_lance"],
        ),
    ):
        if path.is_dir():
            observed_bytes = sum(
                child.stat().st_size
                for child in path.rglob("*")
                if child.is_file()
            )
            exists = True
        else:
            observed_bytes = path.stat().st_size if path.is_file() else None
            exists = path.is_file()
        upstream_checks[name] = {
            "path": str(path),
            "exists": exists,
            "expected_bytes": int(specification["bytes"]),
            "observed_bytes": observed_bytes,
            "passed": exists and observed_bytes == int(specification["bytes"]),
        }
    for family in ("lewm", "pldm"):
        specification = release["training"]["reference_matrix"][
            "initial_checkpoints"
        ][family]
        path = resolve_reacher_initial_checkpoint(
            release,
            family,
            repo_root=root,
        )
        observed = file_sha256(path) if path.is_file() else None
        config_path = resolve_reacher_initial_checkpoint_config(
            release,
            family,
            repo_root=root,
        )
        config_observed = (
            file_sha256(config_path) if config_path.is_file() else None
        )
        upstream_checks[f"{family}_initial_checkpoint"] = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_bytes": int(specification["bytes"]),
            "observed_bytes": path.stat().st_size if path.is_file() else None,
            "expected_sha256": specification["sha256"],
            "observed_sha256": observed,
            "config_path": str(config_path),
            "config_expected_sha256": specification["config_sha256"],
            "config_observed_sha256": config_observed,
            "passed": bool(
                path.is_file()
                and path.stat().st_size == int(specification["bytes"])
                and observed == specification["sha256"]
                and config_observed == specification["config_sha256"]
            ),
        }
    release_checks = {
        "public_test_release_candidate": release["release_status"]
        in {"public_test_release_candidate", "public_test_release"},
        "reference_matrix_completed": release["training"][
            "reference_matrix"
        ]["status"]
        == "completed",
        "original_task_retention_completed": release["scoring"][
            "original_task_retention"
        ]["status"]
        == "completed",
    }
    passed = (
        all(row["passed"] for row in files.values())
        and tree_result["passed"]
        and public_result["passed"]
        and all(data_checks.values())
        and causal_data["passed"]
        and reference_result["passed"]
        and all(row["passed"] for row in upstream_checks.values())
        and all(release_checks.values())
    )
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "full": bool(full),
        "files": files,
        "artifact_tree": tree_result,
        "data_checks": data_checks,
        "causal_data_contract": causal_data,
        "reference_result": reference_result,
        "upstream": upstream_checks,
        "release_checks": release_checks,
        "public_test": public_result,
        "status": "passed" if passed else "failed",
        "passed": passed,
    }


__all__ = [
    "DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG",
    "REACHER_ARM_MASS_MODES",
    "REACHER_ARM_MASS_RELEASE_ID",
    "ReacherArmMassEvalArrays",
    "ReacherArmMassICLEvalDataset",
    "_read_lance_pairs",
    "audit_reacher_arm_mass_icl_release",
    "directory_sha256",
    "file_sha256",
    "load_reacher_arm_mass_icl_release",
    "resolve_reacher_initial_checkpoint",
    "resolve_reacher_initial_checkpoint_config",
    "resolve_reacher_original_h5",
    "resolve_reacher_original_lance",
]
