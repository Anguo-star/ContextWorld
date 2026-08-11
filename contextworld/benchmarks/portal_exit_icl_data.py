from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any

import lance
import numpy as np
from PIL import Image
import yaml

from contextworld.benchmarks.causal_data_contract import (
    audit_causal_data_contract,
)
from contextworld.paths import (
    artifact_root,
    repository_root,
    resolve_contextworld_path,
)


PORTAL_EXIT_RELEASE_ID = "contextworld_tworoom_portal_exit_icl_history3_v1"
DEFAULT_PORTAL_EXIT_RELEASE_CONFIG = (
    repository_root()
    / "configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml"
)
PORTAL_EXIT_MODES = ("near_border", "farther_from_border")


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


def load_portal_exit_icl_release(
    path: Path | str = DEFAULT_PORTAL_EXIT_RELEASE_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported Portal Exit release: {config_path}")
    if payload.get("release_id") != PORTAL_EXIT_RELEASE_ID:
        raise ValueError(f"Unexpected Portal Exit release id: {config_path}")
    if payload.get("release_status") not in {
        "data_ready_training_in_progress",
        "public_test_release_candidate",
        "public_test_release",
    }:
        raise ValueError("Unsupported Portal Exit release status")
    scope = payload.get("scope", {})
    if scope.get("history_tokens") != 3:
        raise ValueError("Portal Exit v1 requires History=3")
    if scope.get("public_test_included") is not True:
        raise ValueError("Portal Exit v1 must include Public Test")
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


def resolve_portal_original_lance(
    release: dict[str, Any],
    *,
    repo_root: Path,
) -> Path:
    return _resolve_upstream(
        release["training"]["upstream"]["original_lance"],
        repo_root=repo_root,
    )


@dataclass(frozen=True)
class PortalExitEvalArrays:
    pair_ids: tuple[str, ...]
    near_border_pixels: np.ndarray
    farther_from_border_pixels: np.ndarray
    raw_action_blocks: np.ndarray
    near_border_states: np.ndarray
    farther_from_border_states: np.ndarray

    @property
    def pair_count(self) -> int:
        return len(self.pair_ids)

    @property
    def low_pixels(self) -> np.ndarray:
        return self.near_border_pixels

    @property
    def high_pixels(self) -> np.ndarray:
        return self.farther_from_border_pixels

    @property
    def low_states(self) -> np.ndarray:
        return self.near_border_states

    @property
    def high_states(self) -> np.ndarray:
        return self.farther_from_border_states


def _decode(value: bytes) -> np.ndarray:
    with Image.open(BytesIO(value)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_lance_pairs(
    path: Path,
    *,
    expected_pairs: int,
    expected_split: str,
) -> PortalExitEvalArrays:
    table = lance.dataset(path).to_table(
        columns=[
            "episode_idx", "step_idx", "pixels", "action", "proprio",
            "pair_id", "hidden_mode", "hidden_portal_exit", "split",
        ]
    )
    episode_indices = np.asarray(table["episode_idx"].to_numpy(), dtype=np.int64)
    step_indices = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
    pixels = table["pixels"].to_pylist()
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["proprio"].to_pylist(), dtype=np.float32)
    pair_ids = table["pair_id"].to_pylist()
    modes = table["hidden_mode"].to_pylist()
    hidden = np.asarray(table["hidden_portal_exit"].to_pylist()).reshape(-1)
    splits = table["split"].to_pylist()
    episodes: dict[str, dict[str, tuple[np.ndarray, ...]]] = {}
    unique_episodes = np.unique(episode_indices)
    if len(unique_episodes) != 2 * expected_pairs:
        raise RuntimeError("Unexpected portal-exit episode count")
    for episode in unique_episodes:
        rows = np.flatnonzero(episode_indices == episode)
        rows = rows[np.argsort(step_indices[rows])]
        if not np.array_equal(step_indices[rows], np.arange(20)):
            raise RuntimeError(f"Episode {episode} is not a complete clip")
        pair_values = {str(pair_ids[index]) for index in rows}
        mode_values = {str(modes[index]) for index in rows}
        split_values = {str(splits[index]) for index in rows}
        hidden_values = {int(hidden[index]) for index in rows}
        if len(pair_values) != 1 or len(mode_values) != 1:
            raise RuntimeError(f"Episode {episode} changes pair or mode")
        pair_id = pair_values.pop()
        mode = mode_values.pop()
        expected_hidden = PORTAL_EXIT_MODES.index(mode) if mode in PORTAL_EXIT_MODES else -1
        if split_values != {expected_split} or hidden_values != {expected_hidden}:
            raise RuntimeError(f"Unexpected split or hidden value for {pair_id}")
        frame_rows = rows[[0, 5, 10, 15]]
        episodes.setdefault(pair_id, {})[mode] = (
            np.stack([_decode(pixels[index]) for index in frame_rows]),
            actions[rows].reshape(4, 5, 2),
            states[frame_rows],
        )
    if len(episodes) != expected_pairs:
        raise RuntimeError("Unexpected portal-exit pair count")
    ordered_ids = tuple(sorted(episodes))
    near_pixels, farther_pixels, action_blocks = [], [], []
    near_states, farther_states = [], []
    for pair_id in ordered_ids:
        pair = episodes[pair_id]
        if set(pair) != set(PORTAL_EXIT_MODES):
            raise RuntimeError(f"Incomplete portal-exit pair {pair_id}")
        near, farther = (pair[mode] for mode in PORTAL_EXIT_MODES)
        if not np.array_equal(near[0][0], farther[0][0]):
            raise RuntimeError(f"Initial frame differs for {pair_id}")
        if not np.array_equal(near[0][2], farther[0][2]):
            raise RuntimeError(f"Query frame differs for {pair_id}")
        if not np.array_equal(near[1], farther[1]):
            raise RuntimeError(f"Actions differ for {pair_id}")
        if np.array_equal(near[0][1], farther[0][1]):
            raise RuntimeError(f"History does not reveal portal exit for {pair_id}")
        if np.array_equal(near[0][3], farther[0][3]):
            raise RuntimeError(f"True futures do not differ for {pair_id}")
        if not np.array_equal(near[2][2], farther[2][2]):
            raise RuntimeError(f"Query state differs for {pair_id}")
        near_pixels.append(near[0])
        farther_pixels.append(farther[0])
        action_blocks.append(near[1])
        near_states.append(near[2])
        farther_states.append(farther[2])
    return PortalExitEvalArrays(
        pair_ids=ordered_ids,
        near_border_pixels=np.stack(near_pixels),
        farther_from_border_pixels=np.stack(farther_pixels),
        raw_action_blocks=np.stack(action_blocks),
        near_border_states=np.stack(near_states),
        farther_from_border_states=np.stack(farther_states),
    )


class PortalExitICLEvalDataset:
    def __init__(
        self,
        *,
        release: dict[str, Any] | None = None,
        release_config: Path | str = DEFAULT_PORTAL_EXIT_RELEASE_CONFIG,
        repo_root: Path | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.release = release or load_portal_exit_icl_release(release_config)
        self.root = resolve_contextworld_path(
            self.release["data"]["artifact_tree"]["root"],
            repo_root=self.repo_root,
        )
        self._arrays: PortalExitEvalArrays | None = None

    @property
    def arrays(self) -> PortalExitEvalArrays:
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
            "portal_exit_modes": list(PORTAL_EXIT_MODES),
            "online_environment_calls": 0,
        }


def audit_portal_exit_icl_release(
    *,
    release_config: Path | str = DEFAULT_PORTAL_EXIT_RELEASE_CONFIG,
    repo_root: Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_portal_exit_icl_release(release_config)
    data_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"], repo_root=root
    )
    files = {}
    for name, entry in release.get("identity", {}).items():
        path = resolve_contextworld_path(entry["path"], repo_root=root)
        observed = file_sha256(path) if path.is_file() else None
        files[f"identity.{name}"] = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": entry.get("sha256"),
            "observed_sha256": observed,
            "passed": path.is_file() and observed == entry.get("sha256"),
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
            "expected_sha256": entry.get("sha256"),
            "observed_sha256": observed,
            "passed": path.is_file() and observed == entry.get("sha256"),
        }
    for name in ("query_catalog", "query_data"):
        entry = release["scoring"]["original_task_retention"][name]
        path = resolve_contextworld_path(entry["path"], repo_root=root)
        observed = file_sha256(path) if path.is_file() else None
        files[f"scoring.original_task_retention.{name}"] = {
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
        and (not full or tree_result["expected_sha256"] == tree_result["observed_sha256"])
    )
    public = PortalExitICLEvalDataset(release=release, repo_root=root)
    public_result = public.describe()
    public_result["passed"] = public.arrays.pair_count == int(
        release["evaluation"]["pair_count"]
    )
    build_payload = yaml.safe_load(
        (data_root / "build_report.json").read_text(encoding="utf-8")
    )
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
    public_reference = reference_summary.get("public_test", {})
    reference_result = {
        "status": reference_summary.get("status"),
        "lewm_passed_training_seeds": sum(
            bool(row.get("passed"))
            for row in public_reference.get("lewm_paired_real_future", [])
        ),
        "pldm_passed_training_seeds": sum(
            bool(row.get("passed"))
            for row in public_reference.get("pldm", [])
        ),
        "original_task_retention_passed_checkpoints": reference_summary.get(
            "original_task_retention", {}
        ).get("passed_checkpoints"),
    }
    response_methods = response_summary.get("methods", {})
    checkpoint_identity_matches = all(
        {
            int(row["training_seed"]): row["checkpoint_sha256"]
            for row in response_methods.get(response_family, [])
        }
        == {
            int(row["seed"]): row["checkpoint_sha256"]
            for row in public_reference.get(reference_family, [])
        }
        and all(
            len(str(row.get("checkpoint_sha256", ""))) == 64
            for row in response_methods.get(response_family, [])
        )
        for response_family, reference_family in (
            ("LeWM", "lewm_paired_real_future"),
            ("PLDM", "pldm"),
        )
    )
    reference_result["latent_response_checkpoint_identity_matches"] = (
        checkpoint_identity_matches
    )
    reference_result["latent_response_gate_passed_training_seeds"] = {
        family: sum(
            row.get("response_gate_passed") is True
            for row in response_methods.get(family, [])
        )
        for family in ("LeWM", "PLDM")
    }
    reference_result["passed"] = bool(
        reference_summary.get("release_id") == release["release_id"]
        and reference_result["status"] == "failed_public_test_0_of_3"
        and reference_result["lewm_passed_training_seeds"] == 0
        and reference_result["pldm_passed_training_seeds"] == 0
        and response_summary.get("public_test_manifest_sha256")
        == release["data"]["manifest_sha256"]
        and response_summary.get("gate")
        == {
            "response_gain_minimum": 0.5,
            "normalized_response_error_exclusive_maximum": 1.0,
            "target_latent_separation_required": True,
        }
        and reference_result[
            "latent_response_gate_passed_training_seeds"
        ]
        == {"LeWM": 3, "PLDM": 0}
        and checkpoint_identity_matches
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
        component_id="portal_exit",
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
            pair["audit"]["checks"]["query_pixels_identical"]
            for pair in published_pairs
        ),
        query_actions_exact=all(
            pair["audit"]["checks"]["query_actions_identical"]
            for pair in published_pairs
        ),
        history_effect_present=(
            min(
                float(split["minimum_history_exit_gap_px"])
                for split in published_splits
            )
            > 0.0
        ),
        true_future_effect_present=(
            min(
                float(split["minimum_true_future_gap_px"])
                for split in published_splits
            )
            > 0.0
        ),
        x0_policy="shared_visible_start",
        x0_static_leakage_check_passed=all(
            pair["audit"]["checks"]["initial_pixels_identical"]
            for pair in published_pairs
        ),
        evidence=(
            files["identity.causal_pair_contract"]["path"],
            str(data_root / "build_report.json"),
        ),
    )
    upstream_specification = release["training"]["upstream"][
        "original_lance"
    ]
    upstream_lance = resolve_portal_original_lance(
        release,
        repo_root=root,
    )
    upstream_bytes = (
        sum(
            child.stat().st_size
            for child in upstream_lance.rglob("*")
            if child.is_file()
        )
        if upstream_lance.is_dir()
        else None
    )
    upstream_result = {
        "path": str(upstream_lance),
        "exists": upstream_lance.is_dir(),
        "expected_bytes": int(upstream_specification["bytes"]),
        "observed_bytes": upstream_bytes,
        "passed": bool(
            upstream_lance.is_dir()
            and upstream_bytes == int(upstream_specification["bytes"])
        ),
    }
    final_candidate = release["release_status"] in {
        "public_test_release_candidate",
        "public_test_release",
    }
    release_checks = {
        "public_test_release_candidate": final_candidate,
        "reference_matrix_completed": release["training"][
            "reference_matrix"
        ]["status"] in {"completed", "completed_failed_prediction_gate"},
        "original_task_retention_completed": release["scoring"][
            "original_task_retention"
        ]["status"] == "completed",
    }
    passed = (
        all(row["passed"] for row in files.values())
        and tree_result["passed"]
        and public_result["passed"]
        and all(data_checks.values())
        and causal_data["passed"]
        and upstream_result["passed"]
        and reference_result["passed"]
        and (not final_candidate or all(release_checks.values()))
    )
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "full": bool(full),
        "files": files,
        "artifact_tree": tree_result,
        "data_checks": data_checks,
        "causal_data_contract": causal_data,
        "upstream_original_lance": upstream_result,
        "reference_result": reference_result,
        "release_checks": release_checks,
        "public_test": public_result,
        "status": "passed" if passed else "failed",
        "passed": passed,
    }


__all__ = [
    "DEFAULT_PORTAL_EXIT_RELEASE_CONFIG",
    "PORTAL_EXIT_MODES",
    "PORTAL_EXIT_RELEASE_ID",
    "PortalExitEvalArrays",
    "PortalExitICLEvalDataset",
    "_read_lance_pairs",
    "audit_portal_exit_icl_release",
    "directory_sha256",
    "file_sha256",
    "load_portal_exit_icl_release",
    "resolve_portal_original_lance",
]
