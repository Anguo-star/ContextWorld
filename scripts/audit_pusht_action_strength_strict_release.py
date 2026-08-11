#!/usr/bin/env python3
"""Audit a strict PushT action-strength Training/Public Test release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import lance
import numpy as np


DEFAULT_RELEASE_ROOT = Path(
    "/tmp/contextworld_action_strength_strict_release"
)
DEFAULT_LEGACY_TRAINING = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "context_world/synthesis/"
    "pusht_hidden_actuation_replay_matched_h3_v2"
)
DEFAULT_LEGACY_PUBLIC_TEST = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "context_world/synthesis/"
    "pusht_hidden_actuation_replay_matched_confirm_h3_v3"
)
MODEL_FRAME_STEPS = (0, 5, 10, 15)
RAW_ROWS_PER_EPISODE = 20


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


def tree_receipt(path: Path) -> dict[str, Any]:
    files = sorted(value for value in path.rglob("*") if value.is_file())
    return {
        "path": str(path),
        "files": len(files),
        "bytes": sum(value.stat().st_size for value in files),
        "sha256": directory_sha256(path),
        "manifest_sha256": file_sha256(path / "manifest.json"),
        "build_report_sha256": file_sha256(path / "build_report.json"),
        "request_sha256": file_sha256(path / "request.json"),
    }


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def manifest_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((path / "manifest.json").read_text())
    build_report = json.loads((path / "build_report.json").read_text())
    request = json.loads((path / "request.json").read_text())
    strict = manifest["strict_causal_chain_audit"]
    receipt = {
        "manifest_passed": manifest.get("passed") is True,
        "build_report_passed": build_report.get("passed") is True,
        "manifest_hash_matches_build_report": (
            build_report["manifest_sha256"]
            == file_sha256(path / "manifest.json")
        ),
        "request_hash_matches_manifest": (
            manifest["request_sha256"]
            == canonical_json_sha256(request)
        ),
        "strict_causal_chain_passed": strict.get("passed") is True,
        "state_installations_after_x0": strict[
            "state_installations_after_x0"
        ],
        "query_simulator_recreated": strict["query_simulator_recreated"],
        "max_pair_full_state_gap": strict["max_pair_full_state_gap"],
        "full_state_tolerance": strict["full_state_tolerance"],
        "full_state_dimensions": strict["full_state_dimensions"],
        "max_pair_query_pixel_difference": strict[
            "max_pair_query_pixel_difference"
        ],
        "max_pair_query_action_difference": strict[
            "max_pair_query_action_difference"
        ],
        "min_history_effect": strict["min_history_effect"],
        "min_true_future_effect": strict["min_true_future_effect"],
    }
    receipt["passed"] = all(
        (
            receipt["manifest_passed"],
            receipt["build_report_passed"],
            receipt["manifest_hash_matches_build_report"],
            receipt["request_hash_matches_manifest"],
            receipt["strict_causal_chain_passed"],
            receipt["state_installations_after_x0"] == 0,
            not receipt["query_simulator_recreated"],
            receipt["max_pair_full_state_gap"]
            <= receipt["full_state_tolerance"],
            receipt["full_state_dimensions"] == 12,
            receipt["max_pair_query_pixel_difference"] == 0,
            receipt["max_pair_query_action_difference"] == 0.0,
        )
    )
    return manifest, receipt


def _update_array_digest(digest: Any, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def compare_model_visible_tables(
    *,
    strict_path: Path,
    legacy_path: Path,
    expected_pairs: int,
) -> dict[str, Any]:
    strict = lance.dataset(strict_path)
    legacy = lance.dataset(legacy_path)
    expected_rows = expected_pairs * 2 * RAW_ROWS_PER_EPISODE
    strict_rows = strict.count_rows()
    legacy_rows = legacy.count_rows()
    model_rows = np.asarray(
        [
            episode * RAW_ROWS_PER_EPISODE + step
            for episode in range(expected_pairs * 2)
            for step in MODEL_FRAME_STEPS
        ],
        dtype=np.int64,
    )
    strict_digest = hashlib.sha256()
    legacy_digest = hashlib.sha256()
    strict_target_digest = hashlib.sha256()
    legacy_target_digest = hashlib.sha256()
    strict_condition_ids: list[str] = []
    legacy_condition_ids: list[str] = []
    pixel_mismatches = 0
    target_pixel_mismatches = 0
    action_mismatches = 0
    pair_id_mismatches = 0
    condition_id_mismatches = 0
    physics_mismatches = 0
    maximum_physics_difference = 0.0

    for start in range(0, len(model_rows), 4096):
        indices = model_rows[start : start + 4096]
        strict_rows_batch = strict.take(
            indices,
            columns=["pixels", "physics_state", "pair_id", "hidden_mode"],
        ).to_pylist()
        legacy_rows_batch = legacy.take(
            indices,
            columns=["pixels", "physics_state", "pair_id", "hidden_mode"],
        ).to_pylist()
        for row_index, strict_row, legacy_row in zip(
            indices,
            strict_rows_batch,
            legacy_rows_batch,
        ):
            strict_blob = bytes(strict_row["pixels"])
            legacy_blob = bytes(legacy_row["pixels"])
            strict_digest.update(strict_blob)
            legacy_digest.update(legacy_blob)
            pixel_mismatches += int(strict_blob != legacy_blob)
            strict_pair = str(strict_row["pair_id"])
            legacy_pair = str(legacy_row["pair_id"])
            strict_condition = (
                f"{strict_pair}/{strict_row['hidden_mode']}"
            )
            legacy_condition = (
                f"{legacy_pair}/{legacy_row['hidden_mode']}"
            )
            pair_id_mismatches += int(strict_pair != legacy_pair)
            condition_id_mismatches += int(
                strict_condition != legacy_condition
            )
            if int(row_index) % RAW_ROWS_PER_EPISODE == 15:
                strict_condition_ids.append(strict_condition)
                legacy_condition_ids.append(legacy_condition)
                strict_target_digest.update(strict_blob)
                legacy_target_digest.update(legacy_blob)
                target_pixel_mismatches += int(
                    strict_blob != legacy_blob
                )
            strict_physics = np.asarray(
                strict_row["physics_state"], dtype=np.float32
            )
            legacy_physics = np.asarray(
                legacy_row["physics_state"], dtype=np.float32
            )
            difference = float(
                np.max(np.abs(strict_physics - legacy_physics))
            )
            physics_mismatches += int(difference != 0.0)
            maximum_physics_difference = max(
                maximum_physics_difference,
                difference,
            )

    for start in range(0, expected_rows, 8192):
        indices = np.arange(
            start,
            min(start + 8192, expected_rows),
            dtype=np.int64,
        )
        strict_actions = np.asarray(
            strict.take(indices, columns=["action"])["action"].to_pylist(),
            dtype=np.float32,
        )
        legacy_actions = np.asarray(
            legacy.take(indices, columns=["action"])["action"].to_pylist(),
            dtype=np.float32,
        )
        _update_array_digest(strict_digest, strict_actions)
        _update_array_digest(legacy_digest, legacy_actions)
        action_mismatches += int(
            np.count_nonzero(strict_actions != legacy_actions)
        )

    result = {
        "expected_pairs": expected_pairs,
        "expected_rows": expected_rows,
        "strict_rows": strict_rows,
        "legacy_rows": legacy_rows,
        "model_frames_compared": len(model_rows),
        "raw_action_rows_compared": expected_rows,
        "pixel_blob_mismatch_count": pixel_mismatches,
        "future_target_blob_mismatch_count": target_pixel_mismatches,
        "action_value_mismatch_count": action_mismatches,
        "pair_id_mismatch_count": pair_id_mismatches,
        "condition_id_mismatch_count": condition_id_mismatches,
        "strict_condition_ids_sha256": canonical_json_sha256(
            strict_condition_ids
        ),
        "legacy_condition_ids_sha256": canonical_json_sha256(
            legacy_condition_ids
        ),
        "condition_ids_exactly_identical": (
            strict_condition_ids == legacy_condition_ids
        ),
        "strict_visible_future_targets_sha256": (
            strict_target_digest.hexdigest()
        ),
        "legacy_visible_future_targets_sha256": (
            legacy_target_digest.hexdigest()
        ),
        "visible_future_targets_exactly_identical": (
            strict_target_digest.digest() == legacy_target_digest.digest()
        ),
        "strict_model_visible_sha256": strict_digest.hexdigest(),
        "legacy_model_visible_sha256": legacy_digest.hexdigest(),
        "model_visible_exactly_identical": (
            strict_digest.digest() == legacy_digest.digest()
        ),
        "model_frame_physics_state_mismatch_count": physics_mismatches,
        "maximum_model_frame_physics_state_difference": (
            maximum_physics_difference
        ),
    }
    result["passed"] = all(
        (
            strict_rows == expected_rows,
            legacy_rows == expected_rows,
            pixel_mismatches == 0,
            target_pixel_mismatches == 0,
            action_mismatches == 0,
            pair_id_mismatches == 0,
            condition_id_mismatches == 0,
            result["condition_ids_exactly_identical"],
            result["visible_future_targets_exactly_identical"],
            result["model_visible_exactly_identical"],
        )
    )
    return result


def split_isolation(
    training: dict[str, Any],
    public_test: dict[str, Any],
) -> dict[str, Any]:
    groups = {
        "training": training["splits"]["train"]["pairs"],
        "development": training["splits"]["validation"]["pairs"],
        "public_test": public_test["splits"]["validation"]["pairs"],
    }
    source_episodes = {
        name: {
            int(row["template"]["source_episode_index"])
            for row in rows
        }
        for name, rows in groups.items()
    }
    query_hashes = {
        "training": set(training["splits"]["train"]["query_hashes"]),
        "development": set(
            training["splits"]["validation"]["query_hashes"]
        ),
        "public_test": set(
            public_test["splits"]["validation"]["query_hashes"]
        ),
    }
    names = tuple(groups)
    episode_overlap = {}
    query_overlap = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            key = f"{left}__{right}"
            episode_overlap[key] = len(
                source_episodes[left] & source_episodes[right]
            )
            query_overlap[key] = len(
                query_hashes[left] & query_hashes[right]
            )
    result = {
        "source_episode_overlap_counts": episode_overlap,
        "query_pixel_hash_overlap_counts": query_overlap,
    }
    result["passed"] = not any(episode_overlap.values()) and not any(
        query_overlap.values()
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument(
        "--training-root",
        type=Path,
        help="Strict Training/Development root when artifacts are separated.",
    )
    parser.add_argument(
        "--public-test-root",
        type=Path,
        help="Strict Public Test root when artifacts are separated.",
    )
    parser.add_argument(
        "--legacy-training", type=Path, default=DEFAULT_LEGACY_TRAINING
    )
    parser.add_argument(
        "--legacy-public-test", type=Path, default=DEFAULT_LEGACY_PUBLIC_TEST
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_root = args.release_root.expanduser().resolve()
    training_root = (
        args.training_root.expanduser().resolve()
        if args.training_root is not None
        else release_root / "training"
    )
    public_root = (
        args.public_test_root.expanduser().resolve()
        if args.public_test_root is not None
        else release_root / "public_test"
    )
    separated_roots = bool(
        args.training_root is not None or args.public_test_root is not None
    )
    canonical_release_root = (
        Path(os.path.commonpath((training_root, public_root)))
        if separated_roots
        else release_root
    )
    output = args.output or (release_root / "release_audit.json")
    training_manifest, training_receipt = manifest_receipt(training_root)
    public_manifest, public_receipt = manifest_receipt(public_root)
    comparisons = {
        "training": compare_model_visible_tables(
            strict_path=training_root / "train.lance",
            legacy_path=args.legacy_training / "train.lance",
            expected_pairs=2048,
        ),
        "development": compare_model_visible_tables(
            strict_path=training_root / "validation.lance",
            legacy_path=args.legacy_training / "validation.lance",
            expected_pairs=256,
        ),
        "public_test": compare_model_visible_tables(
            strict_path=public_root / "validation.lance",
            legacy_path=args.legacy_public_test / "validation.lance",
            expected_pairs=256,
        ),
    }
    isolation = split_isolation(training_manifest, public_manifest)
    report = {
        "schema_version": 1,
        "release_root": str(canonical_release_root),
        "artifact_layout": (
            "separate_training_and_public_test_roots"
            if separated_roots
            else "single_release_root"
        ),
        "training_root": str(training_root),
        "public_test_root": str(public_root),
        "trees": {
            "training": tree_receipt(training_root),
            "public_test": tree_receipt(public_root),
        },
        "manifests": {
            "training": training_receipt,
            "public_test": public_receipt,
        },
        "model_visible_legacy_comparison": comparisons,
        "cross_split_isolation": isolation,
    }
    report["passed"] = all(
        (
            training_receipt["passed"],
            public_receipt["passed"],
            isolation["passed"],
            all(row["passed"] for row in comparisons.values()),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
