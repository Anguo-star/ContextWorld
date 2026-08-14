#!/usr/bin/env python3
"""Build the one-use Cube History=3 v4r1 Public Test after a valid freeze."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import traceback
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import contextworld.evaluation.cube_grasp_rule_h3_v4 as v4_physics  # noqa: E402
from contextworld.benchmarks.cube_grasp_rule_public_contract import (  # noqa: E402
    DEFAULT_FREEZE_RECEIPT,
    DEFAULT_PREREGISTRATION,
    PUBLIC_CANDIDATE_ASSIGNMENT_SEED,
    PUBLIC_CATALOG_INDEX_OFFSET,
    PUBLIC_CATALOG_SEED,
    PUBLIC_PAIR_COUNT,
    PUBLIC_PROFILE_SEED,
    PUBLIC_SPLIT,
    file_identity,
    load_public_authorization,
)
from contextworld.evaluation.cube_grasp_rule_h3 import (  # noqa: E402
    CubeGraspRuleCandidate,
)
from contextworld.evaluation.cube_grasp_rule_h3_v4 import (  # noqa: E402
    CubeGraspRuleV4Candidate,
    make_v4_candidate,
)
from contextworld.paths import portable_contextworld_path  # noqa: E402
import scripts.build_cube_grasp_rule_h3_v4_data as development_builder  # noqa: E402


PROTOCOL = "cube_gripper_carry_rule_history3_v4r1_public_release_v1"
DEFAULT_WORKERS = 16
DEFAULT_JPEG_QUALITY = 95
DEFAULT_STAGING_ROOT = Path("/tmp")
SUCCESS_MARKER = "_SUCCESS.json"
GENERATION_STARTED_MARKER = "_GENERATION_STARTED.json"
GENERATION_FAILURE_MARKER = "_GENERATION_FAILURE.json"
_ORIGINAL_WORKER_INITIALIZE = development_builder._worker_initialize


class _PublicBalancedAcceptance(development_builder._BalancedAcceptance):
    """Public-only tracker without the Development split-universe assertion."""

    def __post_init__(self) -> None:
        if (
            isinstance(self.pair_count, bool)
            or not isinstance(self.pair_count, (int, np.integer))
            or int(self.pair_count) <= 0
            or int(self.pair_count) % len(self.anchors)
        ):
            raise ValueError(
                "Public pair count must be positive and divisible by four"
            )
        self.pair_count = int(self.pair_count)
        self.counts = {anchor: 0 for anchor in self.anchors}


def _public_worker_initialize(quality: int) -> None:
    v4_physics.V4_PROFILE_SPLIT_SEEDS[PUBLIC_SPLIT] = PUBLIC_PROFILE_SEED
    _ORIGINAL_WORKER_INITIALIZE(quality)


@contextmanager
def _public_v4_runtime() -> Iterator[None]:
    old_profile_seeds = dict(v4_physics.V4_PROFILE_SPLIT_SEEDS)
    old_active_splits = development_builder.ACTIVE_SPLITS
    old_catalog_seeds = development_builder.CATALOG_SEEDS
    old_offset = development_builder.FORMAL_CATALOG_INDEX_OFFSET
    old_initializer = development_builder._worker_initialize
    old_acceptance = development_builder._BalancedAcceptance
    try:
        v4_physics.V4_PROFILE_SPLIT_SEEDS[PUBLIC_SPLIT] = PUBLIC_PROFILE_SEED
        development_builder.ACTIVE_SPLITS = (PUBLIC_SPLIT,)
        development_builder.CATALOG_SEEDS = {PUBLIC_SPLIT: PUBLIC_CATALOG_SEED}
        development_builder.FORMAL_CATALOG_INDEX_OFFSET = (
            PUBLIC_CATALOG_INDEX_OFFSET
        )
        development_builder._worker_initialize = _public_worker_initialize
        development_builder._BalancedAcceptance = _PublicBalancedAcceptance
        yield
    finally:
        v4_physics.V4_PROFILE_SPLIT_SEEDS.clear()
        v4_physics.V4_PROFILE_SPLIT_SEEDS.update(old_profile_seeds)
        development_builder.ACTIVE_SPLITS = old_active_splits
        development_builder.CATALOG_SEEDS = old_catalog_seeds
        development_builder.FORMAL_CATALOG_INDEX_OFFSET = old_offset
        development_builder._worker_initialize = old_initializer
        development_builder._BalancedAcceptance = old_acceptance


def _sha256_values(values: Sequence[str]) -> str:
    normalized = sorted(str(value) for value in values)
    return hashlib.sha256(("\n".join(normalized) + "\n").encode()).hexdigest()


def _regular_files(root: Path) -> list[Path]:
    metadata = os.lstat(root)
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise ValueError(f"tree root must be a real directory: {root}")
    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            child = directory_path / name
            child_metadata = os.lstat(child)
            if not stat.S_ISDIR(child_metadata.st_mode) or child.is_symlink():
                raise ValueError(f"tree contains a directory alias: {child}")
        for name in file_names:
            child = directory_path / name
            child_metadata = os.lstat(child)
            if not stat.S_ISREG(child_metadata.st_mode) or child.is_symlink():
                raise ValueError(f"tree contains a non-regular file: {child}")
            files.append(child)
    return sorted(files, key=lambda value: value.relative_to(root).as_posix())


def _tree_identity(
    root: Path, *, excluded_names: frozenset[str] = frozenset()
) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = _regular_files(root)
    total = 0
    entries = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if relative in excluded_names:
            continue
        identity = file_identity(path, logical_path=relative)
        total += int(identity["size_bytes"])
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(identity["sha256"].encode("ascii"))
        digest.update(b"\0")
        entries.append(identity)
    return {
        "file_count": len(entries),
        "size_bytes": total,
        "tree_sha256": digest.hexdigest(),
        "files": entries,
    }


def _identity_matches(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(observed.get(name) == expected.get(name) for name in ("sha256", "size_bytes"))


def _source_identity(source: Path, frozen: Mapping[str, Any]) -> dict[str, Any]:
    observed = file_identity(source, logical_path=str(frozen["path"]))
    if not _identity_matches(observed, frozen):
        raise RuntimeError("source H5 changed before Public candidate selection")
    with h5py.File(source, "r", swmr=True) as handle:
        row_count = int(handle["action"].shape[0])
    expected_rows = int(frozen.get("row_count", -1))
    if row_count != expected_rows:
        raise RuntimeError("source H5 row count changed")
    return {
        **observed,
        "row_count": row_count,
        "content_rehashed_before_public_candidate_selection": True,
    }


def _public_exclusion_audit(
    freeze: Mapping[str, Any], *, freeze_receipt_identity: Mapping[str, Any]
) -> dict[str, Any]:
    raw = freeze.get("public_exclusions")
    if not isinstance(raw, Mapping) or raw.get("checks_passed") is not True:
        raise RuntimeError("freeze receipt lacks validated Public exclusions")
    return {
        **raw,
        "path": str(freeze_receipt_identity["path"]),
        "sha256": str(freeze_receipt_identity["sha256"]),
        "size_bytes": int(freeze_receipt_identity["size_bytes"]),
        "status": str(freeze["status"]),
        "checks_passed": True,
    }


def build_public_catalog(
    source: Path,
    *,
    source_identity: Mapping[str, Any],
    exclusions: Mapping[str, Any],
) -> tuple[list[CubeGraspRuleV4Candidate], dict[str, Any]]:
    excluded_episodes, prior_content = development_builder._prior_exclusion_sets(
        exclusions
    )
    eligible_before = development_builder._eligible_source_rows(source)
    eligible = [row for row in eligible_before if int(row[1]) not in excluded_episodes]
    required = 2 * PUBLIC_PAIR_COUNT
    if len(eligible) < required:
        raise RuntimeError(
            f"only {len(eligible)} eligible source episodes remain for {required} Public candidates"
        )
    order = np.random.default_rng(PUBLIC_CANDIDATE_ASSIGNMENT_SEED).permutation(
        len(eligible)
    )
    assigned = [eligible[int(index)] for index in order[:required]]
    requested_rows = sorted({int(row) for row, _, _ in assigned})
    with h5py.File(source, "r", swmr=True) as handle:
        qpos = np.asarray(handle["qpos"][requested_rows], dtype=np.float64)
        control = np.asarray(handle["control"][requested_rows], dtype=np.float64)
    source_values = {
        row: (qpos[index], control[index])
        for index, row in enumerate(requested_rows)
    }

    anchors = development_builder._anchor_ids()
    candidates: list[CubeGraspRuleV4Candidate] = []
    profile_ids: set[str] = set()
    scene_hashes: set[str] = set()
    pair_hashes: set[str] = set()
    anchor_counts = {anchor: 0 for anchor in anchors}
    for local_index, (source_row, source_episode, source_step) in enumerate(assigned):
        catalog_index = PUBLIC_CATALOG_INDEX_OFFSET + local_index
        generator = np.random.default_rng(
            np.random.SeedSequence([PUBLIC_CATALOG_SEED, local_index])
        )
        source_qpos, source_control = source_values[int(source_row)]
        base = CubeGraspRuleCandidate(
            candidate_id=f"cube-carry-v4r1-public-{local_index:06d}",
            split=PUBLIC_SPLIT,
            catalog_index=catalog_index,
            source_row=int(source_row),
            source_episode=int(source_episode),
            source_step=int(source_step),
            simulator_seed=int(generator.integers(0, 2**31 - 1)),
            task_id=1 + local_index % 5,
            qpos=tuple(float(value) for value in source_qpos),
            control=tuple(float(value) for value in source_control),
            cube_color=tuple(float(value) for value in generator.uniform(0.18, 0.92, 3)),
            target_position=(
                float(generator.uniform(0.32, 0.53)),
                float(generator.uniform(-0.24, 0.24)),
                0.02,
            ),
        )
        candidate = make_v4_candidate(base)
        anchor, profile_id, _, _ = development_builder._profile_from_candidate(
            candidate
        )
        expected_anchor = anchors[catalog_index % len(anchors)]
        if anchor != expected_anchor:
            raise RuntimeError("Public catalog anchor rotation mismatch")
        scene_hash = development_builder.scene_template_content_sha256(candidate)
        pair_hash = development_builder.pair_content_sha256(scene_hash, profile_id)
        overlap = {
            "source_episode": int(source_episode) in excluded_episodes,
            "action_profile_id": profile_id in prior_content["action_profile_ids"],
            "scene_template_content_hash": scene_hash
            in prior_content["scene_template_content_hashes"],
            "pair_content_hash": pair_hash in prior_content["pair_content_hashes"],
        }
        if any(overlap.values()):
            raise RuntimeError(f"Public catalog overlaps frozen prior content: {overlap}")
        if profile_id in profile_ids or scene_hash in scene_hashes or pair_hash in pair_hashes:
            raise RuntimeError("Public candidate catalog contains duplicate content")
        profile_ids.add(profile_id)
        scene_hashes.add(scene_hash)
        pair_hashes.add(pair_hash)
        anchor_counts[anchor] += 1
        candidates.append(candidate)

    return candidates, {
        "source_h5": dict(source_identity),
        "eligible_source_episode_count_before_exclusion": len(eligible_before),
        "excluded_source_episode_count": len(excluded_episodes),
        "eligible_source_episode_count_after_exclusion": len(eligible),
        "candidate_pool_count": len(candidates),
        "candidate_assignment_seed": PUBLIC_CANDIDATE_ASSIGNMENT_SEED,
        "catalog_seed": PUBLIC_CATALOG_SEED,
        "profile_seed": PUBLIC_PROFILE_SEED,
        "catalog_index_offset": PUBLIC_CATALOG_INDEX_OFFSET,
        "catalog_action_anchor_counts": anchor_counts,
        "catalog_action_profile_ids_sha256": _sha256_values(profile_ids),
        "catalog_scene_template_hashes_sha256": _sha256_values(scene_hashes),
        "catalog_pair_content_hashes_sha256": _sha256_values(pair_hashes),
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _publish(
    staged: Path, output: Path, *, success_payload: Mapping[str, Any]
) -> dict[str, Any]:
    if output.is_symlink() or not output.is_dir():
        raise FileExistsError(f"reserved Public output is invalid: {output}")
    existing = {path.name for path in output.iterdir()}
    if existing != {GENERATION_STARTED_MARKER}:
        raise RuntimeError(
            "reserved Public output changed before publication: "
            f"{sorted(existing)}"
        )
    staged_identity = _tree_identity(staged)
    for child in sorted(staged.iterdir(), key=lambda value: value.name):
        destination = output / child.name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Public publication target already exists: {destination}")
        metadata = os.lstat(child)
        if stat.S_ISDIR(metadata.st_mode):
            shutil.copytree(child, destination, copy_function=shutil.copy2)
        elif stat.S_ISREG(metadata.st_mode):
            shutil.copy2(child, destination)
        else:
            raise ValueError(f"Public staged tree contains a special node: {child}")
    for entry in staged_identity["files"]:
        destination = output / str(entry["path"])
        observed = file_identity(destination, logical_path=str(entry["path"]))
        if observed != entry:
            raise RuntimeError(f"Public file changed during publication: {entry['path']}")
    published_before_success = _tree_identity(output)
    marker = output / SUCCESS_MARKER
    _write_json_exclusive(
        marker,
        {
            **success_payload,
            "staged_tree": staged_identity,
            "published_tree_before_success_marker": published_before_success,
        },
    )
    return {
        "success_marker": file_identity(
            marker,
            logical_path=portable_contextworld_path(marker),
        ),
        "published_tree": _tree_identity(output),
    }


def build_public_data(
    *,
    source: Path,
    preregistration: Path,
    freeze_receipt: Path,
    output: Path,
    staging_root: Path,
    workers: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    authorization = load_public_authorization(
        preregistration_path=preregistration,
        freeze_receipt_path=freeze_receipt,
        require_public_absent=True,
    )
    expected_output = authorization.public_root
    if output.expanduser().resolve() != expected_output:
        raise ValueError("Public output differs from the preregistered one-use path")
    if int(workers) != DEFAULT_WORKERS or int(jpeg_quality) != DEFAULT_JPEG_QUALITY:
        raise ValueError("Public workers/JPEG quality differ from the frozen recipe")
    staging_root = staging_root.expanduser().resolve()
    if staging_root != DEFAULT_STAGING_ROOT or not staging_root.is_dir():
        raise ValueError("Public build staging root is frozen at /tmp")

    source = source.expanduser().resolve()
    frozen_source = authorization.freeze_receipt["frozen_inputs"]["source_h5"]
    if source != Path(str(frozen_source["path"])).expanduser().resolve():
        raise ValueError("source H5 path differs from the frozen source")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started_path = output / GENERATION_STARTED_MARKER
    started_identity: dict[str, Any] | None = None
    try:
        _write_json_exclusive(
            started_path,
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL,
                "status": "public_generation_attempt_started_one_use_namespace_reserved",
                "generation_attempt": 1,
                "started_at_utc": _utc_now(),
                "preregistration": file_identity(
                    authorization.preregistration_path,
                    logical_path=authorization.preregistration["identity"][
                        "preregistration_path"
                    ],
                ),
                "freeze_receipt": file_identity(
                    authorization.freeze_receipt_path,
                    logical_path=portable_contextworld_path(
                        authorization.freeze_receipt_path
                    ),
                ),
                "output": portable_contextworld_path(output),
                "public_table_opened": False,
                "public_model_read": False,
                "rerun_authorized": False,
            },
        )
        started_identity = file_identity(
            started_path,
            logical_path=portable_contextworld_path(started_path),
        )
    except BaseException as error:
        failure_path = output / GENERATION_FAILURE_MARKER
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_json_exclusive(
                failure_path,
                {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL,
                    "status": "public_generation_reservation_failed_namespace_consumed_no_rerun",
                    "failed_at_utc": _utc_now(),
                    "generation_started": None,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "public_model_read": False,
                    "public_scored": False,
                    "rerun_authorized": False,
                },
            )
        raise
    try:
        source_receipt = _source_identity(source, frozen_source)
        exclusions = _public_exclusion_audit(
            authorization.freeze_receipt,
            freeze_receipt_identity=authorization.freeze_receipt_identity,
        )

        with _public_v4_runtime():
            candidates, catalog = build_public_catalog(
                source,
                source_identity=source_receipt,
                exclusions=exclusions,
            )
            request = {
                "schema_version": 1,
                "protocol_id": PROTOCOL,
                "split": PUBLIC_SPLIT,
                "pair_count": PUBLIC_PAIR_COUNT,
                "workers": DEFAULT_WORKERS,
                "jpeg_quality": DEFAULT_JPEG_QUALITY,
                "output": portable_contextworld_path(output),
                "generation_started": started_identity,
                "catalog": catalog,
                "public_exclusions": {
                    name: exclusions[name]
                    for name in (
                        "excluded_source_episode_count",
                        "excluded_source_episodes_sha256",
                        "prior_content_exclusions",
                    )
                },
                "training_or_checkpoint_selection": False,
                "threshold_or_recipe_changes": False,
            }
            with tempfile.TemporaryDirectory(
                prefix="contextworld-cube-public-v4r1-", dir=staging_root
            ) as temporary:
                local_root = Path(temporary) / output.name
                local_root.mkdir()
                _write_json_exclusive(local_root / "request.json", request)
                report = development_builder.build_split(
                    local_root,
                    split=PUBLIC_SPLIT,
                    pair_count=PUBLIC_PAIR_COUNT,
                    candidates=candidates,
                    quality=DEFAULT_JPEG_QUALITY,
                    workers=DEFAULT_WORKERS,
                    prior_exclusion_audit=exclusions,
                )
                build_report = {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL,
                    "public_generation_attempt": 1,
                    "split": PUBLIC_SPLIT,
                    "pair_count": PUBLIC_PAIR_COUNT,
                    "request": request,
                    "splits": {PUBLIC_SPLIT: report},
                    "cross_split_isolation": {
                        "source_episode_overlap_with_all_prior_content": report[
                            "prior_episode_and_content_exclusion"
                        ]["accepted_overlap"]["source_episode_count"],
                        "action_profile_overlap_with_all_prior_content": report[
                            "prior_episode_and_content_exclusion"
                        ]["accepted_overlap"]["action_profile_id_count"],
                        "scene_template_overlap_with_all_prior_content": report[
                            "prior_episode_and_content_exclusion"
                        ]["accepted_overlap"]["scene_template_content_hash_count"],
                        "pair_content_overlap_with_all_prior_content": report[
                            "prior_episode_and_content_exclusion"
                        ]["accepted_overlap"]["pair_content_hash_count"],
                        "query_pixel_overlap_with_all_prior_content": report[
                            "prior_episode_and_content_exclusion"
                        ]["accepted_overlap"]["query_pixel_hash_count"],
                    },
                    "causal_data_contract": {
                        "all_pairs_passed": report["all_causal_checks_passed"],
                        "fresh_simulator_replay_passed": report[
                            "fresh_simulator_replay"
                        ]["passed"],
                        "maximum_query_physical_gap": report[
                            "maximum_query_physical_gap"
                        ],
                        "maximum_query_simulator_state_gap": report[
                            "maximum_query_simulator_state_gap"
                        ],
                        "maximum_state_installations_after_x0": report[
                            "maximum_state_installations_after_x0"
                        ],
                        "passed": report["passed"],
                    },
                    "public_test": {
                        "generated": True,
                        "opened_for_integrity_validation": True,
                        "read_by_model": False,
                        "scored": False,
                    },
                    "passed": report["passed"],
                }
                _write_json_exclusive(local_root / "build_report.json", build_report)
                manifest = {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL,
                    "split": PUBLIC_SPLIT,
                    "pair_count": PUBLIC_PAIR_COUNT,
                    "files_before_manifest": _tree_identity(local_root),
                }
                _write_json_exclusive(local_root / "manifest.json", manifest)
                if build_report["passed"] is not True:
                    raise RuntimeError("Public data build did not pass its frozen gates")
                publication = _publish(
                    local_root,
                    output,
                    success_payload={
                        "schema_version": 1,
                        "protocol_id": PROTOCOL,
                        "status": "public_data_generated_and_integrity_validated_not_model_read_or_scored",
                        "preregistration": request["preregistration"],
                        "freeze_receipt": request["freeze_receipt"],
                        "generation_started": request["generation_started"],
                        "manifest": file_identity(
                            local_root / "manifest.json",
                            logical_path=portable_contextworld_path(
                                output / "manifest.json"
                            ),
                        ),
                        "build_report": file_identity(
                            local_root / "build_report.json",
                            logical_path=portable_contextworld_path(
                                output / "build_report.json"
                            ),
                        ),
                        "request": file_identity(
                            local_root / "request.json",
                            logical_path=portable_contextworld_path(
                                output / "request.json"
                            ),
                        ),
                        "public_test": {
                            "generated": True,
                            "hashed": True,
                            "opened_for_integrity_validation": True,
                            "read_by_model": False,
                            "scored": False,
                        },
                    },
                )
        return {
            "status": "public_data_generated_not_model_scored",
            "output": str(output),
            "pair_count": PUBLIC_PAIR_COUNT,
            **publication,
        }
    except BaseException as error:
        failure_path = output / GENERATION_FAILURE_MARKER
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_json_exclusive(
                failure_path,
                {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL,
                    "status": "public_generation_failed_namespace_consumed_no_rerun",
                    "failed_at_utc": _utc_now(),
                    "generation_started": started_identity,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "public_model_read": False,
                    "public_scored": False,
                    "rerun_authorized": False,
                    "next_step": "archive and freeze a distinct recovery authorization",
                },
            )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument(
        "--freeze-receipt", type=Path, default=DEFAULT_FREEZE_RECEIPT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_public_data(
        source=args.source,
        preregistration=args.prereg,
        freeze_receipt=args.freeze_receipt,
        output=args.output,
        staging_root=args.staging_root,
        workers=args.workers,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
