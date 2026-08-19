#!/usr/bin/env python3
"""Build replay-matched paired Push-T hidden-actuation training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / "stable-worldmodel"
for source_root in (CONTEXTWORLD_ROOT, STABLE_WORLD_MODEL_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.evaluation.pusht_hidden_actuation import (  # noqa: E402
    MODE_SCALES,
    PHYSICS_STATE_COMPONENTS,
    array_sha256,
)
from contextworld.evaluation.pusht_replay_matched_hidden_actuation import (  # noqa: E402
    FREE_SPACE_RESPONSE,
    ReplayMatchedHiddenActuationTemplate,
    fast_replay_matched_pair_audit,
    project_recovery_to_nullspace,
    replay_candidate_rows,
    rotate_action_block_to_direction,
    simulate_replay_matched_hidden_actuation,
    validate_replay_matched_pair,
)
from contextworld.paths import artifact_path  # noqa: E402
from stable_worldmodel.data import LanceWriter  # noqa: E402


PROTOCOL = "pusht_action_strength_history3_replay_matched_strict_v3"
SPLITS = ("train", "validation")
DEFAULT_ORIGINAL_DATASET = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/"
    "lewm-pusht/pusht_expert_train.h5"
)
DEFAULT_OUTPUT = artifact_path(
    "synthesis/pusht_hidden_actuation_replay_matched_h3_strict_v3"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def safe_output_path(path: Path) -> Path:
    result = Path(os.path.abspath(path.expanduser()))
    if result.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {result}"
        )
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def as_action_tuple(value: np.ndarray) -> tuple[tuple[float, float], ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (5, 2):
        raise ValueError(f"Unexpected action block shape {array.shape}")
    return tuple(tuple(map(float, row)) for row in array)


def source_episode_partitions(
    episode_count: int,
    *,
    seed: int,
    validation_fraction: float,
) -> dict[str, np.ndarray]:
    if episode_count <= 1:
        raise ValueError("At least two source episodes are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    generator = np.random.default_rng(seed)
    order = generator.permutation(episode_count)
    validation_count = max(
        1,
        int(round(validation_fraction * episode_count)),
    )
    return {
        "validation": np.sort(order[:validation_count]),
        "train": np.sort(order[validation_count:]),
    }


def make_template(
    *,
    split: str,
    pair_index: int,
    source_row: int,
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids_by_row: np.ndarray,
    step_indices: np.ndarray,
    episode_offsets: np.ndarray,
    episode_lengths: np.ndarray,
    seed: int,
) -> ReplayMatchedHiddenActuationTemplate:
    source_episode = int(episode_ids_by_row[source_row])
    episode_position = int(
        np.searchsorted(episode_offsets, source_row, side="right") - 1
    )
    if episode_position < 0:
        raise RuntimeError(f"Could not resolve source row {source_row}")
    final_row = int(
        episode_offsets[episode_position]
        + episode_lengths[episode_position]
        - 1
    )
    state = states[source_row].astype(np.float64)
    goal = states[final_row].astype(np.float64)
    query = actions[source_row : source_row + 5].astype(np.float64)
    filler = actions[source_row + 5 : source_row + 10].astype(
        np.float64
    )
    away = state[:2] - state[2:4]
    probe = rotate_action_block_to_direction(query, away)
    oriented_reference = rotate_action_block_to_direction(filler, away)
    recovery = project_recovery_to_nullspace(
        probe,
        oriented_reference,
    )
    simulator_seed = int(
        np.random.SeedSequence(
            [
                seed,
                source_row,
                pair_index,
                {"train": 1, "validation": 2}[split],
            ]
        ).generate_state(1)[0]
    )
    return ReplayMatchedHiddenActuationTemplate(
        template_id=f"phrm-{split}-{pair_index:05d}",
        source_row_index=int(source_row),
        source_episode_index=source_episode,
        source_step_index=int(step_indices[source_row]),
        agent_position=tuple(map(float, state[:2])),
        block_position=tuple(map(float, state[2:4])),
        block_angle=float(state[4]),
        goal_agent_position=tuple(map(float, goal[:2])),
        goal_block_position=tuple(map(float, goal[2:4])),
        goal_block_angle=float(goal[4]),
        probe_actions=as_action_tuple(probe),
        recovery_actions=as_action_tuple(recovery),
        query_actions=as_action_tuple(query),
        filler_actions=as_action_tuple(filler),
        simulator_seed=simulator_seed,
    )


def episode_rows(
    rollout: dict[str, Any],
    *,
    split: str,
) -> dict[str, list[Any]]:
    rows = rollout["rows"]
    if rows is None:
        raise RuntimeError("Rendered rollout unexpectedly has no rows")
    result = {key: list(value) for key, value in rows.items()}
    result["split"] = [split] * len(result["pixels"])
    result["synthesis_version"] = ["replay_matched_strict_v3"] * len(
        result["pixels"]
    )
    return result


def failure_key(report: dict[str, Any]) -> str:
    if "exception" in report:
        message = str(report.get("message", ""))
        if "common query state" in message:
            return "query_state_residual"
        return f"exception:{report['exception']}"
    failed = [
        name
        for name, passed in report.get("checks", {}).items()
        if not passed
    ]
    return ",".join(failed) or "unknown"


def strict_causal_chain_audit(
    pair_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the no-state-installation contract for one split."""

    audits = [row["audit"] for row in pair_reports]
    if not audits:
        raise ValueError("Strict causal audit requires at least one pair")
    summary = {
        "pair_count": len(audits),
        "state_installations_after_x0": int(
            sum(
                int(row["state_installations_after_x0"])
                for row in audits
            )
        ),
        "query_simulator_recreated": bool(
            any(row["query_simulator_recreated"] for row in audits)
        ),
        "max_pair_full_state_gap": float(
            max(row["query_physics_max_abs_gap"] for row in audits)
        ),
        "max_pair_query_pixel_difference": int(
            max(row["pair_query_pixel_difference"] for row in audits)
        ),
        "max_pair_query_action_difference": float(
            max(row["pair_query_action_difference"] for row in audits)
        ),
        "min_history_effect": float(
            min(row["history_effect"] for row in audits)
        ),
        "min_true_future_effect": float(
            min(row["true_future_effect"] for row in audits)
        ),
        "full_state_tolerance": float(
            min(row["query_physics_tolerance"] for row in audits)
        ),
        "full_state_dimensions": int(audits[0]["full_state_dimensions"]),
        "full_state_components": audits[0]["full_state_components"],
        "query_pixel_difference_unit": "different_uint8_channel_values",
        "query_action_difference_unit": "maximum_absolute_action_value",
        "history_effect_unit": "agent_position_px_at_x1",
        "true_future_effect_unit": "block_position_px_at_x3",
    }
    summary["passed"] = (
        summary["state_installations_after_x0"] == 0
        and not summary["query_simulator_recreated"]
        and summary["max_pair_full_state_gap"]
        <= summary["full_state_tolerance"]
        and summary["max_pair_query_pixel_difference"] == 0
        and summary["max_pair_query_action_difference"] == 0.0
    )
    return summary


def build_split(
    *,
    root: Path,
    split: str,
    pair_count: int,
    candidates: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids_by_row: np.ndarray,
    step_indices: np.ndarray,
    episode_offsets: np.ndarray,
    episode_lengths: np.ndarray,
    seed: int,
    resolution: int,
    jpeg_quality: int,
    maximum_candidate_attempts: int,
    minimum_future_block_gap_px: float = 2.0,
) -> dict[str, Any]:
    generator = np.random.default_rng(
        np.random.SeedSequence(
            [seed, {"train": 101, "validation": 103}[split]]
        )
    )
    order = generator.permutation(candidates)
    table_path = root / f"{split}.lance"
    pair_reports: list[dict[str, Any]] = []
    query_hashes: set[str] = set()
    selected_rows: set[int] = set()
    failure_counts: dict[str, int] = {}
    candidate_cursor = 0
    attempts_total = 0

    def episodes() -> Iterator[dict[str, list[Any]]]:
        nonlocal candidate_cursor, attempts_total
        for pair_index in range(pair_count):
            accepted = False
            for _ in range(maximum_candidate_attempts):
                if candidate_cursor >= len(order):
                    raise RuntimeError(
                        f"Exhausted {split} candidate pool after "
                        f"{attempts_total} attempts"
                    )
                source_row = int(order[candidate_cursor])
                candidate_cursor += 1
                attempts_total += 1
                attempt_started = time.monotonic()
                if attempts_total % 256 == 0:
                    print(
                        f"  {split}: accepted={len(pair_reports)}/"
                        f"{pair_count}, attempts={attempts_total}, "
                        f"failures={failure_counts}",
                        flush=True,
                    )
                try:
                    template = make_template(
                        split=split,
                        pair_index=pair_index,
                        source_row=source_row,
                        states=states,
                        actions=actions,
                        episode_ids_by_row=episode_ids_by_row,
                        step_indices=step_indices,
                        episode_offsets=episode_offsets,
                        episode_lengths=episode_lengths,
                        seed=seed,
                    )
                    fast = fast_replay_matched_pair_audit(
                        template,
                        minimum_future_block_gap_px=(
                            minimum_future_block_gap_px
                        ),
                    )
                    fast_seconds = time.monotonic() - attempt_started
                    if attempts_total <= 5:
                        print(
                            f"  {split}: candidate row={source_row}, "
                            f"fast_pass={fast['passed']}, "
                            f"fast_seconds={fast_seconds:.3f}",
                            flush=True,
                        )
                    if not fast["passed"]:
                        key = failure_key(fast)
                        failure_counts[key] = (
                            failure_counts.get(key, 0) + 1
                        )
                        continue
                    low = simulate_replay_matched_hidden_actuation(
                        template,
                        mode="low_gain",
                        resolution=resolution,
                    )
                    high = simulate_replay_matched_hidden_actuation(
                        template,
                        mode="high_gain",
                        resolution=resolution,
                    )
                    audit = validate_replay_matched_pair(low, high)
                except (AssertionError, RuntimeError, ValueError) as error:
                    key = f"exception:{type(error).__name__}"
                    failure_counts[key] = failure_counts.get(key, 0) + 1
                    continue
                if not audit["passed"]:
                    key = failure_key(audit)
                    failure_counts[key] = failure_counts.get(key, 0) + 1
                    continue
                query_hash = audit["hashes"]["query_pixels"]
                if query_hash in query_hashes:
                    failure_counts["duplicate_query_hash"] = (
                        failure_counts.get("duplicate_query_hash", 0) + 1
                    )
                    continue
                source_query = actions[
                    source_row : source_row + 5
                ].astype(np.float32)
                source_filler = actions[
                    source_row + 5 : source_row + 10
                ].astype(np.float32)
                if not np.array_equal(
                    low["action_blocks"][2],
                    source_query,
                ) or not np.array_equal(
                    low["action_blocks"][3],
                    source_filler,
                ):
                    raise RuntimeError(
                        "Source action receipt changed during synthesis"
                    )

                query_hashes.add(query_hash)
                selected_rows.add(source_row)
                pair_reports.append(
                    {
                        "template": asdict(template),
                        "fast_audit": fast,
                        "audit": audit,
                    }
                )
                if (
                    len(pair_reports) <= 5
                    or len(pair_reports) % 64 == 0
                    or len(pair_reports) == pair_count
                ):
                    print(
                        f"  {split}: accepted={len(pair_reports)}/"
                        f"{pair_count} after {attempts_total} attempts",
                        flush=True,
                    )
                yield episode_rows(low, split=split)
                yield episode_rows(high, split=split)
                accepted = True
                break
            if not accepted:
                raise RuntimeError(
                    f"Could not synthesize {split} pair {pair_index}; "
                    f"attempts={maximum_candidate_attempts}, "
                    f"failures={failure_counts}"
                )

    with LanceWriter(
        table_path,
        jpeg_quality=jpeg_quality,
        mode="error",
    ) as writer:
        writer.write_episodes(episodes())

    if len(pair_reports) != pair_count:
        raise RuntimeError(
            f"Expected {pair_count} {split} pairs, built "
            f"{len(pair_reports)}"
        )
    selected = np.asarray(sorted(selected_rows), dtype=np.int64)
    return {
        "split": split,
        "pair_count": pair_count,
        "episode_count": 2 * pair_count,
        "raw_rows": 2 * pair_count * 20,
        "table_path": table_path.name,
        "source_candidate_count": int(len(candidates)),
        "source_candidate_rows_sha256": array_sha256(
            np.sort(candidates).astype(np.int64)
        ),
        "selected_source_row_count": int(len(selected)),
        "selected_source_rows_sha256": array_sha256(selected),
        "attempts_total": attempts_total,
        "acceptance_rate": pair_count / attempts_total,
        "failure_counts": failure_counts,
        "query_hash_count": len(query_hashes),
        "query_hashes": sorted(query_hashes),
        "pairs": pair_reports,
        "strict_causal_chain_audit": strict_causal_chain_audit(
            pair_reports
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-dataset",
        type=Path,
        default=DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-pairs", type=int, default=2048)
    parser.add_argument("--validation-pairs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--maximum-candidate-attempts",
        type=int,
        default=64,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.original_dataset.expanduser().resolve()
    output = safe_output_path(args.output)
    required = [source]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input(s):\n" + "\n".join(map(str, missing))
        )
    if args.train_pairs <= 0 or args.validation_pairs <= 0:
        raise ValueError("Pair counts must be positive")

    print("Loading original replay state/action arrays", flush=True)
    with h5py.File(source, "r", swmr=True) as handle:
        states = handle["state"][:]
        actions = handle["action"][:]
        episode_ids_by_row = handle["episode_idx"][:]
        step_indices = handle["step_idx"][:]
        episode_offsets = handle["ep_offset"][:]
        episode_lengths = handle["ep_len"][:]
        source_shapes = {
            key: list(handle[key].shape)
            for key in (
                "state",
                "action",
                "episode_idx",
                "step_idx",
                "ep_offset",
                "ep_len",
            )
        }
    partitions = source_episode_partitions(
        len(episode_offsets),
        seed=int(args.seed),
        validation_fraction=float(args.validation_fraction),
    )
    candidates = {
        split: replay_candidate_rows(
            states,
            actions,
            episode_offsets,
            episode_lengths,
            partitions[split],
        )
        for split in SPLITS
    }
    print(
        "Candidate pools: "
        + ", ".join(
            f"{split}={len(rows)}"
            for split, rows in candidates.items()
        ),
        flush=True,
    )
    requested = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "resolution": int(args.resolution),
        "jpeg_quality": int(args.jpeg_quality),
        "pair_counts": {
            "train": int(args.train_pairs),
            "validation": int(args.validation_pairs),
        },
        "source": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "shapes": source_shapes,
            "episode_partition": {
                split: {
                    "count": int(len(values)),
                    "sha256": array_sha256(
                        values.astype(np.int64)
                    ),
                }
                for split, values in partitions.items()
            },
        },
        "mode_scales": MODE_SCALES,
        "strict_causal_contract": {
            "initial_x0_identical_across_modes": True,
            "state_installations_after_x0": 0,
            "query_simulator_recreated": False,
            "x1_to_x2_reached_by_environment_steps_only": True,
            "query_pixels_identical_across_modes": True,
            "query_action_identical_across_modes": True,
            "full_state_tolerance": 1e-5,
            "full_state_dimensions": len(PHYSICS_STATE_COMPONENTS),
            "full_state_components": list(PHYSICS_STATE_COMPONENTS),
        },
        "free_space_response_sha256": array_sha256(
            FREE_SPACE_RESPONSE
        ),
    }

    with tempfile.TemporaryDirectory(
        prefix="pusht-replay-matched-v2-build-",
        dir="/tmp",
    ) as temporary:
        root = Path(temporary) / output.name
        root.mkdir()
        (root / "request.json").write_text(
            json.dumps(requested, indent=2, sort_keys=True) + "\n"
        )
        reports = {}
        for split, pair_count in (
            ("train", int(args.train_pairs)),
            ("validation", int(args.validation_pairs)),
        ):
            print(
                f"Building {split}: {pair_count} pairs",
                flush=True,
            )
            reports[split] = build_split(
                root=root,
                split=split,
                pair_count=pair_count,
                candidates=candidates[split],
                states=states,
                actions=actions,
                episode_ids_by_row=episode_ids_by_row,
                step_indices=step_indices,
                episode_offsets=episode_offsets,
                episode_lengths=episode_lengths,
                seed=int(args.seed),
                resolution=int(args.resolution),
                jpeg_quality=int(args.jpeg_quality),
                maximum_candidate_attempts=int(
                    args.maximum_candidate_attempts
                ),
            )
        train_episodes = set(map(int, partitions["train"]))
        validation_episodes = set(map(int, partitions["validation"]))
        source_episode_overlap = train_episodes & validation_episodes
        train_query_hashes = set(reports["train"]["query_hashes"])
        validation_query_hashes = set(
            reports["validation"]["query_hashes"]
        )
        cross_split = {
            "source_episode_overlap_count": len(
                source_episode_overlap
            ),
            "query_hash_overlap_counts": {
                "train__validation": len(
                    train_query_hashes & validation_query_hashes
                ),
            },
        }
        cross_split["passed"] = (
            cross_split["source_episode_overlap_count"] == 0
            and not any(
                cross_split["query_hash_overlap_counts"].values()
            )
        )
        if not cross_split["passed"]:
            raise RuntimeError(
                f"Cross-split audit failed: {cross_split}"
            )

        all_pair_reports = [
            pair
            for split in SPLITS
            for pair in reports[split]["pairs"]
        ]
        strict_audit = strict_causal_chain_audit(all_pair_reports)
        if not strict_audit["passed"]:
            raise RuntimeError(
                f"Strict causal-chain audit failed: {strict_audit}"
            )

        manifest = {
            **requested,
            "request_sha256": canonical_json_sha256(requested),
            "splits": reports,
            "cross_split_audit": cross_split,
            "strict_causal_chain_audit": strict_audit,
            "passed": (
                cross_split["passed"]
                and strict_audit["passed"]
                and all(
                    pair["audit"]["passed"]
                    for split in SPLITS
                    for pair in reports[split]["pairs"]
                )
            ),
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        summary = {
            "protocol": PROTOCOL,
            "root": str(output),
            "manifest": manifest_path.name,
            "manifest_sha256": file_sha256(manifest_path),
            "passed": manifest["passed"],
            "pair_counts": requested["pair_counts"],
            "source_candidate_counts": {
                split: int(len(rows))
                for split, rows in candidates.items()
            },
            "acceptance_rates": {
                split: reports[split]["acceptance_rate"]
                for split in SPLITS
            },
            "cross_split_audit": cross_split,
            "strict_causal_chain_audit": strict_audit,
        }
        (root / "build_report.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        shutil.copytree(root, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
