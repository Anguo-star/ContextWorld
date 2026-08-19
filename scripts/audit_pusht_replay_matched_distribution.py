#!/usr/bin/env python3
"""Audit Push-T v2 against the preregistered replay distribution contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import wasserstein_distance


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / "stable-worldmodel"
for source_root in (
    CONTEXTWORLD_ROOT,
    STABLE_WORLD_MODEL_ROOT,
    Path(__file__).resolve().parent,
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.evaluation.pusht_hidden_actuation import (  # noqa: E402
    HiddenActuationTemplate,
    action_blocks,
    array_sha256,
)
from contextworld.evaluation.pusht_replay_matched_hidden_actuation import (  # noqa: E402
    replay_candidate_rows,
)
from contextworld.paths import artifact_path  # noqa: E402
from build_pusht_replay_matched_hidden_actuation_h3 import (  # noqa: E402
    DEFAULT_ORIGINAL_DATASET,
    source_episode_partitions,
)


FEATURES = (
    "agent_x",
    "agent_y",
    "block_x",
    "block_y",
    "block_angle_sin",
    "block_angle_cos",
    "agent_block_distance",
    "goal_block_x",
    "goal_block_y",
    "query_action_x",
    "query_action_y",
    "query_action_magnitude",
)
DEFAULT_V1_ROOT = artifact_path("synthesis/pusht_hidden_actuation_h3_v1")
DEFAULT_V2_ROOT = artifact_path(
    "synthesis/pusht_hidden_actuation_replay_matched_h3_v2"
)
DEFAULT_PROTOCOL = CONTEXTWORLD_ROOT / (
    "configs/benchmark/pusht_hidden_actuation_replay_matched_v2.yaml"
)


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def template_features(
    *,
    agent: np.ndarray,
    block: np.ndarray,
    angle: np.ndarray,
    goal_block: np.ndarray,
    query_actions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert state/action populations into the registered 1-D marginals."""

    agent = np.asarray(agent, dtype=np.float64)
    block = np.asarray(block, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    goal_block = np.asarray(goal_block, dtype=np.float64)
    query_actions = np.asarray(query_actions, dtype=np.float64)
    count = agent.shape[0]
    if agent.shape != (count, 2) or block.shape != (count, 2):
        raise ValueError("agent and block must have shape (N, 2)")
    if angle.shape != (count,) or goal_block.shape != (count, 2):
        raise ValueError("angle/goal arrays have incompatible shapes")
    if query_actions.shape != (count, 5, 2):
        raise ValueError("query_actions must have shape (N, 5, 2)")
    magnitudes = np.linalg.norm(query_actions, axis=-1)
    return {
        "agent_x": agent[:, 0],
        "agent_y": agent[:, 1],
        "block_x": block[:, 0],
        "block_y": block[:, 1],
        "block_angle_sin": np.sin(angle),
        "block_angle_cos": np.cos(angle),
        "agent_block_distance": np.linalg.norm(agent - block, axis=1),
        "goal_block_x": goal_block[:, 0],
        "goal_block_y": goal_block[:, 1],
        "query_action_x": query_actions[:, :, 0].reshape(-1),
        "query_action_y": query_actions[:, :, 1].reshape(-1),
        "query_action_magnitude": magnitudes.reshape(-1),
    }


def manifest_features(
    manifest: dict[str, Any],
    *,
    replay_matched: bool,
) -> dict[str, np.ndarray]:
    pairs = manifest["splits"]["train"]["pairs"]
    agents = []
    blocks = []
    angles = []
    goals = []
    queries = []
    for pair in pairs:
        raw = pair["template"]
        agents.append(raw["agent_position"])
        blocks.append(raw["block_position"])
        angles.append(raw["block_angle"])
        goals.append(raw["goal_block_position"])
        if replay_matched:
            queries.append(raw["query_actions"])
        else:
            template = HiddenActuationTemplate(**raw)
            queries.append(action_blocks(template)[2])
    return template_features(
        agent=np.asarray(agents),
        block=np.asarray(blocks),
        angle=np.asarray(angles),
        goal_block=np.asarray(goals),
        query_actions=np.asarray(queries),
    )


def replay_features(
    states: np.ndarray,
    actions: np.ndarray,
    episode_offsets: np.ndarray,
    episode_lengths: np.ndarray,
    candidates: np.ndarray,
) -> dict[str, np.ndarray]:
    candidates = np.asarray(candidates, dtype=np.int64)
    state = states[candidates].astype(np.float64)
    episode_for_row = (
        np.searchsorted(episode_offsets, candidates, side="right") - 1
    )
    final_rows = (
        episode_offsets[episode_for_row]
        + episode_lengths[episode_for_row]
        - 1
    )
    query = np.stack(
        [actions[candidates + step] for step in range(5)],
        axis=1,
    )
    return template_features(
        agent=state[:, :2],
        block=state[:, 2:4],
        angle=state[:, 4],
        goal_block=states[final_rows, 2:4],
        query_actions=query,
    )


def normalized_distances(
    reference: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    result = {}
    for name in FEATURES:
        reference_values = np.asarray(reference[name], dtype=np.float64)
        observed_values = np.asarray(observed[name], dtype=np.float64)
        q25, q75 = np.quantile(reference_values, [0.25, 0.75])
        iqr = float(q75 - q25)
        if not np.isfinite(iqr) or iqr <= 1e-12:
            raise RuntimeError(f"Reference IQR is degenerate for {name}")
        raw = float(
            wasserstein_distance(reference_values, observed_values)
        )
        result[name] = {
            "wasserstein": raw,
            "reference_iqr": iqr,
            "normalized_wasserstein": raw / iqr,
        }
    return result


def distance_summary(
    distances: dict[str, dict[str, float]],
) -> dict[str, float]:
    values = np.asarray(
        [
            distances[name]["normalized_wasserstein"]
            for name in FEATURES
        ],
        dtype=np.float64,
    )
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }


def receipt_audit(
    *,
    manifest: dict[str, Any],
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids_by_row: np.ndarray,
    step_indices: np.ndarray,
    episode_offsets: np.ndarray,
    episode_lengths: np.ndarray,
    candidates_by_split: dict[str, np.ndarray],
    partitions: dict[str, np.ndarray],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    split_reports = {}
    for split in ("train", "validation"):
        pairs = manifest["splits"][split]["pairs"]
        rows = np.asarray(
            [pair["template"]["source_row_index"] for pair in pairs],
            dtype=np.int64,
        )
        rows_unique = len(np.unique(rows)) == len(rows)
        rows_are_candidates = bool(
            np.isin(rows, candidates_by_split[split]).all()
        )
        receipts = []
        for pair in pairs:
            raw = pair["template"]
            row = int(raw["source_row_index"])
            episode_position = int(
                np.searchsorted(episode_offsets, row, side="right") - 1
            )
            final_row = int(
                episode_offsets[episode_position]
                + episode_lengths[episode_position]
                - 1
            )
            receipts.append(
                bool(
                    int(raw["source_episode_index"])
                    == int(episode_ids_by_row[row])
                    and int(raw["source_step_index"])
                    == int(step_indices[row])
                    and np.array_equal(
                        np.asarray(raw["agent_position"], dtype=np.float32),
                        states[row, :2].astype(np.float32),
                    )
                    and np.array_equal(
                        np.asarray(raw["block_position"], dtype=np.float32),
                        states[row, 2:4].astype(np.float32),
                    )
                    and np.float32(raw["block_angle"])
                    == np.float32(states[row, 4])
                    and np.array_equal(
                        np.asarray(
                            raw["goal_block_position"],
                            dtype=np.float32,
                        ),
                        states[final_row, 2:4].astype(np.float32),
                    )
                    and np.array_equal(
                        np.asarray(raw["query_actions"], dtype=np.float32),
                        actions[row : row + 5].astype(np.float32),
                    )
                    and np.array_equal(
                        np.asarray(raw["filler_actions"], dtype=np.float32),
                        actions[row + 5 : row + 10].astype(np.float32),
                    )
                )
            )
        candidate_hash_matches = (
            manifest["splits"][split][
                "source_candidate_rows_sha256"
            ]
            == array_sha256(
                np.sort(candidates_by_split[split]).astype(np.int64)
            )
        )
        partition_hash_matches = (
            manifest["source"]["episode_partition"][split]["sha256"]
            == array_sha256(partitions[split].astype(np.int64))
        )
        split_reports[split] = {
            "pair_count": len(pairs),
            "source_rows_unique": rows_unique,
            "all_source_rows_in_registered_candidate_pool": (
                rows_are_candidates
            ),
            "all_state_goal_and_action_receipts_exact": all(receipts),
            "candidate_pool_hash_matches": candidate_hash_matches,
            "episode_partition_hash_matches": partition_hash_matches,
        }
        checks[f"{split}_source_rows_unique"] = rows_unique
        checks[f"{split}_source_rows_are_candidates"] = (
            rows_are_candidates
        )
        checks[f"{split}_receipts_exact"] = all(receipts)
        checks[f"{split}_candidate_hash"] = candidate_hash_matches
        checks[f"{split}_partition_hash"] = partition_hash_matches

    cross_split = manifest["cross_split_audit"]
    checks["manifest_pair_audits_pass"] = bool(manifest["passed"])
    checks["source_episode_partitions_disjoint"] = (
        len(set(map(int, partitions["train"])) & set(
            map(int, partitions["validation"])
        ))
        == 0
    )
    checks["manifest_cross_split_audit_pass"] = bool(
        cross_split["passed"]
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "splits": split_reports,
        "manifest_cross_split_audit": cross_split,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-dataset",
        type=Path,
        default=DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.original_dataset.expanduser().resolve()
    v1_root = args.v1_root.expanduser().resolve()
    v2_root = args.v2_root.expanduser().resolve()
    protocol = args.protocol.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else v2_root / "distribution_audit.json"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    required = (
        source,
        v1_root / "manifest.json",
        v2_root / "manifest.json",
        protocol,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input(s):\n" + "\n".join(map(str, missing))
        )

    v1_manifest = json.loads((v1_root / "manifest.json").read_text())
    v2_manifest = json.loads((v2_root / "manifest.json").read_text())
    with h5py.File(source, "r", swmr=True) as handle:
        states = handle["state"][:]
        actions = handle["action"][:]
        episode_ids_by_row = handle["episode_idx"][:]
        step_indices = handle["step_idx"][:]
        episode_offsets = handle["ep_offset"][:]
        episode_lengths = handle["ep_len"][:]

    partitions = source_episode_partitions(
        len(episode_offsets),
        seed=int(args.seed),
        validation_fraction=float(args.validation_fraction),
    )
    candidates_by_split = {
        split: replay_candidate_rows(
            states,
            actions,
            episode_offsets,
            episode_lengths,
            partitions[split],
        )
        for split in ("train", "validation")
    }
    reference = replay_features(
        states,
        actions,
        episode_offsets,
        episode_lengths,
        candidates_by_split["train"],
    )
    v1 = manifest_features(v1_manifest, replay_matched=False)
    v2 = manifest_features(v2_manifest, replay_matched=True)
    distances = {
        "v1_narrow": normalized_distances(reference, v1),
        "v2_replay_matched": normalized_distances(reference, v2),
    }
    summaries = {
        name: distance_summary(values)
        for name, values in distances.items()
    }
    v1_median = summaries["v1_narrow"]["median"]
    v2_median = summaries["v2_replay_matched"]["median"]
    median_reduction = 1.0 - v2_median / v1_median
    distribution_checks = {
        "v2_median_at_most_0p25": v2_median <= 0.25,
        "v2_p90_at_most_0p50": (
            summaries["v2_replay_matched"]["p90"] <= 0.50
        ),
        "v2_median_reduction_vs_v1_at_least_0p50": (
            median_reduction >= 0.50
        ),
    }
    receipts = receipt_audit(
        manifest=v2_manifest,
        states=states,
        actions=actions,
        episode_ids_by_row=episode_ids_by_row,
        step_indices=step_indices,
        episode_offsets=episode_offsets,
        episode_lengths=episode_lengths,
        candidates_by_split=candidates_by_split,
        partitions=partitions,
    )
    passed = all(distribution_checks.values()) and receipts["passed"]
    report = {
        "schema_version": 1,
        "status": (
            "passed_preregistered_distribution_and_causal_gates"
            if passed
            else "failed_preregistered_distribution_or_causal_gate"
        ),
        "benchmark": "pusht_hidden_actuation_history3_replay_matched_v2",
        "protocol": {
            "path": str(protocol),
            "sha256": file_sha256(protocol),
        },
        "inputs": {
            "original_replay": str(source),
            "v1_manifest": str(v1_root / "manifest.json"),
            "v1_manifest_sha256": file_sha256(
                v1_root / "manifest.json"
            ),
            "v2_manifest": str(v2_root / "manifest.json"),
            "v2_manifest_sha256": file_sha256(
                v2_root / "manifest.json"
            ),
        },
        "reference_population": {
            "split": "train_source_episode_partition",
            "candidate_count": int(
                len(candidates_by_split["train"])
            ),
            "candidate_rows_sha256": array_sha256(
                np.sort(candidates_by_split["train"]).astype(np.int64)
            ),
        },
        "feature_population_sizes": {
            "reference": {
                name: int(len(reference[name])) for name in FEATURES
            },
            "v1_narrow": {
                name: int(len(v1[name])) for name in FEATURES
            },
            "v2_replay_matched": {
                name: int(len(v2[name])) for name in FEATURES
            },
        },
        "normalized_wasserstein_by_feature": distances,
        "summaries": summaries,
        "median_reduction_vs_v1": median_reduction,
        "distribution_checks": distribution_checks,
        "causal_and_source_receipt_audit": receipts,
        "passed": passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": passed,
                "summaries": summaries,
                "median_reduction_vs_v1": median_reduction,
                "distribution_checks": distribution_checks,
                "receipt_audit_passed": receipts["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
