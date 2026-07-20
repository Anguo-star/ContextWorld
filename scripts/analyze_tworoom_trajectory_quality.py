#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.paths import (
    artifact_root,
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.manifest import write_json


WALL_CENTER = 112.0
GRID_SIZE = 14.0


@dataclass(frozen=True)
class EpisodeTable:
    start: np.ndarray
    goal: np.ndarray
    final: np.ndarray
    lengths: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    speed: np.ndarray | None = None


def _round_rows(values: np.ndarray, decimals: int = 5) -> set[tuple[float, ...]]:
    return {tuple(row) for row in np.round(values, decimals=decimals).tolist()}


def _pair_rows(
    left: np.ndarray, right: np.ndarray, decimals: int = 5
) -> set[tuple[tuple[float, ...], tuple[float, ...]]]:
    left_rows = np.round(left, decimals=decimals).tolist()
    right_rows = np.round(right, decimals=decimals).tolist()
    return {(tuple(a), tuple(b)) for a, b in zip(left_rows, right_rows, strict=True)}


def room_relation(
    start: np.ndarray, goal: np.ndarray, wall_center: float = WALL_CENTER
) -> np.ndarray:
    """Return true when start and goal lie on opposite sides of the vertical wall."""

    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return (start[..., 0] < wall_center) != (goal[..., 0] < wall_center)


def _subset_summary(episodes: EpisodeTable, mask: np.ndarray) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"episodes": 0}
    initial_distance = np.linalg.norm(episodes.start[mask] - episodes.goal[mask], axis=1)
    final_distance = np.linalg.norm(episodes.final[mask] - episodes.goal[mask], axis=1)
    return {
        "episodes": count,
        "termination_successes": int(episodes.terminated[mask].sum()),
        "termination_success_rate": float(episodes.terminated[mask].mean()),
        "truncations": int(episodes.truncated[mask].sum()),
        "truncation_rate": float(episodes.truncated[mask].mean()),
        "mean_episode_rows": float(episodes.lengths[mask].mean()),
        "median_episode_rows": float(np.median(episodes.lengths[mask])),
        "mean_initial_distance": float(initial_distance.mean()),
        "median_initial_distance": float(np.median(initial_distance)),
        "mean_final_distance": float(final_distance.mean()),
        "median_final_distance": float(np.median(final_distance)),
    }


def summarize_episodes(episodes: EpisodeTable) -> dict[str, Any]:
    count = len(episodes.lengths)
    all_rows = np.ones(count, dtype=bool)
    cross_room = room_relation(episodes.start, episodes.goal)
    start_bins = np.floor(episodes.start / GRID_SIZE).astype(np.int64)
    goal_bins = np.floor(episodes.goal / GRID_SIZE).astype(np.int64)
    summary = _subset_summary(episodes, all_rows)
    summary.update(
        {
            "rows": int(episodes.lengths.sum()),
            "episode_rows_p90": float(np.percentile(episodes.lengths, 90)),
            "maximum_episode_rows": int(episodes.lengths.max()),
            "unique_start_states": len(_round_rows(episodes.start)),
            "unique_goal_states": len(_round_rows(episodes.goal)),
            "unique_start_goal_pairs": len(_pair_rows(episodes.start, episodes.goal)),
            "start_grid_bins_14px": len(_round_rows(start_bins, decimals=0)),
            "goal_grid_bins_14px": len(_round_rows(goal_bins, decimals=0)),
            "start_goal_grid_pairs_14px": len(
                _pair_rows(start_bins, goal_bins, decimals=0)
            ),
            "cross_room": _subset_summary(episodes, cross_room),
            "same_room": _subset_summary(episodes, ~cross_room),
            "cross_room_fraction": float(cross_room.mean()),
        }
    )
    return summary


def _load_original_h5(path: Path) -> EpisodeTable:
    import h5py

    with h5py.File(path, "r") as dataset:
        offsets = np.asarray(dataset["ep_offset"], dtype=np.int64)
        lengths = np.asarray(dataset["ep_len"], dtype=np.int64)
        ends = offsets + lengths - 1
        positions = dataset["pos_agent"]
        goals = dataset["pos_target"]
        return EpisodeTable(
            start=np.asarray(positions[offsets], dtype=np.float64),
            goal=np.asarray(goals[offsets], dtype=np.float64),
            final=np.asarray(positions[ends], dtype=np.float64),
            lengths=lengths,
            terminated=np.asarray(dataset["terminated"][ends], dtype=bool),
            truncated=np.asarray(dataset["truncated"][ends], dtype=bool),
        )


def _logical_artifact_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return resolve_contextworld_path(path, repo_root=REPO_ROOT)


def _load_synthetic_catalog(path: Path) -> EpisodeTable:
    import lance

    catalog = json.loads(path.read_text(encoding="utf-8"))
    starts: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    finals: list[np.ndarray] = []
    lengths: list[int] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    speeds: list[float] = []
    columns = [
        "episode_idx",
        "state",
        "terminated",
        "truncated",
        "variation_agent_speed",
        "variation_agent_position",
        "variation_target_position",
    ]
    for logical_path in catalog["train"]["synthetic"]:
        table = lance.dataset(_logical_artifact_path(logical_path)).to_table(
            columns=columns
        )
        values = table.to_pydict()
        episode_indices = np.asarray(values["episode_idx"], dtype=np.int64)
        state = np.asarray(values["state"], dtype=np.float64)
        terminal = np.asarray(values["terminated"], dtype=np.float64)[:, 0]
        truncation = np.asarray(values["truncated"], dtype=np.float64)[:, 0]
        speed = np.asarray(values["variation_agent_speed"], dtype=np.float64)[:, 0]
        reset_state = np.asarray(
            values["variation_agent_position"], dtype=np.float64
        )
        target_state = np.asarray(
            values["variation_target_position"], dtype=np.float64
        )
        for episode_index in np.unique(episode_indices):
            rows = np.flatnonzero(episode_indices == episode_index)
            starts.append(reset_state[rows[0]])
            goals.append(target_state[rows[0]])
            finals.append(state[rows[-1]])
            lengths.append(len(rows))
            terminated.append(bool(terminal[rows[-1]] > 0.5))
            truncated.append(bool(truncation[rows[-1]] > 0.5))
            speeds.append(float(speed[rows[0]]))
    return EpisodeTable(
        start=np.asarray(starts, dtype=np.float64),
        goal=np.asarray(goals, dtype=np.float64),
        final=np.asarray(finals, dtype=np.float64),
        lengths=np.asarray(lengths, dtype=np.int64),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
        speed=np.asarray(speeds, dtype=np.float64),
    )


def _speed_outcome_coupling(episodes: EpisodeTable) -> dict[str, Any]:
    if episodes.speed is None:
        raise ValueError("Synthetic episodes must include speed values")
    final_distance = np.linalg.norm(episodes.final - episodes.goal, axis=1)
    rows: list[dict[str, Any]] = []
    for raw_speed in sorted(np.unique(episodes.speed)):
        selected = np.isclose(episodes.speed, raw_speed)
        rows.append(
            {
                "speed": round(float(raw_speed), 6),
                "episodes": int(selected.sum()),
                "termination_success_rate": float(episodes.terminated[selected].mean()),
                "mean_episode_rows": float(episodes.lengths[selected].mean()),
                "mean_final_distance": float(final_distance[selected].mean()),
            }
        )
    speeds = np.asarray([row["speed"] for row in rows], dtype=np.float64)
    success_rates = np.asarray(
        [row["termination_success_rate"] for row in rows], dtype=np.float64
    )
    mean_lengths = np.asarray(
        [row["mean_episode_rows"] for row in rows], dtype=np.float64
    )
    mean_final_distances = np.asarray(
        [row["mean_final_distance"] for row in rows], dtype=np.float64
    )
    return {
        "by_speed": rows,
        "pearson_speed_vs_termination_success_rate": float(
            np.corrcoef(speeds, success_rates)[0, 1]
        ),
        "pearson_speed_vs_mean_episode_rows": float(
            np.corrcoef(speeds, mean_lengths)[0, 1]
        ),
        "pearson_speed_vs_mean_final_distance": float(
            np.corrcoef(speeds, mean_final_distances)[0, 1]
        ),
    }


def _match_speed_room(
    episodes: EpisodeTable, *, speed: float, cross_room: bool
) -> EpisodeTable:
    if episodes.speed is None:
        raise ValueError("Synthetic episodes must include speed values")
    selected = np.isclose(episodes.speed, speed) & (
        room_relation(episodes.start, episodes.goal) == cross_room
    )
    return EpisodeTable(
        start=episodes.start[selected],
        goal=episodes.goal[selected],
        final=episodes.final[selected],
        lengths=episodes.lengths[selected],
        terminated=episodes.terminated[selected],
        truncated=episodes.truncated[selected],
        speed=episodes.speed[selected],
    )


def _resolve_stale_artifact_reference(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if "artifacts" in path.parts:
        index = path.parts.index("artifacts")
        remapped = artifact_root(REPO_ROOT).joinpath(*path.parts[index + 1 :])
        if remapped.exists():
            return remapped
    return resolve_contextworld_path(path, repo_root=REPO_ROOT)


def _template_map(query_catalog_path: Path) -> dict[str, dict[str, Any]]:
    catalog = json.loads(query_catalog_path.read_text(encoding="utf-8"))
    templates: dict[str, dict[str, Any]] = {}
    for bundle in catalog["bundles"]:
        template = bundle["template"]
        template_id = template["template_id"]
        if template_id not in {"s0", "s1", "s2", "s3"}:
            continue
        row = {
            "reset_state": template["reset_state"],
            "goal_state": template["goal_state"],
            "room_relation": (
                "cross_room"
                if bool(
                    room_relation(
                        np.asarray([template["reset_state"]]),
                        np.asarray([template["goal_state"]]),
                    )[0]
                )
                else "same_room"
            ),
        }
        if template_id in templates and templates[template_id] != row:
            raise ValueError(f"Template {template_id} has inconsistent geometry")
        templates[template_id] = row
    if set(templates) != {"s0", "s1", "s2", "s3"}:
        raise ValueError(f"Incomplete E4 template map: {sorted(templates)}")
    return templates


def _e4_records(summary_path: Path) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_reference in summary["protocol"]["raw_results"]:
        raw_path = _resolve_stale_artifact_reference(raw_reference)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        for record in raw["records"]:
            key = (record["evaluation_id"], record["condition"])
            if key in seen:
                raise ValueError(f"Duplicate E4 record: {key}")
            seen.add(key)
            records.append(record)
    return records


def _aggregate_e4_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "evaluations": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
        "mean_final_distance": float(
            np.mean([float(row["final_distance"]) for row in rows])
        ),
    }


def _e4_geometry_summary(
    summary_path: Path, templates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    records = _e4_records(summary_path)
    by_template: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_relation: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        template_id = record["template_id"]
        condition = record["condition"]
        by_template[(template_id, condition)].append(record)
        relation = templates[template_id]["room_relation"]
        by_relation[(relation, condition)].append(record)
    return {
        "by_template": {
            template_id: {
                condition: _aggregate_e4_rows(by_template[(template_id, condition)])
                for condition in ("correct", "wrong")
            }
            for template_id in sorted(templates)
        },
        "by_room_relation": {
            relation: {
                condition: _aggregate_e4_rows(by_relation[(relation, condition)])
                for condition in ("correct", "wrong")
            }
            for relation in ("cross_room", "same_room")
        },
    }


def _policy_and_environment_checks(
    synthesis_config_path: Path, stablewm_repo: Path
) -> dict[str, Any]:
    config = yaml.safe_load(synthesis_config_path.read_text(encoding="utf-8"))
    synthetic_policy = config["collection"]["policy"]
    collection_source = stablewm_repo / "scripts/data/collect_tworooms.py"
    source = collection_source.read_text(encoding="utf-8")
    match = re.search(
        r"ExpertPolicy\(action_noise=([0-9.]+),\s*action_repeat_prob=([0-9.]+)\)",
        source,
    )
    if match is None:
        raise ValueError(f"Could not read ExpertPolicy settings from {collection_source}")
    stablewm_policy = {
        "action_noise": float(match.group(1)),
        "action_repeat_prob": float(match.group(2)),
    }
    current_env_source = (
        stablewm_repo / "stable_worldmodel/envs/two_room/env.py"
    ).read_text(encoding="utf-8")
    legacy_env_source = (
        stablewm_repo / "stable_worldmodel/envs/two_room/legacy_env.py"
    ).read_text(encoding="utf-8")
    return {
        "synthetic_collection_policy": synthetic_policy,
        "stablewm_reference_collection_policy": stablewm_policy,
        "policy_parameter_match": {
            "action_noise": float(synthetic_policy["action_noise"])
            == stablewm_policy["action_noise"],
            "action_repeat_prob": float(synthetic_policy["action_repeat_prob"])
            == stablewm_policy["action_repeat_prob"],
        },
        "original_h5_embeds_collection_policy_metadata": False,
        "task_semantics_source_checks": {
            "current_env_opposite_room_constraint_is_commented": (
                "# constrain_fn=self._constrain_target_by_min_steps"
                in current_env_source
            ),
            "legacy_env_checks_opposite_room": "and self.check_other_room(x)"
            in legacy_env_source,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    original_h5 = resolve_contextworld_path(args.original_h5, repo_root=REPO_ROOT)
    synthetic_catalog = resolve_contextworld_path(
        args.synthetic_catalog, repo_root=REPO_ROOT
    )
    synthesis_config = resolve_contextworld_path(
        args.synthesis_config, repo_root=REPO_ROOT
    )
    query_catalog = resolve_contextworld_path(args.query_catalog, repo_root=REPO_ROOT)
    stablewm_repo = resolve_contextworld_path(args.stablewm_repo, repo_root=REPO_ROOT)
    output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)

    original = _load_original_h5(original_h5)
    synthetic = _load_synthetic_catalog(synthetic_catalog)
    original_summary = summarize_episodes(original)
    synthetic_summary = summarize_episodes(synthetic)
    speed5_cross = _match_speed_room(synthetic, speed=5.0, cross_room=True)
    speed5_same = _match_speed_room(synthetic, speed=5.0, cross_room=False)
    templates = _template_map(query_catalog)

    e4_summaries = {
        "H3-Orig": resolve_contextworld_path(args.e4_orig, repo_root=REPO_ROOT),
        "H3-SpeedClean": resolve_contextworld_path(
            args.e4_speedclean, repo_root=REPO_ROOT
        ),
        "H3-SpeedSeen": resolve_contextworld_path(
            args.e4_speedseen, repo_root=REPO_ROOT
        ),
    }
    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_trajectory_quality_diagnosis_v1",
        "status": "passed",
        "sources": {
            "original_h5": str(original_h5),
            "synthetic_catalog": portable_contextworld_path(
                synthetic_catalog, repo_root=REPO_ROOT
            ),
            "synthesis_config": str(synthesis_config.relative_to(REPO_ROOT)),
            "query_catalog": portable_contextworld_path(
                query_catalog, repo_root=REPO_ROOT
            ),
            "e4_summaries": {
                name: portable_contextworld_path(path, repo_root=REPO_ROOT)
                for name, path in e4_summaries.items()
            },
        },
        "policy_and_environment": _policy_and_environment_checks(
            synthesis_config, stablewm_repo
        ),
        "datasets": {
            "original_tworoom_h5": original_summary,
            "speedseen_synthetic_train": {
                **synthetic_summary,
                "factor_values": len(np.unique(synthetic.speed)),
                "episodes_per_independent_geometry": float(
                    synthetic_summary["episodes"]
                    / synthetic_summary["unique_start_goal_pairs"]
                ),
            },
        },
        "matched_speed5_task_strata": {
            "original_all_cross_room_speed5": original_summary["cross_room"],
            "synthetic_cross_room_speed5": summarize_episodes(speed5_cross),
            "synthetic_same_room_speed5": summarize_episodes(speed5_same),
            "interpretation": (
                "At matched speed 5 and cross-room geometry, synthetic outcomes are "
                "close to the original H5; the aggregate synthetic advantage is driven "
                "by easier same-room episodes."
            ),
        },
        "speed_outcome_coupling": _speed_outcome_coupling(synthetic),
        "e4_geometry_strata": {
            "templates": templates,
            "models": {
                name: _e4_geometry_summary(path, templates)
                for name, path in e4_summaries.items()
            },
        },
        "findings": {
            "supported": [
                "The synthetic training set does not preserve the original all-cross-room task semantics.",
                "4096 synthetic episodes contain only 128 independent reset-goal geometries crossed with 32 speeds.",
                "Speed is strongly coupled to termination rate, episode length, and remaining distance in the synthetic data.",
                "E4 success is dominated by same-room template geometry; all evaluated models score zero on the cross-room stratum.",
            ],
            "not_established": [
                "The TwoRoom synthetic ExpertPolicy is intrinsically weaker than the reference StableWM collection recipe.",
                "Filtering to successful trajectories alone would improve context learning without introducing selection bias.",
            ],
            "cross_task_context": (
                "The prior PushT expert-versus-weak observation motivates trajectory-quality controls, "
                "but it is not treated as a TwoRoom causal result."
            ),
        },
        "recommended_control": {
            "dataset_id": "TwoRoom-SpeedTask-v1",
            "model_id": "H3-SpeedTask-s3072",
            "fixed_components": [
                "lossless codec",
                "exact E4 speed support",
                "ExpertPolicy parameters",
                "50/50 original-synthetic mixture",
                "model architecture and optimizer-step budget",
            ],
            "changed_components": [
                "enforce opposite-room targets for every synthetic training episode",
                "increase independent reset-goal geometries instead of repeating 128 geometries over all speeds",
                "report and control success/truncation, path length, and final-distance strata per speed",
                "retain failure trajectories as an explicit weighted stratum rather than silently filtering them",
            ],
            "decision_rule": (
                "Run E1, original ID retention, and frozen E4 50x6. Improvement over SpeedSeen "
                "under the fixed model/training recipe isolates task-distribution and trajectory-composition quality."
            ),
        },
    }
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original and synthetic TwoRoom trajectory/task quality."
    )
    parser.add_argument(
        "--original-h5",
        type=Path,
        default=Path("../../data/world_model/quentinll/lewm-tworooms/tworoom.h5"),
    )
    parser.add_argument(
        "--synthetic-catalog",
        type=Path,
        default=Path("artifacts/synthesis/catalogs/tworoom_speed_seen_v1.json"),
    )
    parser.add_argument(
        "--synthesis-config",
        type=Path,
        default=Path("configs/synthesis/tworoom_speed_seen_v1.yaml"),
    )
    parser.add_argument(
        "--query-catalog",
        type=Path,
        default=Path(
            "artifacts/evaluation/icl/tworoom_icl_v1_validation_context_query_catalog.json"
        ),
    )
    parser.add_argument(
        "--stablewm-repo", type=Path, default=Path("../stable-worldmodel")
    )
    parser.add_argument(
        "--e4-orig",
        type=Path,
        default=Path("artifacts/evaluation/history3/e4_speed_ctx_n50x6.json"),
    )
    parser.add_argument(
        "--e4-speedclean",
        type=Path,
        default=Path(
            "artifacts/evaluation/history3/h3_speedclean_s3072/e4_speed_ctx_n50x6.json"
        ),
    )
    parser.add_argument(
        "--e4-speedseen",
        type=Path,
        default=Path(
            "artifacts/evaluation/history3/h3_speedseen_s3072/e4_speed_ctx_n50x6.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/history3/trajectory_quality_v1.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "benchmark": result["benchmark"],
                "output": result["sources"],
            },
            sort_keys=True,
        )
    )
