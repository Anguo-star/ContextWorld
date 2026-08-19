from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.benchmarks.adapters import (
    SpeedICLModelAdapter,
    validate_adapter_protocol,
)
from contextworld.benchmarks.speed_icl_data import (
    DEFAULT_RELEASE_CONFIG,
    HORIZONS,
    SpeedICLEvalDataset,
    load_speed_icl_release,
)
from contextworld.evaluation.icl_model import file_sha256
from contextworld.paths import repository_root


def _mean(values: Iterable[float]) -> float:
    entries = list(values)
    if not entries:
        raise ValueError("Cannot average an empty collection")
    return float(np.mean(entries))


def _loss_summary(records: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[
            (
                float(row["reference_speed"]),
                int(row["eval_seed"]),
                str(row["query_id"]),
            )
        ].append(row)
    by_speed_queries: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for (speed, eval_seed, query_id), rows in grouped.items():
        matching_condition = str(rows[0]["matching_condition"])
        matching = [row for row in rows if row["condition"] == matching_condition]
        if len(matching) != 1 or len(rows) < 2:
            raise RuntimeError(
                f"Incomplete history matrix: {speed} {eval_seed} {query_id}"
            )
        matching_loss = float(
            matching[0]["latent_mse_by_horizon"][str(horizon)]
        )
        other_losses = [
            float(row["latent_mse_by_horizon"][str(horizon)])
            for row in rows
            if row["condition"] != matching_condition
        ]
        other_mean = _mean(other_losses)
        by_speed_queries[speed].append(
            {
                "eval_seed": eval_seed,
                "query_id": query_id,
                "static_query_id": str(rows[0]["static_query_id"]),
                "matching_loss": matching_loss,
                "other_mean_loss": other_mean,
                "matching_beats_other_mean": matching_loss < other_mean,
                "matching_beats_every_other": all(
                    matching_loss < value for value in other_losses
                ),
            }
        )

    by_speed = {}
    for speed, rows in sorted(by_speed_queries.items()):
        matching = _mean(row["matching_loss"] for row in rows)
        other = _mean(row["other_mean_loss"] for row in rows)
        by_seed = {}
        for seed in sorted({int(row["eval_seed"]) for row in rows}):
            selected = [row for row in rows if int(row["eval_seed"]) == seed]
            seed_matching = _mean(row["matching_loss"] for row in selected)
            seed_other = _mean(row["other_mean_loss"] for row in selected)
            by_seed[str(seed)] = {
                "queries": len(selected),
                "matching_loss": seed_matching,
                "other_history_mean_loss": seed_other,
                "matching_history_advantage": seed_other - seed_matching,
                "matching_to_other_loss_ratio": (
                    seed_matching / max(seed_other, 1e-12)
                ),
            }
        condition_means = {
            condition: _mean(
                float(row["latent_mse_by_horizon"][str(horizon)])
                for row in records
                if float(row["reference_speed"]) == speed
                and row["condition"] == condition
            )
            for condition in sorted(
                {
                    str(row["condition"])
                    for row in records
                    if float(row["reference_speed"]) == speed
                }
            )
        }
        matching_condition = str(
            next(
                row["matching_condition"]
                for row in records
                if float(row["reference_speed"]) == speed
            )
        )
        by_speed[str(speed)] = {
            "queries": len(rows),
            "matching_condition": matching_condition,
            "matching_loss": matching,
            "other_history_mean_loss": other,
            "matching_history_advantage": other - matching,
            "matching_to_other_loss_ratio": matching / max(other, 1e-12),
            "relative_loss_reduction": 1.0 - matching / max(other, 1e-12),
            "query_win_rate_vs_other_mean": _mean(
                float(row["matching_beats_other_mean"]) for row in rows
            ),
            "strict_query_win_rate_vs_every_other": _mean(
                float(row["matching_beats_every_other"]) for row in rows
            ),
            "condition_mean_losses": condition_means,
            "matching_below_each_other_history": all(
                condition_means[matching_condition] < value
                for condition, value in condition_means.items()
                if condition != matching_condition
            ),
            "all_eval_seed_directions_positive": all(
                value["matching_history_advantage"] > 0
                for value in by_seed.values()
            ),
            "by_eval_seed": by_seed,
        }
    balanced_ratio = _mean(
        row["matching_to_other_loss_ratio"] for row in by_speed.values()
    )
    return {
        "reference_speed_balanced_matching_to_other_loss_ratio": balanced_ratio,
        "reference_speed_balanced_relative_loss_reduction": 1.0 - balanced_ratio,
        "reference_speed_balanced_query_win_rate_vs_other_mean": _mean(
            row["query_win_rate_vs_other_mean"] for row in by_speed.values()
        ),
        "reference_speed_balanced_strict_query_win_rate_vs_every_other": _mean(
            row["strict_query_win_rate_vs_every_other"]
            for row in by_speed.values()
        ),
        "diagnostic_within_sample_pass": all(
            row["matching_history_advantage"] > 0
            and row["all_eval_seed_directions_positive"]
            for row in by_speed.values()
        ),
        "strict_each_alternative_diagnostic": all(
            row["matching_below_each_other_history"]
            for row in by_speed.values()
        ),
        "by_reference_speed": by_speed,
    }


def _longest_contiguous(passes: dict[str, bool]) -> int:
    longest = 0
    for horizon in HORIZONS:
        if not passes.get(str(horizon), False):
            break
        longest = horizon
    return longest


def _score_track(
    adapter: SpeedICLModelAdapter,
    dataset: SpeedICLEvalDataset,
    *,
    encode_batch_size: int,
    rollout_batch_size: int,
    bundle_batch_size: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    prefix_max_difference = 0.0
    prefix_checked = False
    for bundle_start in range(0, len(dataset), int(bundle_batch_size)):
        bundles = [
            dataset[index]
            for index in range(
                bundle_start,
                min(len(dataset), bundle_start + int(bundle_batch_size)),
            )
        ]
        target_pixels = np.concatenate(
            [bundle.target_pixels for bundle in bundles], axis=0
        )
        target_latents = adapter.encode_pixels(
            target_pixels, batch_size=encode_batch_size
        ).reshape(len(bundles), 5, -1)
        samples = []
        for bundle_index, bundle in enumerate(bundles):
            for condition, history in bundle.histories.items():
                samples.append((bundle_index, bundle, condition, history))
        pixels = np.stack([sample[3].input_pixels for sample in samples])
        actions = np.stack([sample[3].raw_action_blocks for sample in samples])
        predictions = adapter.rollout_latents(
            pixels,
            actions,
            batch_size=rollout_batch_size,
        )
        if not prefix_checked:
            audit_count = min(8, len(samples))
            full = adapter.rollout_latents(
                pixels[:audit_count],
                actions[:audit_count],
                batch_size=audit_count,
            )
            for future_count in (1, 2, 3):
                shorter = adapter.rollout_latents(
                    pixels[:audit_count],
                    actions[:audit_count, : 2 + future_count],
                    batch_size=audit_count,
                )
                difference = float(
                    np.max(
                        np.abs(shorter - full[:, :future_count])
                    )
                )
                prefix_max_difference = max(prefix_max_difference, difference)
            prefix_checked = True
        for prediction, sample in zip(predictions, samples):
            bundle_index, bundle, condition, history = sample
            losses = np.mean(
                np.square(
                    prediction.astype(np.float64)
                    - target_latents[bundle_index].astype(np.float64)
                ),
                axis=-1,
            )
            records.append(
                {
                    "query_id": bundle.query_id,
                    "static_query_id": bundle.static_query_id,
                    "track": bundle.track,
                    "reference_speed": bundle.reference_speed,
                    "matching_condition": bundle.matching_condition,
                    "action_family": bundle.action_family,
                    "eval_seed": bundle.eval_seed,
                    "evaluation_index": bundle.evaluation_index,
                    "condition": condition,
                    "history_speed": history.history_speed,
                    "latent_mse_by_horizon": {
                        str(horizon): float(losses[horizon - 1])
                        for horizon in HORIZONS
                    },
                }
            )
    expected = len(dataset) * len(dataset.conditions)
    if len(records) != expected:
        raise RuntimeError(f"Scored {len(records)} rows, expected {expected}")
    if prefix_max_difference > 1e-6:
        raise RuntimeError(
            f"Future action prefix audit failed: {prefix_max_difference}"
        )
    horizons = {
        str(horizon): _loss_summary(records, horizon) for horizon in HORIZONS
    }
    formal_eligible = dataset.is_full_protocol
    formal_passes = {
        str(horizon): bool(
            formal_eligible
            and horizons[str(horizon)]["diagnostic_within_sample_pass"]
        )
        for horizon in HORIZONS
    }
    for horizon in HORIZONS:
        horizons[str(horizon)]["formal_protocol_eligible"] = formal_eligible
        horizons[str(horizon)]["formal_within_checkpoint_pass"] = (
            formal_passes[str(horizon)] if formal_eligible else None
        )
    return {
        "data": dataset.describe(),
        "condition_trajectories": len(records),
        "horizon_loss_records": len(records) * len(HORIZONS),
        "autoregressive_prefix_audit": {
            "maximum_absolute_difference": prefix_max_difference,
            "passed": prefix_max_difference <= 1e-6,
        },
        "horizons": horizons,
        "longest_contiguous_passing_horizon": (
            _longest_contiguous(formal_passes) if formal_eligible else None
        ),
        "records": records,
    }


def evaluate_speed_icl_model(
    *,
    adapter: SpeedICLModelAdapter,
    model_name: str,
    training_role: str,
    training_seed: int | None,
    release_config: Path | str = DEFAULT_RELEASE_CONFIG,
    repo_root: Path | None = None,
    tracks: list[str] | tuple[str, ...] | None = None,
    eval_seeds: list[int] | tuple[int, ...] | None = None,
    limit_per_reference_speed_per_seed: int | None = None,
    encode_batch_size: int = 64,
    rollout_batch_size: int = 128,
    bundle_batch_size: int = 16,
    include_records: bool = True,
) -> dict[str, Any]:
    """Evaluate one frozen model without requiring any comparison checkpoint."""

    root = (repo_root or repository_root()).resolve()
    release = load_speed_icl_release(release_config)
    selected_tracks = list(tracks or release["scope"]["public_tracks"])
    if len(selected_tracks) != len(set(selected_tracks)):
        raise ValueError(f"Tracks must be unique: {selected_tracks}")
    unknown = set(selected_tracks) - set(release["evaluation"]["tracks"])
    if unknown:
        raise KeyError(f"Unknown tracks: {sorted(unknown)}")
    try:
        protocol = validate_adapter_protocol(
            adapter,
            history_tokens=int(release["scope"]["history_tokens"]),
            action_block_raw_steps=int(
                release["scope"]["action_block_raw_steps"]
            ),
            action_dim=2,
            minimum_future_action_blocks=max(HORIZONS),
            task_name="Speed ICL v1",
        )
    except ValueError as exc:
        raise RuntimeError(
            "Adapter protocol is incompatible with release: "
            f"{adapter.protocol}"
        ) from exc
    before = adapter.frozen_state_hash()
    track_results = {}
    for track in selected_tracks:
        dataset = SpeedICLEvalDataset(
            release=release,
            track=track,
            repo_root=root,
            eval_seeds=eval_seeds,
            limit_per_reference_speed_per_seed=(
                limit_per_reference_speed_per_seed
            ),
        )
        track_results[track] = _score_track(
            adapter,
            dataset,
            encode_batch_size=encode_batch_size,
            rollout_batch_size=rollout_batch_size,
            bundle_batch_size=bundle_batch_size,
        )
        if not include_records:
            del track_results[track]["records"]
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during frozen evaluation")
    full_protocol = (
        set(selected_tracks) == set(release["scope"]["public_tracks"])
        and all(row["data"]["full_protocol"] for row in track_results.values())
    )
    config_path = Path(release["_config_path"])
    return {
        "schema_version": 1,
        "benchmark": release["release_id"],
        "submission_kind": "single_model",
        "status": "passed",
        "full_protocol": full_protocol,
        "formal_claim_level": (
            "descriptive_model_score" if full_protocol else "smoke_only"
        ),
        "release_config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "model": {
            "name": str(model_name),
            "training_role": str(training_role),
            "training_seed": training_seed,
            **adapter.metadata,
        },
        "frozen_weight_audit": {
            "state_hash_before": before,
            "state_hash_after": after,
            "passed": before == after,
        },
        "protocol": {
            "history_tokens": protocol.history_tokens,
            "action_block_raw_steps": protocol.action_block_raw_steps,
            "future_action_blocks": protocol.future_action_blocks,
            "fully_autoregressive": True,
            "teacher_forcing_future_frames": False,
            "target": "adapter_native_encoder_of_frozen_true_future_pixels",
            "online_environment_calls": 0,
        },
        "tracks": track_results,
    }


def _load_result(path: Path | str) -> dict[str, Any]:
    value = Path(path).expanduser().resolve()
    payload = json.loads(value.read_text(encoding="utf-8"))
    if payload.get("submission_kind") != "single_model":
        raise ValueError(f"Not a Speed ICL single-model result: {value}")
    if not payload.get("full_protocol"):
        raise ValueError(f"Method aggregation requires a full result: {value}")
    return {**payload, "_result_path": str(value)}


def aggregate_speed_icl_method(
    *,
    target_results: list[Path | str],
    control_results: list[Path | str],
    method_name: str,
    release_config: Path | str = DEFAULT_RELEASE_CONFIG,
) -> dict[str, Any]:
    """Combine three paired target/control seeds into the formal method score."""

    release = load_speed_icl_release(release_config)
    targets = [_load_result(path) for path in target_results]
    controls = [_load_result(path) for path in control_results]
    if len(targets) != 3 or len(controls) != 3:
        raise ValueError("Formal method score requires 3 target and 3 control results")
    benchmark_ids = {
        row["benchmark"] for row in [*targets, *controls]
    }
    config_hashes = {
        row["release_config"]["sha256"] for row in [*targets, *controls]
    }
    if len(benchmark_ids) != 1 or len(config_hashes) != 1:
        raise ValueError("Submission results use different benchmark releases")
    release_path = Path(release["_config_path"])
    if benchmark_ids != {release["release_id"]} or config_hashes != {
        file_sha256(release_path)
    }:
        raise ValueError("Submission results do not match the selected release")
    if any(
        row["model"]["training_role"] != "multi_speed_target"
        for row in targets
    ):
        raise ValueError("Every target result must use multi_speed_target role")
    if any(
        row["model"]["training_role"] != "single_speed_control"
        for row in controls
    ):
        raise ValueError(
            "Every control result must use single_speed_control role"
        )
    target_by_seed = {int(row["model"]["training_seed"]): row for row in targets}
    control_by_seed = {
        int(row["model"]["training_seed"]): row for row in controls
    }
    if len(target_by_seed) != 3 or set(target_by_seed) != set(control_by_seed):
        raise ValueError("Target/control training seeds must be unique and paired")
    expected_training_seeds = {
        int(value) for value in release["training"]["paired_training_seeds"]
    }
    if set(target_by_seed) != expected_training_seeds:
        raise ValueError(
            "Training seeds do not match the frozen release: "
            f"{sorted(target_by_seed)} != {sorted(expected_training_seeds)}"
        )
    tracks = [str(value) for value in release["scope"]["public_tracks"]]
    if any(set(row["tracks"]) != set(tracks) for row in [*targets, *controls]):
        raise ValueError("Submission results do not contain identical tracks")
    decisions = {}
    summaries = {}
    for track in tracks:
        decisions[track] = {}
        summaries[track] = {}
        for horizon in HORIZONS:
            key = str(horizon)
            target_h = [row["tracks"][track]["horizons"][key] for row in targets]
            control_h = {
                seed: control_by_seed[seed]["tracks"][track]["horizons"][key]
                for seed in sorted(control_by_seed)
            }
            paired = []
            for seed, target in sorted(target_by_seed.items()):
                target_metric = target["tracks"][track]["horizons"][key]
                control_metric = control_h[seed]
                target_reduction = float(
                    target_metric[
                        "reference_speed_balanced_relative_loss_reduction"
                    ]
                )
                control_reduction = float(
                    control_metric[
                        "reference_speed_balanced_relative_loss_reduction"
                    ]
                )
                paired.append(
                    {
                        "training_seed": seed,
                        "target_relative_loss_reduction": target_reduction,
                        "control_relative_loss_reduction": control_reduction,
                        "target_minus_control": target_reduction - control_reduction,
                    }
                )
            within = all(
                row["formal_within_checkpoint_pass"] is True for row in target_h
            )
            attribution = all(
                row["target_minus_control"] > 0 for row in paired
            )
            decisions[track][key] = bool(within and attribution)
            summaries[track][key] = {
                "target_models": 3,
                "control_models": 3,
                "mean_target_loss_ratio": _mean(
                    row[
                        "reference_speed_balanced_matching_to_other_loss_ratio"
                    ]
                    for row in target_h
                ),
                "mean_target_query_win_rate": _mean(
                    row[
                        "reference_speed_balanced_query_win_rate_vs_other_mean"
                    ]
                    for row in target_h
                ),
                "mean_target_strict_query_win_rate": _mean(
                    row[
                        "reference_speed_balanced_strict_query_win_rate_vs_every_other"
                    ]
                    for row in target_h
                ),
                "paired_training_seed_effects": paired,
                "all_target_within_checkpoint_gates_pass": within,
                "all_paired_training_effects_positive": attribution,
            }
    longest = {
        track: _longest_contiguous(values) for track, values in decisions.items()
    }
    core_tracks = [str(value) for value in release["scoring"]["core_claim_tracks"]]
    core_one_step_passed = all(
        decisions.get(track, {}).get("1", False) for track in core_tracks
    )
    extrapolation_tracks = [
        str(value) for value in release["scoring"]["extrapolation_tracks"]
    ]
    bilateral_extrapolation_passed = all(
        decisions.get(track, {}).get("1", False)
        for track in extrapolation_tracks
    )
    return {
        "schema_version": 1,
        "benchmark": next(iter(benchmark_ids)),
        "submission_kind": "complete_method",
        "method_name": str(method_name),
        "status": "passed",
        "formal_claim_level": (
            "training_attributed_speed_icl"
            if core_one_step_passed
            else "speed_icl_not_demonstrated"
        ),
        "target_results": [row["_result_path"] for row in targets],
        "control_results": [row["_result_path"] for row in controls],
        "training_seeds": sorted(target_by_seed),
        "summaries": summaries,
        "decision": {
            "formal_pass_by_track_and_horizon": decisions,
            "longest_contiguous_passing_horizon_by_track": longest,
            "core_in_range_one_step_passed": bool(core_one_step_passed),
            "bilateral_one_step_extrapolation_passed": bool(
                bilateral_extrapolation_passed
            ),
        },
    }


def aggregate_speed_icl_planning(
    *,
    result_paths: list[Path | str],
    release_config: Path | str = DEFAULT_RELEASE_CONFIG,
) -> dict[str, Any]:
    """Aggregate fixed-candidate or CEM cell files without pooling tracks."""

    release = load_speed_icl_release(release_config)
    planning = release["planning"]
    rows = []
    for value in result_paths:
        path = Path(value).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "passed" or not payload.get(
            "count_audit", {}
        ).get("passed"):
            raise ValueError(f"Planning result did not pass: {path}")
        rows.append({**payload, "_result_path": str(path)})
    benchmarks = {row["benchmark"] for row in rows}
    mode_by_benchmark = {
        "tworoom_history3_speed_fixed_candidate_v2": "fixed_candidate",
        "tworoom_history3_speed_closed_loop_v2": "cem",
    }
    if len(benchmarks) != 1 or next(iter(benchmarks)) not in mode_by_benchmark:
        raise ValueError(f"Planning results mix unsupported modes: {benchmarks}")
    mode = mode_by_benchmark[next(iter(benchmarks))]
    model_hashes = {row.get("model", {}).get("sha256") for row in rows}
    if None in model_hashes or len(model_hashes) != 1:
        raise ValueError("Planning results must use one checkpoint SHA256")
    if any(
        not row.get("frozen_weight_audit", {}).get("passed") for row in rows
    ):
        raise ValueError("A planning result failed its frozen-weight audit")
    release_path = Path(release["_config_path"])
    expected_release_hash = file_sha256(release_path)
    expected_normalizer = release["evaluation"]["normalizer_sha256"]
    expected_commit = release["runtime"]["stable_worldmodel"]["expected_ref"]

    def resource_contract_passes(row: dict[str, Any]) -> bool:
        stamp = row.get("contextworld_release", {})
        if stamp != {
            "release_id": release["release_id"],
            "release_config_sha256": expected_release_hash,
            "planning_mode": mode,
            "catalog_sha256": planning["tracks"][str(row["track"])][
                "catalog_sha256"
            ],
        }:
            return False
        if row.get("normalizer", {}).get("sha256") != expected_normalizer:
            return False
        if row.get("stable_worldmodel", {}).get("commit") != expected_commit:
            return False
        protocol = row.get("protocol", {})
        common = bool(
            int(protocol.get("action_block", -1))
            == int(release["scope"]["action_block_raw_steps"])
            and int(protocol.get("history_size", -1))
            == int(release["scope"]["history_tokens"])
        )
        if mode == "fixed_candidate":
            fixed = planning["fixed_candidate"]
            return bool(
                common
                and int(protocol.get("candidates", -1))
                == int(fixed["candidates"])
                and int(protocol.get("horizon_action_blocks", -1))
                == int(fixed["horizon_action_blocks"])
                and protocol.get("same_candidate_bank_across_conditions")
                is True
                and protocol.get("regret_uses_exact_query_dynamics") is True
            )
        cem = planning["cem"]
        return bool(
            common
            and int(protocol.get("cem_samples", -1))
            == int(cem["candidates"])
            and int(protocol.get("cem_iterations", -1))
            == int(cem["iterations"])
            and int(protocol.get("cem_topk", -1)) == int(cem["topk"])
            and int(protocol.get("eval_budget_raw_steps", -1))
            == int(cem["execution_budget_raw_steps"])
            and list(protocol.get("deadline_budgets_raw_steps", []))
            == list(cem["deadline_budgets_raw_steps"])
            and int(protocol.get("horizon_action_blocks", -1))
            == int(cem["horizon_action_blocks"])
            and int(protocol.get("receding_horizon_action_blocks", -1))
            == int(cem["receding_horizon_action_blocks"])
            and protocol.get("same_query_and_cem_seed_across_conditions")
            is True
        )

    resource_checks = [resource_contract_passes(row) for row in rows]
    expected_seeds = {int(value) for value in planning["eval_seeds"]}
    expected_count = int(
        planning["evaluations_per_speed_condition_per_seed"]
    )
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["track"]), float(row["query_speed"]))].append(row)
    summaries = {}
    observed_tracks = {key[0] for key in grouped}
    expected_tracks = set(planning["tracks"])
    full_protocol = observed_tracks == expected_tracks and all(resource_checks)
    for track in sorted(observed_tracks):
        if track not in planning["tracks"]:
            raise ValueError(f"Unknown planning track: {track}")
        summaries[track] = {}
        expected_speeds = {
            float(value) for value in planning["tracks"][track]["speeds"]
        }
        observed_speeds = {speed for name, speed in grouped if name == track}
        full_protocol = full_protocol and observed_speeds == expected_speeds
        for speed in sorted(observed_speeds):
            selected = grouped[(track, speed)]
            observed_seeds = {int(row["eval_seed"]) for row in selected}
            if len(selected) != len(observed_seeds):
                raise ValueError(
                    f"Duplicate planning cell for {track} speed={speed}"
                )
            if mode == "fixed_candidate":
                counts_match = all(
                    int(row["count_audit"].get("records", -1))
                    == expected_count
                    for row in selected
                )
            else:
                counts_match = all(
                    int(
                        row["count_audit"].get(
                            "evaluations_per_condition", -1
                        )
                    )
                    == expected_count
                    for row in selected
                )
            cell_full = (
                observed_seeds == expected_seeds
                and len(selected) == len(expected_seeds)
                and counts_match
            )
            full_protocol = full_protocol and cell_full
            conditions: dict[str, list[dict[str, Any]]] = defaultdict(list)
            if mode == "fixed_candidate":
                for row in selected:
                    for record in row["records"]:
                        for condition, values in record["conditions"].items():
                            conditions[str(condition)].append(values)
                condition_summary = {
                    condition: {
                        "history_speed": float(values[0]["history_speed"]),
                        "history_relation": str(values[0]["history_relation"]),
                        "evaluations": len(values),
                        "mean_exact_query_dynamics_regret_px": _mean(
                            float(row["exact_query_dynamics_regret_px"])
                            for row in values
                        ),
                        "mean_cost_vs_true_distance_spearman": _mean(
                            float(row["cost_vs_true_distance_spearman"])
                            for row in values
                        ),
                    }
                    for condition, values in sorted(conditions.items())
                }
            else:
                for row in selected:
                    for record in row["records"]:
                        conditions[str(record["condition"])].append(record)
                budgets = [
                    int(value)
                    for value in planning["cem"]["deadline_budgets_raw_steps"]
                ]
                condition_summary = {}
                for condition, values in sorted(conditions.items()):
                    condition_summary[condition] = {
                        "history_speed": float(values[0]["history_speed"]),
                        "history_relation": str(values[0]["history_relation"]),
                        "evaluations": len(values),
                        "mean_final_distance_px": _mean(
                            float(row["final_distance"]) for row in values
                        ),
                        "mean_normalized_distance_auc": _mean(
                            float(row["trajectory"]["normalized_distance_auc"])
                            for row in values
                        ),
                        "success_rate_by_execution_budget": {
                            str(budget): _mean(
                                float(
                                    row["trajectory"][
                                        "success_by_budget_raw_steps"
                                    ][str(budget)]
                                )
                                for row in values
                            )
                            for budget in budgets
                        },
                    }
            summaries[track][str(speed)] = {
                "full_protocol_cell": cell_full,
                "eval_seeds": sorted(observed_seeds),
                "conditions": condition_summary,
            }
    return {
        "schema_version": 1,
        "benchmark": release["release_id"],
        "submission_kind": "planning_support",
        "planning_mode": mode,
        "status": "passed",
        "full_protocol": bool(full_protocol),
        "formal_claim_level": (
            "supporting_utility_metrics" if full_protocol else "smoke_only"
        ),
        "model": {"checkpoint_sha256": str(next(iter(model_hashes)))},
        "resource_contract": {
            "passed": all(resource_checks),
            "release_config_sha256": expected_release_hash,
            "normalizer_sha256": expected_normalizer,
            "stable_worldmodel_commit": expected_commit,
        },
        "result_files": [row["_result_path"] for row in rows],
        "tracks": summaries,
    }


__all__ = [
    "aggregate_speed_icl_method",
    "aggregate_speed_icl_planning",
    "evaluate_speed_icl_model",
]
