#!/usr/bin/env python3
"""Score frozen true-future latent loss for speed extrapolation/multi-step."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_model import file_sha256, state_dict_sha256
from contextworld.evaluation.protocol import (
    frozen_normalizer_process,
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from scripts.eval_tworoom_speed_next_latent import (
    _array_sha256,
    _encode_registry,
    _normalize_actions,
    _preprocess_pixels,
    _register_pixels,
)


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
HORIZONS = (1, 2, 3, 5)


def _load_assets(
    catalog_path: Path,
    *,
    action_standardizer: Any,
    eval_seeds: list[int],
    expected_per_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    conditions = list(catalog["summary"]["history_conditions"])
    speeds = [float(value) for value in catalog["summary"]["reference_speeds"]]
    registry: dict[str, np.ndarray] = {}
    assets = []
    payload_hashes = []
    for bundle in sorted(catalog["bundles"], key=lambda row: row["query_id"]):
        payload_path = resolve_contextworld_path(bundle["payload"], repo_root=ROOT)
        observed_hash = file_sha256(payload_path)
        if observed_hash != bundle["payload_sha256"]:
            raise RuntimeError(f"Payload hash mismatch: {payload_path}")
        payload_hashes.append(observed_hash)
        with np.load(payload_path, allow_pickle=False) as payload:
            query_pixels = np.asarray(payload["query_pixels"], dtype=np.uint8)
            future_actions = np.asarray(payload["future_actions"], dtype=np.float32)
            future_pixels = np.asarray(payload["future_pixels"], dtype=np.uint8)
            targets = np.asarray(payload["future_next_pixels"], dtype=np.uint8)
            if future_actions.shape != (5, 5, 2):
                raise RuntimeError(f"Unexpected future actions: {payload_path}")
            if targets.shape[0] != 5 or future_pixels.shape[0] != 5:
                raise RuntimeError(f"Unexpected future pixels: {payload_path}")
            if not np.array_equal(future_pixels[0], query_pixels):
                raise RuntimeError(f"Future does not start at query: {payload_path}")
            if not np.array_equal(future_pixels[1:], targets[:-1]):
                raise RuntimeError(f"Future continuity failed: {payload_path}")
            if _array_sha256(query_pixels) != bundle["query_pixels_sha256"]:
                raise RuntimeError(f"Query hash mismatch: {payload_path}")
            if _array_sha256(future_actions) != bundle["future_actions_sha256"]:
                raise RuntimeError(f"Future action hash mismatch: {payload_path}")
            for horizon in HORIZONS:
                expected = bundle["target_pixels_sha256_by_horizon"][str(horizon)]
                if _array_sha256(targets[horizon - 1]) != expected:
                    raise RuntimeError(
                        f"Target hash mismatch h={horizon}: {payload_path}"
                    )
            query_key = _register_pixels(registry, query_pixels)[0]
            target_keys = _register_pixels(registry, targets)
            samples = {}
            for condition in conditions:
                prefix = f"context_b2_{condition}"
                context_pixels = np.asarray(
                    payload[f"{prefix}_pixels"], dtype=np.uint8
                )
                context_actions = np.asarray(
                    payload[f"{prefix}_actions"], dtype=np.float32
                )
                context_next = np.asarray(
                    payload[f"{prefix}_next_pixels"], dtype=np.uint8
                )
                if context_pixels.shape[0] != 2 or context_actions.shape != (2, 5, 2):
                    raise RuntimeError(f"Unexpected context shape: {payload_path}")
                if not np.array_equal(context_pixels[1], context_next[0]):
                    raise RuntimeError(f"Context continuity failed: {payload_path}")
                if not np.array_equal(context_next[-1], query_pixels):
                    raise RuntimeError(f"Context does not end at query: {payload_path}")
                input_keys = [*_register_pixels(registry, context_pixels), query_key]
                raw_actions = np.concatenate([context_actions, future_actions], axis=0)
                samples[condition] = {
                    "input_pixel_keys": input_keys,
                    "normalized_actions": _normalize_actions(
                        raw_actions, action_standardizer
                    ),
                    "history_speed": float(
                        bundle["conditions"][condition]["factors"]["agent.speed"]
                    ),
                }
        assets.append(
            {
                "query_id": str(bundle["query_id"]),
                "static_query_id": str(bundle["static_query_id"]),
                "template_id": str(bundle["template"]["template_id"]),
                "reference_speed": float(bundle["query_factors"]["agent.speed"]),
                "matching_condition": str(bundle["matching_condition"]),
                "action_family": str(bundle["query_action_family"]),
                "eval_seed": int(bundle["eval_seed"]),
                "evaluation_index": int(bundle["evaluation_index"]),
                "target_keys": target_keys,
                "samples": samples,
            }
        )
    expected_per_speed = len(eval_seeds) * int(expected_per_seed)
    by_speed_seed = {
        (speed, seed): sum(
            row["reference_speed"] == speed and row["eval_seed"] == seed
            for row in assets
        )
        for speed in speeds
        for seed in eval_seeds
    }
    if set(by_speed_seed.values()) != {int(expected_per_seed)}:
        raise RuntimeError(f"Speed/seed count mismatch: {by_speed_seed}")
    if any(
        len(
            {
                row["static_query_id"]
                for row in assets
                if row["reference_speed"] == speed and row["eval_seed"] == seed
            }
        )
        != expected_per_seed
        for speed in speeds
        for seed in eval_seeds
    ):
        raise RuntimeError("Repeated static query inside deterministic seed cell")
    if any(
        sum(row["reference_speed"] == speed for row in assets) != expected_per_speed
        for speed in speeds
    ):
        raise RuntimeError("Reference-speed count mismatch")
    assignments: dict[str, tuple[int, int]] = {}
    for row in assets:
        static_id = str(row["static_query_id"])
        assignment = (int(row["eval_seed"]), int(row["evaluation_index"]))
        previous = assignments.setdefault(static_id, assignment)
        if previous != assignment:
            raise RuntimeError(
                f"Static query assignment changed by speed: {static_id}"
            )
    disjoint_by_speed = {}
    for speed in speeds:
        seed_sets = {
            seed: {
                str(row["static_query_id"])
                for row in assets
                if row["reference_speed"] == speed
                and row["eval_seed"] == seed
            }
            for seed in eval_seeds
        }
        union = set().union(*seed_sets.values())
        disjoint = len(union) == sum(len(values) for values in seed_sets.values())
        disjoint_by_speed[str(speed)] = bool(
            disjoint and len(union) == expected_per_speed
        )
    if not all(disjoint_by_speed.values()):
        raise RuntimeError(
            f"Eval-seed query partitions overlap: {disjoint_by_speed}"
        )
    if len(set(payload_hashes)) != len(assets):
        raise RuntimeError("Payloads are not unique within the catalog")
    return assets, registry, {
        "catalog": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "bundles": len(assets),
        "conditions": conditions,
        "reference_speeds": speeds,
        "unique_pixels": len(registry),
        "unique_payload_hashes": len(set(payload_hashes)),
        "unique_queries_by_reference_speed_and_eval_seed": {
            f"v{speed:g}/s{seed}": count
            for (speed, seed), count in sorted(by_speed_seed.items())
        },
        "eval_seed_query_partitions_disjoint_by_reference_speed": (
            disjoint_by_speed
        ),
        "static_query_assignments_paired_across_reference_speeds": True,
        "all_eval_seed_queries_are_disjoint": all(disjoint_by_speed.values()),
        "online_environment_calls": 0,
    }


def _samples(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for asset in assets:
        for condition, sample in asset["samples"].items():
            rows.append(
                {
                    "query_id": asset["query_id"],
                    "static_query_id": asset["static_query_id"],
                    "template_id": asset["template_id"],
                    "reference_speed": asset["reference_speed"],
                    "matching_condition": asset["matching_condition"],
                    "action_family": asset["action_family"],
                    "eval_seed": asset["eval_seed"],
                    "evaluation_index": asset["evaluation_index"],
                    "condition": condition,
                    "history_speed": sample["history_speed"],
                    "input_pixel_keys": sample["input_pixel_keys"],
                    "normalized_actions": sample["normalized_actions"],
                    "target_keys": asset["target_keys"],
                }
            )
    return rows


def _rollout(
    model: Any,
    rows: list[dict[str, Any]],
    registry: dict[str, np.ndarray],
    *,
    device: str,
):
    import torch

    pixels = np.stack(
        [
            np.stack([registry[key] for key in row["input_pixel_keys"]])
            for row in rows
        ]
    )
    batch, frames = pixels.shape[:2]
    transformed = _preprocess_pixels(
        pixels.reshape(-1, *pixels.shape[2:]), device
    ).reshape(batch, frames, 3, pixels.shape[2], pixels.shape[3])
    actions = torch.from_numpy(
        np.stack([row["normalized_actions"] for row in rows])
    ).to(device=device, dtype=next(model.parameters()).dtype)
    output = model.rollout(
        {"pixels": transformed[:, None]},
        actions[:, None],
        history_size=3,
    )["predicted_emb"][:, 0]
    predictions = output[:, 3:]
    expected_future = actions.shape[1] - 2
    if predictions.shape[1] != expected_future:
        raise RuntimeError(
            f"Expected {expected_future} predicted futures: {predictions.shape}"
        )
    return predictions


def _score(
    model: Any,
    assets: list[dict[str, Any]],
    registry: dict[str, np.ndarray],
    embeddings: dict[str, Any],
    *,
    device: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    samples = _samples(assets)
    records = []
    maximum_prefix_difference = 0.0
    prefix_checked = False
    with torch.inference_mode():
        for start in range(0, len(samples), int(batch_size)):
            chunk = samples[start : start + int(batch_size)]
            predicted = _rollout(model, chunk, registry, device=device)
            targets = torch.stack(
                [
                    torch.stack([embeddings[key] for key in row["target_keys"]])
                    for row in chunk
                ]
            )
            losses = F.mse_loss(predicted, targets, reduction="none").mean(dim=-1)
            if not prefix_checked:
                audit_chunk = chunk[: min(8, len(chunk))]
                audit_full = _rollout(
                    model, audit_chunk, registry, device=device
                )
                for future_count in (1, 2, 3):
                    truncated = [
                        {
                            **row,
                            "normalized_actions": row["normalized_actions"][
                                : 2 + future_count
                            ],
                        }
                        for row in audit_chunk
                    ]
                    shorter = _rollout(model, truncated, registry, device=device)
                    difference = float(
                        torch.max(
                            torch.abs(
                                shorter[:, :future_count]
                                - audit_full[:, :future_count]
                            )
                        ).item()
                    )
                    maximum_prefix_difference = max(
                        maximum_prefix_difference, difference
                    )
                prefix_checked = True
            for row, row_losses in zip(chunk, losses):
                records.append(
                    {
                        **{
                            key: value
                            for key, value in row.items()
                            if key
                            not in {
                                "input_pixel_keys",
                                "normalized_actions",
                                "target_keys",
                            }
                        },
                        "latent_mse_by_horizon": {
                            str(horizon): float(row_losses[horizon - 1].item())
                            for horizon in HORIZONS
                        },
                    }
                )
    if maximum_prefix_difference > 1e-6:
        raise RuntimeError(
            "Future actions changed a shared rollout prefix: "
            f"{maximum_prefix_difference}"
        )
    return records, {
        "shared_prefix_max_abs_difference": maximum_prefix_difference,
        "passed": maximum_prefix_difference <= 1e-6,
    }


def _paired_rows(
    records: list[dict[str, Any]], horizon: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        key = (
            int(row["eval_seed"]),
            float(row["reference_speed"]),
            int(row["evaluation_index"]),
            str(row["query_id"]),
        )
        grouped[key][str(row["condition"])] = row
    paired = []
    for key, values in grouped.items():
        matching_condition = str(next(iter(values.values()))["matching_condition"])
        if matching_condition not in values or len(values) < 2:
            raise RuntimeError(f"Incomplete history matrix: {key}")
        matching = float(
            values[matching_condition]["latent_mse_by_horizon"][str(horizon)]
        )
        other = [
            float(row["latent_mse_by_horizon"][str(horizon)])
            for condition, row in values.items()
            if condition != matching_condition
        ]
        paired.append(
            {
                "eval_seed": key[0],
                "reference_speed": key[1],
                "evaluation_index": key[2],
                "query_id": key[3],
                "static_query_id": str(next(iter(values.values()))["static_query_id"]),
                "matching_loss": matching,
                "other_history_mean_loss": float(np.mean(other)),
                "matching_history_advantage": float(np.mean(other) - matching),
            }
        )
    return paired


def _ratio_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    matching = float(np.mean([row["matching_loss"] for row in rows]))
    other = float(np.mean([row["other_history_mean_loss"] for row in rows]))
    advantage = other - matching
    return {
        "pairs": len(rows),
        "matching_loss": matching,
        "other_history_mean_loss": other,
        "matching_history_advantage": advantage,
        "relative_loss_reduction": float(advantage / max(other, 1e-12)),
    }


def _bootstrap_ratio(
    rows: list[dict[str, Any]], *, seed: int, samples: int = 10000
) -> dict[str, float | int]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row["static_query_id"])].append(row)
    clusters = sorted(by_cluster)
    values = np.asarray(
        [
            [
                np.mean([row["matching_loss"] for row in by_cluster[key]]),
                np.mean(
                    [row["other_history_mean_loss"] for row in by_cluster[key]]
                ),
            ]
            for key in clusters
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    sampled = values[indices].mean(axis=1)
    ratios = (sampled[:, 1] - sampled[:, 0]) / np.maximum(
        sampled[:, 1], 1e-12
    )
    return {
        "clusters": len(clusters),
        "ci_low": float(np.quantile(ratios, 0.025)),
        "ci_high": float(np.quantile(ratios, 0.975)),
    }


def summarize_records(
    records: list[dict[str, Any]],
    *,
    conditions: list[str],
    eval_seeds: list[int],
    expected_per_seed: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    speeds = sorted({float(row["reference_speed"]) for row in records})
    by_horizon = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        paired = _paired_rows(records, horizon)
        by_speed = {}
        within_pass = True
        strict_pass = True
        for speed_index, speed in enumerate(speeds):
            selected = [row for row in paired if row["reference_speed"] == speed]
            seed_rows = {
                str(seed): _ratio_summary(
                    [row for row in selected if row["eval_seed"] == seed]
                )
                for seed in eval_seeds
            }
            summary = _ratio_summary(selected)
            summary["relative_loss_reduction_ci"] = _bootstrap_ratio(
                selected,
                seed=bootstrap_seed + 100 * horizon_index + speed_index,
            )
            summary["by_eval_seed"] = seed_rows
            summary["all_eval_seed_directions_positive"] = all(
                row["matching_history_advantage"] > 0 for row in seed_rows.values()
            )
            condition_means = {
                condition: float(
                    np.mean(
                        [
                            row["latent_mse_by_horizon"][str(horizon)]
                            for row in records
                            if row["reference_speed"] == speed
                            and row["condition"] == condition
                        ]
                    )
                )
                for condition in conditions
            }
            matching_condition = str(
                next(
                    row["matching_condition"]
                    for row in records
                    if row["reference_speed"] == speed
                )
            )
            summary["matching_condition"] = matching_condition
            summary["condition_mean_losses"] = condition_means
            summary["matching_below_each_other_history"] = all(
                condition_means[matching_condition] < value
                for condition, value in condition_means.items()
                if condition != matching_condition
            )
            speed_pass = (
                summary["matching_history_advantage"] > 0
                and summary["all_eval_seed_directions_positive"]
            )
            within_pass = within_pass and speed_pass
            strict_pass = strict_pass and summary[
                "matching_below_each_other_history"
            ]
            by_speed[str(speed)] = summary
        balanced = float(
            np.mean(
                [row["relative_loss_reduction"] for row in by_speed.values()]
            )
        )
        by_horizon[str(horizon)] = {
            "reference_speed_balanced_relative_loss_reduction": balanced,
            "by_reference_speed": by_speed,
            "formal_within_checkpoint_pass": bool(within_pass),
            "strict_each_alternative_pass": bool(strict_pass),
        }
    counts: dict[tuple[float, str, int], int] = defaultdict(int)
    query_ids: dict[tuple[float, str, int], set[str]] = defaultdict(set)
    static_ids_by_speed_condition: dict[
        tuple[float, str], set[str]
    ] = defaultdict(set)
    for row in records:
        key = (
            float(row["reference_speed"]),
            str(row["condition"]),
            int(row["eval_seed"]),
        )
        counts[key] += 1
        query_ids[key].add(str(row["query_id"]))
        static_ids_by_speed_condition[
            (float(row["reference_speed"]), str(row["condition"]))
        ].add(str(row["static_query_id"]))
    expected_cells = len(speeds) * len(conditions) * len(eval_seeds)
    expected_across_seeds = len(eval_seeds) * int(expected_per_seed)
    independent_across_seeds = (
        len(static_ids_by_speed_condition) == len(speeds) * len(conditions)
        and all(
            len(values) == expected_across_seeds
            for values in static_ids_by_speed_condition.values()
        )
    )
    count_pass = (
        len(counts) == expected_cells
        and set(counts.values()) == {int(expected_per_seed)}
        and all(len(query_ids[key]) == count for key, count in counts.items())
        and independent_across_seeds
    )
    return {
        "by_horizon": by_horizon,
        "count_audit": {
            "condition_trajectories": len(records),
            "horizon_loss_records": len(records) * len(HORIZONS),
            "speed_history_cells": len(speeds) * len(conditions),
            "records_per_cell": len(eval_seeds) * int(expected_per_seed),
            "records_per_eval_seed_cell": int(expected_per_seed),
            "unique_static_queries_per_speed_condition_across_eval_seeds": (
                expected_across_seeds
            ),
            "all_queries_unique_within_seed_cells": all(
                len(query_ids[key]) == count for key, count in counts.items()
            ),
            "eval_seed_query_partitions_are_disjoint": (
                independent_across_seeds
            ),
            "passed": count_pass,
        },
    }


def _find_model(config: dict[str, Any], slug: str) -> tuple[str, dict[str, Any]]:
    for group, models in config["models"].items():
        for model in models:
            if model["slug"] == slug:
                return str(group), dict(model)
    raise KeyError(f"Unknown model: {slug}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = file_sha256(config_path)
    if config.get("status") != (
        "preregistered_before_catalog_generation_and_model_scoring"
    ):
        raise ValueError("v5 config is not frozen before execution")
    group, model_row = _find_model(config, args.model)
    checkpoint = resolve_contextworld_path(model_row["checkpoint"], repo_root=ROOT)
    normalizer = resolve_contextworld_path(
        config["evaluation"]["normalizer"], repo_root=ROOT
    )
    output = resolve_contextworld_path(args.output, repo_root=ROOT)
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, args.stablewm_repo, args.stablewm_ref
    )
    if stable_commit != config["stable_worldmodel"]["expected_ref"]:
        raise RuntimeError(f"StableWM commit mismatch: {stable_commit}")
    build_report_path = resolve_contextworld_path(
        config["artifacts"]["build_report"], repo_root=ROOT
    )
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    if build_report.get("status") != "passed":
        raise RuntimeError("Catalog build report did not pass")
    if build_report["config"]["sha256"] != config_hash:
        raise RuntimeError("Catalog build report/config hash mismatch")
    if build_report["stable_worldmodel"]["commit"] != stable_commit:
        raise RuntimeError("Catalog build report/StableWM commit mismatch")
    process = frozen_normalizer_process(normalizer)
    model = load_pretrained_cost_model(
        checkpoint,
        swm,
        cache_dir=artifact_path("evaluation/model_cache", repo_root=ROOT),
    )
    protocol = infer_model_protocol(model, action_dim=2)
    if protocol != {"action_block": 5, "history_size": 3}:
        raise RuntimeError(f"Unexpected model protocol: {protocol}")
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    setattr(model, "history_size", 3)
    setattr(model, "interpolate_pos_encoding", True)
    before = state_dict_sha256(model)
    eval_seeds = [int(value) for value in config["evaluation"]["eval_seeds"]]
    per_seed = int(
        config["evaluation"]["unique_queries_per_reference_speed_per_seed"]
    )
    tracks = {}
    for track_index, (track_name, track) in enumerate(
        config["data"]["tracks"].items()
    ):
        catalog = resolve_contextworld_path(track["catalog"], repo_root=ROOT)
        observed_catalog_hash = file_sha256(catalog)
        expected_catalog_hash = build_report["tracks"][str(track_name)][
            "catalog_sha256"
        ]
        if observed_catalog_hash != expected_catalog_hash:
            raise RuntimeError(f"Catalog/build report hash mismatch: {catalog}")
        print(f"[track] {track_name} catalog={catalog}", flush=True)
        assets, registry, data_audit = _load_assets(
            catalog,
            action_standardizer=process["action"],
            eval_seeds=eval_seeds,
            expected_per_seed=per_seed,
        )
        embeddings = _encode_registry(
            model,
            registry,
            device=args.device,
            batch_size=args.encode_batch_size,
        )
        records, prefix_audit = _score(
            model,
            assets,
            registry,
            embeddings,
            device=args.device,
            batch_size=args.rollout_batch_size,
        )
        summary = summarize_records(
            records,
            conditions=data_audit["conditions"],
            eval_seeds=eval_seeds,
            expected_per_seed=per_seed,
            bootstrap_seed=args.bootstrap_seed + 1000 * track_index,
        )
        if not summary["count_audit"]["passed"]:
            raise RuntimeError(f"Count audit failed: {track_name}")
        tracks[str(track_name)] = {
            "data_audit": data_audit,
            "autoregressive_prefix_audit": prefix_audit,
            "summary": summary,
            "records": records,
        }
        del embeddings
    after = state_dict_sha256(model)
    if before != after:
        raise RuntimeError("Model weights changed during evaluation")
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "config": {"path": str(config_path), "sha256": config_hash},
        "build_report": {
            "path": str(build_report_path),
            "sha256": file_sha256(build_report_path),
        },
        "model": {
            "group": group,
            "slug": model_row["slug"],
            "training_seed": int(model_row["training_seed"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
        },
        "normalizer": {"path": str(normalizer), "sha256": file_sha256(normalizer)},
        "stable_worldmodel": {"repo": str(stable_repo), "commit": stable_commit},
        "protocol": {
            **protocol,
            "fully_autoregressive": True,
            "teacher_forcing_future_frames": False,
            "target": "frozen_offline_true_future_pixels",
            "target_horizons_action_blocks": list(HORIZONS),
            "target_encoding": "current_checkpoint_frozen_encoder",
            "online_environment_during_scoring": False,
        },
        "frozen_weight_audit": {
            "state_dict_sha256_before": before,
            "state_dict_sha256_after": after,
            "passed": before == after,
        },
        "online_environment_calls": 0,
        "tracks": tracks,
    }
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/benchmark/tworoom_speed_multistep_extrap_v5.yaml",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-seed", type=int, default=2026072114)
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "model": result["model"]["slug"],
                "tracks": {
                    name: {
                        horizon: row["formal_within_checkpoint_pass"]
                        for horizon, row in track["summary"]["by_horizon"].items()
                    }
                    for name, track in result["tracks"].items()
                },
            },
            sort_keys=True,
        )
    )
