#!/usr/bin/env python3
"""Score History-3 speed ICL against frozen offline next frames."""

from __future__ import annotations

import argparse
import hashlib
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

from contextworld.evaluation.icl_model import (
    file_sha256,
    state_dict_sha256,
)
from contextworld.evaluation.protocol import (
    frozen_normalizer_process,
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
CONDITIONS = ("history_low", "history_mid", "history_high")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _pixel_key(value: np.ndarray) -> str:
    return _array_sha256(np.asarray(value, dtype=np.uint8))


def _register_pixels(
    registry: dict[str, np.ndarray], value: np.ndarray
) -> list[str]:
    pixels = np.asarray(value, dtype=np.uint8)
    if pixels.ndim == 3:
        pixels = pixels[None]
    keys = []
    for frame in pixels:
        key = _pixel_key(frame)
        registry.setdefault(key, frame.copy())
        keys.append(key)
    return keys


def _normalize_actions(value: np.ndarray, standardizer: Any) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[1:] != (5, 2):
        raise ValueError(f"Expected (tokens,5,2) actions, got {actions.shape}")
    normalized = standardizer.transform(actions.reshape(-1, 2))
    return np.asarray(normalized, dtype=np.float32).reshape(len(actions), 10)


def _preprocess_pixels(value: np.ndarray, device: str):
    import torch

    pixels = torch.from_numpy(np.asarray(value, dtype=np.uint8)).to(device)
    pixels = pixels.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(
        (0.485, 0.456, 0.406), device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        (0.229, 0.224, 0.225), device=device
    ).view(1, 3, 1, 1)
    return (pixels - mean) / std


def _load_track_assets(
    catalog_path: Path,
    *,
    action_standardizer: Any,
    expected_queries_per_speed: int,
    eval_seeds: list[int],
    expected_queries_per_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    registry: dict[str, np.ndarray] = {}
    assets = []
    payload_hashes = []
    target_hashes = []
    for bundle in sorted(catalog["bundles"], key=lambda row: row["query_id"]):
        payload_path = resolve_contextworld_path(
            bundle["payload"], repo_root=ROOT
        )
        observed_payload_hash = file_sha256(payload_path)
        if observed_payload_hash != bundle["payload_sha256"]:
            raise RuntimeError(f"Payload hash mismatch: {payload_path}")
        payload_hashes.append(observed_payload_hash)
        with np.load(payload_path, allow_pickle=False) as payload:
            query_pixels = np.asarray(payload["query_pixels"], dtype=np.uint8)
            target_pixels = np.asarray(payload["target_pixels"], dtype=np.uint8)
            if _array_sha256(query_pixels) != bundle["query_pixels_sha256"]:
                raise RuntimeError(f"Query pixel hash mismatch: {payload_path}")
            if _array_sha256(target_pixels) != bundle["target_pixels_sha256"]:
                raise RuntimeError(f"Target pixel hash mismatch: {payload_path}")
            query_key = _register_pixels(registry, query_pixels)[0]
            target_key = _register_pixels(registry, target_pixels)[0]
            target_hashes.append(bundle["target_pixels_sha256"])
            query_action = np.asarray(
                payload["query_action"], dtype=np.float32
            )[None]
            samples = {}
            for condition in CONDITIONS:
                if condition not in bundle["conditions"]:
                    raise RuntimeError(
                        f"Missing {condition} in {bundle['query_id']}"
                    )
                prefix = f"context_b2_{condition}"
                context_pixels = np.asarray(
                    payload[f"{prefix}_pixels"], dtype=np.uint8
                )
                context_next = np.asarray(
                    payload[f"{prefix}_next_pixels"], dtype=np.uint8
                )
                if not np.array_equal(context_next[-1], query_pixels):
                    raise RuntimeError(
                        f"Context does not end at query: "
                        f"{bundle['query_id']} {condition}"
                    )
                context_keys = _register_pixels(registry, context_pixels)
                context_actions = np.asarray(
                    payload[f"{prefix}_actions"], dtype=np.float32
                )
                raw_actions = np.concatenate(
                    [context_actions, query_action], axis=0
                )
                samples[condition] = {
                    "pixel_keys": [*context_keys, query_key],
                    "actions": _normalize_actions(
                        raw_actions, action_standardizer
                    ),
                    "history_speed": float(
                        bundle["conditions"][condition]["factors"][
                            "agent.speed"
                        ]
                    ),
                }
        assets.append(
            {
                "query_id": str(bundle["query_id"]),
                "static_query_id": str(bundle["static_query_id"]),
                "template_id": str(bundle["template"]["template_id"]),
                "reference_speed": float(
                    bundle["query_factors"]["agent.speed"]
                ),
                "matching_condition": str(bundle["same_speed_condition"]),
                "eval_seed": int(bundle["eval_seed"]),
                "evaluation_index": int(bundle["evaluation_index"]),
                "target_key": target_key,
                "samples": samples,
            }
        )
    speeds = sorted({asset["reference_speed"] for asset in assets})
    if len(speeds) != 3:
        raise RuntimeError(f"Expected three reference speeds: {speeds}")
    by_speed = {
        speed: sum(asset["reference_speed"] == speed for asset in assets)
        for speed in speeds
    }
    if set(by_speed.values()) != {int(expected_queries_per_speed)}:
        raise RuntimeError(
            f"Expected {expected_queries_per_speed} assets per speed: "
            f"{by_speed}"
        )
    by_speed_seed = {
        (speed, seed): sum(
            asset["reference_speed"] == speed
            and asset["eval_seed"] == seed
            for asset in assets
        )
        for speed in speeds
        for seed in eval_seeds
    }
    if set(by_speed_seed.values()) != {int(expected_queries_per_seed)}:
        raise RuntimeError(
            "Unexpected unique query count by speed/eval seed: "
            f"{by_speed_seed}"
        )
    unique_static_by_speed_seed = {
        key: len(
            {
                asset["static_query_id"]
                for asset in assets
                if asset["reference_speed"] == key[0]
                and asset["eval_seed"] == key[1]
            }
        )
        for key in by_speed_seed
    }
    if unique_static_by_speed_seed != by_speed_seed:
        raise RuntimeError("Duplicate static query inside an eval-seed cell")
    audit = {
        "catalog": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "bundles": len(assets),
        "reference_speeds": speeds,
        "bundles_by_reference_speed": {
            str(speed): count for speed, count in by_speed.items()
        },
        "unique_queries_by_reference_speed_and_eval_seed": {
            f"v{speed:g}/s{seed}": count
            for (speed, seed), count in sorted(by_speed_seed.items())
        },
        "all_eval_seed_queries_are_disjoint": True,
        "unique_pixels": len(registry),
        "unique_payload_hashes": len(set(payload_hashes)),
        "unique_target_hashes": len(set(target_hashes)),
        "online_environment_calls": 0,
    }
    return assets, registry, audit


def _encode_registry(
    model: Any,
    registry: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    keys = list(registry)
    encoded = {}
    with torch.inference_mode():
        for start in range(0, len(keys), batch_size):
            chunk = keys[start : start + batch_size]
            tensor = _preprocess_pixels(
                np.stack([registry[key] for key in chunk]), device
            ).unsqueeze(1)
            values = model.encode({"pixels": tensor})["emb"][:, 0]
            for key, value in zip(chunk, values):
                encoded[key] = value.detach().clone()
    return encoded


def _score_unique_assets(
    model: Any,
    assets: list[dict[str, Any]],
    embeddings: dict[str, Any],
    *,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    samples = []
    for asset in assets:
        for condition in CONDITIONS:
            sample = asset["samples"][condition]
            samples.append(
                {
                    "query_id": asset["query_id"],
                    "static_query_id": asset["static_query_id"],
                    "template_id": asset["template_id"],
                    "reference_speed": asset["reference_speed"],
                    "matching_condition": asset["matching_condition"],
                    "eval_seed": asset["eval_seed"],
                    "evaluation_index": asset["evaluation_index"],
                    "condition": condition,
                    "history_speed": sample["history_speed"],
                    "pixel_keys": sample["pixel_keys"],
                    "target_key": asset["target_key"],
                    "actions": sample["actions"],
                }
            )
    rows = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            pixels = torch.stack(
                [
                    torch.stack([embeddings[key] for key in row["pixel_keys"]])
                    for row in chunk
                ]
            )
            actions = torch.from_numpy(
                np.stack([row["actions"] for row in chunk])
            ).to(device)
            predicted = model.predict(
                pixels, model.action_encoder(actions)
            )[:, -1]
            targets = torch.stack(
                [embeddings[row["target_key"]] for row in chunk]
            )
            losses = F.mse_loss(
                predicted, targets, reduction="none"
            ).mean(dim=-1)
            for row, loss in zip(chunk, losses):
                rows.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"pixel_keys", "target_key", "actions"}
                    }
                    | {"next_frame_latent_mse": float(loss.item())}
                )
    return rows


def _scheduled_records(
    unique_rows: list[dict[str, Any]],
    *,
    eval_seeds: list[int],
    evaluations: int,
) -> list[dict[str, Any]]:
    records = []
    expected_seeds = set(map(int, eval_seeds))
    observed_seeds = {int(row["eval_seed"]) for row in unique_rows}
    if observed_seeds != expected_seeds:
        raise RuntimeError(
            f"Eval seed mismatch: {observed_seeds} != {expected_seeds}"
        )
    counts: dict[tuple[float, int, str], int] = defaultdict(int)
    query_ids: dict[tuple[float, int, str], set[str]] = defaultdict(set)
    for row in unique_rows:
        key = (
            float(row["reference_speed"]),
            int(row["eval_seed"]),
            str(row["condition"]),
        )
        counts[key] += 1
        query_ids[key].add(str(row["query_id"]))
        records.append(
            {
                **row,
                "evaluation_id": (
                    f"s{row['eval_seed']}-v{row['reference_speed']:g}-"
                    f"e{row['evaluation_index']:03d}-{row['query_id']}"
                ),
            }
        )
    if set(counts.values()) != {int(evaluations)}:
        raise RuntimeError(f"Expected {evaluations} rows per seed cell: {counts}")
    if any(len(query_ids[key]) != count for key, count in counts.items()):
        raise RuntimeError("A deterministic next-latent cell repeats a query")
    return sorted(
        records,
        key=lambda row: (
            float(row["reference_speed"]),
            int(row["eval_seed"]),
            int(row["evaluation_index"]),
            str(row["condition"]),
        ),
    )


def _paired_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(dict)
    for row in records:
        key = (
            int(row["eval_seed"]),
            float(row["reference_speed"]),
            int(row["evaluation_index"]),
            str(row["query_id"]),
        )
        grouped[key][row["condition"]] = row
    result = []
    for key, values in grouped.items():
        if set(values) != set(CONDITIONS):
            raise RuntimeError(f"Incomplete paired histories: {key}")
        matching_condition = next(iter(values.values()))["matching_condition"]
        matching = float(
            values[matching_condition]["next_frame_latent_mse"]
        )
        other = [
            float(row["next_frame_latent_mse"])
            for condition, row in values.items()
            if condition != matching_condition
        ]
        result.append(
            {
                "eval_seed": key[0],
                "reference_speed": key[1],
                "evaluation_index": key[2],
                "query_id": key[3],
                "static_query_id": next(iter(values.values()))[
                    "static_query_id"
                ],
                "matching_loss": matching,
                "other_history_mean_loss": float(np.mean(other)),
                "matching_history_advantage": float(np.mean(other) - matching),
            }
        )
    return result


def _ratio_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    matching = float(np.mean([row["matching_loss"] for row in rows]))
    other = float(
        np.mean([row["other_history_mean_loss"] for row in rows])
    )
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
) -> dict[str, float]:
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[row["static_query_id"]].append(row)
    clusters = sorted(by_cluster)
    values = np.asarray(
        [
            [
                np.mean([row["matching_loss"] for row in by_cluster[key]]),
                np.mean(
                    [
                        row["other_history_mean_loss"]
                        for row in by_cluster[key]
                    ]
                ),
            ]
            for key in clusters
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
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
    bootstrap_seed: int,
    expected_eval_seeds: int = 6,
    expected_per_seed: int = 50,
) -> dict[str, Any]:
    paired = _paired_rows(records)
    speeds = sorted({float(row["reference_speed"]) for row in paired})
    eval_seeds = sorted({int(row["eval_seed"]) for row in paired})
    by_speed = {}
    all_seed_directions = True
    all_competitors = True
    for speed_index, speed in enumerate(speeds):
        speed_pairs = [
            row for row in paired if row["reference_speed"] == speed
        ]
        unique_records = [
            row for row in records if row["reference_speed"] == speed
        ]
        condition_means = {
            condition: float(
                np.mean(
                    [
                        row["next_frame_latent_mse"]
                        for row in unique_records
                        if row["condition"] == condition
                    ]
                )
            )
            for condition in CONDITIONS
        }
        matching_condition = next(
            row["matching_condition"] for row in unique_records
        )
        competitor_pass = all(
            condition_means[matching_condition] < value
            for condition, value in condition_means.items()
            if condition != matching_condition
        )
        seed_summaries = {}
        for eval_seed in eval_seeds:
            selected = [
                row for row in speed_pairs if row["eval_seed"] == eval_seed
            ]
            seed_summaries[str(eval_seed)] = _ratio_summary(selected)
        seed_direction_pass = all(
            row["matching_history_advantage"] > 0
            for row in seed_summaries.values()
        )
        summary = _ratio_summary(speed_pairs)
        summary["relative_loss_reduction_ci"] = _bootstrap_ratio(
            speed_pairs,
            seed=bootstrap_seed + speed_index,
        )
        summary["matching_condition"] = matching_condition
        summary["condition_mean_losses"] = condition_means
        summary["matching_below_each_other_history"] = competitor_pass
        summary["by_eval_seed"] = seed_summaries
        summary["all_eval_seed_directions_positive"] = seed_direction_pass
        by_speed[str(speed)] = summary
        all_competitors = all_competitors and competitor_pass
        all_seed_directions = all_seed_directions and seed_direction_pass
    overall = _ratio_summary(paired)
    overall["relative_loss_reduction_ci"] = _bootstrap_ratio(
        paired, seed=bootstrap_seed + 100
    )
    counts = defaultdict(int)
    for row in records:
        counts[(row["reference_speed"], row["condition"])] += 1
    expected_per_cell = int(expected_eval_seeds) * int(expected_per_seed)
    seed_cell_counts = defaultdict(int)
    seed_cell_queries: dict[tuple[float, str, int], set[str]] = defaultdict(set)
    for row in records:
        key = (
            float(row["reference_speed"]),
            str(row["condition"]),
            int(row["eval_seed"]),
        )
        seed_cell_counts[key] += 1
        seed_cell_queries[key].add(str(row["query_id"]))
    unique_queries_pass = all(
        len(seed_cell_queries[key]) == count
        for key, count in seed_cell_counts.items()
    )
    count_pass = (
        len(eval_seeds) == int(expected_eval_seeds)
        and len(counts) == 9
        and set(counts.values()) == {expected_per_cell}
        and len(seed_cell_counts) == 9 * int(expected_eval_seeds)
        and set(seed_cell_counts.values()) == {int(expected_per_seed)}
        and unique_queries_pass
        and len(records) == 9 * expected_per_cell
    )
    return {
        "overall": overall,
        "by_reference_speed": by_speed,
        "decision": {
            "matching_below_each_other_history_all_speeds": all_competitors,
            "all_eval_seed_directions_positive": all_seed_directions,
            "passed": bool(all_competitors and all_seed_directions),
        },
        "count_audit": {
            "records": len(records),
            "matrix_cells": len(counts),
            "records_per_cell": {
                f"v{speed:g}/{condition}": count
                for (speed, condition), count in sorted(counts.items())
            },
            "expected_records_per_cell": expected_per_cell,
            "records_per_eval_seed_cell": {
                f"v{speed:g}/{condition}/s{seed}": count
                for (speed, condition, seed), count in sorted(
                    seed_cell_counts.items()
                )
            },
            "expected_records_per_eval_seed_cell": int(expected_per_seed),
            "all_queries_unique_within_eval_seed_cells": unique_queries_pass,
            "passed": count_pass,
        },
    }


def _find_model(config: dict[str, Any], slug: str) -> tuple[str, dict[str, Any]]:
    for group, models in config["models"].items():
        for model in models:
            if model["slug"] == slug:
                return str(group), dict(model)
    raise KeyError(f"Unknown model slug: {slug}")


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
    if config.get("status") != (
        "preregistered_before_independent_catalog_generation_and_scoring"
    ):
        raise ValueError("Config is not frozen before execution")
    group, model_row = _find_model(config, args.model)
    checkpoint = resolve_contextworld_path(
        model_row["checkpoint"], repo_root=ROOT
    )
    normalizer = resolve_contextworld_path(
        config["evaluation"]["normalizer"], repo_root=ROOT
    )
    output = resolve_contextworld_path(args.output, repo_root=ROOT)
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, args.stablewm_repo, args.stablewm_ref
    )
    process = frozen_normalizer_process(normalizer)
    loaded = load_pretrained_cost_model(
        checkpoint,
        swm,
        cache_dir=artifact_path("evaluation/model_cache", repo_root=ROOT),
    )
    protocol = infer_model_protocol(loaded, action_dim=2)
    if protocol != {"action_block": 5, "history_size": 3}:
        raise RuntimeError(f"Unexpected model protocol: {protocol}")
    loaded = loaded.to(args.device).eval()
    loaded.requires_grad_(False)
    setattr(loaded, "history_size", 3)
    setattr(loaded, "interpolate_pos_encoding", True)
    before = state_dict_sha256(loaded)
    eval_seeds = [int(value) for value in config["evaluation"]["eval_seeds"]]
    evaluations = int(
        config["evaluation"][
            "evaluations_per_reference_speed_per_history_per_seed"
        ]
    )
    tracks = {}
    total_environment_calls = 0
    for track_index, (track, track_config) in enumerate(
        config["data"]["tracks"].items()
    ):
        catalog = resolve_contextworld_path(
            track_config["catalog"], repo_root=ROOT
        )
        print(f"[track] {track} catalog={catalog}", flush=True)
        assets, registry, audit = _load_track_assets(
            catalog,
            action_standardizer=process["action"],
            expected_queries_per_speed=int(
                config["evaluation"]["unique_queries_per_reference_speed"]
            ),
            eval_seeds=eval_seeds,
            expected_queries_per_seed=int(
                config["evaluation"][
                    "unique_queries_per_reference_speed_per_seed"
                ]
            ),
        )
        embeddings = _encode_registry(
            loaded,
            registry,
            device=args.device,
            batch_size=args.encode_batch_size,
        )
        unique_rows = _score_unique_assets(
            loaded,
            assets,
            embeddings,
            device=args.device,
            batch_size=args.predictor_batch_size,
        )
        records = _scheduled_records(
            unique_rows,
            eval_seeds=eval_seeds,
            evaluations=evaluations,
        )
        summary = summarize_records(
            records,
            bootstrap_seed=args.bootstrap_seed + 1000 * track_index,
            expected_eval_seeds=len(eval_seeds),
            expected_per_seed=evaluations,
        )
        if not summary["count_audit"]["passed"]:
            raise RuntimeError(f"Count audit failed for {track}")
        tracks[track] = {
            "data_audit": audit,
            "unique_predictions": len(unique_rows),
            "summary": summary,
            "records": records,
        }
        total_environment_calls += int(audit["online_environment_calls"])
        del embeddings
    after = state_dict_sha256(loaded)
    if before != after:
        raise RuntimeError("Model weights changed during evaluation")
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "model": {
            "group": group,
            "slug": model_row["slug"],
            "training_seed": int(model_row["training_seed"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
        },
        "normalizer": {
            "path": str(normalizer),
            "sha256": file_sha256(normalizer),
        },
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "protocol": {
            **protocol,
            "target": "frozen_offline_query_next_frame_pixels",
            "target_encoding": "current_checkpoint_frozen_encoder",
            "metric": "native_next_frame_latent_mse",
            "eval_seeds": eval_seeds,
            "evaluations_per_matrix_cell_per_seed": evaluations,
            "online_environment_during_scoring": False,
        },
        "frozen_weight_audit": {
            "state_dict_sha256_before": before,
            "state_dict_sha256_after": after,
            "passed": before == after,
        },
        "online_environment_calls": total_environment_calls,
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
        / "configs/benchmark/tworoom_speed_next_latent_v4.yaml",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--predictor-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
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
                    name: row["summary"]["decision"]
                    for name, row in result["tracks"].items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
