#!/usr/bin/env python3
"""Score visual-door queries against frozen offline true-future latents."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.door_visual import (
    DIRECTIONS,
    HORIZONS,
    INPUT_CONDITIONS,
    TASKS,
    array_sha256,
    checkpoint_cell_summary,
)
from contextworld.evaluation.icl_model import file_sha256, state_dict_sha256
from contextworld.evaluation.sealed_test_gate import (
    canonical_door_split_root,
    require_canonical_split_path,
    require_sealed_test_gate,
)
from contextworld.evaluation.protocol import (
    frozen_normalizer_process,
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from scripts.eval_tworoom_speed_next_latent import (
    _encode_registry,
    _normalize_actions,
    _preprocess_pixels,
    _register_pixels,
)


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
DEFAULT_NORMALIZER = "artifacts/splits/tworoom_original_train_s3072_normalizer.json"


def _load_assets(
    catalog_path: Path,
    *,
    action_standardizer: Any,
    eval_seeds: list[int],
    expected_per_seed: int,
    require_formal_counts: bool,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    registry: dict[str, np.ndarray] = {}
    assets = []
    payload_hashes = set()
    for bundle in sorted(catalog["bundles"], key=lambda row: row["query_id"]):
        payload_path = resolve_contextworld_path(bundle["payload"], repo_root=ROOT)
        observed_hash = file_sha256(payload_path)
        if observed_hash != bundle["payload_sha256"]:
            raise RuntimeError(f"Payload hash mismatch: {payload_path}")
        if observed_hash in payload_hashes:
            raise RuntimeError(f"Repeated payload hash: {payload_path}")
        payload_hashes.add(observed_hash)
        with np.load(payload_path, allow_pickle=False) as payload:
            history_pixels = np.asarray(payload["history_pixels"], dtype=np.uint8)
            history_actions = np.asarray(payload["history_actions"], dtype=np.float32)
            query_pixels = np.asarray(payload["query_pixels"], dtype=np.uint8)
            future_actions = np.asarray(payload["future_actions"], dtype=np.float32)
            future_pixels = np.asarray(payload["future_pixels"], dtype=np.uint8)
            targets = np.asarray(payload["future_next_pixels"], dtype=np.uint8)
            if history_pixels.shape != (3, 224, 224, 3):
                raise RuntimeError(f"Unexpected history pixels: {payload_path}")
            if history_actions.shape != (2, 5, 2):
                raise RuntimeError(f"Unexpected history actions: {payload_path}")
            if future_actions.shape != (5, 5, 2):
                raise RuntimeError(f"Unexpected future actions: {payload_path}")
            if future_pixels.shape != (5, 224, 224, 3) or targets.shape != (
                5,
                224,
                224,
                3,
            ):
                raise RuntimeError(f"Unexpected future pixels: {payload_path}")
            if not np.array_equal(history_pixels[-1], query_pixels):
                raise RuntimeError(f"History does not end at query: {payload_path}")
            if not np.array_equal(future_pixels[0], query_pixels):
                raise RuntimeError(f"Future does not start at query: {payload_path}")
            if not np.array_equal(future_pixels[1:], targets[:-1]):
                raise RuntimeError(f"Future continuity failed: {payload_path}")
            if array_sha256(query_pixels) != bundle["query_pixels_sha256"]:
                raise RuntimeError(f"Query hash mismatch: {payload_path}")
            if array_sha256(history_pixels) != bundle["history_pixels_sha256"]:
                raise RuntimeError(f"History hash mismatch: {payload_path}")
            if array_sha256(future_actions) != bundle["future_actions_sha256"]:
                raise RuntimeError(f"Action hash mismatch: {payload_path}")
            for horizon in HORIZONS:
                if array_sha256(targets[horizon - 1]) != bundle[
                    "target_pixels_sha256_by_horizon"
                ][str(horizon)]:
                    raise RuntimeError(
                        f"Target hash mismatch h={horizon}: {payload_path}"
                    )
                if np.array_equal(query_pixels, targets[horizon - 1]):
                    raise RuntimeError(
                        f"Unchanged true future h={horizon}: {payload_path}"
                    )
            query_key = _register_pixels(registry, query_pixels)[0]
            history_keys = _register_pixels(registry, history_pixels)
            target_keys = _register_pixels(registry, targets)
            assets.append(
                {
                    "query_id": str(bundle["query_id"]),
                    "static_query_id": str(bundle["static_query_id"]),
                    "template_id": str(bundle["template_id"]),
                    "track": str(bundle["track"]),
                    "task": str(bundle["task"]),
                    "direction": str(bundle["direction"]),
                    "door_position": int(bundle["door_position"]),
                    "eval_seed": int(bundle["eval_seed"]),
                    "evaluation_index": int(bundle["evaluation_index"]),
                    "query_key": query_key,
                    "target_keys": target_keys,
                    "samples": {
                        "query_only": {
                            "input_pixel_keys": [query_key],
                            "normalized_actions": _normalize_actions(
                                future_actions, action_standardizer
                            ),
                            "history_size": 1,
                        },
                        "natural_history3": {
                            "input_pixel_keys": history_keys,
                            "normalized_actions": _normalize_actions(
                                np.concatenate(
                                    [history_actions, future_actions], axis=0
                                ),
                                action_standardizer,
                            ),
                            "history_size": 3,
                        },
                    },
                }
            )
    by_cell: dict[tuple[int, str, int], int] = {}
    for door in sorted({row["door_position"] for row in assets}):
        for task in TASKS:
            for seed in eval_seeds:
                by_cell[(door, task, seed)] = sum(
                    row["door_position"] == door
                    and row["task"] == task
                    and row["eval_seed"] == seed
                    for row in assets
                )
    if require_formal_counts and set(by_cell.values()) != {int(expected_per_seed)}:
        raise RuntimeError(f"Door/task/seed count mismatch: {by_cell}")
    if require_formal_counts:
        for door, task, seed in by_cell:
            direction_counts = {
                direction: sum(
                    row["door_position"] == door
                    and row["task"] == task
                    and row["eval_seed"] == seed
                    and row["direction"] == direction
                    for row in assets
                )
                for direction in DIRECTIONS
            }
            if set(direction_counts.values()) != {int(expected_per_seed) // 2}:
                raise RuntimeError(
                    f"Direction imbalance door={door}/task={task}/seed={seed}: "
                    f"{direction_counts}"
                )
    for door in sorted({row["door_position"] for row in assets}):
        for task in TASKS:
            seed_sets = [
                {
                    row["static_query_id"]
                    for row in assets
                    if row["door_position"] == door
                    and row["task"] == task
                    and row["eval_seed"] == seed
                }
                for seed in eval_seeds
            ]
            if sum(map(len, seed_sets)) != len(set().union(*seed_sets)):
                raise RuntimeError(f"Eval seed query overlap: door={door}, task={task}")
    return assets, registry, {
        "catalog": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "bundles": len(assets),
        "input_conditions": list(INPUT_CONDITIONS),
        "scored_sequences": len(assets) * len(INPUT_CONDITIONS),
        "horizon_losses": len(assets) * len(INPUT_CONDITIONS) * len(HORIZONS),
        "unique_payloads": len(payload_hashes),
        "unique_registered_pixels": len(registry),
        "door_task_seed_counts": {
            f"door{door}/{task}/seed{seed}": count
            for (door, task, seed), count in sorted(by_cell.items())
        },
        "formal_counts_required": require_formal_counts,
        "formal_direction_balance": (
            "25_left_to_right_plus_25_right_to_left"
            if require_formal_counts
            else "smoke_only_not_formal"
        ),
        "eval_seed_queries_disjoint_in_every_door_task": True,
    }


def _sample_rows(assets: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    rows = []
    for asset in assets:
        sample = asset["samples"][condition]
        rows.append(
            {
                **{
                    key: value
                    for key, value in asset.items()
                    if key not in {"samples", "query_key", "target_keys"}
                },
                "query_key": asset["query_key"],
                "target_keys": asset["target_keys"],
                "input_condition": condition,
                **sample,
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

    history_sizes = {int(row["history_size"]) for row in rows}
    if len(history_sizes) != 1:
        raise ValueError("A rollout batch must use one input condition")
    history_size = next(iter(history_sizes))
    pixels = np.stack(
        [
            np.stack([registry[key] for key in row["input_pixel_keys"]])
            for row in rows
        ]
    )
    batch, frames = pixels.shape[:2]
    if frames != history_size:
        raise RuntimeError("Visible frame count and history size disagree")
    transformed = _preprocess_pixels(
        pixels.reshape(-1, *pixels.shape[2:]), device
    ).reshape(batch, frames, 3, pixels.shape[2], pixels.shape[3])
    actions = torch.from_numpy(
        np.stack([row["normalized_actions"] for row in rows])
    ).to(device=device, dtype=next(model.parameters()).dtype)
    output = model.rollout(
        {"pixels": transformed[:, None]},
        actions[:, None],
        history_size=history_size,
    )["predicted_emb"][:, 0]
    predictions = output[:, history_size:]
    expected_future = actions.shape[1] - (history_size - 1)
    if predictions.shape[1] != expected_future:
        raise RuntimeError(
            f"Expected {expected_future} futures, got {predictions.shape}"
        )
    return predictions


def _audit_prefix_causality(
    model: Any,
    rows: list[dict[str, Any]],
    registry: dict[str, np.ndarray],
    *,
    device: str,
) -> dict[str, Any]:
    """Verify causality without changing the rollout tensor shape.

    Comparing a short rollout with a long rollout is not a valid bitwise
    causality check on CUDA: the different tensor shapes can select different
    numerical kernels even when the model is causal.  Instead, keep the full
    action sequence length fixed, preserve the prefix needed for the first
    ``future_count`` predictions, and deterministically perturb only the later
    action suffix.  A causal rollout must leave the shared prediction prefix
    unchanged.
    """

    import torch

    audit_rows = rows[: min(8, len(rows))]
    if not audit_rows:
        raise ValueError("Prefix causality audit needs at least one row")
    history_sizes = {int(row["history_size"]) for row in audit_rows}
    if len(history_sizes) != 1:
        raise ValueError("Prefix causality audit requires one history size")
    history_size = next(iter(history_sizes))
    reference = _rollout(model, audit_rows, registry, device=device)
    maximum_prefix_difference = 0.0
    minimum_suffix_perturbation = float("inf")
    checked_future_counts = []
    for future_count in (1, 2, 3):
        action_prefix_tokens = history_size - 1 + future_count
        variants = []
        for row in audit_rows:
            actions = np.asarray(
                row["normalized_actions"], dtype=np.float32
            ).copy()
            suffix = actions[action_prefix_tokens:]
            if suffix.size == 0:
                raise RuntimeError(
                    "Prefix causality audit requires a non-empty action suffix"
                )
            perturbation = np.where(
                np.arange(suffix.size).reshape(suffix.shape) % 2 == 0,
                np.float32(0.5),
                np.float32(-0.5),
            ).astype(np.float32)
            actions[action_prefix_tokens:] = suffix + perturbation
            minimum_suffix_perturbation = min(
                minimum_suffix_perturbation,
                float(np.min(np.abs(perturbation))),
            )
            variants.append({**row, "normalized_actions": actions})
        perturbed = _rollout(model, variants, registry, device=device)
        difference = float(
            torch.max(
                torch.abs(
                    perturbed[:, :future_count]
                    - reference[:, :future_count]
                )
            ).item()
        )
        maximum_prefix_difference = max(maximum_prefix_difference, difference)
        checked_future_counts.append(future_count)
    passed = bool(
        maximum_prefix_difference <= 1e-6
        and minimum_suffix_perturbation >= 0.5
    )
    return {
        "method": "same_length_future_action_suffix_perturbation",
        "checked_future_counts": checked_future_counts,
        "shared_prefix_max_abs_difference": maximum_prefix_difference,
        "minimum_suffix_action_perturbation": minimum_suffix_perturbation,
        "tolerance": 1e-6,
        "passed": passed,
    }


def _score_condition(
    model: Any,
    rows: list[dict[str, Any]],
    registry: dict[str, np.ndarray],
    embeddings: dict[str, Any],
    *,
    device: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    records = []
    with torch.inference_mode():
        prefix_audit = _audit_prefix_causality(
            model, rows, registry, device=device
        )
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start : start + int(batch_size)]
            predicted = _rollout(model, chunk, registry, device=device)
            targets = torch.stack(
                [
                    torch.stack([embeddings[key] for key in row["target_keys"]])
                    for row in chunk
                ]
            )
            query_embeddings = torch.stack(
                [embeddings[row["query_key"]] for row in chunk]
            )[:, None]
            losses = F.mse_loss(predicted, targets, reduction="none").mean(dim=-1)
            baseline = F.mse_loss(
                query_embeddings.expand_as(targets), targets, reduction="none"
            ).mean(dim=-1)
            if torch.any(baseline <= 1e-12):
                raise RuntimeError("Unchanged-frame latent baseline is zero")
            normalized = losses / baseline
            for row, row_losses, row_baseline, row_normalized in zip(
                chunk, losses, baseline, normalized
            ):
                records.append(
                    {
                        **{
                            key: value
                            for key, value in row.items()
                            if key
                            not in {
                                "query_key",
                                "target_keys",
                                "input_pixel_keys",
                                "normalized_actions",
                                "history_size",
                            }
                        },
                        "latent_mse_by_horizon": {
                            str(horizon): float(row_losses[horizon - 1].item())
                            for horizon in HORIZONS
                        },
                        "unchanged_baseline_mse_by_horizon": {
                            str(horizon): float(row_baseline[horizon - 1].item())
                            for horizon in HORIZONS
                        },
                        "normalized_error_by_horizon": {
                            str(horizon): float(row_normalized[horizon - 1].item())
                            for horizon in HORIZONS
                        },
                    }
                )
    return records, prefix_audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(1)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = file_sha256(config_path)
    gate_audit = require_sealed_test_gate(
        split=args.split,
        config_path=config_path,
        config=config,
        manifest_path=getattr(args, "sealed_test_gate", None),
        repo_root=ROOT,
    )
    artifact_root = require_canonical_split_path(
        args.artifact_root,
        canonical=canonical_door_split_root(
            config, split=args.split, repo_root=ROOT
        ),
        split=args.split,
        label="Door latent artifact root",
    )
    build_report_path = artifact_root / "catalogs" / "build_report.json"
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    if build_report["config"]["sha256"] != config_hash:
        raise RuntimeError("Build report/config hash mismatch")
    if str(build_report.get("evaluation_split")) != args.split:
        raise RuntimeError("Build report/evaluator split mismatch")
    formal = build_report["status"] == "passed"
    if not formal and not args.allow_smoke_catalog:
        raise RuntimeError("Refusing to score a non-formal smoke catalog")
    normalizer = resolve_contextworld_path(args.normalizer, repo_root=ROOT)
    checkpoint = resolve_contextworld_path(args.checkpoint, repo_root=ROOT)
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, args.stablewm_repo, args.stablewm_ref
    )
    if stable_commit != build_report["stable_worldmodel"]["commit"]:
        raise RuntimeError("StableWM/build report commit mismatch")
    process = frozen_normalizer_process(normalizer)
    model = load_pretrained_cost_model(
        checkpoint,
        swm,
        cache_dir=artifact_path("evaluation/model_cache", repo_root=ROOT),
    )
    protocol = infer_model_protocol(model, action_dim=2)
    if protocol != {"action_block": 5, "history_size": 3}:
        raise RuntimeError(f"Unexpected checkpoint protocol: {protocol}")
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    setattr(model, "interpolate_pos_encoding", True)
    before = state_dict_sha256(model)
    eval_seeds = [int(value) for value in config["evaluation_data"]["eval_seeds"]]
    expected_per_seed = int(
        config["evaluation_data"]["unique_queries_per_door_per_task_per_seed"]
    )
    tracks = {}
    all_records = []
    for track_name, track_report in build_report["tracks"].items():
        catalog_path = Path(track_report["catalog"]).resolve()
        if file_sha256(catalog_path) != track_report["catalog_sha256"]:
            raise RuntimeError(f"Catalog/build report mismatch: {catalog_path}")
        assets, registry, data_audit = _load_assets(
            catalog_path,
            action_standardizer=process["action"],
            eval_seeds=eval_seeds,
            expected_per_seed=expected_per_seed,
            require_formal_counts=formal,
        )
        embeddings = _encode_registry(
            model,
            registry,
            device=args.device,
            batch_size=args.encode_batch_size,
        )
        prefix_audits = {}
        track_records = []
        for condition in INPUT_CONDITIONS:
            records, prefix = _score_condition(
                model,
                _sample_rows(assets, condition),
                registry,
                embeddings,
                device=args.device,
                batch_size=args.rollout_batch_size,
            )
            if not prefix["passed"]:
                raise RuntimeError(
                    f"Autoregressive prefix audit failed: {track_name}/{condition}"
                )
            prefix_audits[condition] = prefix
            track_records.extend(records)
        tracks[track_name] = {
            "data_audit": data_audit,
            "autoregressive_prefix_audit": prefix_audits,
            "records": track_records,
        }
        all_records.extend(track_records)
        del embeddings
    after = state_dict_sha256(model)
    if before != after:
        raise RuntimeError("Model weights changed during evaluation")
    split_name = str(build_report.get("evaluation_split", "validation"))
    count_key = (
        "validation_counts_per_checkpoint"
        if split_name == "validation"
        else "sealed_test_counts_per_checkpoint"
    )
    expected_sequences = int(
        config["evaluation_data"][count_key][
            "scored_sequences"
        ]
    )
    if formal and len(all_records) != expected_sequences:
        raise RuntimeError(
            f"Expected {expected_sequences} scored sequences, got {len(all_records)}"
        )
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed" if formal else "smoke_only",
        "evaluation_split": split_name,
        "sealed_test_gate": gate_audit,
        "config": {"path": str(config_path), "sha256": config_hash},
        "build_report": {
            "path": str(build_report_path),
            "sha256": file_sha256(build_report_path),
        },
        "model": {
            "slug": args.model_slug,
            "group": args.group,
            "training_seed": int(args.training_seed),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
        },
        "normalizer": {"path": str(normalizer), "sha256": file_sha256(normalizer)},
        "stable_worldmodel": {"repo": str(stable_repo), "commit": stable_commit},
        "protocol": {
            **protocol,
            "query_only_visible_frames": 1,
            "natural_history3_visible_frames": 3,
            "fully_autoregressive": True,
            "teacher_forcing_future_frames": False,
            "target": "frozen_offline_true_future_pixels",
            "target_encoding": "current_checkpoint_frozen_encoder",
            "future_horizons_action_blocks": list(HORIZONS),
            "normalization_baseline": "unchanged_query_frame_same_checkpoint",
            "raw_latent_mse_cross_checkpoint_ranking": False,
            "online_environment_during_scoring": False,
        },
        "frozen_weight_audit": {
            "state_dict_sha256_before": before,
            "state_dict_sha256_after": after,
            "passed": before == after,
        },
        "online_environment_calls": 0,
        "count_audit": {
            "scored_sequences": len(all_records),
            "horizon_losses": len(all_records) * len(HORIZONS),
            "formal_expected_scored_sequences": expected_sequences,
            "passed": (not formal or len(all_records) == expected_sequences),
        },
        "checkpoint_summary": checkpoint_cell_summary(
            all_records, eval_seeds=eval_seeds
        ),
        "tracks": tracks,
    }
    output = args.output.resolve()
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml",
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--split", choices=("validation", "sealed_test"), default="validation"
    )
    parser.add_argument("--sealed-test-gate", type=Path)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument(
        "--group",
        required=True,
        choices=["original_reference", "fixed_door_control", "multi_door_target"],
    )
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalizer", default=DEFAULT_NORMALIZER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--allow-smoke-catalog", action="store_true")
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "model": result["model"],
                "count_audit": result["count_audit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
