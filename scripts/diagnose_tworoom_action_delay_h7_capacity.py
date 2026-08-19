#!/usr/bin/env python3
"""Test whether the failed History-7 predictor can learn explicit delay binding.

The diagnostic freezes each failed checkpoint's Encoder and Projector, caches
their native latents, and trains only Predictor/PredProj/ActionEncoder.  Three
small controlled variants separate same-query triplet support from the weight
placed on the final transition that actually requires long history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMHistory7Adapter,
)
from contextworld.evaluation.action_delay import array_sha256
from contextworld.evaluation.action_delay_h7_domain_score import (
    load_domain_catalog,
    load_domain_track_assets,
)
from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.evaluation.icl_model import state_dict_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from scripts.diagnose_tworoom_action_delay_h7_checkpoint import (
    StableWorldModelPLDMHistory7DiagnosticAdapter,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_capacity_diagnostic_v1.yaml"
)
DELAYS = (0, 4, 8)
VARIANTS = ("unpaired_final", "paired_full", "paired_final")


@dataclass(frozen=True)
class LatentCache:
    history: np.ndarray
    target: np.ndarray
    actions: np.ndarray
    source_indices: np.ndarray
    query_ids: tuple[str, ...]
    asset_sha256s: tuple[str, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _selected_assets(
    catalog: dict[str, Any],
    *,
    tracks: list[str],
    queries_per_track: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for track in tracks:
        rows = load_domain_track_assets(
            catalog,
            track=track,
            repo_root=ROOT,
        )
        _require(
            len(rows) >= queries_per_track,
            f"Not enough queries in {track}",
        )
        selected.extend(rows[:queries_per_track])
    _require(
        len(selected) == len(tracks) * queries_per_track,
        "Selected query count changed",
    )
    _require(
        len({row["query_id"] for row in selected}) == len(selected),
        "Selected query identifiers are not unique",
    )
    return selected


def _build_cache(
    adapter: Any,
    assets: list[dict[str, Any]],
    *,
    batch_size: int,
) -> LatentCache:
    histories = np.stack(
        [np.asarray(row["history_pixels"], dtype=np.uint8) for row in assets]
    )
    targets = np.stack(
        [
            np.asarray(row["true_future_pixels"][:, 0], dtype=np.uint8)
            for row in assets
        ]
    )
    _require(
        histories.shape[1:3] == (len(DELAYS), 7),
        f"Unexpected history cache shape: {histories.shape}",
    )
    _require(
        targets.shape[1] == len(DELAYS),
        f"Unexpected target cache shape: {targets.shape}",
    )
    history_latents = np.asarray(
        adapter.encode_pixels(
            histories.reshape(-1, *histories.shape[-3:]),
            batch_size=batch_size,
        ),
        dtype=np.float32,
    ).reshape(len(assets), len(DELAYS), 7, -1)
    target_latents = np.asarray(
        adapter.encode_pixels(
            targets.reshape(-1, *targets.shape[-3:]),
            batch_size=batch_size,
        ),
        dtype=np.float32,
    ).reshape(len(assets), len(DELAYS), -1)
    raw_actions = np.stack(
        [
            np.asarray(row["action_blocks"][:7], dtype=np.float32)
            for row in assets
        ]
    )
    normalized_actions = np.asarray(
        adapter._normalize_actions(raw_actions),
        dtype=np.float32,
    )
    delay_index = {delay: index for index, delay in enumerate(DELAYS)}
    source_indices = np.asarray(
        [delay_index[int(row["source_delay"])] for row in assets],
        dtype=np.int64,
    )
    return LatentCache(
        history=history_latents,
        target=target_latents,
        actions=normalized_actions,
        source_indices=source_indices,
        query_ids=tuple(str(row["query_id"]) for row in assets),
        asset_sha256s=tuple(str(row["asset_sha256"]) for row in assets),
    )


def _cache_identity(cache: LatentCache) -> dict[str, Any]:
    digest = hashlib.sha256()
    for value in cache.query_ids:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for value in cache.asset_sha256s:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return {
        "queries": len(cache.query_ids),
        "query_ids_sha256": digest.hexdigest(),
        "history_latents_sha256": array_sha256(cache.history),
        "target_latents_sha256": array_sha256(cache.target),
        "normalized_actions_sha256": array_sha256(cache.actions),
        "source_indices_sha256": array_sha256(cache.source_indices),
    }


def _alignment_metrics(
    predicted: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    pairs = ((0, 1), (0, 2), (1, 2))
    left = np.asarray([row[0] for row in pairs], dtype=np.int64)
    right = np.asarray([row[1] for row in pairs], dtype=np.int64)
    prediction_delta = predicted[:, left] - predicted[:, right]
    target_delta = targets[:, left] - targets[:, right]
    prediction_norm_sq = np.sum(prediction_delta**2, axis=-1)
    target_norm_sq = np.sum(target_delta**2, axis=-1)
    dot = np.sum(prediction_delta * target_delta, axis=-1)
    valid = (prediction_norm_sq > 1e-18) & (target_norm_sq > 1e-18)
    cosine = dot[valid] / np.sqrt(
        prediction_norm_sq[valid] * target_norm_sq[valid]
    )
    prediction_pair_mse = float(np.mean(prediction_delta**2))
    target_pair_mse = float(np.mean(target_delta**2))
    prediction_centered = predicted - predicted.mean(axis=1, keepdims=True)
    target_centered = targets - targets.mean(axis=1, keepdims=True)
    centered_dot = np.sum(
        prediction_centered * target_centered,
        axis=(1, 2),
    )
    prediction_centered_norm = np.sum(
        prediction_centered**2,
        axis=(1, 2),
    )
    target_centered_norm = np.sum(target_centered**2, axis=(1, 2))
    centered_valid = (
        prediction_centered_norm > 1e-18
    ) & (target_centered_norm > 1e-18)
    centered_cosine = centered_dot[centered_valid] / np.sqrt(
        prediction_centered_norm[centered_valid]
        * target_centered_norm[centered_valid]
    )
    return {
        "target_pair_mse": target_pair_mse,
        "prediction_pair_mse": prediction_pair_mse,
        "prediction_to_target_pair_magnitude_ratio": float(
            np.sqrt(
                prediction_pair_mse / max(target_pair_mse, 1e-18)
            )
        ),
        "pair_direction_cosine_mean": float(np.mean(cosine)),
        "pair_direction_positive_fraction": float(
            np.mean(dot[valid] > 0)
        ),
        "centered_delay_pattern_cosine_mean": float(
            np.mean(centered_cosine)
        ),
        "target_centered_variance": float(
            np.mean(target_centered**2)
        ),
        "prediction_centered_variance": float(
            np.mean(prediction_centered**2)
        ),
    }


def _selection_metrics(
    predicted: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    _require(
        predicted.shape == targets.shape
        and predicted.ndim == 3
        and predicted.shape[1] == len(DELAYS),
        f"Unexpected evaluation arrays: {predicted.shape}/{targets.shape}",
    )
    losses = np.mean(
        (predicted[:, :, None] - targets[:, None]) ** 2,
        axis=-1,
    )
    selected_target = np.argmin(losses, axis=2)
    selected_history = np.argmin(losses, axis=1)
    identity = np.arange(len(DELAYS), dtype=np.int64)[None]
    matching = np.diagonal(losses, axis1=1, axis2=2)
    other_mean = np.stack(
        [
            np.mean(
                np.delete(losses[:, :, target_index], target_index, axis=1),
                axis=1,
            )
            for target_index in range(len(DELAYS))
        ],
        axis=1,
    )
    strict = np.stack(
        [
            matching[:, target_index]
            < np.min(
                np.delete(
                    losses[:, :, target_index],
                    target_index,
                    axis=1,
                ),
                axis=1,
            )
            for target_index in range(len(DELAYS))
        ],
        axis=1,
    )
    counts = Counter(
        int(value) for value in selected_target.reshape(-1)
    )
    return {
        "query_target_units": int(predicted.shape[0] * len(DELAYS)),
        "exact_target_selection_rate": float(
            np.mean(selected_target == identity)
        ),
        "exact_history_selection_rate": float(
            np.mean(selected_history == identity)
        ),
        "matching_history_strict_win_rate": float(np.mean(strict)),
        "mean_matching_history_loss": float(np.mean(matching)),
        "mean_other_history_loss": float(np.mean(other_mean)),
        "mean_history_margin": float(
            np.mean(other_mean - matching)
        ),
        "selected_target_counts": {
            str(delay): int(counts[index])
            for index, delay in enumerate(DELAYS)
        },
        "selected_target_rates": {
            str(delay): float(
                counts[index] / selected_target.size
            )
            for index, delay in enumerate(DELAYS)
        },
        "latent_alignment": _alignment_metrics(predicted, targets),
    }


def _predict_h1(
    model: Any,
    cache: LatentCache,
    *,
    device: str,
    batch_queries: int,
) -> np.ndarray:
    import torch

    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(cache.query_ids), batch_queries):
            history = torch.from_numpy(
                cache.history[start : start + batch_queries]
            ).to(device=device)
            actions = torch.from_numpy(
                cache.actions[start : start + batch_queries]
            ).to(device=device)
            query_count = history.shape[0]
            history = history.reshape(
                query_count * len(DELAYS),
                7,
                -1,
            )
            actions = (
                actions[:, None]
                .expand(-1, len(DELAYS), -1, -1)
                .reshape(query_count * len(DELAYS), 7, -1)
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=str(device).startswith("cuda"),
            ):
                act_emb = model.action_encoder(actions)
                prediction = model.predict(history, act_emb)[:, -1]
            outputs.append(
                prediction.float()
                .cpu()
                .numpy()
                .reshape(query_count, len(DELAYS), -1)
            )
    return np.concatenate(outputs, axis=0)


def _evaluate(
    model: Any,
    cache: LatentCache,
    *,
    device: str,
) -> dict[str, Any]:
    predicted = _predict_h1(
        model,
        cache,
        device=device,
        batch_queries=32,
    )
    return _selection_metrics(predicted, cache.target)


def _train_variant(
    model: Any,
    initial_state: dict[str, Any],
    train_cache: LatentCache,
    heldout_cache: LatentCache,
    *,
    variant: str,
    device: str,
    seed: int,
    steps: int,
    examples_per_step: int,
    learning_rate: float,
    weight_decay: float,
    snapshot_steps: tuple[int, ...],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    _require(variant in VARIANTS, f"Unknown variant: {variant}")
    model.load_state_dict(initial_state, strict=True)
    model.requires_grad_(False)
    trainable_modules = ("predictor", "pred_proj", "action_encoder")
    parameters = []
    for name in trainable_modules:
        module = getattr(model, name)
        module.requires_grad_(True)
        parameters.extend(
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    history_all = torch.from_numpy(train_cache.history).to(device=device)
    target_all = torch.from_numpy(train_cache.target).to(device=device)
    actions_all = torch.from_numpy(train_cache.actions).to(device=device)
    source_indices = torch.from_numpy(
        train_cache.source_indices
    ).to(device=device)
    delay_indices = torch.arange(
        len(DELAYS),
        device=device,
        dtype=torch.long,
    )
    snapshots: dict[str, Any] = {}
    loss_trace = []

    def snapshot(step: int, loss: float | None) -> None:
        snapshots[str(step)] = {
            "optimizer_step": int(step),
            "latest_training_loss": loss,
            "train": _evaluate(
                model,
                train_cache,
                device=device,
            ),
            "heldout": _evaluate(
                model,
                heldout_cache,
                device=device,
            ),
        }

    snapshot(0, None)
    latest_loss = None
    for step in range(1, steps + 1):
        model.train()
        model.encoder.eval()
        model.projector.eval()
        if variant.startswith("paired_"):
            _require(
                examples_per_step % len(DELAYS) == 0,
                "Paired batch size must be divisible by three",
            )
            query_numpy = generator.integers(
                0,
                len(train_cache.query_ids),
                size=examples_per_step // len(DELAYS),
            )
            query_index = torch.as_tensor(
                np.repeat(query_numpy, len(DELAYS)),
                device=device,
                dtype=torch.long,
            )
            history_index = delay_indices.repeat(len(query_numpy))
        else:
            query_numpy = generator.integers(
                0,
                len(train_cache.query_ids),
                size=examples_per_step,
            )
            query_index = torch.as_tensor(
                query_numpy,
                device=device,
                dtype=torch.long,
            )
            history_index = source_indices[query_index]

        history = history_all[query_index, history_index]
        target = target_all[query_index, history_index]
        actions = actions_all[query_index]
        target_sequence = torch.cat(
            [history[:, 1:], target[:, None]],
            dim=1,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=str(device).startswith("cuda"),
        ):
            act_emb = model.action_encoder(actions)
            prediction = model.predict(history, act_emb)
            loss = (
                functional.mse_loss(
                    prediction[:, -1].float(),
                    target.float(),
                )
                if variant.endswith("_final")
                else functional.mse_loss(
                    prediction.float(),
                    target_sequence.float(),
                )
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        latest_loss = float(loss.detach().cpu())
        if step == 1 or step % 16 == 0 or step == steps:
            loss_trace.append(
                {
                    "optimizer_step": step,
                    "loss": latest_loss,
                }
            )
        if step in snapshot_steps:
            snapshot(step, latest_loss)

    final_state_sha256 = state_dict_sha256(model)
    return {
        "variant": variant,
        "optimizer_steps": steps,
        "examples_per_step": examples_per_step,
        "trainable_modules": list(trainable_modules),
        "frozen_modules": ["encoder", "projector"],
        "initial_state_sha256": state_dict_sha256_from_state(initial_state),
        "final_state_sha256": final_state_sha256,
        "loss_trace": loss_trace,
        "snapshots": snapshots,
        "final": snapshots[str(steps)],
    }


def state_dict_sha256_from_state(state: dict[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-family",
        choices=("lewm", "pldm"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stablewm-repo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(
        config.get("benchmark")
        == "tworoom_action_delay_history7_capacity_diagnostic_v1",
        "Expected the frozen History-7 capacity diagnostic config",
    )
    _require(
        config.get("status")
        == "preregistered_before_capacity_diagnostic",
        "Capacity diagnostic was not preregistered",
    )
    checkpoint_spec = config["inputs"]["failed_checkpoints"][
        args.model_family
    ]
    checkpoint = resolve_contextworld_path(
        checkpoint_spec["path"],
        repo_root=ROOT,
    )
    _require(
        file_sha256(checkpoint) == checkpoint_spec["sha256"],
        "Failed-checkpoint hash changed",
    )
    normalizer_spec = config["inputs"]["normalizer"]
    normalizer = resolve_contextworld_path(
        normalizer_spec["path"],
        repo_root=ROOT,
    )
    _require(
        file_sha256(normalizer) == normalizer_spec["sha256"],
        "Normalizer hash changed",
    )
    catalog_spec = config["inputs"]["paired_domain_catalog"]
    catalog_path = resolve_contextworld_path(
        catalog_spec["path"],
        repo_root=ROOT,
    )
    _require(
        file_sha256(catalog_path) == catalog_spec["sha256"],
        "Paired domain catalog hash changed",
    )
    catalog = load_domain_catalog(catalog_path)

    adapter_class = (
        StableWorldModelLeWMHistory7Adapter
        if args.model_family == "lewm"
        else StableWorldModelPLDMHistory7DiagnosticAdapter
    )
    adapter = adapter_class.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=str(
            args.stablewm_repo
            or config["inputs"]["stable_worldmodel"]["repo"]
        ),
        stablewm_ref=str(
            config["inputs"]["stable_worldmodel"]["commit"]
        ),
        device=args.device,
    )
    model = adapter.model
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    initial_state_sha256 = state_dict_sha256_from_state(initial_state)
    _require(
        initial_state_sha256 == adapter.frozen_state_hash(),
        "Initial model state hash mismatch",
    )

    query_count = int(config["data"]["queries_per_track"])
    train_assets = _selected_assets(
        catalog,
        tracks=list(config["data"]["train_tracks"]),
        queries_per_track=query_count,
    )
    heldout_assets = _selected_assets(
        catalog,
        tracks=list(config["data"]["heldout_tracks"]),
        queries_per_track=query_count,
    )
    train_cache = _build_cache(
        adapter,
        train_assets,
        batch_size=int(args.encode_batch_size),
    )
    heldout_cache = _build_cache(
        adapter,
        heldout_assets,
        batch_size=int(args.encode_batch_size),
    )
    del train_assets, heldout_assets
    _require(
        not set(train_cache.query_ids) & set(heldout_cache.query_ids),
        "Train and heldout query identifiers overlap",
    )
    training = config["training"]
    configured_variants = tuple(str(value) for value in training["variants"])
    _require(
        configured_variants
        and set(configured_variants).issubset(VARIANTS),
        f"Unsupported configured variants: {configured_variants}",
    )
    results = {}
    for variant in configured_variants:
        print(f"[h7-capacity] start {args.model_family} {variant}", flush=True)
        results[variant] = _train_variant(
            model,
            initial_state,
            train_cache,
            heldout_cache,
            variant=variant,
            device=str(args.device),
            seed=int(training["seed"]),
            steps=int(training["optimizer_steps"]),
            examples_per_step=int(training["examples_per_step"]),
            learning_rate=float(training["optimizer"]["learning_rate"]),
            weight_decay=float(training["optimizer"]["weight_decay"]),
            snapshot_steps=tuple(
                int(value) for value in training["snapshots"]
            ),
        )
        final = results[variant]["final"]
        print(
            f"[h7-capacity] completed {variant}: "
            f"train_target={final['train']['exact_target_selection_rate']:.4f} "
            f"heldout_target="
            f"{final['heldout']['exact_target_selection_rate']:.4f}",
            flush=True,
        )

    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed_post_hoc_capacity_diagnostic",
        "claim_boundary": config["claim_boundary"],
        "model_family": str(args.model_family),
        "identity": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "normalizer": str(normalizer),
            "normalizer_sha256": file_sha256(normalizer),
            "catalog": str(catalog_path),
            "catalog_sha256": file_sha256(catalog_path),
            "initial_model_state_sha256": initial_state_sha256,
        },
        "cache_identity": {
            "train": _cache_identity(train_cache),
            "heldout": _cache_identity(heldout_cache),
        },
        "training_protocol": training,
        "variants": results,
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "model_family": args.model_family,
                "output": str(output),
                "final": {
                    variant: {
                        split: {
                            "exact_target_selection_rate": result[
                                "final"
                            ][split]["exact_target_selection_rate"],
                            "exact_history_selection_rate": result[
                                "final"
                            ][split]["exact_history_selection_rate"],
                            "magnitude_ratio": result["final"][split][
                                "latent_alignment"
                            ][
                                "prediction_to_target_pair_magnitude_ratio"
                            ],
                        }
                        for split in ("train", "heldout")
                    }
                    for variant, result in results.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
