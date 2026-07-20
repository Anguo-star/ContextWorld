from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from contextworld.paths import portable_contextworld_path, resolve_contextworld_path

from .icl_model import (
    _bootstrap_scenario_mean,
    _encode_pixel_registry,
    _pixel_key,
    file_sha256,
    state_dict_sha256,
)
from .protocol import ColumnStandardizer, load_catalog_regime, scenario_seed
from .tworoom import DOOR_COLUMN, SPEED_COLUMN


FRAMESKIP = 5
NUM_STEPS = 4
HISTORY_SIZE = 3
ACTION_DIM = 2


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _relative_path(path: Path, repo_root: Path) -> str:
    return portable_contextworld_path(path, repo_root=repo_root)


def _clip_sha256(clip: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(clip):
        value = np.ascontiguousarray(clip[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _to_hwc_uint8(pixels: np.ndarray) -> np.ndarray:
    value = np.asarray(pixels)
    if value.ndim != 4:
        raise ValueError(f"Expected four-dimensional pixels, got {value.shape}")
    if value.shape[1] == 3:
        value = value.transpose(0, 2, 3, 1)
    if value.shape[-1] != 3:
        raise ValueError(f"Expected RGB pixels, got {value.shape}")
    if value.dtype != np.uint8:
        raise ValueError(f"Expected uint8 pixels, got {value.dtype}")
    return np.ascontiguousarray(value)


def history_token_slice(context_budget: int) -> slice:
    """Return the suffix ending at the shared query token.

    A four-frame training clip is ``[c1, c2, query, target]``.  StableWM's
    history size counts the query token, while the benchmark's K counts only
    prior completed transitions.
    """

    if context_budget not in (0, 1, 2):
        raise ValueError(f"Unsupported context budget: {context_budget}")
    return slice(HISTORY_SIZE - context_budget - 1, HISTORY_SIZE)


def _factor_value(column: str, value: np.ndarray) -> float | int:
    rows = np.asarray(value).reshape(NUM_STEPS, -1)
    if not np.array_equal(rows, np.repeat(rows[:1], NUM_STEPS, axis=0)):
        raise ValueError(f"Factor {column} changes inside a natural-history clip")
    first = rows[0]
    if column == SPEED_COLUMN:
        if first.size != 1:
            raise ValueError(f"Unexpected speed shape: {value.shape}")
        return float(first[0])
    if column == DOOR_COLUMN:
        if first.size not in (1, 3) or not np.all(first == first[0]):
            raise ValueError(f"Unexpected door shape/value: {value.shape}, {first}")
        return int(first[0])
    raise KeyError(column)


def _load_clip(dataset: Any, *, episode: int, start: int) -> dict[str, np.ndarray]:
    end = start + NUM_STEPS * FRAMESKIP
    loaded = dataset.load_chunk(
        np.asarray([episode], dtype=np.int64),
        np.asarray([start], dtype=np.int64),
        np.asarray([end], dtype=np.int64),
    )[0]
    return {key: np.asarray(value) for key, value in loaded.items()}


def _scenario_dataset(swm: Any, path: Path) -> tuple[Any, list[str]]:
    probe = swm.data.LanceDataset(path=path, frameskip=1, num_steps=1)
    factor_columns = [
        column for column in (SPEED_COLUMN, DOOR_COLUMN) if column in probe.column_names
    ]
    keys = ["pixels", "action", "state", *factor_columns]
    missing = [key for key in ("pixels", "action", "state") if key not in probe.column_names]
    if missing:
        raise KeyError(f"{path} is missing natural-history columns {missing}")
    dataset = swm.data.LanceDataset(
        path=path,
        frameskip=FRAMESKIP,
        num_steps=NUM_STEPS,
        keys_to_load=keys,
    )
    return dataset, factor_columns


def build_natural_history_catalog(
    *,
    swm: Any,
    repo_root: Path,
    sources: list[dict[str, Any]],
    generator_seed: int = 20260714,
    clips_per_scenario: int = 3,
) -> dict[str, Any]:
    """Freeze one training-equivalent contiguous clip per episode.

    Only validation regimes should be provided.  No test scenario is read or
    materialized by this builder.
    """

    clips: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    excluded_episodes: list[dict[str, Any]] = []
    span = NUM_STEPS * FRAMESKIP
    if clips_per_scenario <= 0:
        raise ValueError("clips_per_scenario must be positive")
    for source in sources:
        family = str(source["family"])
        regime = str(source["regime"])
        catalog_path = resolve_contextworld_path(
            source["catalog"], repo_root=repo_root
        )
        paths = load_catalog_regime(catalog_path, regime, repo_root=repo_root)
        source_records.append(
            {
                "family": family,
                "regime": regime,
                "catalog": _relative_path(catalog_path, repo_root),
                "catalog_sha256": file_sha256(catalog_path),
                "scenarios": len(paths),
            }
        )
        for path in paths:
            dataset, factor_columns = _scenario_dataset(swm, path)
            rng = np.random.default_rng(scenario_seed(generator_seed, path))
            lengths = np.asarray(dataset.lengths, dtype=np.int64)
            eligible = np.flatnonzero(lengths >= span)
            if len(eligible) < clips_per_scenario:
                raise ValueError(
                    f"Need {clips_per_scenario} eligible episodes for {path}, found {len(eligible)}"
                    )
            selected = sorted(
                int(value)
                for value in rng.choice(
                    eligible, size=clips_per_scenario, replace=False
                )
            )
            for episode, length in enumerate(lengths):
                if episode not in selected:
                    excluded_episodes.append(
                        {
                            "source_scenario_id": path.stem,
                            "episode": int(episode),
                            "length": int(length),
                            "reason": (
                                "shorter_than_training_equivalent_span"
                                if int(length) < span
                                else "scenario_balance_cap"
                            ),
                        }
                    )
            for episode in selected:
                length = int(lengths[episode])
                start = int(rng.integers(0, int(length) - span + 1))
                clip = _load_clip(dataset, episode=episode, start=start)
                expected_shapes = {
                    "pixels": (NUM_STEPS, 3, 224, 224),
                    "action": (NUM_STEPS, FRAMESKIP * ACTION_DIM),
                }
                for key, expected in expected_shapes.items():
                    if tuple(clip[key].shape) != expected:
                        raise ValueError(
                            f"Unexpected {key} shape for {path}: {clip[key].shape}, expected={expected}"
                        )
                factors: dict[str, float | int] = {}
                if SPEED_COLUMN in clip:
                    factors["agent.speed"] = _factor_value(SPEED_COLUMN, clip[SPEED_COLUMN])
                if DOOR_COLUMN in clip:
                    factors["door.position"] = _factor_value(DOOR_COLUMN, clip[DOOR_COLUMN])
                identity = {
                    "family": family,
                    "scenario": path.name,
                    "episode": int(episode),
                    "start": start,
                    "factors": factors,
                }
                clip_id = "twnh-" + hashlib.sha256(
                    _canonical_json(identity).encode("utf-8")
                ).hexdigest()[:16]
                clips.append(
                    {
                        "clip_id": clip_id,
                        "family": family,
                        "regime": regime,
                        "source_scenario_id": path.stem,
                        "source_lance": _relative_path(path, repo_root),
                        "episode": int(episode),
                        "start_row": start,
                        "end_row_exclusive": start + span,
                        "query_token_index": 2,
                        "target_frame_index": 3,
                        "factors": factors,
                        "factor_columns": factor_columns,
                        "clip_sha256": _clip_sha256(clip),
                    }
                )

    family_counts: dict[str, int] = defaultdict(int)
    scenario_counts: dict[str, set[str]] = defaultdict(set)
    for clip in clips:
        family_counts[clip["family"]] += 1
        scenario_counts[clip["family"]].add(clip["source_scenario_id"])
    return {
        "schema_version": 1,
        "benchmark": "contextworld_tworoom_natural_history_v1",
        "catalog_kind": "training_equivalent_contiguous_validation_history",
        "split": "validation",
        "generator_seed": int(generator_seed),
        "protocol": {
            "frameskip": FRAMESKIP,
            "num_steps": NUM_STEPS,
            "model_history_tokens": HISTORY_SIZE,
            "supported_context_budgets": [0, 1, 2],
            "clip_layout": ["context_1", "context_2", "query", "target"],
            "context_source": "same_episode_same_environment_parameters",
            "one_clip_per_episode": True,
            "clips_per_scenario": int(clips_per_scenario),
            "short_episodes_are_excluded_not_padded": True,
            "arbitrary_cross_trajectory_splicing": False,
            "factor_labels_exposed_to_model": False,
            "k0_vs_k1_k2_is_descriptive_due_to_sequence_length": True,
            "causal_context_controls_live_in": "strict_paired_context_query_catalog",
        },
        "sources": source_records,
        "counts": {
            "clips": len(clips),
            "by_family": dict(sorted(family_counts.items())),
            "scenarios_by_family": {
                family: len(values) for family, values in sorted(scenario_counts.items())
            },
        },
        "excluded_episodes": excluded_episodes,
        "clips": sorted(clips, key=lambda item: item["clip_id"]),
    }


def validate_natural_history_catalog(
    catalog: dict[str, Any],
    *,
    swm: Any,
    repo_root: Path,
    family: str | None = None,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0
    cached: dict[Path, tuple[Any, list[str]]] = {}
    seen: set[str] = set()
    scenario_counts: dict[str, int] = defaultdict(int)
    expected_per_scenario = int(catalog.get("protocol", {}).get("clips_per_scenario", 0))
    if catalog.get("split") != "validation":
        failures.append({"clip_id": "catalog", "error": "Catalog split is not validation"})
    if expected_per_scenario <= 0:
        failures.append(
            {"clip_id": "catalog", "error": "Invalid clips_per_scenario"}
        )
    clips = [
        entry
        for entry in catalog.get("clips", [])
        if family is None or entry["family"] == family
    ]
    if not clips:
        available = sorted(
            {entry["family"] for entry in catalog.get("clips", [])}
        )
        raise ValueError(f"No clips for family={family!r}; available={available}")
    for entry in clips:
        clip_id = str(entry.get("clip_id"))
        try:
            if clip_id in seen:
                raise ValueError(f"Duplicate clip_id: {clip_id}")
            seen.add(clip_id)
            if not str(entry.get("regime", "")).startswith("validation"):
                raise ValueError(f"Non-validation regime: {entry.get('regime')}")
            scenario_id = str(entry["source_scenario_id"])
            scenario_counts[scenario_id] += 1
            path = resolve_contextworld_path(
                entry["source_lance"], repo_root=repo_root
            )
            if path not in cached:
                cached[path] = _scenario_dataset(swm, path)
            dataset, factor_columns = cached[path]
            episode = int(entry["episode"])
            start = int(entry["start_row"])
            if episode < 0 or episode >= len(dataset.lengths):
                raise IndexError(f"Invalid episode {episode}")
            if start < 0 or start + NUM_STEPS * FRAMESKIP > int(dataset.lengths[episode]):
                raise IndexError(f"Invalid clip bounds episode={episode}, start={start}")
            clip = _load_clip(dataset, episode=episode, start=start)
            if _clip_sha256(clip) != entry["clip_sha256"]:
                raise ValueError("Clip hash mismatch")
            if factor_columns != entry["factor_columns"]:
                raise ValueError(
                    f"Factor column mismatch: {factor_columns} != {entry['factor_columns']}"
                )
            observed: dict[str, float | int] = {}
            if SPEED_COLUMN in clip:
                observed["agent.speed"] = _factor_value(SPEED_COLUMN, clip[SPEED_COLUMN])
            if DOOR_COLUMN in clip:
                observed["door.position"] = _factor_value(DOOR_COLUMN, clip[DOOR_COLUMN])
            if observed != entry["factors"]:
                raise ValueError(f"Factor mismatch: {observed} != {entry['factors']}")
            _to_hwc_uint8(clip["pixels"])
            if tuple(clip["action"].shape) != (NUM_STEPS, FRAMESKIP * ACTION_DIM):
                raise ValueError(f"Unexpected action shape: {clip['action'].shape}")
            checked += 1
        except Exception as exc:  # report every bad frozen clip in one pass
            failures.append({"clip_id": clip_id, "error": str(exc)})
    for scenario_id, count in sorted(scenario_counts.items()):
        if count != expected_per_scenario:
            failures.append(
                {
                    "clip_id": "catalog",
                    "error": (
                        f"Scenario {scenario_id} has {count} clips; "
                        f"expected {expected_per_scenario}"
                    ),
                }
            )
    return {
        "schema_version": 1,
        "passed": not failures and checked == len(clips),
        "clips_checked": checked,
        "families": sorted({entry["family"] for entry in clips}),
        "scenarios_checked": len(scenario_counts),
        "clips_per_scenario": expected_per_scenario,
        "failures": failures,
        "training_equivalent_layout": True,
        "test_split_read": False,
    }


def _normalize_flat_action_blocks(
    blocks: np.ndarray, standardizer: ColumnStandardizer
) -> np.ndarray:
    value = np.asarray(blocks, dtype=np.float32)
    if value.shape != (NUM_STEPS, FRAMESKIP * ACTION_DIM):
        raise ValueError(f"Unexpected natural-history action shape: {value.shape}")
    normalized = standardizer.transform(value.reshape(-1, ACTION_DIM)).astype(np.float32)
    return normalized.reshape(NUM_STEPS, -1)


def _aggregate_natural_records(
    records: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    metrics = (
        "query_latent_mse",
        "query_latent_cosine_distance",
        "sequence_latent_mse",
        "persistence_latent_mse",
        "context_output_shift_from_k0",
    )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["family"], record["context_budget"])].append(record)
    output: list[dict[str, Any]] = []
    for (family, budget), entries in sorted(grouped.items()):
        item: dict[str, Any] = {
            "family": family,
            "context_budget": budget,
            "clips": len(entries),
            "scenarios": len({entry["source_scenario_id"] for entry in entries}),
        }
        for metric_index, metric in enumerate(metrics):
            by_scenario: dict[str, list[float]] = defaultdict(list)
            for entry in entries:
                by_scenario[entry["source_scenario_id"]].append(float(entry[metric]))
            item[metric] = _bootstrap_scenario_mean(
                by_scenario,
                seed=seed + 101 * budget + metric_index + len(family),
            )
        output.append(item)
    return output


def _natural_contrasts(
    records: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    lookup = {
        (record["clip_id"], record["context_budget"]): record for record in records
    }
    output: list[dict[str, Any]] = []
    families = sorted({record["family"] for record in records})
    for family in families:
        clip_ids = sorted(
            {record["clip_id"] for record in records if record["family"] == family}
        )
        for budget in (1, 2):
            absolute: dict[str, list[float]] = defaultdict(list)
            relative: dict[str, list[float]] = defaultdict(list)
            for clip_id in clip_ids:
                none = lookup[(clip_id, 0)]
                context = lookup[(clip_id, budget)]
                scenario = context["source_scenario_id"]
                gain = none["query_latent_mse"] - context["query_latent_mse"]
                absolute[scenario].append(float(gain))
                relative[scenario].append(
                    float(gain / max(none["query_latent_mse"], 1e-12))
                )
            output.append(
                {
                    "family": family,
                    "context_budget": budget,
                    "clips": len(clip_ids),
                    "interpretation": "descriptive_same_trajectory_gain_not_causal_length_matched",
                    "query_prediction_gain": _bootstrap_scenario_mean(
                        absolute, seed=seed + 1009 * budget + len(family)
                    ),
                    "relative_query_prediction_gain": _bootstrap_scenario_mean(
                        relative, seed=seed + 2017 * budget + len(family)
                    ),
                }
            )
    return output


def evaluate_frozen_natural_history(
    *,
    model: Any,
    checkpoint_path: Path,
    catalog: dict[str, Any],
    swm: Any,
    repo_root: Path,
    original_h5: Path,
    action_standardizer: ColumnStandardizer,
    device: str,
    encode_batch_size: int = 64,
    predictor_batch_size: int = 128,
    seed: int = 3072,
    family: str | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen model on real contiguous OOD histories."""

    import torch
    import torch.nn.functional as F

    validation = validate_natural_history_catalog(
        catalog, swm=swm, repo_root=repo_root, family=family
    )
    if not validation["passed"]:
        raise RuntimeError(f"Natural-history catalog validation failed: {validation}")

    selected_clips = [
        entry
        for entry in catalog["clips"]
        if family is None or entry["family"] == family
    ]
    if not selected_clips:
        available = sorted({entry["family"] for entry in catalog["clips"]})
        raise ValueError(f"No clips for family={family!r}; available={available}")

    model = model.to(device).eval()
    model.requires_grad_(False)
    before_hash = state_dict_sha256(model)

    pixel_registry: dict[str, np.ndarray] = {}
    assets: list[dict[str, Any]] = []
    datasets: dict[Path, tuple[Any, list[str]]] = {}
    all_normalized_actions: list[np.ndarray] = []
    for entry in selected_clips:
        path = resolve_contextworld_path(
            entry["source_lance"], repo_root=repo_root
        )
        if path not in datasets:
            datasets[path] = _scenario_dataset(swm, path)
        dataset, _ = datasets[path]
        clip = _load_clip(
            dataset, episode=int(entry["episode"]), start=int(entry["start_row"])
        )
        pixels = _to_hwc_uint8(clip["pixels"])
        pixel_keys: list[str] = []
        for pixel in pixels:
            key = _pixel_key(pixel)
            pixel_registry.setdefault(key, pixel.copy())
            pixel_keys.append(key)
        normalized_actions = _normalize_flat_action_blocks(
            clip["action"], action_standardizer
        )
        all_normalized_actions.append(normalized_actions)
        assets.append(
            {
                **entry,
                "pixel_keys": pixel_keys,
                "normalized_actions": normalized_actions,
            }
        )

    embeddings = _encode_pixel_registry(
        model,
        pixel_registry,
        device=device,
        batch_size=encode_batch_size,
    )
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for budget in (0, 1, 2):
            token_slice = history_token_slice(budget)
            for start in range(0, len(assets), predictor_batch_size):
                chunk = assets[start : start + predictor_batch_size]
                input_embeddings = torch.stack(
                    [
                        torch.stack(
                            [embeddings[key] for key in asset["pixel_keys"][token_slice]]
                        )
                        for asset in chunk
                    ]
                )
                actions = torch.from_numpy(
                    np.stack(
                        [asset["normalized_actions"][token_slice] for asset in chunk]
                    )
                ).to(device)
                predicted = model.predict(
                    input_embeddings, model.action_encoder(actions)
                )
                target_embeddings = torch.stack(
                    [
                        torch.stack(
                            [
                                embeddings[key]
                                for key in asset["pixel_keys"][
                                    token_slice.start + 1 : HISTORY_SIZE + 1
                                ]
                            ]
                        )
                        for asset in chunk
                    ]
                )
                for asset, prediction, targets in zip(
                    chunk, predicted, target_embeddings
                ):
                    query = embeddings[asset["pixel_keys"][2]]
                    target = embeddings[asset["pixel_keys"][3]]
                    records.append(
                        {
                            **{
                                key: value
                                for key, value in asset.items()
                                if key not in {"pixel_keys", "normalized_actions"}
                            },
                            "context_budget": budget,
                            "input_tokens": budget + 1,
                            "query_latent_mse": float(
                                F.mse_loss(prediction[-1], target).item()
                            ),
                            "query_latent_cosine_distance": float(
                                (1.0 - F.cosine_similarity(prediction[-1][None], target[None])).item()
                            ),
                            "sequence_latent_mse": float(
                                F.mse_loss(prediction, targets).item()
                            ),
                            "persistence_latent_mse": float(
                                F.mse_loss(query, target).item()
                            ),
                            "_prediction": prediction[-1].detach().cpu().numpy(),
                        }
                    )

    k0_predictions = {
        record["clip_id"]: record["_prediction"]
        for record in records
        if record["context_budget"] == 0
    }
    for record in records:
        baseline = k0_predictions[record["clip_id"]]
        record["context_output_shift_from_k0"] = float(
            np.mean((record["_prediction"] - baseline) ** 2)
        )
        del record["_prediction"]

    after_hash = state_dict_sha256(model)
    if before_hash != after_hash:
        raise RuntimeError("Model state changed during natural-history evaluation")
    normalized = np.concatenate(all_normalized_actions, axis=0)

    scenario_summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["source_scenario_id"], record["context_budget"])].append(record)
    for (scenario, budget), entries in sorted(grouped.items()):
        scenario_summaries.append(
            {
                "family": entries[0]["family"],
                "regime": entries[0]["regime"],
                "source_scenario_id": scenario,
                "factors": entries[0]["factors"],
                "context_budget": budget,
                "clips": len(entries),
                "query_latent_mse": float(
                    np.mean([entry["query_latent_mse"] for entry in entries])
                ),
                "sequence_latent_mse": float(
                    np.mean([entry["sequence_latent_mse"] for entry in entries])
                ),
            }
        )

    return {
        "schema_version": 1,
        "benchmark": "contextworld_tworoom_natural_history_v1",
        "run_kind": "validation_diagnostic",
        "status": "passed",
        "model_id": "M_orig",
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": file_sha256(checkpoint_path),
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "frozen_weight_audit": {
            "requires_grad_false": not any(
                parameter.requires_grad for parameter in model.parameters()
            ),
            "optimizer_created": False,
            "state_dict_sha256_before": before_hash,
            "state_dict_sha256_after": after_hash,
            "passed": before_hash == after_hash,
        },
        "catalog_validation": validation,
        "data": {
            "clips": len(assets),
            "families": sorted({asset["family"] for asset in assets}),
            "unique_pixels_encoded": len(pixel_registry),
            "normalization_source": str(original_h5.resolve()),
            "context_source": "same_episode_same_environment_parameters",
            "factor_values_exposed_to_model": False,
            "test_split_read": False,
            "normalized_action_rms": float(np.sqrt(np.mean(normalized**2))),
            "normalized_action_max_abs": float(np.max(np.abs(normalized))),
        },
        "metrics": {
            "query_latent_mse": "last-token prediction MSE against the shared next-frame native latent",
            "sequence_latent_mse": "mean native latent MSE over every predicted suffix transition",
            "natural_history_gain": "loss_k0_minus_loss_k; descriptive, not a length-matched causal estimate",
            "aggregation": "clip_then_scenario_balanced_with_scenario_bootstrap_ci",
        },
        "aggregates": _aggregate_natural_records(records, seed=seed),
        "contrasts": _natural_contrasts(records, seed=seed),
        "scenario_summaries": scenario_summaries,
        "raw_records": records,
    }


__all__ = [
    "ACTION_DIM",
    "FRAMESKIP",
    "HISTORY_SIZE",
    "NUM_STEPS",
    "build_natural_history_catalog",
    "evaluate_frozen_natural_history",
    "history_token_slice",
    "validate_natural_history_catalog",
]
