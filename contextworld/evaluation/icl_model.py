from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from contextworld.paths import resolve_contextworld_path

from .protocol import ColumnStandardizer, infer_model_protocol


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(model: Any) -> str:
    """Hash parameters and persistent buffers in a device-independent order."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _pixel_key(pixel: np.ndarray) -> str:
    value = np.ascontiguousarray(pixel)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _normalize_action_blocks(
    blocks: np.ndarray,
    standardizer: ColumnStandardizer,
) -> np.ndarray:
    blocks = np.asarray(blocks, dtype=np.float32)
    if blocks.ndim != 3 or blocks.shape[1:] != (5, 2):
        raise ValueError(f"Expected (tokens,5,2) raw actions, got {blocks.shape}")
    normalized = standardizer.transform(blocks.reshape(-1, 2)).astype(np.float32)
    return normalized.reshape(blocks.shape[0], -1)


def _register_pixels(
    registry: dict[str, np.ndarray], pixels: np.ndarray
) -> list[str]:
    values = np.asarray(pixels, dtype=np.uint8)
    if values.ndim == 3:
        values = values[None]
    keys = []
    for value in values:
        key = _pixel_key(value)
        registry.setdefault(key, value.copy())
        keys.append(key)
    return keys


def _condition_order(bundle: dict[str, Any]) -> list[str]:
    if bundle["family"] == "speed_door_composition":
        return ["correct", "wrong_speed", "irrelevant_door", "wrong_both"]
    return ["correct", "wrong", "irrelevant"]


def _build_samples(
    catalog: dict[str, Any],
    *,
    repo_root: Path,
    action_standardizer: ColumnStandardizer,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    pixel_registry: dict[str, np.ndarray] = {}
    bundle_assets: dict[str, dict[str, Any]] = {}

    for bundle in catalog["bundles"]:
        payload_path = resolve_contextworld_path(
            bundle["payload"], repo_root=repo_root
        )
        if file_sha256(payload_path) != bundle["payload_sha256"]:
            raise RuntimeError(f"Payload hash mismatch: {payload_path}")
        with np.load(payload_path, allow_pickle=False) as payload:
            query_key = _register_pixels(pixel_registry, payload["query_pixels"])[0]
            target_key = _register_pixels(pixel_registry, payload["target_pixels"])[0]
            candidate_keys = _register_pixels(pixel_registry, payload["candidate_pixels"])
            candidate_names = [entry["name"] for entry in bundle["candidates"]]
            bundle_assets[bundle["query_id"]] = {
                "query_key": query_key,
                "target_key": target_key,
                "candidate_keys": candidate_keys,
                "candidate_names": candidate_names,
                "correct_candidate_index": int(bundle["correct_candidate_index"]),
            }

            base = {
                "query_id": bundle["query_id"],
                "paired_group_id": bundle["paired_group_id"],
                "source_scenario_id": bundle["source_scenario_id"],
                "family": bundle["family"],
                "regime": bundle["regime"],
                "query_factors": bundle["query_factors"],
            }
            query_actions = payload["query_action"][None]
            samples.append(
                {
                    **base,
                    "condition": "none",
                    "context_budget": 0,
                    "pixel_keys": [query_key],
                    "actions": _normalize_action_blocks(
                        query_actions, action_standardizer
                    ),
                }
            )

            for budget in (1, 2):
                for condition in _condition_order(bundle):
                    prefix = f"context_b{budget}_{condition}"
                    context_keys = _register_pixels(
                        pixel_registry, payload[f"{prefix}_pixels"]
                    )
                    raw_actions = np.concatenate(
                        [payload[f"{prefix}_actions"], query_actions], axis=0
                    )
                    samples.append(
                        {
                            **base,
                            "condition": condition,
                            "context_budget": budget,
                            "pixel_keys": [*context_keys, query_key],
                            "actions": _normalize_action_blocks(
                                raw_actions, action_standardizer
                            ),
                        }
                    )
                if budget == 2:
                    prefix = "context_b2_correct"
                    permutation = bundle["shuffled"]["permutation"]
                    pixels = payload[f"{prefix}_pixels"]
                    actions = payload[f"{prefix}_actions"]
                    shuffled_keys = _register_pixels(
                        pixel_registry, pixels[permutation]
                    )
                    raw_actions = np.concatenate(
                        [actions[permutation], query_actions], axis=0
                    )
                    samples.append(
                        {
                            **base,
                            "condition": "shuffled",
                            "context_budget": 2,
                            "pixel_keys": [*shuffled_keys, query_key],
                            "actions": _normalize_action_blocks(
                                raw_actions, action_standardizer
                            ),
                        }
                    )

    return samples, pixel_registry, bundle_assets


def _preprocess_pixels(pixels: np.ndarray, device: str):
    import torch

    value = torch.from_numpy(np.asarray(pixels, dtype=np.uint8)).to(device)
    value = value.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (value - mean) / std


def _encode_pixel_registry(
    model: Any,
    registry: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    keys = list(registry)
    embeddings: dict[str, Any] = {}
    with torch.inference_mode():
        for start in range(0, len(keys), batch_size):
            chunk = keys[start : start + batch_size]
            pixels = np.stack([registry[key] for key in chunk])
            tensor = _preprocess_pixels(pixels, device).unsqueeze(1)
            encoded = model.encode({"pixels": tensor})["emb"][:, 0]
            for key, value in zip(chunk, encoded):
                embeddings[key] = value.detach().clone()
    return embeddings


def _bootstrap_scenario_mean(
    values: dict[str, list[float]],
    *,
    seed: int,
    samples: int = 2000,
) -> dict[str, float | int]:
    scenario_values = np.asarray(
        [np.mean(entries) for entries in values.values()], dtype=np.float64
    )
    if scenario_values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "scenarios": 0}
    mean = float(scenario_values.mean())
    if scenario_values.size == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "scenarios": 1}
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, scenario_values.size, size=(samples, scenario_values.size)
    )
    boot = scenario_values[indices].mean(axis=1)
    return {
        "mean": mean,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "scenarios": int(scenario_values.size),
    }


def _aggregate_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    metrics = (
        "latent_mse",
        "latent_cosine_distance",
        "persistence_latent_mse",
        "counterfactual_accuracy",
        "counterfactual_margin",
        "context_output_shift_from_none",
    )
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (record["family"], record["context_budget"], record["condition"])
        ].append(record)

    aggregates: list[dict[str, Any]] = []
    for (family, budget, condition), entries in sorted(grouped.items()):
        item: dict[str, Any] = {
            "family": family,
            "context_budget": budget,
            "condition": condition,
            "queries": len(entries),
            "scenarios": len({entry["source_scenario_id"] for entry in entries}),
        }
        for metric_index, metric in enumerate(metrics):
            by_scenario: dict[str, list[float]] = defaultdict(list)
            for entry in entries:
                value = entry.get(metric)
                if value is not None and np.isfinite(value):
                    by_scenario[entry["source_scenario_id"]].append(float(value))
            item[metric] = _bootstrap_scenario_mean(
                by_scenario,
                seed=seed + metric_index + 101 * budget + len(condition),
            )
        aggregates.append(item)
    return aggregates


def _aggregate_contrasts(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    lookup = {
        (record["query_id"], record["context_budget"], record["condition"]): record
        for record in records
    }
    contrasts: list[dict[str, Any]] = []
    families = sorted({record["family"] for record in records})
    for family in families:
        query_ids = sorted(
            {record["query_id"] for record in records if record["family"] == family}
        )
        wrong_name = "wrong_speed" if family == "speed_door_composition" else "wrong"
        irrelevant_name = "irrelevant_door" if family == "speed_door_composition" else "irrelevant"
        for budget in (1, 2):
            metric_values: dict[str, dict[str, list[float]]] = {
                "prediction_icl_gain": defaultdict(list),
                "wrong_context_separation": defaultdict(list),
                "irrelevant_context_gap": defaultdict(list),
                "counterfactual_icl_gain": defaultdict(list),
            }
            if budget == 2:
                metric_values["shuffled_context_separation"] = defaultdict(list)
            queries = 0
            for query_id in query_ids:
                none = lookup[(query_id, 0, "none")]
                correct = lookup[(query_id, budget, "correct")]
                wrong = lookup[(query_id, budget, wrong_name)]
                irrelevant = lookup[(query_id, budget, irrelevant_name)]
                scenario = correct["source_scenario_id"]
                metric_values["prediction_icl_gain"][scenario].append(
                    none["latent_mse"] - correct["latent_mse"]
                )
                metric_values["wrong_context_separation"][scenario].append(
                    wrong["latent_mse"] - correct["latent_mse"]
                )
                metric_values["irrelevant_context_gap"][scenario].append(
                    irrelevant["latent_mse"] - correct["latent_mse"]
                )
                metric_values["counterfactual_icl_gain"][scenario].append(
                    correct["counterfactual_accuracy"]
                    - none["counterfactual_accuracy"]
                )
                if budget == 2:
                    shuffled = lookup[(query_id, 2, "shuffled")]
                    metric_values["shuffled_context_separation"][scenario].append(
                        shuffled["latent_mse"] - correct["latent_mse"]
                    )
                queries += 1
            item: dict[str, Any] = {
                "family": family,
                "context_budget": budget,
                "queries": queries,
            }
            for metric_index, (metric, values) in enumerate(metric_values.items()):
                item[metric] = _bootstrap_scenario_mean(
                    values,
                    seed=seed + 1009 * budget + metric_index + len(family),
                )
            contrasts.append(item)
    return contrasts


def _speed_factor_curve(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for record in records:
        if record["family"] == "speed" and record["condition"] == "none":
            grouped[float(record["query_factors"]["agent.speed"])].append(
                float(record["latent_mse"])
            )
    reference = float(np.mean(grouped[5.0])) if 5.0 in grouped else float("nan")
    return [
        {
            "agent_speed": speed,
            "latent_mse": float(np.mean(values)),
            "delta_from_speed_5": float(np.mean(values) - reference),
        }
        for speed, values in sorted(grouped.items())
    ]


def evaluate_frozen_tworoom_icl(
    *,
    model: Any,
    checkpoint_path: Path,
    catalog_path: Path,
    repo_root: Path,
    original_h5: Path,
    action_standardizer: ColumnStandardizer,
    device: str,
    encode_batch_size: int = 64,
    predictor_batch_size: int = 128,
    seed: int = 3072,
    family: str | None = None,
) -> dict[str, Any]:
    """Evaluate native latent prediction under controlled context prefixes."""

    import torch
    import torch.nn.functional as F

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if family is not None:
        selected_bundles = [
            bundle for bundle in catalog["bundles"] if bundle["family"] == family
        ]
        if not selected_bundles:
            available = sorted({bundle["family"] for bundle in catalog["bundles"]})
            raise ValueError(
                f"No bundles for family={family!r}; available={available}"
            )
        catalog = {**catalog, "bundles": selected_bundles}
    protocol = infer_model_protocol(model, action_dim=2)
    maximum_context = int(protocol["history_size"]) - 1
    if maximum_context != 2:
        raise RuntimeError(
            f"Expected a 3-token model window and two prior transitions, got {protocol}"
        )
    if catalog["protocol"]["maximum_prior_context_transitions"] != maximum_context:
        raise RuntimeError("Catalog/model context capacity mismatch")

    model = model.to(device).eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Failed to freeze model parameters")
    before_hash = state_dict_sha256(model)
    samples, pixel_registry, bundle_assets = _build_samples(
        catalog,
        repo_root=repo_root,
        action_standardizer=action_standardizer,
    )
    pixel_embeddings = _encode_pixel_registry(
        model,
        pixel_registry,
        device=device,
        batch_size=encode_batch_size,
    )

    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_length[len(sample["pixel_keys"])].append(sample)

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for length, entries in sorted(by_length.items()):
            for start in range(0, len(entries), predictor_batch_size):
                chunk = entries[start : start + predictor_batch_size]
                embeddings = torch.stack(
                    [
                        torch.stack([pixel_embeddings[key] for key in entry["pixel_keys"]])
                        for entry in chunk
                    ]
                )
                actions = torch.from_numpy(
                    np.stack([entry["actions"] for entry in chunk])
                ).to(device)
                action_embeddings = model.action_encoder(actions)
                predicted = model.predict(embeddings, action_embeddings)[:, -1]

                for entry, prediction in zip(chunk, predicted):
                    assets = bundle_assets[entry["query_id"]]
                    target = pixel_embeddings[assets["target_key"]]
                    query = pixel_embeddings[assets["query_key"]]
                    candidates = torch.stack(
                        [pixel_embeddings[key] for key in assets["candidate_keys"]]
                    )
                    candidate_losses = (
                        (candidates - prediction.unsqueeze(0)).square().mean(dim=-1)
                    )
                    selected = int(candidate_losses.argmin().item())
                    correct_index = int(assets["correct_candidate_index"])
                    wrong_mask = torch.arange(
                        candidate_losses.numel(), device=candidate_losses.device
                    ) != correct_index
                    counterfactual_margin = float(
                        (
                            candidate_losses[wrong_mask].min()
                            - candidate_losses[correct_index]
                        ).item()
                    )
                    record = {
                        **{key: value for key, value in entry.items() if key not in {"pixel_keys", "actions"}},
                        "input_tokens": length,
                        "raw_context_actions": int(entry["context_budget"] * 5),
                        "latent_mse": float(
                            F.mse_loss(prediction, target, reduction="mean").item()
                        ),
                        "latent_cosine_distance": float(
                            (1.0 - F.cosine_similarity(prediction[None], target[None])).item()
                        ),
                        "persistence_latent_mse": float(
                            F.mse_loss(query, target, reduction="mean").item()
                        ),
                        "candidate_names": assets["candidate_names"],
                        "candidate_latent_mse": [
                            float(value) for value in candidate_losses.detach().cpu().tolist()
                        ],
                        "selected_candidate_index": selected,
                        "selected_candidate": assets["candidate_names"][selected],
                        "correct_candidate_index": correct_index,
                        "counterfactual_margin": counterfactual_margin,
                        "counterfactual_accuracy": float(counterfactual_margin > 0.0),
                        "_prediction": prediction.detach().cpu().numpy(),
                    }
                    records.append(record)

    none_predictions = {
        record["query_id"]: record["_prediction"]
        for record in records
        if record["condition"] == "none"
    }
    for record in records:
        baseline = none_predictions[record["query_id"]]
        record["context_output_shift_from_none"] = float(
            np.mean((record["_prediction"] - baseline) ** 2)
        )
        del record["_prediction"]

    after_hash = state_dict_sha256(model)
    frozen = before_hash == after_hash
    if not frozen:
        raise RuntimeError("Model state changed during frozen evaluation")

    aggregates = _aggregate_records(records, seed)
    contrasts = _aggregate_contrasts(records, seed)
    contrast_lookup = {
        (entry["family"], entry["context_budget"]): entry for entry in contrasts
    }
    signals: dict[str, Any] = {}
    evaluated_families = sorted({record["family"] for record in records})
    for evaluated_family in evaluated_families:
        k2 = contrast_lookup[(evaluated_family, 2)]
        gain = float(k2["prediction_icl_gain"]["mean"])
        separation = float(k2["wrong_context_separation"]["mean"])
        signals[evaluated_family] = {
            "prediction_icl_gain_k2": gain,
            "wrong_context_separation_k2": separation,
            "positive_correct_context_gain": gain > 0.0,
            "correct_beats_wrong_context": separation > 0.0,
        }

    return {
        "schema_version": 1,
        "benchmark": "contextworld_tworoom_icl_v1",
        "run_kind": "validation_diagnostic",
        "status": "passed",
        "model_id": "M_orig",
        "track": "paired_context_prediction",
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
            "passed": frozen,
        },
        "model_protocol": {
            **protocol,
            "maximum_prior_context_transitions": maximum_context,
            "supported_context_budgets": [0, 1, 2],
            "k1_speed_context_is_intentionally_nonidentifying": True,
            "k2_speed_context_is_diagnostic": True,
        },
        "data": {
            "catalog": str(catalog_path.resolve()),
            "catalog_sha256": file_sha256(catalog_path),
            "bundles": len(catalog["bundles"]),
            "families": evaluated_families,
            "unique_pixels_encoded": len(pixel_registry),
            "normalization_source": str(original_h5.resolve()),
            "factor_values_exposed_to_model": False,
        },
        "metrics": {
            "latent_mse": "mean squared error in native 192-D target latent",
            "prediction_icl_gain": "loss_none_minus_loss_correct",
            "wrong_context_separation": "loss_wrong_minus_loss_correct",
            "aggregation": "query_then_scenario_balanced_with_scenario_bootstrap_ci",
        },
        "aggregates": aggregates,
        "contrasts": contrasts,
        "speed_t0_factor_curve": _speed_factor_curve(records),
        "diagnostic_signals": signals,
        "raw_records": records,
    }


__all__ = [
    "evaluate_frozen_tworoom_icl",
    "file_sha256",
    "state_dict_sha256",
]
