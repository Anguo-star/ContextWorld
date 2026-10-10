#!/usr/bin/env python3
"""Frozen-checkpoint native objective and predictor-gradient audit for Delay H7.

This is a deterministic, no-update diagnostic over the existing start-0
Training panel. It evaluates all seven native one-step prediction positions,
decomposes native MSE into condition response error and common bias, and
compares the final (current-query) position with the other six positions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CONTEXTWORLD = Path(__file__).resolve().parents[1]
DIAGNOSTIC_HELPER = CONTEXTWORLD / "scripts/diagnose_cross_task_decisions.py"
DELAYS = tuple(range(11))
STRATA = ("left/up", "left/down", "right/up", "right/down")
HISTORY = 7
POSITIONS = 7
MSE_CLOSURE_ATOL = 2.0e-6
GRADIENT_CLOSURE_RELATIVE_MAX = 1.0e-3
CANONICAL_MAX_ABS = 3.0e-4
CANONICAL_RELATIVE_L2 = 3.0e-5


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "contextworld_diagnose_cross_task_decisions", DIAGNOSTIC_HELPER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ContextWorld helper: {DIAGNOSTIC_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _select_scenes(manifest: Mapping[str, Any], split: str, per_stratum: int):
    scenes = manifest["splits"][split]["scenes"]
    selected: dict[str, list[dict[str, Any]]] = {key: [] for key in STRATA}
    for entry in scenes:
        key = f"{entry['room']}/{entry['direction']}"
        if key not in selected:
            raise ValueError(f"Unexpected panel stratum: {key}")
        if len(selected[key]) < per_stratum:
            selected[key].append(entry)
    missing = {key: len(rows) for key, rows in selected.items() if len(rows) != per_stratum}
    if missing:
        raise ValueError(f"Insufficient scenes in panel strata: {missing}")
    # Preserve the manifest's existing order, including its source-identity
    # selection order. No model output is used to choose scenes.
    chosen_ids = {str(row["scene_id"]) for rows in selected.values() for row in rows}
    return [row for row in scenes if str(row["scene_id"]) in chosen_ids]


def _tensor_tuple_norm(values: Sequence[Any | None]) -> float:
    total = 0.0
    for value in values:
        if value is not None:
            total += float(value.detach().float().square().sum().cpu())
    return math.sqrt(max(0.0, total))


def _tensor_tuple_dot(
    left: Sequence[Any | None], right: Sequence[Any | None]
) -> float:
    total = 0.0
    for one, two in zip(left, right, strict=True):
        if one is not None and two is not None:
            total += float((one.detach().float() * two.detach().float()).sum().cpu())
    return total


def _gradient_closure(
    reference: Sequence[Any | None],
    components: Sequence[Sequence[Any | None]],
) -> dict[str, float]:
    squared_error = 0.0
    squared_reference = 0.0
    max_abs = 0.0
    for index, value in enumerate(reference):
        component_sum = None
        for rows in components:
            gradient = rows[index]
            if gradient is not None:
                component_sum = (
                    gradient.detach().float()
                    if component_sum is None
                    else component_sum + gradient.detach().float()
                )
        if value is None:
            if component_sum is not None:
                residual = component_sum
                max_abs = max(max_abs, float(residual.abs().max().cpu()))
                squared_error += float(residual.square().sum().cpu())
            continue
        residual = value.detach().float()
        squared_reference += float(residual.square().sum().cpu())
        if component_sum is not None:
            residual = residual - component_sum
        max_abs = max(max_abs, float(residual.abs().max().cpu()))
        squared_error += float(residual.square().sum().cpu())
    residual_norm = math.sqrt(max(0.0, squared_error))
    reference_norm = math.sqrt(squared_reference)
    component_norm_sum = sum(_tensor_tuple_norm(row) for row in components)
    return {
        "max_abs": max_abs,
        "relative_l2": residual_norm / max(reference_norm, 1.0e-30),
        "relative_to_component_norm_sum": residual_norm / max(reference_norm, component_norm_sum, 1.0e-30),
        "reference_norm": reference_norm,
        "component_norm_sum": component_norm_sum,
    }


def _pairwise_gradient_summary(
    gradients: Mapping[str, Sequence[Any | None]],
) -> dict[str, Any]:
    norms = {name: _tensor_tuple_norm(value) for name, value in gradients.items()}
    pairs: dict[str, Any] = {}
    for left, right in itertools.combinations(gradients, 2):
        dot = _tensor_tuple_dot(gradients[left], gradients[right])
        denominator = norms[left] * norms[right]
        pairs[f"{left}__{right}"] = {
            "dot": dot,
            "cosine": dot / denominator if denominator > 0.0 else None,
        }
    return {"norms": norms, "pairs": pairs}


def _cpu_gradient_copy(values: Sequence[Any | None]):
    return [
        None if value is None else value.detach().to(device="cpu", dtype=__import__("torch").float32)
        for value in values
    ]


def _accumulate_gradient(
    total: list[Any], values: Sequence[Any | None]
) -> None:
    for index, value in enumerate(values):
        if value is not None:
            total[index].add_(value.detach().to(device="cpu", dtype=total[index].dtype))


def _array_norm(values: Sequence[Any]) -> float:
    return math.sqrt(
        sum(float(value.square().sum()) for value in values if value is not None)
    )


def _array_dot(left: Sequence[Any], right: Sequence[Any]) -> float:
    return sum(
        float((one * two).sum())
        for one, two in zip(left, right, strict=True)
        if one is not None and two is not None
    )


def _aggregate_gradient_summary(
    totals: Mapping[str, Sequence[Any]],
    per_scene_norm_sums: Mapping[str, float],
    scene_count: int,
) -> dict[str, Any]:
    means = {
        name: [tensor / float(scene_count) for tensor in values]
        for name, values in totals.items()
    }
    norms = {name: _array_norm(values) for name, values in means.items()}
    pairs: dict[str, Any] = {}
    for left, right in itertools.combinations(means, 2):
        dot = _array_dot(means[left], means[right])
        denominator = norms[left] * norms[right]
        pairs[f"{left}__{right}"] = {
            "dot": dot,
            "cosine": dot / denominator if denominator > 0.0 else None,
        }
    cancellation = {}
    for name, mean_norm in norms.items():
        sum_norm = float(per_scene_norm_sums[name])
        sum_vector_norm = mean_norm * float(scene_count)
        coherence = sum_vector_norm / sum_norm if sum_norm > 0.0 else None
        cancellation[name] = {
            "sum_scene_gradient_norms": sum_norm,
            "norm_of_sum_scene_gradients": sum_vector_norm,
            "coherence_ratio": coherence,
            "cancellation_fraction": (
                1.0 - coherence if coherence is not None else None
            ),
        }
    return {"mean_gradient_norms": norms, "pairwise": pairs, "scene_cancellation": cancellation}


def _loss_decomposition(prediction: Any, target: Any):
    """Return native per-position MSE = response MSE + common-bias MSE.

    Axis 0 is the eleven hidden delay conditions, axis 1 is the seven native
    prediction positions, and any remaining axes are the model's latent axes.
    """
    import torch.nn.functional as functional

    prediction = prediction.double()
    target = target.double()
    error = (prediction - target).square()
    feature_axes = tuple(range(2, error.ndim))
    condition_and_feature_axes = (0, *feature_axes)
    pred_centered = prediction - prediction.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    response = (pred_centered - target_centered).square().mean(
        dim=condition_and_feature_axes
    )
    target_energy = target_centered.square().mean(dim=condition_and_feature_axes)
    pred_response_energy = pred_centered.square().mean(dim=condition_and_feature_axes)
    mean_delta = prediction.mean(dim=0) - target.mean(dim=0)
    common = mean_delta.square().mean(dim=tuple(range(1, mean_delta.ndim)))
    native_by_position = error.mean(dim=(0, *feature_axes))
    native = functional.mse_loss(prediction, target)
    return {
        "native": native,
        "native_by_position": native_by_position,
        "response_by_position": response,
        "common_by_position": common,
        "target_response_energy_by_position": target_energy,
        "prediction_response_energy_by_position": pred_response_energy,
    }


def _evaluate_scene(
    *,
    torch: Any,
    functional: Any,
    model: Any,
    adapter: Any,
    family: str,
    history_np: np.ndarray,
    target_np: np.ndarray,
    raw_actions: np.ndarray,
    scene_id: str,
    named_parameters: Sequence[tuple[str, Any]],
    helper: Any,
):
    delays = len(history_np)
    if delays != len(DELAYS) or history_np.shape[1] != HISTORY:
        raise ValueError(
            f"Expected [11,7,...] history; got {history_np.shape}"
        )
    if target_np.shape[0] != delays or target_np.shape[1] != 1:
        raise ValueError(f"Expected [11,1,...] targets; got {target_np.shape}")
    if raw_actions.shape[:2] != (delays, HISTORY):
        raise ValueError(f"Expected [11,7,...] action blocks; got {raw_actions.shape}")

    history = torch.from_numpy(np.asarray(history_np, dtype=np.float32)).to(adapter.device)
    future = torch.from_numpy(np.asarray(target_np, dtype=np.float32)).to(adapter.device)
    target_sequence = torch.cat([history[:, 1:], future], dim=1).detach()
    normalized_actions = torch.from_numpy(
        adapter._normalize_actions(raw_actions)
    ).to(adapter.device)

    if family == "dinowm":
        if tuple(model.extra_encoders.keys()) != ("action",):
            raise ValueError(
                "DINO-WM diagnostic requires only the native action stream; "
                f"got {tuple(model.extra_encoders.keys())}"
            )
        channels = int(model.backbone.config.hidden_size)
        if history.shape[-1] % channels or future.shape[-1] % channels:
            raise ValueError("DINO flattened visual latent is not patch×channel aligned")
        patches = history.shape[-1] // channels
        if future.shape[-1] // channels != patches:
            raise ValueError("DINO history and target patch counts differ")
        visual_history = history.reshape(delays, HISTORY, patches, channels)
        visual_target = target_sequence.reshape(delays, POSITIONS, patches, channels)
        action_embedding = model.extra_encoders["action"](normalized_actions)
        predictor_input = torch.cat(
            [
                visual_history,
                action_embedding[:, :, None, :].expand(-1, -1, patches, -1),
            ],
            dim=-1,
        )
        predicted_full = model.predict(predictor_input)
        prediction = predicted_full[..., :channels]
        target = visual_target
        prediction_for_eval = prediction.reshape(delays, POSITIONS, -1)
    elif family in {"lewm", "pldm"}:
        action_encoder = model.action_encoder
        action_embedding = action_encoder(normalized_actions)
        prediction = model.predict(history, action_embedding)
        target = target_sequence
        prediction_for_eval = prediction
    else:
        raise ValueError(f"Unsupported model family: {family}")

    if tuple(prediction.shape)[:2] != (delays, POSITIONS):
        raise ValueError(
            f"Native predictor did not return all seven positions: {tuple(prediction.shape)}"
        )
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )

    decomposition = _loss_decomposition(prediction, target)
    native_by_position = decomposition["native_by_position"]
    response_by_position = decomposition["response_by_position"]
    common_by_position = decomposition["common_by_position"]
    if not torch.allclose(
        native_by_position,
        response_by_position + common_by_position,
        atol=MSE_CLOSURE_ATOL,
        rtol=2.0e-5,
    ):
        raise RuntimeError(
            f"{scene_id}: condition-centered MSE decomposition failed"
        )
    if not torch.allclose(
        decomposition["native"], native_by_position.mean(),
        atol=MSE_CLOSURE_ATOL,
        rtol=2.0e-5,
    ):
        raise RuntimeError(f"{scene_id}: seven-position native MSE reduction failed")

    # The benchmark adapter's free first-step path must agree with the final
    # native predictor position before any gradient statistic is accepted.
    adapter_prediction = helper.predict_all(
        adapter,
        history_np,
        target_np,
        raw_actions,
        family,
        modes=("free",),
    )["free"][:, 0]
    direct_endpoint = prediction_for_eval[:, -1].detach().float().cpu().numpy()
    canonical_delta = direct_endpoint - adapter_prediction
    canonical_max_abs = float(np.abs(canonical_delta).max())
    canonical_relative_l2 = float(
        np.linalg.norm(canonical_delta)
        / max(float(np.linalg.norm(adapter_prediction)), 1.0e-12)
    )
    if (
        canonical_max_abs >= CANONICAL_MAX_ABS
        or canonical_relative_l2 >= CANONICAL_RELATIVE_L2
    ):
        raise RuntimeError(
            f"{scene_id}: native adapter endpoint mismatch: "
            f"max_abs={canonical_max_abs}, relative_l2={canonical_relative_l2}"
        )

    parameters = tuple(parameter for _, parameter in named_parameters)
    native_loss = decomposition["native"]
    final_response = response_by_position[-1] / float(POSITIONS)
    final_common = common_by_position[-1] / float(POSITIONS)
    earlier = native_by_position[:-1].sum() / float(POSITIONS)
    final_native = native_by_position[-1] / float(POSITIONS)

    def gradients(value: Any, *, retain_graph: bool):
        if not value.requires_grad:
            return tuple(None for _ in parameters)
        return torch.autograd.grad(
            value,
            parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        )

    grad_rows = {
        "final_response": gradients(final_response, retain_graph=True),
        "final_common": gradients(final_common, retain_graph=True),
        "earlier_six": gradients(earlier, retain_graph=True),
        "final_native": gradients(final_native, retain_graph=True),
        "native_all_seven": gradients(native_loss, retain_graph=False),
    }
    final_closure = _gradient_closure(
        grad_rows["final_native"],
        [grad_rows["final_response"], grad_rows["final_common"]],
    )
    native_closure = _gradient_closure(
        grad_rows["native_all_seven"],
        [
            grad_rows["final_response"],
            grad_rows["final_common"],
            grad_rows["earlier_six"],
        ],
    )
    for label, check in (("final", final_closure), ("native", native_closure)):
        if (
            not math.isfinite(check["relative_to_component_norm_sum"])
            or check["relative_to_component_norm_sum"] > GRADIENT_CLOSURE_RELATIVE_MAX
            or check["max_abs"] > 1.0e-5
        ):
            raise RuntimeError(f"{scene_id}: {label} gradient closure failed: {check}")

    grad_summary = _pairwise_gradient_summary(grad_rows)
    grad_summary["closures"] = {
        "final_native_equals_final_response_plus_final_common": final_closure,
        "native_equals_final_response_plus_final_common_plus_earlier_six": native_closure,
    }

    scalar = lambda value: float(value.detach().double().cpu())
    vector = lambda value: [float(item) for item in value.detach().double().cpu().tolist()]
    return {
        "scene_id": scene_id,
        "canonical_adapter_endpoint": {
            "max_abs": canonical_max_abs,
            "relative_l2": canonical_relative_l2,
        },
        "native_mse": scalar(native_loss),
        "per_position": {
            "native_mse": vector(native_by_position),
            "response_mse": vector(response_by_position),
            "common_bias_mse": vector(common_by_position),
            "target_response_energy": vector(
                decomposition["target_response_energy_by_position"]
            ),
            "prediction_response_energy": vector(
                decomposition["prediction_response_energy_by_position"]
            ),
            "decomposition_max_abs": float(
                (native_by_position - response_by_position - common_by_position)
                .detach()
                .abs()
                .max()
                .cpu()
            ),
        },
        "gradient": grad_summary,
        "_gradient_tensors": {
            name: _cpu_gradient_copy(value) for name, value in grad_rows.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scenes-per-stratum", type=int, default=4)
    parser.add_argument("--split", choices=("training", "development"), default="training")
    args = parser.parse_args()

    if args.scenes_per_stratum <= 0:
        raise ValueError("--scenes-per-stratum must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite {result_path}")

    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    helper = _load_helper()
    manifest_path = args.panel / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = json.loads(args.models.read_text(encoding="utf-8"))
    spec_matches = [entry for entry in specs if entry["id"] == args.id]
    if len(spec_matches) != 1:
        raise ValueError(f"Expected exactly one model spec for {args.id}")
    model_spec = spec_matches[0]
    if (
        model_spec.get("task") != "action_delay"
        or model_spec.get("regime") != "scratch"
        or int(model_spec.get("training_seed", -1)) != 3072
    ):
        raise ValueError("This diagnostic is frozen to an ActionDelay T1 seed-3072 checkpoint")
    family = str(model_spec["family"])
    if family not in {"dinowm", "lewm", "pldm"}:
        raise ValueError(f"Unsupported model family: {family}")
    observed_checkpoint_sha = _sha256(Path(model_spec["checkpoint"]))
    if observed_checkpoint_sha != model_spec["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash does not match model catalog")

    adapter = helper.load_adapter(
        model_spec,
        manifest["normalization"],
        args.output,
        args.device,
    )
    model = adapter.model
    model.eval()
    state_hash_before = adapter.frozen_state_hash()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.predictor.requires_grad_(True)
    model.eval()
    named_parameters = [
        (f"predictor.{name}", parameter)
        for name, parameter in model.predictor.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise RuntimeError("No predictor parameters are available for gradients")
    parameter_count = sum(parameter.numel() for _, parameter in named_parameters)

    scenes = _select_scenes(manifest, args.split, args.scenes_per_stratum)
    per_scene: list[dict[str, Any]] = []
    aggregate_names = (
        "final_response",
        "final_common",
        "earlier_six",
        "final_native",
        "native_all_seven",
    )
    aggregate_totals = {
        name: [
            torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            for _, parameter in named_parameters
        ]
        for name in aggregate_names
    }
    aggregate_norm_sums = {name: 0.0 for name in aggregate_names}
    source_hashes = []

    for index, entry in enumerate(scenes):
        panel_path = args.panel / entry["path"]
        source_sha = _sha256(panel_path)
        if source_sha != entry["sha256"]:
            raise ValueError(f"Panel NPZ hash mismatch: {panel_path}")
        source_hashes.append(source_sha)
        with np.load(panel_path, allow_pickle=False) as data:
            pixels = data["history_pixels"]
            future_pixels = data["future_pixels"]
            raw_actions = data["action_blocks"]
        if pixels.shape[:2] != (len(DELAYS), HISTORY):
            raise ValueError(f"Unexpected history panel shape: {pixels.shape}")
        if future_pixels.shape[:2] != (len(DELAYS), 1):
            raise ValueError(f"Unexpected target panel shape: {future_pixels.shape}")
        if raw_actions.shape[:2] != (len(DELAYS), HISTORY):
            raise ValueError(f"Unexpected action panel shape: {raw_actions.shape}")

        # Cache the same native image latents used by the frozen adapter.
        history_latent, target_latent = helper.encode_unique(
            adapter, pixels, future_pixels
        )
        result = _evaluate_scene(
            torch=torch,
            functional=functional,
            model=model,
            adapter=adapter,
            family=family,
            history_np=history_latent,
            target_np=target_latent,
            raw_actions=raw_actions,
            scene_id=str(entry["scene_id"]),
            named_parameters=named_parameters,
            helper=helper,
        )
        grad_rows = result.pop("_gradient_tensors")
        for name in aggregate_names:
            _accumulate_gradient(aggregate_totals[name], grad_rows[name])
            aggregate_norm_sums[name] += _array_norm(grad_rows[name])
        result["panel_source_sha256"] = source_sha
        result["room"] = entry["room"]
        result["direction"] = entry["direction"]
        per_scene.append(result)
        print(
            f"{index + 1}/{len(scenes)} {entry['scene_id']} "
            f"native_mse={result['native_mse']:.7g}",
            flush=True,
        )

    state_hash_after = adapter.frozen_state_hash()
    if state_hash_before != state_hash_after:
        raise RuntimeError("Model parameters or buffers changed during diagnostic")

    aggregate = _aggregate_gradient_summary(
        aggregate_totals,
        aggregate_norm_sums,
        len(scenes),
    )
    rows_per_stratum = {
        key: sum(f"{row['room']}/{row['direction']}" == key for row in per_scene)
        for key in STRATA
    }
    result = {
        "schema": "contextworld.delay_native_objective_gradient.v1",
        "scope": {
            "purpose": "Frozen-checkpoint native objective and predictor-only gradient decomposition",
            "no_training_or_optimizer_step": True,
            "split": args.split,
            "training_seed": 3072,
            "scene_count": len(per_scene),
            "scenes_per_stratum": args.scenes_per_stratum,
            "stratum_counts": rows_per_stratum,
            "delays_equal_weighted": list(DELAYS),
            "history_frames": HISTORY,
            "native_prediction_positions": POSITIONS,
            "current_query_position_index": POSITIONS - 1,
            "physical_endpoint_step": 5,
            "mode": "eval; float32 model forward, float64 diagnostic loss reduction; dropout disabled; cached and detached image latents",
            "gradient_identity_tolerance": {"max_abs": 1.0e-5, "relative_to_component_norm_sum": GRADIENT_CLOSURE_RELATIVE_MAX},
            "gradient_parameters": "model.predictor only; other modules frozen",
            "interpretation_limit": "Local gradients at fixed weights do not establish causal training effects or broad task-family capability.",
        },
        "provenance": {
            "model_id": model_spec["id"],
            "family": family,
            "checkpoint": model_spec["checkpoint"],
            "checkpoint_sha256": observed_checkpoint_sha,
            "stable_repo": model_spec["stable_repo"],
            "stable_ref": model_spec["stable_ref"],
            "adapter_stable_commit": str(getattr(adapter, "stable_commit", "")),
            "panel_manifest": str(manifest_path.resolve()),
            "panel_manifest_sha256": _sha256(manifest_path),
            "model_catalog": str(args.models.resolve()),
            "model_catalog_sha256": _sha256(args.models),
            "panel_source_sha256s": source_hashes,
            "state_hash_before": state_hash_before,
            "state_hash_after": state_hash_after,
            "predictor_parameter_count": parameter_count,
            "parameter_names_sha256": hashlib.sha256(
                "\n".join(name for name, _ in named_parameters).encode("utf-8")
            ).hexdigest(),
        },
        "validation": {
            "canonical_adapter_endpoint_max_abs_maximum": max(
                row["canonical_adapter_endpoint"]["max_abs"] for row in per_scene
            ),
            "canonical_adapter_endpoint_relative_l2_maximum": max(
                row["canonical_adapter_endpoint"]["relative_l2"] for row in per_scene
            ),
            "state_unchanged": state_hash_before == state_hash_after,
            "no_optimizer_or_parameter_update": True,
        },
        "aggregate_sufficient_statistics": {
            "mean_native_mse_by_position": np.mean(
                [row["per_position"]["native_mse"] for row in per_scene], axis=0
            ).tolist(),
            "mean_response_mse_by_position": np.mean(
                [row["per_position"]["response_mse"] for row in per_scene], axis=0
            ).tolist(),
            "mean_common_bias_mse_by_position": np.mean(
                [row["per_position"]["common_bias_mse"] for row in per_scene], axis=0
            ).tolist(),
            "mean_target_response_energy_by_position": np.mean(
                [row["per_position"]["target_response_energy"] for row in per_scene], axis=0
            ).tolist(),
            "mean_prediction_response_energy_by_position": np.mean(
                [row["per_position"]["prediction_response_energy"] for row in per_scene], axis=0
            ).tolist(),
        },
        "predictor_gradient_aggregate": aggregate,
        "scenes": per_scene,
    }
    _write_json(result_path, result)
    print(f"wrote {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
