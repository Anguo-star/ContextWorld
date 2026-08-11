#!/usr/bin/env python3
"""Closed-loop scalar CEM for the paired Push-T hidden-actuation task.

The planner receives the full History=3 pixels and the first two action
blocks.  It optimizes the amplitude of the final contact block, renders no
counterfactual labels, and is scored by executing the selected raw action in
the real Push-T simulator under the episode's persistent hidden gain.

The action family is deliberately one-dimensional:

    [a * contact_axis, a * contact_axis, 0, 0, 0],  a in [0, 1].

This removes irrelevant lateral-search failures while retaining the causal
question.  The low-gain mode needs a much larger amplitude than the high-gain
mode to reach the same visible goal.  A current-query-only planner receives
identical input for the two modes and therefore cannot select both actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / "stable-worldmodel"
for source_root in (CONTEXTWORLD_ROOT, STABLE_WORLD_MODEL_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from stable_worldmodel.envs.pusht.env import PushT  # noqa: E402
from contextworld.evaluation.pusht_hidden_actuation import (  # noqa: E402
    MODE_SCALES,
    HiddenActuationTemplate,
    _variation_values,
)
from contextworld.paths import artifact_path  # noqa: E402


DEFAULT_DATA_ROOT = artifact_path(
    "synthesis/pusht_hidden_actuation_h3_v1"
)
DEFAULT_ORIGINAL_DATASET = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/"
    "pusht_expert_train.h5"
)
DEFAULT_EVAL_SEEDS = (4096, 5120, 6144, 7168, 8192, 9216)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def action_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r", swmr=True) as handle:
        values = handle["action"]
        count = int(values.shape[0])
        total = np.zeros(2, dtype=np.float64)
        square = np.zeros(2, dtype=np.float64)
        for start in range(0, count, 200_000):
            batch = values[start : start + 200_000].astype(np.float64)
            total += batch.sum(axis=0)
            square += np.square(batch).sum(axis=0)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - np.square(mean), 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def preprocess_pixels(
    pixels: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    values = pixels.to(device=device, non_blocking=True).float().div_(255.0)
    mean = torch.as_tensor(
        IMAGENET_MEAN,
        dtype=values.dtype,
        device=device,
    ).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(
        IMAGENET_STD,
        dtype=values.dtype,
        device=device,
    ).view(1, 1, 3, 1, 1)
    return (values - mean) / std


def normalize_action(
    actions: torch.Tensor,
    *,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    original_shape = actions.shape
    values = actions.reshape(*original_shape[:-1], -1, 2)
    mean_tensor = torch.as_tensor(
        mean,
        dtype=values.dtype,
        device=values.device,
    )
    std_tensor = torch.as_tensor(
        std,
        dtype=values.dtype,
        device=values.device,
    )
    values = (values - mean_tensor) / std_tensor.clamp_min(1e-8)
    return values.reshape(original_shape)


def instantiate_model() -> torch.nn.Module:
    cfg = OmegaConf.load(
        STABLE_WORLD_MODEL_ROOT / "scripts/train/config/lewm.yaml"
    )
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = 10
    return hydra.utils.instantiate(cfg.model)


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("state_dict", payload)
    model_prefixed = {
        name[len("model.") :]: value
        for name, value in source.items()
        if name.startswith("model.")
    }
    return model_prefixed or dict(source)


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    model = instantiate_model()
    model.load_state_dict(checkpoint_state(path), strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


@dataclass(frozen=True)
class Condition:
    condition_id: str
    pair_index: int
    mode: str
    template: HiddenActuationTemplate
    contact_direction: tuple[float, float]
    history_pixels: torch.Tensor
    fixed_action_blocks: torch.Tensor
    goal_pixels: torch.Tensor


def load_conditions(
    root: Path,
    *,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[Condition]:
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    conditions = []
    for pair_index, row in enumerate(manifest["splits"]["eval"]["pairs"]):
        template = HiddenActuationTemplate(**row["template"])
        payload_path = root / "eval_payloads" / row["eval_payload"]["path"]
        with np.load(payload_path) as payload:
            low_pixels = torch.from_numpy(
                payload["low_pixels"].copy()
            ).permute(0, 3, 1, 2)
            high_pixels = torch.from_numpy(
                payload["high_pixels"].copy()
            ).permute(0, 3, 1, 2)
            low_actions = torch.from_numpy(
                payload["low_actions"].copy()
            ).reshape(4, 10)
            high_actions = torch.from_numpy(
                payload["high_actions"].copy()
            ).reshape(4, 10)
            goal_pixels = torch.from_numpy(
                payload["goal_pixels"].copy()
            ).permute(2, 0, 1)
        if not torch.equal(low_actions, high_actions):
            raise RuntimeError(f"Unequal paired actions: {payload_path}")
        fixed = normalize_action(
            low_actions[:2].float(),
            mean=mean,
            std=std,
        )
        for mode, pixels in (
            ("low_gain", low_pixels),
            ("high_gain", high_pixels),
        ):
            conditions.append(
                Condition(
                    condition_id=f"{template.template_id}/{mode}",
                    pair_index=pair_index,
                    mode=mode,
                    template=template,
                    contact_direction=template.contact_direction,
                    history_pixels=pixels[:3],
                    fixed_action_blocks=fixed,
                    goal_pixels=goal_pixels,
                )
            )
    return conditions


@torch.no_grad()
def encode_observations(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, pixels.size(0), batch_size):
        values = preprocess_pixels(
            pixels[start : start + batch_size],
            device=device,
        )
        batch, frames = values.shape[:2]
        raw = model.encoder(
            values.flatten(0, 1),
            interpolate_pos_encoding=True,
        ).last_hidden_state[:, 0]
        projected = model.projector(raw).reshape(batch, frames, -1)
        outputs.append(projected.float())
    return torch.cat(outputs)


@torch.no_grad()
def cache_model_inputs(
    model: torch.nn.Module,
    conditions: list[Condition],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    histories = torch.stack([row.history_pixels for row in conditions])
    goals = torch.stack([row.goal_pixels for row in conditions])[:, None]
    history_embedding = encode_observations(
        model,
        histories,
        device=device,
        batch_size=batch_size,
    )
    goal_embedding = encode_observations(
        model,
        goals,
        device=device,
        batch_size=batch_size,
    )[:, 0]
    fixed_actions = torch.stack(
        [row.fixed_action_blocks for row in conditions]
    ).to(device)
    directions = torch.as_tensor(
        [row.contact_direction for row in conditions],
        dtype=torch.float32,
        device=device,
    )
    return {
        "history_embedding": history_embedding,
        "goal_embedding": goal_embedding,
        "fixed_actions": fixed_actions,
        "directions": directions,
    }


@torch.no_grad()
def candidate_cost(
    model: torch.nn.Module,
    cache: dict[str, torch.Tensor],
    indices: torch.Tensor,
    amplitude: torch.Tensor,
    *,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> torch.Tensor:
    """Score B x S scalar amplitudes against the common visible goal."""

    batch, samples = amplitude.shape
    directions = cache["directions"][indices]
    raw = torch.zeros(
        batch,
        samples,
        5,
        2,
        dtype=torch.float32,
        device=amplitude.device,
    )
    raw[:, :, :2] = (
        amplitude[:, :, None, None]
        * directions[:, None, None, :]
    )
    query_block = normalize_action(
        raw.reshape(batch, samples, 10),
        mean=action_mean,
        std=action_std,
    )
    fixed = cache["fixed_actions"][indices, None].expand(
        batch,
        samples,
        2,
        10,
    )
    actions = torch.cat([fixed, query_block[:, :, None]], dim=2)
    actions = actions.flatten(0, 1)
    history = cache["history_embedding"][indices, None].expand(
        batch,
        samples,
        3,
        -1,
    )
    history = history.flatten(0, 1)
    action_embedding = model.action_encoder(actions)
    prediction = model.predict(history, action_embedding)[:, -1]
    prediction = prediction.reshape(batch, samples, -1)
    goal = cache["goal_embedding"][indices, None]
    return (prediction - goal).square().sum(dim=-1)


@torch.no_grad()
def solve_cem(
    model: torch.nn.Module,
    cache: dict[str, torch.Tensor],
    *,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    seed: int,
    num_samples: int,
    iterations: int,
    topk: int,
    batch_size: int,
    initial_mean: float,
    initial_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = cache["history_embedding"].device
    count = cache["history_embedding"].size(0)
    selected = torch.empty(count, dtype=torch.float32)
    selected_cost = torch.empty(count, dtype=torch.float32)
    if not 0 < topk <= num_samples:
        raise ValueError("topk must be in [1, num_samples]")

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        indices = torch.arange(start, stop, device=device)
        current_batch = stop - start
        mean = torch.full(
            (current_batch,),
            initial_mean,
            device=device,
        )
        std = torch.full(
            (current_batch,),
            initial_std,
            device=device,
        )
        generator = torch.Generator(device=device).manual_seed(
            seed + start * 1_000_003
        )
        for _ in range(iterations):
            noise = torch.randn(
                current_batch,
                num_samples,
                generator=generator,
                device=device,
            )
            candidates = (
                mean[:, None] + std[:, None] * noise
            ).clamp_(0.0, 1.0)
            candidates[:, 0] = mean
            if num_samples > 1:
                candidates[:, 1] = 0.0
            if num_samples > 2:
                candidates[:, 2] = 1.0
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                costs = candidate_cost(
                    model,
                    cache,
                    indices,
                    candidates,
                    action_mean=action_mean,
                    action_std=action_std,
                )
            elite_indices = torch.topk(
                costs,
                k=topk,
                dim=1,
                largest=False,
            ).indices
            elites = candidates.gather(1, elite_indices)
            mean = elites.mean(dim=1)
            std = elites.std(dim=1).clamp_min_(1e-4)
        final_cost = candidate_cost(
            model,
            cache,
            indices,
            mean[:, None],
            action_mean=action_mean,
            action_std=action_std,
        )[:, 0]
        selected[start:stop] = mean.cpu()
        selected_cost[start:stop] = final_cost.float().cpu()
    return selected, selected_cost


def execute_amplitude(
    condition: Condition,
    amplitude: float,
) -> dict[str, Any]:
    env = PushT(
        resolution=224,
        with_target=True,
        render_mode="rgb_array",
    )
    env.action_scale = float(MODE_SCALES[condition.mode])
    direction = np.asarray(condition.contact_direction, dtype=np.float32)
    actions = np.zeros((5, 2), dtype=np.float32)
    actions[:2] = np.float32(amplitude) * direction
    try:
        env.reset(
            seed=int(condition.template.simulator_seed),
            options={
                "variation": (),
                "variation_values": _variation_values(condition.template),
                "state": condition.template.reset_state,
                "goal_state": condition.template.goal_state,
            },
        )
        contact_steps = 0
        for action in actions:
            _, _, _, _, info = env.step(action)
            contact_steps += int(info["n_contacts"] > 0)
        final_state = np.asarray(env._get_obs(), dtype=np.float64)
        success, state_distance = env.eval_state(
            condition.template.goal_state,
            final_state,
        )
    finally:
        env.close()
    goal = condition.template.goal_state
    angle_gap = abs(float(goal[4]) - float(final_state[4])) % (2 * np.pi)
    angle_gap = min(angle_gap, 2 * np.pi - angle_gap)
    return {
        "amplitude": float(amplitude),
        "success": bool(success),
        "state_distance": float(state_distance),
        "position_distance": float(
            np.linalg.norm(goal[:4] - final_state[:4])
        ),
        "agent_position_error": float(
            np.linalg.norm(goal[:2] - final_state[:2])
        ),
        "block_position_error": float(
            np.linalg.norm(goal[2:4] - final_state[2:4])
        ),
        "block_angle_error": float(angle_gap),
        "contact_steps": int(contact_steps),
        "final_state": final_state.tolist(),
    }


def oracle_surface(
    conditions: list[Condition],
    *,
    grid_size: int,
) -> list[dict[str, Any]]:
    grid = np.linspace(0.0, 1.0, grid_size)
    rows = []
    for index, condition in enumerate(conditions, start=1):
        outcomes = [
            execute_amplitude(condition, float(amplitude))
            for amplitude in grid
        ]
        best = min(outcomes, key=lambda row: row["state_distance"])
        successful = [row for row in outcomes if row["success"]]
        rows.append(
            {
                "condition_id": condition.condition_id,
                "pair_index": condition.pair_index,
                "mode": condition.mode,
                "best": best,
                "any_success": bool(successful),
                "successful_amplitude_min": (
                    min(row["amplitude"] for row in successful)
                    if successful
                    else None
                ),
                "successful_amplitude_max": (
                    max(row["amplitude"] for row in successful)
                    if successful
                    else None
                ),
            }
        )
        if index % 20 == 0:
            print(
                f"[oracle] {index}/{len(conditions)} conditions",
                flush=True,
            )
    return rows


def summarize_records(
    records: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
) -> dict[str, Any]:
    oracle_by_id = {row["condition_id"]: row for row in oracle}
    for row in records:
        row["oracle"] = oracle_by_id[row["condition_id"]]
        row["absolute_amplitude_regret"] = abs(
            row["execution"]["amplitude"]
            - row["oracle"]["best"]["amplitude"]
        )
    by_mode = {}
    for mode in MODE_SCALES:
        selected = [row for row in records if row["mode"] == mode]
        by_mode[mode] = {
            "count": len(selected),
            "success_rate": float(
                np.mean([row["execution"]["success"] for row in selected])
            ),
            "mean_selected_amplitude": float(
                np.mean(
                    [row["execution"]["amplitude"] for row in selected]
                )
            ),
            "mean_oracle_amplitude": float(
                np.mean(
                    [row["oracle"]["best"]["amplitude"] for row in selected]
                )
            ),
            "mean_state_distance": float(
                np.mean(
                    [row["execution"]["state_distance"] for row in selected]
                )
            ),
        }

    paired: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in records:
        key = (int(row["eval_seed"]), int(row["pair_index"]))
        paired.setdefault(key, {})[row["mode"]] = row
    ordered = []
    mode_correct = []
    for modes in paired.values():
        low = modes["low_gain"]
        high = modes["high_gain"]
        ordered.append(
            low["execution"]["amplitude"]
            > high["execution"]["amplitude"]
        )
        low_own = abs(
            low["execution"]["amplitude"]
            - low["oracle"]["best"]["amplitude"]
        )
        low_other = abs(
            low["execution"]["amplitude"]
            - high["oracle"]["best"]["amplitude"]
        )
        high_own = abs(
            high["execution"]["amplitude"]
            - high["oracle"]["best"]["amplitude"]
        )
        high_other = abs(
            high["execution"]["amplitude"]
            - low["oracle"]["best"]["amplitude"]
        )
        mode_correct.extend([low_own < low_other, high_own < high_other])
    return {
        "record_count": len(records),
        "query_seed_pairs": len(paired),
        "real_environment_success_rate": float(
            np.mean([row["execution"]["success"] for row in records])
        ),
        "mean_absolute_amplitude_regret": float(
            np.mean([row["absolute_amplitude_regret"] for row in records])
        ),
        "correct_low_greater_than_high_rate": float(np.mean(ordered)),
        "oracle_mode_action_classification_rate": float(
            np.mean(mode_correct)
        ),
        "by_mode": by_mode,
        "oracle_grid_success_rate": float(
            np.mean([row["any_success"] for row in oracle])
        ),
    }


def parse_models(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--model values must be NAME=CHECKPOINT")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate model name: {name!r}")
        result[name] = Path(raw_path).expanduser().resolve()
    if not result:
        raise ValueError("At least one --model is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--original-dataset",
        type=Path,
        default=DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="NAME=CHECKPOINT; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--eval-seeds",
        default=",".join(map(str, DEFAULT_EVAL_SEEDS)),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--cem-batch-size", type=int, default=10)
    parser.add_argument("--cem-samples", type=int, default=300)
    parser.add_argument("--cem-iterations", type=int, default=30)
    parser.add_argument("--cem-topk", type=int, default=30)
    parser.add_argument("--cem-initial-mean", type=float, default=0.5)
    parser.add_argument("--cem-initial-std", type=float, default=0.35)
    parser.add_argument("--oracle-grid-size", type=int, default=101)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = parse_models(args.model)
    eval_seeds = tuple(
        int(value)
        for value in args.eval_seeds.split(",")
        if value.strip()
    )
    if not eval_seeds or len(eval_seeds) != len(set(eval_seeds)):
        raise ValueError("--eval-seeds must be a non-empty unique list")
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    data_root = args.data_root.expanduser().resolve()
    original_dataset = args.original_dataset.expanduser().resolve()
    required = [
        data_root / "manifest.json",
        data_root / "eval_payloads",
        original_dataset,
        *models.values(),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input(s):\n" + "\n".join(map(str, missing))
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    mean, std = action_stats(original_dataset)
    conditions = load_conditions(data_root, mean=mean, std=std)
    print(
        f"Calibrating real-environment oracle for {len(conditions)} conditions",
        flush=True,
    )
    oracle = oracle_surface(
        conditions,
        grid_size=args.oracle_grid_size,
    )
    oracle_path = output / "oracle_surface.json"
    oracle_path.write_text(
        json.dumps(oracle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result_rows = []
    for model_index, (name, checkpoint) in enumerate(
        models.items(),
        start=1,
    ):
        print(f"[{model_index}/{len(models)}] loading {name}", flush=True)
        set_seed(0)
        model = load_model(checkpoint, device)
        cache = cache_model_inputs(
            model,
            conditions,
            device=device,
            batch_size=args.encode_batch_size,
        )
        records = []
        started = time.monotonic()
        for eval_seed in eval_seeds:
            print(f"[{name}] CEM seed={eval_seed}", flush=True)
            amplitude, cost = solve_cem(
                model,
                cache,
                action_mean=mean,
                action_std=std,
                seed=eval_seed,
                num_samples=args.cem_samples,
                iterations=args.cem_iterations,
                topk=args.cem_topk,
                batch_size=args.cem_batch_size,
                initial_mean=args.cem_initial_mean,
                initial_std=args.cem_initial_std,
            )
            for index, condition in enumerate(conditions):
                execution = execute_amplitude(
                    condition,
                    float(amplitude[index]),
                )
                records.append(
                    {
                        "model": name,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": file_sha256(checkpoint),
                        "eval_seed": eval_seed,
                        "condition_id": condition.condition_id,
                        "pair_index": condition.pair_index,
                        "mode": condition.mode,
                        "hidden_action_scale": MODE_SCALES[condition.mode],
                        "selected_predicted_cost": float(cost[index]),
                        "execution": execution,
                    }
                )
        summary = summarize_records(records, oracle)
        row = {
            "model": name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "elapsed_seconds": time.monotonic() - started,
            "summary": summary,
            "records": records,
        }
        result_rows.append(row)
        model_path = output / f"{name}.json"
        model_path.write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{name}] success="
            f"{summary['real_environment_success_rate']:.3f} "
            "mode_action="
            f"{summary['oracle_mode_action_classification_rate']:.3f} "
            "ordered="
            f"{summary['correct_low_greater_than_high_rate']:.3f}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "status": "closed_loop_real_environment_cem",
        "data": {
            "root": str(data_root),
            "manifest": str(data_root / "manifest.json"),
            "manifest_sha256": file_sha256(data_root / "manifest.json"),
            "condition_count": len(conditions),
        },
        "action_normalization": {
            "source": str(original_dataset),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "cem": {
            "eval_seeds": eval_seeds,
            "samples": args.cem_samples,
            "iterations": args.cem_iterations,
            "topk": args.cem_topk,
            "initial_mean": args.cem_initial_mean,
            "initial_std": args.cem_initial_std,
            "amplitude_bounds": [0.0, 1.0],
            "action_family": (
                "[a*contact_axis, a*contact_axis, 0, 0, 0]"
            ),
        },
        "oracle": {
            "grid_size": args.oracle_grid_size,
            "path": str(oracle_path),
            "sha256": file_sha256(oracle_path),
        },
        "models": result_rows,
    }
    report_path = output / "aggregate.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": file_sha256(report_path),
                "summaries": {
                    row["model"]: row["summary"] for row in result_rows
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
