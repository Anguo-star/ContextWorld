#!/usr/bin/env python3
"""Run scalar CEM and real execution on replay-matched hidden dynamics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / "stable-worldmodel"
for source_root in (
    CONTEXTWORLD_ROOT,
    STABLE_WORLD_MODEL_ROOT,
    Path(__file__).resolve().parent,
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.evaluation.pusht_hidden_actuation import (  # noqa: E402
    MODE_SCALES,
)
from contextworld.evaluation.pusht_replay_matched_hidden_actuation import (  # noqa: E402
    ReplayMatchedHiddenActuationTemplate,
    _variation_values,
)
from contextworld.paths import artifact_path  # noqa: E402
from stable_worldmodel.data import LanceDataset  # noqa: E402
from stable_worldmodel.envs.pusht.env import PushT  # noqa: E402
from eval_pusht_hidden_actuation_cem import (  # noqa: E402
    action_stats,
    encode_observations,
    file_sha256,
    load_model,
    normalize_action,
    set_seed,
)


DEFAULT_DATA_ROOT = artifact_path(
    "synthesis/pusht_hidden_actuation_replay_matched_confirm_h3_v3"
)
DEFAULT_ORIGINAL_DATASET = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/"
    "pusht_expert_train.h5"
)


@dataclass(frozen=True)
class Condition:
    condition_id: str
    pair_index: int
    mode: str
    template: ReplayMatchedHiddenActuationTemplate
    history_pixels: torch.Tensor
    fixed_action_blocks: torch.Tensor
    base_query_actions: torch.Tensor
    goal_pixels: torch.Tensor
    target_state: np.ndarray


def load_conditions(
    root: Path,
    *,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[Condition]:
    manifest = json.loads((root / "manifest.json").read_text())
    pairs = manifest["splits"]["validation"]["pairs"]
    dataset = LanceDataset(
        path=root / "validation.lance",
        frameskip=5,
        num_steps=4,
        keys_to_load=["pixels", "action", "state"],
    )
    samples = [dataset[index] for index in range(len(dataset))]
    if len(samples) != 2 * len(pairs):
        raise RuntimeError("Lance samples do not match manifest pairs")
    conditions = []
    for pair_index, row in enumerate(pairs):
        template = ReplayMatchedHiddenActuationTemplate(**row["template"])
        low = samples[2 * pair_index]
        high = samples[2 * pair_index + 1]
        if not torch.equal(low["pixels"][0], high["pixels"][0]):
            raise RuntimeError("Pair initial pixels differ")
        if not torch.equal(low["pixels"][2], high["pixels"][2]):
            raise RuntimeError("Pair query pixels differ")
        if not torch.equal(low["action"], high["action"]):
            raise RuntimeError("Pair actions differ")
        fixed = normalize_action(
            low["action"][:2].float(),
            mean=mean,
            std=std,
        )
        base_query = low["action"][2].float().reshape(5, 2)
        source_query = torch.as_tensor(
            template.query_actions,
            dtype=torch.float32,
        )
        if not torch.equal(base_query, source_query):
            raise RuntimeError("Stored query differs from source receipt")
        target_state = low["state"][3].double().numpy().copy()
        for mode, sample in (
            ("low_gain", low),
            ("high_gain", high),
        ):
            conditions.append(
                Condition(
                    condition_id=f"{template.template_id}/{mode}",
                    pair_index=pair_index,
                    mode=mode,
                    template=template,
                    history_pixels=sample["pixels"][:3],
                    fixed_action_blocks=fixed,
                    base_query_actions=base_query,
                    goal_pixels=low["pixels"][3],
                    target_state=target_state,
                )
            )
    return conditions


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
    return {
        "history_embedding": history_embedding,
        "goal_embedding": goal_embedding,
        "fixed_actions": torch.stack(
            [row.fixed_action_blocks for row in conditions]
        ).to(device),
        "base_query_actions": torch.stack(
            [row.base_query_actions for row in conditions]
        ).to(device),
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
    batch, samples = amplitude.shape
    base = cache["base_query_actions"][indices]
    raw = amplitude[:, :, None, None] * base[:, None]
    query = normalize_action(
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
    actions = torch.cat([fixed, query[:, :, None]], dim=2)
    history = cache["history_embedding"][indices, None].expand(
        batch,
        samples,
        3,
        -1,
    )
    prediction = model.predict(
        history.flatten(0, 1),
        model.action_encoder(actions.flatten(0, 1)),
    )[:, -1]
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
) -> tuple[torch.Tensor, torch.Tensor]:
    device = cache["history_embedding"].device
    count = cache["history_embedding"].size(0)
    selected = torch.empty(count, dtype=torch.float32)
    selected_cost = torch.empty(count, dtype=torch.float32)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        indices = torch.arange(start, stop, device=device)
        current = stop - start
        mean = torch.full((current,), 0.5, device=device)
        std = torch.full((current,), 0.35, device=device)
        generator = torch.Generator(device=device).manual_seed(
            seed + start * 1_000_003
        )
        for _ in range(iterations):
            candidates = (
                mean[:, None]
                + std[:, None]
                * torch.randn(
                    current,
                    num_samples,
                    generator=generator,
                    device=device,
                )
            ).clamp_(0.0, 1.0)
            candidates[:, 0] = mean
            candidates[:, 1] = 0.0
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


def angle_gap(left: float, right: float) -> float:
    delta = abs(float(left) - float(right)) % (2 * np.pi)
    return float(min(delta, 2 * np.pi - delta))


def execute_amplitude(
    condition: Condition,
    amplitude: float,
) -> dict[str, Any]:
    env = PushT(
        resolution=32,
        with_target=True,
        render_mode="rgb_array",
    )
    env.render = lambda: None
    env.action_scale = float(MODE_SCALES[condition.mode])
    query_actions = (
        np.float32(amplitude)
        * np.asarray(
            condition.template.query_actions,
            dtype=np.float32,
        )
    )
    try:
        env.reset(
            seed=int(condition.template.simulator_seed),
            options={
                "variation": (),
                "variation_values": _variation_values(
                    condition.template
                ),
                "state": condition.template.reset_state,
                "goal_state": condition.template.goal_state,
            },
        )
        # Reproduce x0 -> x1 -> x2 in the same simulator before applying the
        # candidate query.  Resetting directly to x2 would violate the strict
        # causal contract and would discard the tiny, real residual state at
        # the naturally reached common query.
        prefix_actions = np.concatenate(
            [
                np.asarray(
                    condition.template.probe_actions,
                    dtype=np.float32,
                ),
                np.asarray(
                    condition.template.recovery_actions,
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        for action in prefix_actions:
            env.step(action)
        contact_steps = 0
        for action in query_actions:
            _, _, _, _, info = env.step(action)
            contact_steps += int(info["n_contacts"] > 0)
        final = np.asarray(env._get_obs(), dtype=np.float64)
    finally:
        env.close()
    target = np.asarray(condition.target_state, dtype=np.float64)
    agent_error = float(np.linalg.norm(final[:2] - target[:2]))
    block_error = float(np.linalg.norm(final[2:4] - target[2:4]))
    rotation_error = angle_gap(final[4], target[4])
    visible_distance = float(
        np.linalg.norm(
            [agent_error, block_error, 40.0 * rotation_error]
        )
    )
    return {
        "amplitude": float(amplitude),
        "visible_state_distance": visible_distance,
        "agent_position_error_px": agent_error,
        "block_position_error_px": block_error,
        "block_angle_error_rad": rotation_error,
        "contact_steps": contact_steps,
        "state_installations_after_x0": 0,
        "query_simulator_recreated": False,
        "final_state": final.tolist(),
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
        best = min(
            outcomes,
            key=lambda row: row["visible_state_distance"],
        )
        rows.append(
            {
                "condition_id": condition.condition_id,
                "pair_index": condition.pair_index,
                "mode": condition.mode,
                "best": best,
            }
        )
        if index % 64 == 0:
            print(
                f"[oracle] {index}/{len(conditions)} conditions",
                flush=True,
            )
    return rows


def summarize(
    records: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
) -> dict[str, Any]:
    oracle_by_id = {row["condition_id"]: row for row in oracle}
    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    mode_decisions = []
    regrets = []
    for row in records:
        own = oracle_by_id[row["condition_id"]]["best"]["amplitude"]
        pair_oracles = {
            value["mode"]: value["best"]["amplitude"]
            for value in oracle
            if value["pair_index"] == row["pair_index"]
        }
        other_mode = (
            "high_gain" if row["mode"] == "low_gain" else "low_gain"
        )
        own_regret = abs(row["execution"]["amplitude"] - own)
        other_regret = abs(
            row["execution"]["amplitude"]
            - pair_oracles[other_mode]
        )
        row["oracle_amplitude"] = own
        row["absolute_amplitude_regret"] = own_regret
        row["mode_action_correct"] = own_regret < other_regret
        mode_decisions.append(row["mode_action_correct"])
        regrets.append(own_regret)
        by_pair.setdefault(row["pair_index"], {})[row["mode"]] = row
    ordered = [
        modes["low_gain"]["execution"]["amplitude"]
        > modes["high_gain"]["execution"]["amplitude"]
        for modes in by_pair.values()
    ]
    distances = [
        row["execution"]["visible_state_distance"] for row in records
    ]
    return {
        "record_count": len(records),
        "pair_count": len(by_pair),
        "oracle_mode_action_classification_rate": float(
            np.mean(mode_decisions)
        ),
        "correct_low_greater_than_high_rate": float(np.mean(ordered)),
        "mean_absolute_amplitude_regret": float(np.mean(regrets)),
        "mean_executed_visible_state_distance": float(
            np.mean(distances)
        ),
        "executed_visible_state_within_5px_rate": float(
            np.mean(np.asarray(distances) <= 5.0)
        ),
        "executed_visible_state_within_10px_rate": float(
            np.mean(np.asarray(distances) <= 10.0)
        ),
        "executed_visible_state_within_15px_rate": float(
            np.mean(np.asarray(distances) <= 15.0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--original-dataset",
        type=Path,
        default=DEFAULT_ORIGINAL_DATASET,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--oracle-only",
        action="store_true",
        help="Build the strict physical oracle without loading a model.",
    )
    parser.add_argument(
        "--oracle-path",
        type=Path,
        help=(
            "Reuse a previously built strict physical oracle. Condition IDs "
            "must exactly match the selected Public Test."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=13312)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--cem-batch-size", type=int, default=16)
    parser.add_argument("--cem-samples", type=int, default=300)
    parser.add_argument("--cem-iterations", type=int, default=30)
    parser.add_argument("--cem-topk", type=int, default=30)
    parser.add_argument("--oracle-grid-size", type=int, default=101)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    original = args.original_dataset.expanduser().resolve()
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else None
    )
    output = Path(os.path.abspath(args.output.expanduser()))
    required = (
        root / "manifest.json",
        root / "validation.lance",
    )
    if not args.oracle_only:
        if checkpoint is None:
            raise ValueError("--checkpoint is required unless --oracle-only")
        required = (*required, original, checkpoint)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input(s):\n" + "\n".join(map(str, missing))
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if (
        not args.oracle_only
        and device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA requested but unavailable")

    if args.oracle_only:
        # Normalized actions are not used while constructing the physical
        # oracle; avoid reading the large replay file for irrelevant stats.
        mean = np.zeros(2, dtype=np.float32)
        std = np.ones(2, dtype=np.float32)
    else:
        mean, std = action_stats(original)
    conditions = load_conditions(root, mean=mean, std=std)
    if args.oracle_path is None:
        print(
            f"Calibrating physical oracle for {len(conditions)} conditions",
            flush=True,
        )
        oracle = oracle_surface(
            conditions,
            grid_size=int(args.oracle_grid_size),
        )
        oracle_path = output / "oracle_surface.json"
        oracle_path.write_text(
            json.dumps(oracle, indent=2, sort_keys=True) + "\n"
        )
    else:
        if args.oracle_only:
            raise ValueError("--oracle-only and --oracle-path are exclusive")
        oracle_path = args.oracle_path.expanduser().resolve()
        if not oracle_path.is_file():
            raise FileNotFoundError(oracle_path)
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        expected_ids = {row.condition_id for row in conditions}
        observed_ids = {str(row["condition_id"]) for row in oracle}
        if len(oracle) != len(expected_ids) or observed_ids != expected_ids:
            raise RuntimeError(
                "Provided oracle condition IDs do not match Public Test"
            )

    if args.oracle_only:
        condition_ids = [row["condition_id"] for row in oracle]
        report = {
            "schema_version": 1,
            "status": "strict_causal_action_strength_oracle_completed",
            "data": {
                "root": str(root),
                "manifest_sha256": file_sha256(root / "manifest.json"),
                "condition_count": len(conditions),
            },
            "causal_execution": {
                "prefix_replayed_before_each_candidate": True,
                "state_installations_after_x0": 0,
                "query_simulator_recreated": False,
            },
            "oracle": {
                "path": str(oracle_path),
                "sha256": file_sha256(oracle_path),
                "grid_size": int(args.oracle_grid_size),
                "condition_ids_unique": len(condition_ids)
                == len(set(condition_ids)),
            },
        }
        report["passed"] = bool(
            len(conditions) == 512
            and report["oracle"]["condition_ids_unique"]
        )
        report_path = output / "oracle_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(1)
        return

    set_seed(0)
    assert checkpoint is not None
    model = load_model(checkpoint, device)
    cache = cache_model_inputs(
        model,
        conditions,
        device=device,
        batch_size=int(args.encode_batch_size),
    )
    started = time.monotonic()
    amplitude, cost = solve_cem(
        model,
        cache,
        action_mean=mean,
        action_std=std,
        seed=int(args.seed),
        num_samples=int(args.cem_samples),
        iterations=int(args.cem_iterations),
        topk=int(args.cem_topk),
        batch_size=int(args.cem_batch_size),
    )
    records = []
    for index, condition in enumerate(conditions):
        records.append(
            {
                "condition_id": condition.condition_id,
                "pair_index": condition.pair_index,
                "mode": condition.mode,
                "selected_predicted_cost": float(cost[index]),
                "execution": execute_amplitude(
                    condition,
                    float(amplitude[index]),
                ),
            }
        )
    summary = summarize(records, oracle)
    report = {
        "schema_version": 1,
        "status": "replay_matched_hidden_cem_real_execution",
        "data": {
            "root": str(root),
            "manifest_sha256": file_sha256(root / "manifest.json"),
            "condition_count": len(conditions),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "cem": {
            "seed": int(args.seed),
            "samples": int(args.cem_samples),
            "iterations": int(args.cem_iterations),
            "topk": int(args.cem_topk),
            "amplitude_bounds": [0.0, 1.0],
            "action_family": "exact_source_query_times_amplitude",
        },
        "physical_oracle": {
            "grid_size": int(args.oracle_grid_size),
            "state_metric": (
                "norm(agent_position_error, block_position_error, "
                "40*block_angle_error)"
            ),
            "path": str(oracle_path),
            "sha256": file_sha256(oracle_path),
        },
        "elapsed_seconds": time.monotonic() - started,
        "summary": summary,
        "records": records,
    }
    report_path = output / "aggregate.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": file_sha256(report_path),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
