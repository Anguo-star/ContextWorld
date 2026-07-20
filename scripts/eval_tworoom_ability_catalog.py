#!/usr/bin/env python3
"""Evaluate one frozen original-ability planning catalog and model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.icl_model import state_dict_sha256
from contextworld.evaluation.protocol import (
    frozen_normalizer_process,
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.evaluation.tworoom import (
    TWOROOM_EVAL_ENV_ID,
    register_tworoom_eval_env,
    tworoom_eval_callables,
)
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from scripts.eval_tworoom_step1 import image_transform


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _h5_dataset(swm: Any, path: Path):
    return swm.data.HDF5Dataset(
        path=path,
        frameskip=1,
        num_steps=1,
        keys_to_load=["pixels", "action", "proprio", "pos_agent"],
        keys_to_cache=["action", "proprio"],
    )


def _lance_dataset(swm: Any, path: Path):
    return swm.data.LanceDataset(
        path=path,
        frameskip=1,
        num_steps=1,
        keys_to_load=[
            "pixels",
            "action",
            "proprio",
            "state",
            "goal_state",
            "variation_agent_speed",
        ],
    )


def _original_callables() -> list[dict[str, Any]]:
    return [
        {
            "method": "_set_state",
            "args": {"state": {"value": "pos_agent", "in_dataset": True}},
        },
        {
            "method": "_set_goal_state",
            "args": {
                "goal_state": {
                    "value": "goal_pos_agent",
                    "in_dataset": True,
                }
            },
        },
    ]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _run_group(
    *,
    args: argparse.Namespace,
    swm: Any,
    model: Any,
    protocol: dict[str, int],
    process: dict[str, Any],
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    source_kind = entries[0]["source_kind"]
    source_path = resolve_contextworld_path(
        entries[0]["source_path"], repo_root=REPO_ROOT
    )
    if source_kind == "original_h5":
        dataset = _h5_dataset(swm, source_path)
        env_id = "swm/TwoRoom-v1"
        callables = _original_callables()
        state_key = "pos_agent"
    elif source_kind == "synthetic_lance":
        dataset = _lance_dataset(swm, source_path)
        env_id = TWOROOM_EVAL_ENV_ID
        callables = tworoom_eval_callables()
        state_key = "state"
    else:
        raise ValueError(f"Unsupported catalog source kind: {source_kind}")

    episodes = [int(entry["episode"]) for entry in entries]
    starts = [int(entry["start_step"]) for entry in entries]
    goal_offset = int(entries[0]["goal_offset"])
    cem_seed = int(entries[0]["cem_group_seed"])
    if any(
        int(entry["goal_offset"]) != goal_offset
        or int(entry["cem_group_seed"]) != cem_seed
        for entry in entries
    ):
        raise ValueError("Catalog group has inconsistent goal offsets or CEM seeds")

    world = swm.World(
        env_id,
        num_envs=len(entries),
        max_episode_steps=2 * args.eval_budget,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=args.cem_samples,
        var_scale=1.0,
        n_steps=args.cem_steps,
        topk=args.cem_topk,
        device=args.device,
        seed=cem_seed,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=args.horizon,
            receding_horizon=args.receding_horizon,
            history_len=protocol["history_size"],
            action_block=protocol["action_block"],
            warm_start=True,
        ),
        process=process,
        transform={
            "pixels": image_transform(224),
            "goal": image_transform(224),
        },
    )
    try:
        world.set_policy(policy)
        started = time.time()
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=episodes,
            start_steps=starts,
            goal_offset=goal_offset,
            eval_budget=args.eval_budget,
            callables=callables,
            video=None,
        )
        elapsed = time.time() - started
        chunks = dataset.load_chunk(
            np.asarray(episodes, dtype=np.int64),
            np.asarray(starts, dtype=np.int64),
            np.asarray(starts, dtype=np.int64) + goal_offset + 1,
        )
        initial_states = np.stack(
            [_to_numpy(chunk[state_key][0]) for chunk in chunks]
        ).astype(np.float32)
        goal_states = np.stack(
            [_to_numpy(chunk[state_key][-1]) for chunk in chunks]
        ).astype(np.float32)
        final_states = np.stack(
            [
                _to_numpy(env.unwrapped.agent_position).astype(np.float32)
                for env in world.envs.envs
            ]
        )
        successes = np.asarray(metrics["episode_successes"], dtype=bool)
        records = []
        for entry, initial, goal, final, success in zip(
            entries,
            initial_states,
            goal_states,
            final_states,
            successes,
            strict=True,
        ):
            records.append(
                {
                    **entry,
                    "initial_state": initial.tolist(),
                    "goal_state": goal.tolist(),
                    "final_state": final.tolist(),
                    "final_distance": float(np.linalg.norm(final - goal)),
                    "success": bool(success),
                    "room_relation": (
                        "cross_room"
                        if (initial[0] < 112.0) != (goal[0] < 112.0)
                        else "same_room"
                    ),
                    "group_elapsed_seconds": elapsed,
                }
            )
        return records
    finally:
        world.close()
        del solver, policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    catalog_path = resolve_contextworld_path(args.catalog, repo_root=REPO_ROOT)
    checkpoint = resolve_contextworld_path(args.checkpoint, repo_root=REPO_ROOT)
    normalizer = resolve_contextworld_path(args.normalizer, repo_root=REPO_ROOT)
    output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in catalog["entries"]
        if int(entry["eval_seed"]) == args.seed
    ]
    if len(entries) != 50:
        raise ValueError(f"Expected 50 catalog entries for seed {args.seed}")
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    register_tworoom_eval_env()
    process = frozen_normalizer_process(normalizer)
    model = load_pretrained_cost_model(
        checkpoint,
        swm,
        cache_dir=artifact_path("evaluation/model_cache", repo_root=REPO_ROOT),
    )
    protocol = infer_model_protocol(model, action_dim=2)
    if protocol != {"action_block": 5, "history_size": 3}:
        raise RuntimeError(f"Unexpected model protocol: {protocol}")
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    setattr(model, "history_size", protocol["history_size"])
    setattr(model, "interpolate_pos_encoding", True)
    before = state_dict_sha256(model)

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[
            (
                entry["source_kind"],
                entry["source_path"],
                int(entry["cem_group_seed"]),
            )
        ].append(entry)
    records: list[dict[str, Any]] = []
    for index, values in enumerate(grouped.values(), start=1):
        print(
            f"[{index}/{len(grouped)}] {values[0]['source_kind']} "
            f"{values[0]['source_path']} n={len(values)}",
            flush=True,
        )
        records.extend(
            _run_group(
                args=args,
                swm=swm,
                model=model,
                protocol=protocol,
                process=process,
                entries=values,
            )
        )
    records.sort(key=lambda item: int(item["evaluation_index"]))
    after = state_dict_sha256(model)
    if before != after:
        raise RuntimeError("Model weights changed during frozen evaluation")
    successes = sum(record["success"] for record in records)
    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_original_ability_planning_v1",
        "status": "passed",
        "catalog": {
            "path": str(catalog_path),
            "sha256": _sha256(catalog_path),
            "kind": catalog["catalog"],
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "normalizer": {
            "path": str(normalizer),
            "sha256": _sha256(normalizer),
        },
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "protocol": {
            **protocol,
            "eval_seed": args.seed,
            "evaluations": len(records),
            "eval_budget": args.eval_budget,
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "cem_samples": args.cem_samples,
            "cem_steps": args.cem_steps,
            "cem_topk": args.cem_topk,
        },
        "frozen_weight_audit": {
            "state_dict_sha256_before": before,
            "state_dict_sha256_after": after,
            "passed": before == after,
        },
        "aggregate": {
            "successes": int(successes),
            "evaluations": len(records),
            "success_rate": float(successes / len(records)),
            "mean_final_distance": float(
                np.mean([record["final_distance"] for record in records])
            ),
        },
        "raw_records": records,
    }
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=Path(
            "artifacts/splits/tworoom_original_train_s3072_normalizer.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--cem-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--cem-topk", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["aggregate"], sort_keys=True))
