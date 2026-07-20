#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.icl_catalog import validate_context_query_catalog
from contextworld.evaluation.icl_model import file_sha256, state_dict_sha256
from contextworld.evaluation.icl_planning import (
    CONTEXT_PIXELS_KEY,
    FixedContextCostModel,
    FixedContextPolicy,
    PairedQueryDataset,
    QueryEpisode,
)
from contextworld.evaluation.protocol import (
    EvaluationStarts,
    factor_readback_audit,
    frozen_normalizer_process,
    infer_model_protocol,
    load_legacy_cost_model,
    load_pretrained_cost_model,
    original_h5_process,
)
from contextworld.evaluation.tworoom import (
    TWOROOM_EVAL_ENV_ID,
    register_tworoom_eval_env,
    tworoom_eval_callables,
)
from contextworld.synthesis.manifest import write_json
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.stablewm import load_stable_worldmodel


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
DEFAULT_CHECKPOINT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/"
    "lewm-tworooms/ckpt/tworoom_lewm_20260430/"
    "tworoom_lewm_20260430_epoch_10_object.ckpt"
)
DEFAULT_LEGACY_CODE = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/code/wm_exp"
)
DEFAULT_ORIGINAL_H5 = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/"
    "lewm-tworooms/tworoom.h5"
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _generator_sha256(generator: Any) -> str:
    state = generator.get_state().detach().cpu().contiguous().numpy()
    return _array_sha256(state)


def image_transform(img_size: int):
    import stable_pretraining as spt
    import torch
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def _fixed_variation_values(speed: float, door_position: int) -> dict[str, Any]:
    return {
        "agent.speed": np.asarray([speed], dtype=np.float32),
        "door.position": np.asarray(
            [door_position, door_position, door_position], dtype=np.int64
        ),
        "door.size": np.asarray([14, 14, 14], dtype=np.int64),
        "door.number": 1,
        "wall.axis": 1,
        "wall.thickness": 10,
        "rendering.render_target": 0,
    }


def _goal_pixels_and_query_audit(
    *,
    speed: float,
    door_position: int,
    simulator_seed: int,
    query_state: np.ndarray,
    goal_state: np.ndarray,
    expected_query_pixels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    env = TwoRoomEnv(render_mode="rgb_array")
    try:
        env.reset(
            seed=simulator_seed,
            options={
                "variation": (),
                "variation_values": _fixed_variation_values(
                    speed, door_position
                ),
                "state": np.asarray(query_state, dtype=np.float32),
                "target_state": np.asarray(goal_state, dtype=np.float32),
            },
        )
        rendered_query = np.asarray(env.render(), dtype=np.uint8)
        if not np.array_equal(rendered_query, expected_query_pixels):
            raise RuntimeError(
                "Strict query replay differs from the catalog payload: "
                f"max_error={np.abs(rendered_query.astype(np.int16) - expected_query_pixels.astype(np.int16)).max()}"
            )
        goal_pixels = (
            env._target_img.detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        )
        return goal_pixels.copy(), {
            "query_pixels_exact_replay": True,
            "query_pixels_sha256": _array_sha256(rendered_query),
            "goal_pixels_sha256": _array_sha256(goal_pixels),
        }
    finally:
        env.close()


def _load_selected_queries(
    catalog_path: Path,
    *,
    speeds: list[float],
    templates: list[str],
) -> list[dict[str, Any]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    speed_bundles = [
        bundle for bundle in catalog["bundles"] if bundle["family"] == "speed"
    ]
    selected: list[dict[str, Any]] = []
    for speed in speeds:
        for template in templates:
            matches = [
                bundle
                for bundle in speed_bundles
                if np.isclose(
                    float(bundle["query_factors"]["agent.speed"]),
                    float(speed),
                    rtol=0.0,
                    atol=1e-6,
                )
                and bundle["template"]["template_id"] == template
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one speed={speed}, template={template} bundle; "
                    f"found {len(matches)}"
                )
            selected.append(matches[0])
    query_ids = [bundle["query_id"] for bundle in selected]
    if len(set(query_ids)) != len(query_ids):
        raise RuntimeError("Quick selection contains duplicate query IDs")
    return selected


def _balanced_evaluation_schedule(
    assets: list[dict[str, Any]],
    *,
    num_eval: int | None,
    eval_seed: int,
) -> list[dict[str, Any]]:
    """Cover every base query before reusing one in a randomized new round."""

    count = len(assets) if num_eval is None else int(num_eval)
    if count <= 0:
        raise ValueError("--num-eval must be positive")
    rng = np.random.default_rng(eval_seed)
    asset_indices: list[int] = []
    while len(asset_indices) < count:
        order = (
            np.arange(len(assets), dtype=np.int64)
            if num_eval is None
            else rng.permutation(len(assets))
        )
        remaining = count - len(asset_indices)
        asset_indices.extend(int(value) for value in order[:remaining])

    occurrences: dict[str, int] = {}
    schedule: list[dict[str, Any]] = []
    for evaluation_index, asset_index in enumerate(asset_indices):
        asset = assets[asset_index]
        query_id = asset["episode"].query_id
        repeat_index = occurrences.get(query_id, 0)
        occurrences[query_id] = repeat_index + 1
        cem_seed = (
            int(eval_seed)
            if num_eval is None
            else int(
                np.random.SeedSequence(
                    [int(eval_seed), int(evaluation_index), int(asset_index)]
                ).generate_state(1)[0]
            )
        )
        schedule.append(
            {
                "evaluation_id": f"s{eval_seed}-e{evaluation_index:03d}-{query_id}",
                "evaluation_index": evaluation_index,
                "repeat_index": repeat_index,
                "cem_seed": cem_seed,
                "asset": asset,
            }
        )
    return schedule


def _load_query_assets(
    bundle: dict[str, Any],
    *,
    process: dict[str, Any],
) -> dict[str, Any]:
    payload_path = resolve_contextworld_path(
        bundle["payload"], repo_root=REPO_ROOT
    )
    if file_sha256(payload_path) != bundle["payload_sha256"]:
        raise RuntimeError(f"Payload hash mismatch: {payload_path}")
    with np.load(payload_path, allow_pickle=False) as payload:
        query_pixels = np.asarray(payload["query_pixels"], dtype=np.uint8).copy()
        query_state = np.asarray(payload["query_state"], dtype=np.float32).copy()
        goal_state = np.asarray(
            bundle["template"]["goal_state"], dtype=np.float32
        )
        speed = float(bundle["query_factors"]["agent.speed"])
        door = int(bundle["query_factors"]["door.position"])
        goal_pixels, render_audit = _goal_pixels_and_query_audit(
            speed=speed,
            door_position=door,
            simulator_seed=int(bundle["simulator_seed"]),
            query_state=query_state,
            goal_state=goal_state,
            expected_query_pixels=query_pixels,
        )

        contexts: dict[str, dict[str, Any]] = {}
        for condition in ("correct", "wrong"):
            prefix = f"context_b2_{condition}"
            pixels = np.asarray(payload[f"{prefix}_pixels"], dtype=np.uint8).copy()
            raw_actions = np.asarray(
                payload[f"{prefix}_actions"], dtype=np.float32
            ).copy()
            next_pixels = np.asarray(
                payload[f"{prefix}_next_pixels"], dtype=np.uint8
            )
            next_states = np.asarray(
                payload[f"{prefix}_next_states"], dtype=np.float32
            )
            if not np.array_equal(next_states[-1], query_state):
                raise RuntimeError(
                    f"{bundle['query_id']} {condition} context does not end at query state"
                )
            if not np.array_equal(next_pixels[-1], query_pixels):
                raise RuntimeError(
                    f"{bundle['query_id']} {condition} context does not end at query pixels"
                )
            normalized = process["action"].transform(
                raw_actions.reshape(-1, 2)
            ).astype(np.float32).reshape(2, 10)
            contexts[condition] = {
                "pixels": pixels,
                "raw_actions": raw_actions,
                "normalized_actions": normalized,
                "pixels_sha256": _array_sha256(pixels),
                "raw_actions_sha256": _array_sha256(raw_actions),
                "normalized_actions_sha256": _array_sha256(normalized),
                "source_factors": bundle["conditions"][condition]["factors"],
            }

    if contexts["correct"]["raw_actions_sha256"] != contexts["wrong"]["raw_actions_sha256"]:
        raise RuntimeError("Correct/wrong fixed context actions are not paired")
    return {
        "bundle": bundle,
        "episode": QueryEpisode(
            query_id=bundle["query_id"],
            scenario_id=bundle["source_scenario_id"],
            template_id=bundle["template"]["template_id"],
            speed=speed,
            door_position=door,
            simulator_seed=int(bundle["simulator_seed"]),
            query_pixels=query_pixels,
            goal_pixels=goal_pixels,
            query_state=query_state,
            goal_state=goal_state,
        ),
        "contexts": contexts,
        "render_audit": render_audit,
        "payload": str(payload_path),
        "payload_sha256": bundle["payload_sha256"],
    }


def _run_one(
    *,
    args: argparse.Namespace,
    swm: Any,
    model: Any,
    process: dict[str, Any],
    protocol: dict[str, int],
    asset: dict[str, Any],
    condition: str,
    evaluation_id: str,
    evaluation_index: int,
    repeat_index: int,
    cem_seed: int,
) -> dict[str, Any]:
    import torch

    episode = asset["episode"]
    context = asset["contexts"][condition]
    dataset = PairedQueryDataset([episode])
    world = swm.World(
        TWOROOM_EVAL_ENV_ID,
        num_envs=1,
        max_episode_steps=2 * args.eval_budget,
        image_shape=(args.img_size, args.img_size),
        render_mode="rgb_array",
    )
    prompted_model = FixedContextCostModel(
        model, history_size=protocol["history_size"]
    )
    solver = swm.solver.CEMSolver(
        model=prompted_model,
        batch_size=args.cem_batch_size,
        num_samples=args.cem_num_samples,
        var_scale=args.cem_var_scale,
        n_steps=args.cem_steps,
        topk=args.cem_topk,
        device=args.device,
        seed=cem_seed,
    )
    rng_before = _generator_sha256(solver.torch_gen)
    config = swm.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        history_len=protocol["history_size"],
        action_block=protocol["action_block"],
        warm_start=True,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform={
            "pixels": image_transform(args.img_size),
            "goal": image_transform(args.img_size),
            CONTEXT_PIXELS_KEY: image_transform(args.img_size),
        },
    )
    trajectory_steps: list[dict[str, Any]] = []
    prompted_policy = FixedContextPolicy(
        policy,
        context_pixels=context["pixels"][None],
        context_actions=context["normalized_actions"][None],
        trace_steps=trajectory_steps,
    )
    starts = EvaluationStarts(episodes=[0], steps=[0])
    try:
        world.set_policy(prompted_policy)
        started = time.time()
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=[0],
            start_steps=[0],
            goal_offset=1,
            eval_budget=args.eval_budget,
            callables=tworoom_eval_callables(),
            video=None,
        )
        elapsed = time.time() - started
        factor_audit = factor_readback_audit(dataset, world, starts)
        if not factor_audit["passed"]:
            raise RuntimeError(
                f"Factor readback failed for {episode.query_id}: {factor_audit}"
            )
        final_state = (
            world.envs.envs[0]
            .unwrapped.agent_position.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        success = bool(np.asarray(metrics["episode_successes"])[0])
        final_distance = float(
            np.linalg.norm(final_state - np.asarray(episode.goal_state))
        )
        states = [
            np.asarray(step["state"], dtype=np.float32)
            for step in trajectory_steps
        ]
        actions = [
            np.asarray(step["action"], dtype=np.float32)
            for step in trajectory_steps
        ]
        states.append(final_state)
        state_array = np.stack(states)
        action_array = (
            np.stack(actions)
            if actions
            else np.empty((0, 2), dtype=np.float32)
        )
        goal_state = np.asarray(episode.goal_state, dtype=np.float32)
        distances = np.linalg.norm(state_array - goal_state[None], axis=1)
        path_length = float(
            np.linalg.norm(np.diff(state_array, axis=0), axis=1).sum()
        )
        initial_distance = float(distances[0])
        path_efficiency = (
            float(initial_distance / path_length)
            if success and path_length > 0.0
            else None
        )
        progress_per_path_length = (
            float((initial_distance - final_distance) / path_length)
            if path_length > 0.0
            else 0.0
        )
        return {
            "evaluation_id": evaluation_id,
            "evaluation_index": int(evaluation_index),
            "repeat_index": int(repeat_index),
            "eval_seed": int(args.seed),
            "query_id": episode.query_id,
            "source_scenario_id": episode.scenario_id,
            "template_id": episode.template_id,
            "speed": float(episode.speed),
            "door_position": int(episode.door_position),
            "condition": condition,
            "success": success,
            "final_state": final_state.tolist(),
            "goal_state": np.asarray(episode.goal_state).tolist(),
            "final_distance": final_distance,
            "trajectory": {
                "raw_steps_executed": len(actions),
                "states": state_array.tolist(),
                "actions": action_array.tolist(),
                "goal_distances": distances.tolist(),
                "steps_to_success": len(actions) if success else None,
                "path_length": path_length,
                "path_efficiency_success_only": path_efficiency,
                "progress_per_path_length": progress_per_path_length,
                "distance_auc_raw_step_mean": float(np.mean(distances)),
                "normalized_distance_auc": (
                    float(np.mean(distances) / initial_distance)
                    if initial_distance > 0.0
                    else 0.0
                ),
            },
            "elapsed_seconds": elapsed,
            "cem_seed": int(cem_seed),
            "cem_rng_state_sha256_before": rng_before,
            "cem_rng_state_sha256_after": _generator_sha256(solver.torch_gen),
            "cem_solve_calls": int(prompted_model.get_cost_calls // args.cem_steps),
            "get_cost_calls": int(prompted_model.get_cost_calls),
            "fixed_context": {
                "budget": 2,
                "pixels_sha256": context["pixels_sha256"],
                "raw_actions_sha256": context["raw_actions_sha256"],
                "normalized_actions_sha256": context[
                    "normalized_actions_sha256"
                ],
                "source_factors": context["source_factors"],
            },
            "factor_readback": factor_audit,
        }
    finally:
        world.close()
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()


def _pair_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {
        (record["evaluation_id"], record["condition"]): record
        for record in records
    }
    evaluation_ids = sorted(
        {record["evaluation_id"] for record in records},
        key=lambda value: lookup[(value, "correct")]["evaluation_index"],
    )
    pairs = []
    for evaluation_id in evaluation_ids:
        correct = lookup[(evaluation_id, "correct")]
        wrong = lookup[(evaluation_id, "wrong")]
        pairs.append(
            {
                "evaluation_id": evaluation_id,
                "evaluation_index": correct["evaluation_index"],
                "repeat_index": correct["repeat_index"],
                "query_id": correct["query_id"],
                "source_scenario_id": correct["source_scenario_id"],
                "template_id": correct["template_id"],
                "speed": correct["speed"],
                "correct_success": correct["success"],
                "wrong_success": wrong["success"],
                "correct_only_success": bool(
                    correct["success"] and not wrong["success"]
                ),
                "wrong_only_success": bool(
                    wrong["success"] and not correct["success"]
                ),
                "wrong_minus_correct_final_distance": float(
                    wrong["final_distance"] - correct["final_distance"]
                ),
                "same_initial_cem_rng_state": (
                    correct["cem_rng_state_sha256_before"]
                    == wrong["cem_rng_state_sha256_before"]
                ),
                "same_fixed_context_actions": (
                    correct["fixed_context"]["normalized_actions_sha256"]
                    == wrong["fixed_context"]["normalized_actions_sha256"]
                ),
            }
        )
    correct_successes = sum(pair["correct_success"] for pair in pairs)
    wrong_successes = sum(pair["wrong_success"] for pair in pairs)
    correct_distances = [
        lookup[(evaluation_id, "correct")]["final_distance"]
        for evaluation_id in evaluation_ids
    ]
    wrong_distances = [
        lookup[(evaluation_id, "wrong")]["final_distance"]
        for evaluation_id in evaluation_ids
    ]
    distance_deltas = [
        pair["wrong_minus_correct_final_distance"] for pair in pairs
    ]
    by_speed: dict[str, Any] = {}
    for speed in sorted({float(pair["speed"]) for pair in pairs}):
        speed_pairs = [pair for pair in pairs if float(pair["speed"]) == speed]
        by_speed[f"{speed:g}"] = {
            "queries": len(speed_pairs),
            "correct_successes": int(
                sum(pair["correct_success"] for pair in speed_pairs)
            ),
            "wrong_successes": int(
                sum(pair["wrong_success"] for pair in speed_pairs)
            ),
            "correct_minus_wrong_success_rate_points": float(
                100.0
                * sum(
                    int(pair["correct_success"]) - int(pair["wrong_success"])
                    for pair in speed_pairs
                )
                / len(speed_pairs)
            ),
            "wrong_minus_correct_mean_final_distance": float(
                np.mean(
                    [
                        pair["wrong_minus_correct_final_distance"]
                        for pair in speed_pairs
                    ]
                )
            ),
        }
    return {
        "queries": len(pairs),
        "correct": {
            "successes": int(correct_successes),
            "success_rate": 100.0 * correct_successes / len(pairs),
            "mean_final_distance": float(np.mean(correct_distances)),
        },
        "wrong": {
            "successes": int(wrong_successes),
            "success_rate": 100.0 * wrong_successes / len(pairs),
            "mean_final_distance": float(np.mean(wrong_distances)),
        },
        "correct_minus_wrong_success_rate_points": float(
            100.0 * (correct_successes - wrong_successes) / len(pairs)
        ),
        "wrong_minus_correct_mean_final_distance": float(
            np.mean(wrong_distances) - np.mean(correct_distances)
        ),
        "mean_absolute_paired_final_distance_difference": float(
            np.mean(np.abs(distance_deltas))
        ),
        "correct_lower_final_distance_pairs": int(
            sum(delta > 0.0 for delta in distance_deltas)
        ),
        "wrong_lower_final_distance_pairs": int(
            sum(delta < 0.0 for delta in distance_deltas)
        ),
        "correct_only_successes": int(
            sum(pair["correct_only_success"] for pair in pairs)
        ),
        "wrong_only_successes": int(
            sum(pair["wrong_only_success"] for pair in pairs)
        ),
        "both_successes": int(
            sum(pair["correct_success"] and pair["wrong_success"] for pair in pairs)
        ),
        "neither_successes": int(
            sum(
                not pair["correct_success"] and not pair["wrong_success"]
                for pair in pairs
            )
        ),
        "by_speed": by_speed,
        "pairs": pairs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    args.catalog = resolve_contextworld_path(args.catalog, repo_root=REPO_ROOT)
    args.output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.horizon * 5 > args.eval_budget:
        raise ValueError("horizon * action_block exceeds eval budget")

    swm, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    register_tworoom_eval_env()
    validation = validate_context_query_catalog(
        args.catalog.resolve(),
        repo_root=REPO_ROOT,
        replay_simulator=not args.skip_catalog_replay,
        family="speed",
    )
    if not validation["passed"]:
        raise RuntimeError(f"Catalog validation failed: {validation['failures'][:5]}")

    process = (
        frozen_normalizer_process(args.normalizer.resolve())
        if args.normalizer is not None
        else original_h5_process(args.original_h5.resolve())
    )
    checkpoint = args.checkpoint.resolve()
    if checkpoint.suffix.lower() == ".pt":
        model = load_pretrained_cost_model(
            checkpoint,
            swm,
            cache_dir=artifact_path(
                "evaluation/model_cache", repo_root=REPO_ROOT
            ),
        )
        checkpoint_serialization = "stablewm_pretrained"
    else:
        model = load_legacy_cost_model(
            checkpoint, args.legacy_code_root.resolve()
        )
        checkpoint_serialization = "legacy_object"
    protocol = infer_model_protocol(model, action_dim=2)
    if protocol != {"action_block": 5, "history_size": 3}:
        raise RuntimeError(f"E4 requires history=3/action_block=5, got {protocol}")
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Failed to freeze model")
    setattr(model, "history_size", protocol["history_size"])
    setattr(model, "interpolate_pos_encoding", True)
    weight_hash_before = state_dict_sha256(model)

    selected = _load_selected_queries(
        args.catalog.resolve(), speeds=args.speeds, templates=args.templates
    )
    assets = [
        _load_query_assets(bundle, process=process) for bundle in selected
    ]
    schedule = _balanced_evaluation_schedule(
        assets, num_eval=args.num_eval, eval_seed=args.seed
    )
    records: list[dict[str, Any]] = []
    total = len(schedule) * 2
    run_index = 0
    for scheduled in schedule:
        asset = scheduled["asset"]
        episode = asset["episode"]
        for condition in ("correct", "wrong"):
            run_index += 1
            print(
                f"[{run_index}/{total}] speed={episode.speed:g} "
                f"template={episode.template_id} "
                f"repeat={scheduled['repeat_index']} condition={condition}",
                flush=True,
            )
            records.append(
                _run_one(
                    args=args,
                    swm=swm,
                    model=model,
                    process=process,
                    protocol=protocol,
                    asset=asset,
                    condition=condition,
                    evaluation_id=scheduled["evaluation_id"],
                    evaluation_index=scheduled["evaluation_index"],
                    repeat_index=scheduled["repeat_index"],
                    cem_seed=scheduled["cem_seed"],
                )
            )

    weight_hash_after = state_dict_sha256(model)
    if weight_hash_before != weight_hash_after:
        raise RuntimeError("Model state changed during frozen E4 evaluation")
    pair_metrics = _pair_metrics(records)
    pair_audit_passed = all(
        pair["same_initial_cem_rng_state"] and pair["same_fixed_context_actions"]
        for pair in pair_metrics["pairs"]
    )
    output = {
        "schema_version": 1,
        "benchmark": "contextworld_tworoom_history3_e4_v1",
        "experiment_id": "E4",
        "display_name": "paired-context planning",
        "run_kind": args.run_kind,
        "status": "passed",
        "evidence_level": (
            "confirmation" if args.run_kind == "confirmation" else "qualitative_only"
        ),
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "serialization": checkpoint_serialization,
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "frozen_weight_audit": {
            "requires_grad_false": not any(
                parameter.requires_grad for parameter in model.parameters()
            ),
            "optimizer_created": False,
            "state_dict_sha256_before": weight_hash_before,
            "state_dict_sha256_after": weight_hash_after,
            "passed": weight_hash_before == weight_hash_after,
        },
        "catalog_validation": validation,
        "selection": {
            "family": "speed",
            "speeds": [float(value) for value in args.speeds],
            "templates": args.templates,
            "query_ids": [asset["episode"].query_id for asset in assets],
            "unique_base_queries": len(assets),
            "evaluations_per_condition": len(schedule),
            "reused_evaluations_per_condition": max(
                0, len(schedule) - len(assets)
            ),
            "schedule": [
                {
                    "evaluation_id": scheduled["evaluation_id"],
                    "evaluation_index": scheduled["evaluation_index"],
                    "repeat_index": scheduled["repeat_index"],
                    "cem_seed": scheduled["cem_seed"],
                    "query_id": scheduled["asset"]["episode"].query_id,
                }
                for scheduled in schedule
            ],
            "query_manifest": [
                {
                    "query_id": asset["episode"].query_id,
                    "source_scenario_id": asset["episode"].scenario_id,
                    "speed": float(asset["episode"].speed),
                    "template_id": asset["episode"].template_id,
                    "door_position": int(asset["episode"].door_position),
                }
                for asset in assets
            ],
            "conditions": ["correct", "wrong"],
            "context_budget": 2,
            "render_audits": {
                asset["episode"].query_id: asset["render_audit"] for asset in assets
            },
        },
        "protocol": {
            **protocol,
            "eval_seed": args.seed,
            "eval_budget": args.eval_budget,
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "cem_batch_size": args.cem_batch_size,
            "cem_num_samples": args.cem_num_samples,
            "cem_var_scale": args.cem_var_scale,
            "cem_steps": args.cem_steps,
            "cem_topk": args.cem_topk,
            "normalization_source": str(
                args.normalizer.resolve()
                if args.normalizer is not None
                else args.original_h5.resolve()
            ),
            "model_input_layout": [
                "context_observation_1",
                "context_observation_2",
                "live_query_observation",
            ],
            "action_layout": {
                "fixed_context_blocks": 2,
                "cem_optimized_future_blocks": args.horizon,
                "model_rollout_blocks": 2 + args.horizon,
            },
            "cem_pairing": (
                "fresh solver with identical seed per evaluation and condition; "
                "identical random stream for every shared replan"
            ),
            "variation_restore_order": ["variation", "state", "goal"],
        },
        "pairing_audit": {
            "identical_query_and_goal_by_construction": True,
            "identical_cem_initial_rng_state_per_pair": all(
                pair["same_initial_cem_rng_state"]
                for pair in pair_metrics["pairs"]
            ),
            "identical_fixed_context_actions_per_pair": all(
                pair["same_fixed_context_actions"]
                for pair in pair_metrics["pairs"]
            ),
            "fixed_context_actions_not_cem_variables": True,
            "factor_readback_passed": all(
                record["factor_readback"]["passed"] for record in records
            ),
            "passed": pair_audit_passed,
        },
        "aggregate": pair_metrics,
        "records": records,
    }
    write_json(args.output.resolve(), output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E4 fixed-K=2 paired correct/wrong context planning probe"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=artifact_path(
            "evaluation/icl/tworoom_icl_v1_validation_context_query_catalog.json",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--legacy-code-root", type=Path, default=DEFAULT_LEGACY_CODE
    )
    parser.add_argument("--original-h5", type=Path, default=DEFAULT_ORIGINAL_H5)
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_path(
            "evaluation/history3/e4_speed_ctx_quick_s42.json",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-eval", type=int)
    parser.add_argument(
        "--run-kind",
        choices=("qualitative_probe", "confirmation"),
        default="qualitative_probe",
    )
    parser.add_argument("--speeds", type=float, nargs="+", default=[3.1, 5.0, 7.0])
    parser.add_argument("--templates", nargs="+", default=["s0", "s1", "s2"])
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--cem-batch-size", type=int, default=1)
    parser.add_argument("--cem-num-samples", type=int, default=300)
    parser.add_argument("--cem-var-scale", type=float, default=1.0)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--cem-topk", type=int, default=30)
    parser.add_argument("--skip-catalog-replay", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(parsed.output.resolve()),
                "aggregate": {
                    key: value
                    for key, value in result["aggregate"].items()
                    if key != "pairs"
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
