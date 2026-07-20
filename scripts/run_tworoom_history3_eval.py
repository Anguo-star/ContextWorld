#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = (
    REPO_ROOT / "configs/benchmark/tworoom_history3_eval_v1.yaml"
)


from contextworld.paths import resolve_contextworld_path


def _repo_path(value: str | Path) -> Path:
    return resolve_contextworld_path(value, repo_root=REPO_ROOT)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported eval config schema: {config.get('schema_version')}")
    return config


def _require(path: Path, kind: str) -> None:
    predicate = path.is_file if kind == "file" else path.is_dir
    if not predicate():
        raise FileNotFoundError(f"Required {kind} does not exist: {path}")


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def _run_command(command: list[str], *, dry_run: bool) -> None:
    print(f"\n$ {_command_text(command)}", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _check(
    *,
    config: dict[str, Any],
    checkpoint: Path,
    legacy_code_root: Path,
    original_h5: Path,
    stablewm_repo: Path,
    device: str,
    load_model: bool,
) -> dict[str, Any]:
    required_files = [checkpoint, original_h5]
    groups = config["evaluation_groups"]
    for name in ("e1_paired_prediction", "e2_natural_prediction"):
        required_files.append(_repo_path(groups[name]["catalog"]))
    for family in groups["e3_no_context_planning"]["families"]:
        required_files.append(_repo_path(family["catalog"]))
    required_files.append(
        _repo_path(groups["e4_paired_context_planning"]["catalog"])
    )
    for path in required_files:
        _require(path, "file")
    _require(legacy_code_root, "dir")
    _require(stablewm_repo, "dir")

    from contextworld.synthesis.stablewm import load_stable_worldmodel

    swm, resolved_stablewm, stablewm_commit = load_stable_worldmodel(
        REPO_ROOT,
        str(stablewm_repo),
        config["runtime"]["stable_worldmodel_commit"],
    )
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "legacy_code_root": str(legacy_code_root),
        "original_h5": str(original_h5),
        "stable_worldmodel_repo": str(resolved_stablewm),
        "stable_worldmodel_commit": stablewm_commit,
        "device": device,
    }

    if load_model:
        from contextworld.evaluation.protocol import (
            infer_model_protocol,
            load_legacy_cost_model,
            load_pretrained_cost_model,
        )

        if checkpoint.suffix.lower() == ".pt":
            model = load_pretrained_cost_model(
                checkpoint,
                swm,
                cache_dir=_repo_path("artifacts/evaluation/model_cache"),
            )
            serialization = "stablewm_pretrained"
        else:
            model = load_legacy_cost_model(checkpoint, legacy_code_root)
            serialization = "legacy_object"
        protocol = infer_model_protocol(model, action_dim=2)
        expected = {
            "action_block": int(config["model"]["action_block"]),
            "history_size": int(config["model"]["history_size"]),
        }
        if protocol != expected:
            raise RuntimeError(
                f"Checkpoint protocol mismatch: expected={expected}, observed={protocol}"
            )
        result.update(
            {
                "model_class": f"{type(model).__module__}.{type(model).__name__}",
                "parameters": sum(value.numel() for value in model.parameters()),
                "model_protocol": protocol,
                "checkpoint_serialization": serialization,
            }
        )
        del model
        gc.collect()

    if device.startswith("cuda") and load_model:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {device}, but CUDA is unavailable in this process. "
                "Use --dry-run to print commands or run on a GPU worker."
            )
    print(json.dumps({"check": result}, indent=2, sort_keys=True), flush=True)
    return result


def _common_model_args(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    legacy_code_root: Path,
    original_h5: Path,
    stablewm_repo: Path,
    stablewm_ref: str,
) -> list[str]:
    return [
        "--checkpoint",
        str(checkpoint),
        "--legacy-code-root",
        str(legacy_code_root),
        "--original-h5",
        str(original_h5),
        "--stablewm-repo",
        str(stablewm_repo),
        "--stablewm-ref",
        stablewm_ref,
        "--device",
        args.device,
        "--seed",
        str(args.prediction_seed),
    ]


def _prediction_commands(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    checkpoint: Path,
    legacy_code_root: Path,
    original_h5: Path,
    stablewm_repo: Path,
    output_dir: Path,
) -> dict[str, list[str]]:
    python = args.python
    stablewm_ref = config["runtime"]["stable_worldmodel_commit"]
    common = _common_model_args(
        args,
        checkpoint=checkpoint,
        legacy_code_root=legacy_code_root,
        original_h5=original_h5,
        stablewm_repo=stablewm_repo,
        stablewm_ref=stablewm_ref,
    )
    groups = config["evaluation_groups"]
    paired = groups["e1_paired_prediction"]
    natural = groups["e2_natural_prediction"]
    active_family = config["study_scope"]["active_family"]
    if paired["families"] != [active_family] or natural["families"] != [active_family]:
        raise RuntimeError("Prediction suites must match study_scope.active_family")
    paired_command = [
        python,
        str(_repo_path(paired["entrypoint"])),
        "--catalog",
        str(_repo_path(paired["catalog"])),
        "--output",
        str(output_dir / paired["output_name"]),
        "--encode-batch-size",
        str(paired["encode_batch_size"]),
        "--predictor-batch-size",
        str(paired["predictor_batch_size"]),
        "--family",
        active_family,
        *common,
    ]
    if args.skip_catalog_replay:
        paired_command.append("--skip-catalog-replay")
    natural_command = [
        python,
        str(_repo_path(natural["entrypoint"])),
        "--catalog",
        str(_repo_path(natural["catalog"])),
        "--output",
        str(output_dir / natural["output_name"]),
        "--encode-batch-size",
        str(natural["encode_batch_size"]),
        "--predictor-batch-size",
        str(natural["predictor_batch_size"]),
        "--family",
        active_family,
        *common,
    ]
    return {"e1-paired-pred": paired_command, "e2-natural-pred": natural_command}


def _planning_runs(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    checkpoint: Path,
    legacy_code_root: Path,
    original_h5: Path,
    stablewm_repo: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    planning = config["evaluation_groups"]["e3_no_context_planning"]
    profile = planning["profiles"][args.planning_profile]
    eval_seeds = (
        args.eval_seeds
        if args.eval_seeds is not None
        else [int(value) for value in profile["eval_seeds"]]
    )
    num_eval = (
        args.num_eval
        if args.num_eval is not None
        else int(profile["num_eval_per_seed"])
    )
    max_scenarios = None if args.all_scenarios else args.max_scenarios
    if max_scenarios is None and not args.all_scenarios:
        max_scenarios = profile.get("max_scenarios")
    if args.all_scenarios or args.max_scenarios is not None:
        scenario_indices = None
    elif args.scenario_indices is not None:
        scenario_indices = args.scenario_indices
    else:
        scenario_indices = profile.get("scenario_indices")
    groups: list[dict[str, Any]] = []
    for family in planning["families"]:
        stem = f"e3_{family['name']}_noctx_{args.planning_profile}"
        runs = []
        for eval_seed in eval_seeds:
            output = output_dir / f"{stem}_s{eval_seed}.json"
            command = [
                args.python,
                str(_repo_path(planning["entrypoint"])),
                "--catalog",
                str(_repo_path(family["catalog"])),
                "--regime",
                family["regime"],
                "--output",
                str(output),
                "--run-kind",
                profile["run_kind"],
                "--stablewm-repo",
                str(stablewm_repo),
                "--stablewm-ref",
                config["runtime"]["stable_worldmodel_commit"],
                "--policy-checkpoint",
                str(checkpoint),
                "--legacy-code-root",
                str(legacy_code_root),
                "--original-h5",
                str(original_h5),
                "--device",
                args.device,
                "--seed",
                str(eval_seed),
                "--img-size",
                str(config["model"]["image_size"]),
                "--num-eval",
                str(num_eval),
                "--goal-offset",
                str(profile["goal_offset"]),
                "--eval-budget",
                str(profile["eval_budget"]),
                "--horizon",
                str(profile["horizon"]),
                "--receding-horizon",
                str(profile["receding_horizon"]),
                "--cem-batch-size",
                str(profile["cem_batch_size"]),
                "--cem-num-samples",
                str(profile["cem_num_samples"]),
                "--cem-var-scale",
                str(profile["cem_var_scale"]),
                "--cem-steps",
                str(profile["cem_steps"]),
                "--cem-topk",
                str(profile["cem_topk"]),
            ]
            if max_scenarios is not None:
                command.extend(["--max-scenarios", str(max_scenarios)])
            if scenario_indices is not None:
                command.extend(
                    [
                        "--scenario-indices",
                        *[str(index) for index in scenario_indices],
                    ]
                )
            runs.append({"seed": eval_seed, "output": output, "command": command})
        groups.append(
            {
                "experiment_id": planning["experiment_id"],
                "family": family["name"],
                "profile": args.planning_profile,
                "num_eval_per_seed": num_eval,
                "eval_seeds": eval_seeds,
                "runs": runs,
                "summary": output_dir / f"{stem}.json",
                "original_tworoom_reference": planning.get(
                    "original_tworoom_reference"
                ),
            }
        )
    return groups


def _aggregate_planning_runs(group: dict[str, Any]) -> dict[str, Any]:
    payloads = []
    expected_scenarios: set[str] | None = None
    for run in group["runs"]:
        with run["output"].open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("status") != "passed":
            raise RuntimeError(f"Planning run did not pass: {run['output']}")
        if payload["protocol"]["eval_seed"] != run["seed"]:
            raise RuntimeError(f"Eval seed mismatch in {run['output']}")
        if payload["aggregate"]["evaluations"] != group["num_eval_per_seed"]:
            raise RuntimeError(
                f"Expected {group['num_eval_per_seed']} evaluations in "
                f"{run['output']}, found {payload['aggregate']['evaluations']}"
            )
        scenarios = {value["scenario"] for value in payload["scenarios"]}
        if expected_scenarios is None:
            expected_scenarios = scenarios
        elif scenarios != expected_scenarios:
            raise RuntimeError("Planning seed runs cover different scenario sets")
        payloads.append(payload)

    seed_rates = [
        float(payload["aggregate"]["scenario_balanced_success_rate"])
        for payload in payloads
    ]
    total_evaluations = sum(
        int(payload["aggregate"]["evaluations"]) for payload in payloads
    )
    total_successes = sum(
        int(payload["aggregate"]["successes"]) for payload in payloads
    )
    std = statistics.pstdev(seed_rates) if len(seed_rates) > 1 else 0.0

    scenario_rows = []
    for scenario in sorted(expected_scenarios or []):
        records = [
            next(
                value
                for value in payload["scenarios"]
                if value["scenario"] == scenario
            )
            for payload in payloads
        ]
        evaluations = sum(int(value["evaluations"]) for value in records)
        successes = sum(sum(value["successes"]) for value in records)
        scenario_rows.append(
            {
                "scenario": scenario,
                "evaluations": evaluations,
                "successes": successes,
                "pooled_success_rate": 100.0 * successes / evaluations,
                "mean_seed_success_rate": statistics.fmean(
                    float(value["success_rate"]) for value in records
                ),
                "seed_success_rates": {
                    str(seed): float(value["success_rate"])
                    for seed, value in zip(group["eval_seeds"], records)
                },
            }
        )

    output = {
        "schema_version": 1,
        "benchmark": "contextworld_tworoom_history3_eval_v1",
        "experiment_id": group["experiment_id"],
        "family": group["family"],
        "profile": group["profile"],
        "evidence_level": {
            "smoke": "plumbing_only",
            "quick": "qualitative_only",
            "full": "formal_confirmation",
        }[group["profile"]],
        "status": "passed",
        "policy": payloads[0]["policy"],
        "original_tworoom_reference": group.get(
            "original_tworoom_reference"
        ),
        "protocol": {
            "eval_seeds": group["eval_seeds"],
            "num_eval_per_seed": group["num_eval_per_seed"],
            "total_evaluations": total_evaluations,
            "aggregation": "per_scenario_then_equal_scenario_mean_then_seed_mean",
            "raw_results": [str(run["output"]) for run in group["runs"]],
        },
        "aggregate": {
            "evaluations": total_evaluations,
            "successes": total_successes,
            "success_rate": statistics.fmean(seed_rates),
            "mean_seed_success_rate": statistics.fmean(seed_rates),
            "std_seed_success_rate": std,
            "sem_seed_success_rate": std / math.sqrt(len(seed_rates)),
            "pooled_success_rate": 100.0 * total_successes / total_evaluations,
            "seed_success_rates": {
                str(seed): rate
                for seed, rate in zip(group["eval_seeds"], seed_rates)
            },
            "factor_readback_passed": all(
                payload["aggregate"]["factor_readback_passed"]
                for payload in payloads
            ),
        },
        "scenarios": scenario_rows,
    }
    from contextworld.synthesis.manifest import write_json

    write_json(group["summary"], output)
    print(
        json.dumps(
            {
                "planning_summary": str(group["summary"]),
                "aggregate": output["aggregate"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return output


def _e4_runs(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    checkpoint: Path,
    legacy_code_root: Path,
    original_h5: Path,
    stablewm_repo: Path,
    output_dir: Path,
) -> dict[str, Any]:
    planning = config["evaluation_groups"]["e4_paired_context_planning"]
    profile = planning["formal_confirmation"]
    eval_seeds = (
        args.eval_seeds
        if args.eval_seeds is not None
        else [int(value) for value in profile["eval_seeds"]]
    )
    num_eval = (
        args.num_eval
        if args.num_eval is not None
        else int(profile["num_eval_per_condition_per_seed"])
    )
    stem = f"e4_speed_ctx_n{num_eval}x{len(eval_seeds)}"
    runs = []
    for eval_seed in eval_seeds:
        output = output_dir / f"e4_speed_ctx_n{num_eval}_s{eval_seed}.json"
        command = [
            args.python,
            str(_repo_path(planning["entrypoint"])),
            "--catalog",
            str(_repo_path(planning["catalog"])),
            "--checkpoint",
            str(checkpoint),
            "--legacy-code-root",
            str(legacy_code_root),
            "--original-h5",
            str(original_h5),
            "--stablewm-repo",
            str(stablewm_repo),
            "--stablewm-ref",
            config["runtime"]["stable_worldmodel_commit"],
            "--device",
            args.device,
            "--seed",
            str(eval_seed),
            "--num-eval",
            str(num_eval),
            "--run-kind",
            "confirmation",
            "--speeds",
            *[str(value) for value in profile["speeds"]],
            "--templates",
            *[str(value) for value in profile["templates"]],
            "--eval-budget",
            str(profile["eval_budget"]),
            "--img-size",
            str(config["model"]["image_size"]),
            "--horizon",
            str(profile["horizon"]),
            "--receding-horizon",
            str(profile["receding_horizon"]),
            "--cem-batch-size",
            str(profile["cem_batch_size"]),
            "--cem-num-samples",
            str(profile["cem_num_samples"]),
            "--cem-var-scale",
            str(profile["cem_var_scale"]),
            "--cem-steps",
            str(profile["cem_steps"]),
            "--cem-topk",
            str(profile["cem_topk"]),
            "--output",
            str(output),
        ]
        if args.skip_catalog_replay:
            command.append("--skip-catalog-replay")
        runs.append({"seed": eval_seed, "output": output, "command": command})
    return {
        "experiment_id": planning["experiment_id"],
        "family": planning["family"],
        "num_eval_per_condition_per_seed": num_eval,
        "unique_base_queries": int(profile["unique_base_queries"]),
        "eval_seeds": eval_seeds,
        "runs": runs,
        "summary": output_dir / f"{stem}.json",
    }


def _exact_paired_sign_test(correct_only: int, wrong_only: int) -> dict[str, Any]:
    discordant = int(correct_only + wrong_only)
    if discordant == 0:
        return {"discordant_pairs": 0, "two_sided_p_value": 1.0}
    smaller = min(correct_only, wrong_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (
        2**discordant
    )
    return {
        "discordant_pairs": discordant,
        "two_sided_p_value": min(1.0, 2.0 * tail),
    }


def _aggregate_e4_runs(group: dict[str, Any]) -> dict[str, Any]:
    payloads = []
    all_pairs: list[dict[str, Any]] = []
    for run in group["runs"]:
        with run["output"].open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("status") != "passed":
            raise RuntimeError(f"E4 run did not pass: {run['output']}")
        if int(payload["protocol"]["eval_seed"]) != int(run["seed"]):
            raise RuntimeError(f"E4 eval seed mismatch in {run['output']}")
        if (
            int(payload["aggregate"]["queries"])
            != group["num_eval_per_condition_per_seed"]
        ):
            raise RuntimeError(f"E4 evaluation count mismatch in {run['output']}")
        if (
            int(payload["selection"]["unique_base_queries"])
            != group["unique_base_queries"]
        ):
            raise RuntimeError(f"E4 base query count mismatch in {run['output']}")
        if not (
            payload["pairing_audit"]["passed"]
            and payload["frozen_weight_audit"]["passed"]
        ):
            raise RuntimeError(f"E4 audit failed in {run['output']}")
        for pair in payload["aggregate"]["pairs"]:
            all_pairs.append({**pair, "eval_seed": int(run["seed"])})
        payloads.append(payload)

    correct_seed_rates = [
        float(payload["aggregate"]["correct"]["success_rate"])
        for payload in payloads
    ]
    wrong_seed_rates = [
        float(payload["aggregate"]["wrong"]["success_rate"])
        for payload in payloads
    ]
    delta_seed_rates = [
        correct - wrong
        for correct, wrong in zip(correct_seed_rates, wrong_seed_rates)
    ]
    total = len(all_pairs)
    correct_successes = sum(bool(pair["correct_success"]) for pair in all_pairs)
    wrong_successes = sum(bool(pair["wrong_success"]) for pair in all_pairs)
    correct_only = sum(
        bool(pair["correct_only_success"]) for pair in all_pairs
    )
    wrong_only = sum(bool(pair["wrong_only_success"]) for pair in all_pairs)
    distance_deltas = [
        float(pair["wrong_minus_correct_final_distance"]) for pair in all_pairs
    ]
    by_speed: dict[str, Any] = {}
    for speed in sorted({float(pair["speed"]) for pair in all_pairs}):
        entries = [pair for pair in all_pairs if float(pair["speed"]) == speed]
        c_success = sum(bool(pair["correct_success"]) for pair in entries)
        w_success = sum(bool(pair["wrong_success"]) for pair in entries)
        by_speed[f"{speed:g}"] = {
            "evaluations_per_condition": len(entries),
            "correct_successes": c_success,
            "wrong_successes": w_success,
            "correct_success_rate": 100.0 * c_success / len(entries),
            "wrong_success_rate": 100.0 * w_success / len(entries),
            "correct_minus_wrong_success_rate_points": (
                100.0 * (c_success - w_success) / len(entries)
            ),
            "wrong_minus_correct_mean_final_distance": statistics.fmean(
                float(pair["wrong_minus_correct_final_distance"])
                for pair in entries
            ),
        }

    output = {
        "schema_version": 1,
        "benchmark": "contextworld_tworoom_history3_e4_multiseed_v1",
        "experiment_id": group["experiment_id"],
        "family": group["family"],
        "status": "passed",
        "evidence_level": (
            f"confirmation_{group['num_eval_per_condition_per_seed']}x"
            f"{len(group['eval_seeds'])}_with_query_reuse"
        ),
        "protocol": {
            "eval_seeds": group["eval_seeds"],
            "num_eval_per_condition_per_seed": group[
                "num_eval_per_condition_per_seed"
            ],
            "unique_base_queries_per_seed": group["unique_base_queries"],
            "reused_evaluations_per_condition_per_seed": (
                group["num_eval_per_condition_per_seed"]
                - group["unique_base_queries"]
            ),
            "total_evaluations_per_condition": total,
            "raw_results": [str(run["output"]) for run in group["runs"]],
        },
        "audits": {
            "pairing_passed_all_seeds": all(
                payload["pairing_audit"]["passed"] for payload in payloads
            ),
            "frozen_weights_passed_all_seeds": all(
                payload["frozen_weight_audit"]["passed"] for payload in payloads
            ),
            "catalog_validation_passed_all_seeds": all(
                payload["catalog_validation"]["passed"] for payload in payloads
            ),
        },
        "aggregate": {
            "evaluations_per_condition": total,
            "correct": {
                "successes": correct_successes,
                "pooled_success_rate": 100.0 * correct_successes / total,
                "mean_seed_success_rate": statistics.fmean(correct_seed_rates),
                "seed_success_rates": dict(
                    zip(map(str, group["eval_seeds"]), correct_seed_rates)
                ),
            },
            "wrong": {
                "successes": wrong_successes,
                "pooled_success_rate": 100.0 * wrong_successes / total,
                "mean_seed_success_rate": statistics.fmean(wrong_seed_rates),
                "seed_success_rates": dict(
                    zip(map(str, group["eval_seeds"]), wrong_seed_rates)
                ),
            },
            "correct_minus_wrong_success_rate_points": (
                100.0 * (correct_successes - wrong_successes) / total
            ),
            "seed_success_rate_differences": dict(
                zip(map(str, group["eval_seeds"]), delta_seed_rates)
            ),
            "mean_seed_success_rate_difference": statistics.fmean(
                delta_seed_rates
            ),
            "std_seed_success_rate_difference": (
                statistics.pstdev(delta_seed_rates)
                if len(delta_seed_rates) > 1
                else 0.0
            ),
            "correct_only_successes": correct_only,
            "wrong_only_successes": wrong_only,
            "both_successes": sum(
                bool(pair["correct_success"] and pair["wrong_success"])
                for pair in all_pairs
            ),
            "neither_successes": sum(
                bool(not pair["correct_success"] and not pair["wrong_success"])
                for pair in all_pairs
            ),
            "paired_sign_test": _exact_paired_sign_test(
                correct_only, wrong_only
            ),
            "wrong_minus_correct_mean_final_distance": statistics.fmean(
                distance_deltas
            ),
            "mean_absolute_paired_final_distance_difference": statistics.fmean(
                abs(value) for value in distance_deltas
            ),
            "correct_lower_final_distance_pairs": sum(
                value > 0.0 for value in distance_deltas
            ),
            "wrong_lower_final_distance_pairs": sum(
                value < 0.0 for value in distance_deltas
            ),
            "by_speed": by_speed,
        },
    }
    from contextworld.synthesis.manifest import write_json

    write_json(group["summary"], output)
    print(
        json.dumps(
            {
                "e4_summary": str(group["summary"]),
                "aggregate": output["aggregate"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run speed-only E1-E4 for the frozen TwoRoom history=3 M_orig checkpoint."
        )
    )
    parser.add_argument(
        "--suite",
        choices=(
            "check",
            "e1-paired-pred",
            "e2-natural-pred",
            "e1-e2-pred",
            "e3-no-context-plan",
            "e4-context-plan",
            "key-planning",
            "all",
        ),
        default="check",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--legacy-code-root", type=Path)
    parser.add_argument("--original-h5", type=Path)
    parser.add_argument("--stablewm-repo", type=Path)
    parser.add_argument("--device")
    parser.add_argument(
        "--prediction-seed",
        "--seed",
        dest="prediction_seed",
        type=int,
        help="Seed for deterministic prediction-catalog replay and bootstrap only.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--planning-profile",
        choices=("smoke", "quick", "full"),
        default="smoke",
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        help="Planning eval seeds; formal default is 42 43 44 45 46 47.",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        help="Total planning evaluations per eval seed across selected scenarios.",
    )
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--scenario-indices", type=int, nargs="+")
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Override the smoke profile's one-scenario cap.",
    )
    parser.add_argument("--skip-catalog-replay", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse passing raw seed outputs instead of rerunning them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print commands without loading the model or using CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    model = config["model"]
    runtime = config["runtime"]
    args.device = args.device or runtime["device"]
    args.prediction_seed = (
        args.prediction_seed
        if args.prediction_seed is not None
        else int(model["prediction_seed"])
    )
    if args.num_eval is not None and args.num_eval <= 0:
        raise ValueError("--num-eval must be positive")
    if args.eval_seeds is not None and len(set(args.eval_seeds)) != len(
        args.eval_seeds
    ):
        raise ValueError("--eval-seeds must not contain duplicates")
    if args.max_scenarios is not None and args.max_scenarios <= 0:
        raise ValueError("--max-scenarios must be positive")
    if args.scenario_indices is not None:
        if any(index < 0 for index in args.scenario_indices):
            raise ValueError("--scenario-indices must be non-negative")
        if len(set(args.scenario_indices)) != len(args.scenario_indices):
            raise ValueError("--scenario-indices must not contain duplicates")
    if args.all_scenarios and args.max_scenarios is not None:
        raise ValueError("Use either --all-scenarios or --max-scenarios, not both")
    if args.scenario_indices is not None and (
        args.all_scenarios or args.max_scenarios is not None
    ):
        raise ValueError(
            "Use only one of --scenario-indices, --max-scenarios, or "
            "--all-scenarios"
        )

    checkpoint = _repo_path(args.checkpoint or model["checkpoint"])
    legacy_code_root = _repo_path(args.legacy_code_root or model["legacy_code_root"])
    original_h5 = _repo_path(args.original_h5 or model["original_h5"])
    stablewm_repo = _repo_path(
        args.stablewm_repo or runtime["stable_worldmodel_repo"]
    )
    output_dir = _repo_path(args.output_dir or runtime["output_dir"])

    _check(
        config=config,
        checkpoint=checkpoint,
        legacy_code_root=legacy_code_root,
        original_h5=original_h5,
        stablewm_repo=stablewm_repo,
        device=args.device,
        load_model=not args.dry_run,
    )
    if args.suite == "check":
        return
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    prediction = _prediction_commands(
        args,
        config,
        checkpoint=checkpoint,
        legacy_code_root=legacy_code_root,
        original_h5=original_h5,
        stablewm_repo=stablewm_repo,
        output_dir=output_dir,
    )
    if args.suite in {"e1-paired-pred", "e1-e2-pred", "all"}:
        _run_command(prediction["e1-paired-pred"], dry_run=args.dry_run)
    if args.suite in {"e2-natural-pred", "e1-e2-pred", "all"}:
        _run_command(prediction["e2-natural-pred"], dry_run=args.dry_run)
    if args.suite in {"e3-no-context-plan", "key-planning", "all"}:
        for group in _planning_runs(
            args,
            config,
            checkpoint=checkpoint,
            legacy_code_root=legacy_code_root,
            original_h5=original_h5,
            stablewm_repo=stablewm_repo,
            output_dir=output_dir,
        ):
            for run in group["runs"]:
                if args.reuse_existing and run["output"].is_file():
                    print(f"# reuse -> {run['output']}", flush=True)
                else:
                    _run_command(run["command"], dry_run=args.dry_run)
            if args.dry_run:
                print(f"# aggregate -> {group['summary']}", flush=True)
            else:
                _aggregate_planning_runs(group)
    if args.suite in {"e4-context-plan", "key-planning"}:
        group = _e4_runs(
            args,
            config,
            checkpoint=checkpoint,
            legacy_code_root=legacy_code_root,
            original_h5=original_h5,
            stablewm_repo=stablewm_repo,
            output_dir=output_dir,
        )
        for run in group["runs"]:
            if args.reuse_existing and run["output"].is_file():
                print(f"# reuse -> {run['output']}", flush=True)
            else:
                _run_command(run["command"], dry_run=args.dry_run)
        if args.dry_run:
            print(f"# aggregate -> {group['summary']}", flush=True)
        else:
            _aggregate_e4_runs(group)


if __name__ == "__main__":
    main()
