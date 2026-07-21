from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contextworld.benchmarks.adapters import StableWorldModelLeWMAdapter
from contextworld.benchmarks.speed_icl_data import (
    DEFAULT_RELEASE_CONFIG,
    audit_speed_icl_release,
    build_speed_icl_training_data,
    export_speed_icl_artifacts,
    load_speed_icl_release,
    resolve_original_h5,
)
from contextworld.benchmarks.speed_icl_score import (
    aggregate_speed_icl_method,
    aggregate_speed_icl_planning,
    evaluate_speed_icl_model,
)
from contextworld.evaluation.icl_model import file_sha256
from contextworld.paths import repository_root, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


ROOT = repository_root()


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        write_json(output.expanduser().resolve(), payload)
        print(
            json.dumps(
                {
                    "status": payload.get("status"),
                    "submission_kind": payload.get("submission_kind"),
                    "output": str(output.expanduser().resolve()),
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def _adapter(args: argparse.Namespace) -> StableWorldModelLeWMAdapter:
    release = load_speed_icl_release(args.release_config)
    runtime = release["runtime"]["stable_worldmodel"]
    normalizer = resolve_contextworld_path(
        release["evaluation"]["normalizer"], repo_root=ROOT
    )
    return StableWorldModelLeWMAdapter.from_checkpoint(
        args.checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=args.stablewm_repo or runtime["repo"],
        stablewm_ref=args.stablewm_ref or runtime["expected_ref"],
        device=args.device,
    )


def _run_eval(args: argparse.Namespace, *, smoke: bool) -> dict[str, Any]:
    adapter = _adapter(args)
    return evaluate_speed_icl_model(
        adapter=adapter,
        model_name=args.model_name,
        training_role=args.training_role,
        training_seed=args.training_seed,
        release_config=args.release_config,
        repo_root=ROOT,
        tracks=(None if smoke else args.tracks),
        eval_seeds=([42] if smoke else args.eval_seeds),
        limit_per_reference_speed_per_seed=(
            1 if smoke else args.limit_per_reference_speed_per_seed
        ),
        encode_batch_size=args.encode_batch_size,
        rollout_batch_size=args.rollout_batch_size,
        bundle_batch_size=args.bundle_batch_size,
        include_records=args.include_records,
    )


def _cmd_info(args: argparse.Namespace) -> dict[str, Any]:
    release = load_speed_icl_release(args.release_config)
    return {
        key: value
        for key, value in release.items()
        if not key.startswith("_")
    }


def _cmd_audit(args: argparse.Namespace) -> dict[str, Any]:
    return audit_speed_icl_release(
        release_config=args.release_config,
        repo_root=ROOT,
        original_h5=args.original_h5,
        verify_all_eval_payloads=args.full,
    )


def _cmd_train_plan(args: argparse.Namespace) -> dict[str, Any]:
    release = load_speed_icl_release(args.release_config)
    recipe = release["training"]["recipes"][args.recipe]
    expected_sizes = {
        "epoch_size": int(recipe["epoch_size_global"]),
        "validation_epoch_size": int(recipe["validation_epoch_size"]),
    }
    for name, expected in expected_sizes.items():
        observed = getattr(args, name)
        if observed is not None and int(observed) != expected:
            raise ValueError(
                f"{name} is frozen at {expected} for {args.recipe}"
            )
    allowed_seeds = {int(value) for value in recipe["training_seeds"]}
    if int(args.seed) not in allowed_seeds:
        raise ValueError(
            f"Training seed for {args.recipe} must be one of "
            f"{sorted(allowed_seeds)}"
        )
    runtime = release["runtime"]["stable_worldmodel"]
    stable_repo_arg = args.stablewm_repo or runtime["repo"]
    stable_ref = args.stablewm_ref or runtime["expected_ref"]
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, stable_repo_arg, stable_ref
    )
    grouped = build_speed_icl_training_data(
        swm,
        recipe=args.recipe,
        original_h5=args.original_h5,
        release_config=args.release_config,
        repo_root=ROOT,
        epoch_size=args.epoch_size,
        validation_epoch_size=args.validation_epoch_size,
        seed=args.seed,
        expected_stablewm_commit=stable_commit,
    )
    original = resolve_original_h5(
        release, repo_root=ROOT, explicit=args.original_h5
    )
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "status": "passed",
        "recipe": args.recipe,
        "profile": recipe["profile"],
        "expected_optimizer_steps": int(recipe["optimizer_steps"]),
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "original_h5": str(original),
        "train_length": len(grouped.train),
        "validation_length": len(grouped.val),
        "metadata": grouped.metadata,
    }


def _cmd_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    return aggregate_speed_icl_method(
        target_results=args.target,
        control_results=args.control,
        method_name=args.method_name,
        release_config=args.release_config,
    )


def _cmd_export(args: argparse.Namespace) -> dict[str, Any]:
    return export_speed_icl_artifacts(
        args.destination,
        release_config=args.release_config,
        repo_root=ROOT,
        mode=args.mode,
        include_single_speed_control=not args.without_control,
    )


def _cmd_planning_cell(args: argparse.Namespace) -> dict[str, Any]:
    release = load_speed_icl_release(args.release_config)
    planning = release["planning"]
    if args.track not in planning["tracks"]:
        raise KeyError(f"Unknown planning track: {args.track}")
    speeds = [float(value) for value in planning["tracks"][args.track]["speeds"]]
    if not any(abs(float(args.query_speed) - value) <= 1e-6 for value in speeds):
        raise ValueError(
            f"query-speed must be one of {speeds} for {args.track}"
        )
    if int(args.seed) not in {
        int(value) for value in planning["eval_seeds"]
    }:
        raise ValueError(f"Unsupported eval seed: {args.seed}")
    maximum = int(planning["evaluations_per_speed_condition_per_seed"])
    if not 1 <= int(args.num_eval) <= maximum:
        raise ValueError(f"num-eval must be in [1,{maximum}]")
    runtime = release["runtime"]["stable_worldmodel"]
    catalog = resolve_contextworld_path(
        planning["tracks"][args.track]["catalog"], repo_root=ROOT
    )
    normalizer = resolve_contextworld_path(
        release["evaluation"]["normalizer"], repo_root=ROOT
    )
    stablewm_repo = args.stablewm_repo or runtime["repo"]
    stablewm_ref = args.stablewm_ref or runtime["expected_ref"]
    common = {
        "catalog": catalog,
        "checkpoint": args.checkpoint.expanduser().resolve(),
        "normalizer": normalizer,
        "output": args.output.expanduser().resolve(),
        "query_speed": float(args.query_speed),
        "seed": int(args.seed),
        "num_eval": int(args.num_eval),
        "device": args.device,
        "stablewm_repo": stablewm_repo,
        "stablewm_ref": stablewm_ref,
        "skip_catalog_replay": True,
    }
    if args.mode == "fixed":
        from scripts.eval_tworoom_speed_cube_fixed_candidate import run

        fixed = planning["fixed_candidate"]
        namespace = argparse.Namespace(
            **common,
            candidates=int(fixed["candidates"]),
            horizon=int(fixed["horizon_action_blocks"]),
        )
    else:
        from scripts.eval_tworoom_speed_cube_planning import run

        cem = planning["cem"]
        namespace = argparse.Namespace(
            **common,
            run_kind=("confirmation" if args.num_eval == 50 else "qualitative_probe"),
            eval_budget=int(cem["execution_budget_raw_steps"]),
            deadline_budgets=[
                int(value) for value in cem["deadline_budgets_raw_steps"]
            ],
            img_size=224,
            horizon=int(cem["horizon_action_blocks"]),
            receding_horizon=int(cem["receding_horizon_action_blocks"]),
            cem_batch_size=1,
            cem_num_samples=int(cem["candidates"]),
            cem_var_scale=1.0,
            cem_steps=int(cem["iterations"]),
            cem_topk=int(cem["topk"]),
        )
    payload = run(namespace)
    release_path = Path(release["_config_path"])
    payload["contextworld_release"] = {
        "release_id": release["release_id"],
        "release_config_sha256": file_sha256(release_path),
        "planning_mode": (
            "fixed_candidate" if args.mode == "fixed" else "cem"
        ),
        "catalog_sha256": planning["tracks"][args.track][
            "catalog_sha256"
        ],
    }
    return payload


def _cmd_aggregate_planning(args: argparse.Namespace) -> dict[str, Any]:
    return aggregate_speed_icl_planning(
        result_paths=args.input,
        release_config=args.release_config,
    )


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stablewm-repo", default=None)
    parser.add_argument("--stablewm-ref", default=None)


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--training-role",
        choices=(
            "original_reference",
            "single_speed_control",
            "multi_speed_target",
            "external_model",
        ),
        default="external_model",
    )
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--rollout-batch-size", type=int, default=128)
    parser.add_argument("--bundle-batch-size", type=int, default=16)
    parser.add_argument("--include-records", action="store_true")
    _add_runtime_args(parser)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-speed",
        description="Train-data audit and frozen evaluation for Speed ICL v1",
    )
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_RELEASE_CONFIG,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Print the frozen release contract")
    info.add_argument("--output", type=Path, default=None)

    audit = subparsers.add_parser("audit", help="Verify local train/eval artifacts")
    audit.add_argument("--original-h5", type=Path, default=None)
    audit.add_argument(
        "--full",
        action="store_true",
        help=(
            "Hash original H5, both training trees, 4,200 offline Eval "
            "payloads, and all planning payloads"
        ),
    )
    audit.add_argument("--output", type=Path, default=None)

    export = subparsers.add_parser(
        "export", help="Create a portable artifact-root tree for distribution"
    )
    export.add_argument("--destination", type=Path, required=True)
    export.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    export.add_argument("--without-control", action="store_true")
    export.add_argument("--output", type=Path, default=None)

    train_plan = subparsers.add_parser(
        "train-plan", help="Build and audit a Stable-WorldModel training mixture"
    )
    train_plan.add_argument(
        "--recipe",
        choices=(
            "original_reference",
            "single_speed_control",
            "multi_speed_target",
        ),
        required=True,
    )
    train_plan.add_argument("--original-h5", type=Path, default=None)
    train_plan.add_argument("--epoch-size", type=int, default=None)
    train_plan.add_argument("--validation-epoch-size", type=int, default=None)
    train_plan.add_argument("--seed", type=int, default=3072)
    train_plan.add_argument("--output", type=Path, default=None)
    _add_runtime_args(train_plan)

    smoke = subparsers.add_parser(
        "smoke", help="Run one query per speed on every track"
    )
    _add_eval_args(smoke)
    smoke.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "eval", help="Run a single frozen Stable-WorldModel checkpoint"
    )
    _add_eval_args(evaluate)
    evaluate.add_argument(
        "--tracks",
        nargs="+",
        default=None,
        choices=(
            "seen_for_multi",
            "unseen_interpolation",
            "extrapolation_low",
            "extrapolation_high",
        ),
    )
    evaluate.add_argument("--eval-seeds", nargs="+", type=int, default=None)
    evaluate.add_argument(
        "--limit-per-reference-speed-per-seed", type=int, default=None
    )
    evaluate.add_argument("--output", type=Path, required=True)

    aggregate = subparsers.add_parser(
        "aggregate", help="Aggregate three paired target/control model seeds"
    )
    aggregate.add_argument("--method-name", required=True)
    aggregate.add_argument("--target", type=Path, action="append", required=True)
    aggregate.add_argument("--control", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)

    planning = subparsers.add_parser(
        "planning-cell",
        help="Run one fixed-candidate or CEM speed/query/seed cell",
    )
    planning.add_argument("--mode", choices=("fixed", "cem"), required=True)
    planning.add_argument(
        "--track",
        choices=("seen_for_multi", "unseen_interpolation"),
        required=True,
    )
    planning.add_argument("--query-speed", type=float, required=True)
    planning.add_argument("--seed", type=int, required=True)
    planning.add_argument("--num-eval", type=int, default=50)
    planning.add_argument("--checkpoint", type=Path, required=True)
    planning.add_argument("--device", default="cuda:0")
    planning.add_argument("--output", type=Path, required=True)
    _add_runtime_args(planning)

    aggregate_planning = subparsers.add_parser(
        "aggregate-planning",
        help="Aggregate fixed-candidate or CEM cell result files",
    )
    aggregate_planning.add_argument(
        "--input", type=Path, action="append", required=True
    )
    aggregate_planning.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "info":
        payload = _cmd_info(args)
    elif args.command == "audit":
        payload = _cmd_audit(args)
    elif args.command == "train-plan":
        payload = _cmd_train_plan(args)
    elif args.command == "export":
        payload = _cmd_export(args)
    elif args.command == "smoke":
        payload = _run_eval(args, smoke=True)
    elif args.command == "eval":
        payload = _run_eval(args, smoke=False)
    elif args.command == "aggregate":
        payload = _cmd_aggregate(args)
    elif args.command == "planning-cell":
        payload = _cmd_planning_cell(args)
    elif args.command == "aggregate-planning":
        payload = _cmd_aggregate_planning(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
