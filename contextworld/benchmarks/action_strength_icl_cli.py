from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contextworld.benchmarks.action_strength_icl_data import (
    DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
    action_strength_icl_evaluation_plans,
    action_strength_icl_training_plan,
    audit_action_strength_icl_release,
    load_action_strength_icl_release,
)
from contextworld.benchmarks.action_strength_icl_score import (
    evaluate_action_strength_icl_model,
    score_action_strength_icl_results,
    score_action_strength_planning_submission,
    score_action_strength_retention_report,
)
from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMActionStrengthAdapter,
    StableWorldModelPLDMActionStrengthAdapter,
)
from contextworld.paths import repository_root
from contextworld.synthesis.manifest import write_json


ROOT = repository_root()


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    if output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    target = output.expanduser().resolve()
    write_json(target, payload)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "output": str(target),
                "submission_kind": payload.get("submission_kind"),
            },
            sort_keys=True,
        )
    )


def _adapter(args: argparse.Namespace):
    release = load_action_strength_icl_release(args.release_config)
    runtime = release["runtime"]["stable_worldmodel"]
    normalization = release["evaluation"]["action_normalization"]
    adapter_class = (
        StableWorldModelLeWMActionStrengthAdapter
        if args.adapter == "lewm"
        else StableWorldModelPLDMActionStrengthAdapter
    )
    return adapter_class.from_checkpoint(
        args.checkpoint,
        action_mean=normalization["mean"],
        action_std=normalization["std_population"],
        repo_root=ROOT,
        stablewm_repo=args.stablewm_repo or runtime["repo"],
        stablewm_ref=args.stablewm_ref or runtime["expected_ref"],
        device=args.device,
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", choices=("lewm", "pldm"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--training-recipe", default="external_method")
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stablewm-repo", default=None)
    parser.add_argument("--stablewm-ref", default=None)
    parser.add_argument(
        "--without-records",
        action="store_true",
        help=(
            "Omit pair records. Such a result cannot be independently "
            "rescored by the score command."
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-action-strength",
        description=(
            "Training-data audit and frozen evaluation for PushT "
            "History=3 Pusher Movement Scale ICL"
        ),
    )
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser(
        "info",
        help="Print the frozen Pusher Movement Scale release contract",
    )
    info.add_argument("--output", type=Path, default=None)

    audit = subparsers.add_parser(
        "audit",
        help="Verify paired training data and independent confirmation",
    )
    audit.add_argument(
        "--full",
        action="store_true",
        help="Hash both artifact trees and decode all 256 query pairs",
    )
    audit.add_argument("--output", type=Path, default=None)

    train_plan = subparsers.add_parser(
        "train-plan",
        help="Print a Stable-WorldModel reference or fair-control command",
    )
    train_plan.add_argument("--recipe", required=True)
    train_plan.add_argument("--training-seed", type=int, required=True)
    train_plan.add_argument("--training-output", type=Path, required=True)
    train_plan.add_argument("--output", type=Path, default=None)

    evaluation_plans = subparsers.add_parser(
        "eval-plans",
        help="Print the frozen action-planning and standard-retention commands",
    )
    evaluation_plans.add_argument("--checkpoint", type=Path, required=True)
    evaluation_plans.add_argument("--model-name", required=True)
    evaluation_plans.add_argument("--result-root", type=Path, required=True)
    evaluation_plans.add_argument("--output", type=Path, default=None)

    evaluate = subparsers.add_parser(
        "eval",
        help="Score one checkpoint on all 256 independent query pairs",
    )
    _add_model_args(evaluate)
    evaluate.add_argument("--output", type=Path, required=True)

    score = subparsers.add_parser(
        "score",
        help="Rescore one checkpoint or aggregate three training seeds",
    )
    score.add_argument("--input", type=Path, action="append", required=True)
    score.add_argument("--method-name", required=True)
    score.add_argument("--output", type=Path, required=True)

    planning = subparsers.add_parser(
        "score-planning",
        help=(
            "Score 512 selected action amplitudes against the frozen "
            "physical oracle"
        ),
    )
    planning.add_argument("--submission", type=Path, required=True)
    planning.add_argument("--output", type=Path, required=True)

    retention = subparsers.add_parser(
        "score-retention",
        help="Validate one model from a frozen 300-episode standard PushT CEM run",
    )
    retention.add_argument("--report", type=Path, required=True)
    retention.add_argument("--model-name", required=True)
    retention.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "info":
        release = load_action_strength_icl_release(args.release_config)
        payload = {
            key: value
            for key, value in release.items()
            if not key.startswith("_")
        }
    elif args.command == "audit":
        payload = audit_action_strength_icl_release(
            release_config=args.release_config,
            repo_root=ROOT,
            full=args.full,
        )
    elif args.command == "train-plan":
        payload = action_strength_icl_training_plan(
            args.recipe,
            training_seed=args.training_seed,
            output=args.training_output,
            release_config=args.release_config,
            repo_root=ROOT,
        )
    elif args.command == "eval-plans":
        payload = action_strength_icl_evaluation_plans(
            checkpoint=args.checkpoint,
            model_name=args.model_name,
            output_root=args.result_root,
            release_config=args.release_config,
            repo_root=ROOT,
        )
    elif args.command == "eval":
        payload = evaluate_action_strength_icl_model(
            adapter=_adapter(args),
            model_name=args.model_name,
            training_recipe=args.training_recipe,
            training_seed=args.training_seed,
            release_config=args.release_config,
            repo_root=ROOT,
            batch_size=args.batch_size,
            include_records=not args.without_records,
        )
    elif args.command == "score":
        payload = score_action_strength_icl_results(
            result_paths=args.input,
            method_name=args.method_name,
            release_config=args.release_config,
        )
    elif args.command == "score-planning":
        payload = score_action_strength_planning_submission(
            submission_path=args.submission,
            release_config=args.release_config,
            repo_root=ROOT,
        )
    elif args.command == "score-retention":
        payload = score_action_strength_retention_report(
            report_path=args.report,
            model_name=args.model_name,
            release_config=args.release_config,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
