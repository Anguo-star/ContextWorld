from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contextworld.benchmarks.action_delay_icl_data import (
    DEFAULT_ACTION_DELAY_RELEASE_CONFIG,
    action_delay_icl_training_plan,
    audit_action_delay_icl_release,
    load_action_delay_icl_release,
)
from contextworld.benchmarks.action_delay_icl_score import (
    evaluate_action_delay_icl_model,
    score_action_delay_icl_results,
)
from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMHistory7Adapter,
    StableWorldModelPLDMHistory7Adapter,
)
from contextworld.paths import repository_root, resolve_contextworld_path
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
    release = load_action_delay_icl_release(args.release_config)
    normalizer = resolve_contextworld_path(
        release["evaluation"]["normalizer"],
        repo_root=ROOT,
    )
    runtime = release["runtime"]["stable_worldmodel"]
    adapter_class = (
        StableWorldModelLeWMHistory7Adapter
        if args.adapter == "lewm"
        else StableWorldModelPLDMHistory7Adapter
    )
    return adapter_class.from_checkpoint(
        args.checkpoint,
        normalizer=normalizer,
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stablewm-repo", default=None)
    parser.add_argument("--stablewm-ref", default=None)
    parser.add_argument(
        "--without-records",
        action="store_true",
        help=(
            "Omit per-query loss records. Such a result cannot be "
            "independently rescored by the score command."
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-action-delay",
        description=(
            "Training-data audit and frozen offline scoring for TwoRoom "
            "History=7 Action Delay ICL"
        ),
    )
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_ACTION_DELAY_RELEASE_CONFIG,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser(
        "info",
        help="Print the frozen release contract",
    )
    info.add_argument("--output", type=Path, default=None)

    audit = subparsers.add_parser(
        "audit",
        help="Verify training and offline Validation artifacts",
    )
    audit.add_argument(
        "--full",
        action="store_true",
        help="Also hash and decode all 300 frozen Eval payloads",
    )
    audit.add_argument("--output", type=Path, default=None)

    train_plan = subparsers.add_parser(
        "train-plan",
        help="Print one frozen Stable-WorldModel reference training command",
    )
    train_plan.add_argument(
        "--recipe",
        required=True,
        help="Recipe name listed by the release config",
    )
    train_plan.add_argument("--training-seed", type=int, required=True)
    train_plan.add_argument("--output", type=Path, default=None)

    evaluate = subparsers.add_parser(
        "eval",
        help="Score one checkpoint on all 300 frozen queries",
    )
    _add_model_args(evaluate)
    evaluate.add_argument("--output", type=Path, required=True)

    score = subparsers.add_parser(
        "score",
        help="Rescore one result or aggregate a three-seed method",
    )
    score.add_argument("--input", type=Path, action="append", required=True)
    score.add_argument("--method-name", required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "info":
        release = load_action_delay_icl_release(args.release_config)
        payload = {
            key: value
            for key, value in release.items()
            if not key.startswith("_")
        }
    elif args.command == "audit":
        payload = audit_action_delay_icl_release(
            release_config=args.release_config,
            repo_root=ROOT,
            full=args.full,
        )
    elif args.command == "train-plan":
        payload = action_delay_icl_training_plan(
            args.recipe,
            training_seed=args.training_seed,
            release_config=args.release_config,
            repo_root=ROOT,
        )
    elif args.command == "eval":
        payload = evaluate_action_delay_icl_model(
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
        payload = score_action_delay_icl_results(
            result_paths=args.input,
            method_name=args.method_name,
            release_config=args.release_config,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
