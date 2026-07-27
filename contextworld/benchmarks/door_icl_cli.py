from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMAdapter,
    StableWorldModelPLDMAdapter,
)
from contextworld.benchmarks.door_icl_data import (
    DEFAULT_DOOR_RELEASE_CONFIG,
    audit_door_icl_release,
    door_icl_training_plan,
    export_door_icl_artifacts,
    load_door_icl_release,
)
from contextworld.benchmarks.door_icl_score import (
    evaluate_door_icl_model,
    score_door_icl_results,
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
    release = load_door_icl_release(args.release_config)
    normalizer = resolve_contextworld_path(
        release["evaluation"]["normalizer"],
        repo_root=ROOT,
    )
    runtime = release["runtime"]["stable_worldmodel"]
    adapter_class = (
        StableWorldModelLeWMAdapter
        if args.adapter == "lewm"
        else StableWorldModelPLDMAdapter
    )
    return adapter_class.from_checkpoint(
        args.checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=args.stablewm_repo or runtime["repo"],
        stablewm_ref=args.stablewm_ref or runtime["expected_ref"],
        device=args.device,
    )


def _run_eval(args: argparse.Namespace, *, smoke: bool) -> dict[str, Any]:
    release = load_door_icl_release(args.release_config)
    eval_seeds = (
        [int(release["evaluation"]["eval_seeds"][0])]
        if smoke
        else args.eval_seeds
    )
    return evaluate_door_icl_model(
        adapter=_adapter(args),
        model_name=args.model_name,
        training_recipe=args.training_recipe,
        training_seed=args.training_seed,
        release_config=args.release_config,
        repo_root=ROOT,
        eval_seeds=eval_seeds,
        limit_per_seed=(1 if smoke else args.limit_per_seed),
        batch_size=args.batch_size,
        include_records=not args.without_records,
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
            "Omit per-query records. Such output cannot be independently "
            "rescored by the public score command."
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-door",
        description=(
            "Frozen train-data audit and offline Validation scoring for "
            "TwoRoom History=3 door-rule ICL v1"
        ),
    )
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_DOOR_RELEASE_CONFIG,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser(
        "info",
        help="Print the frozen public release contract",
    )
    info.add_argument("--output", type=Path, default=None)

    audit = subparsers.add_parser(
        "audit",
        help="Verify local train, Validation and reference-result artifacts",
    )
    audit.add_argument(
        "--full",
        action="store_true",
        help="Hash every file and validate all 300 offline payload arrays",
    )
    audit.add_argument("--output", type=Path, default=None)

    export = subparsers.add_parser(
        "export",
        help="Build a portable local release-candidate artifact root",
    )
    export.add_argument("--destination", type=Path, required=True)
    export.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
    )
    export.add_argument("--output", type=Path, default=None)

    train_plan = subparsers.add_parser(
        "train-plan",
        help="Print one frozen Stable-WorldModel reference training command",
    )
    train_plan.add_argument(
        "--recipe",
        choices=(
            "lewm_joint",
            "lewm_fixed_representation",
            "pldm_joint",
        ),
        required=True,
    )
    train_plan.add_argument("--training-seed", type=int, required=True)
    train_plan.add_argument("--output", type=Path, default=None)

    smoke = subparsers.add_parser(
        "smoke",
        help="Score one query as a non-formal runtime check",
    )
    _add_model_args(smoke)
    smoke.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "eval",
        help="Score one frozen checkpoint on Validation",
    )
    _add_model_args(evaluate)
    evaluate.add_argument("--eval-seeds", nargs="+", type=int, default=None)
    evaluate.add_argument("--limit-per-seed", type=int, default=None)
    evaluate.add_argument("--output", type=Path, required=True)

    score = subparsers.add_parser(
        "score",
        help=(
            "Independently rescore one checkpoint result or aggregate three "
            "training seeds"
        ),
    )
    score.add_argument("--input", type=Path, action="append", required=True)
    score.add_argument("--method-name", required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "info":
        release = load_door_icl_release(args.release_config)
        payload = {
            key: value
            for key, value in release.items()
            if not key.startswith("_")
        }
    elif args.command == "audit":
        payload = audit_door_icl_release(
            release_config=args.release_config,
            repo_root=ROOT,
            full=args.full,
        )
    elif args.command == "export":
        payload = export_door_icl_artifacts(
            args.destination,
            release_config=args.release_config,
            repo_root=ROOT,
            mode=args.mode,
        )
    elif args.command == "train-plan":
        payload = door_icl_training_plan(
            args.recipe,
            training_seed=args.training_seed,
            release_config=args.release_config,
            repo_root=ROOT,
        )
    elif args.command == "smoke":
        payload = _run_eval(args, smoke=True)
    elif args.command == "eval":
        payload = _run_eval(args, smoke=False)
    elif args.command == "score":
        payload = score_door_icl_results(
            result_paths=args.input,
            method_name=args.method_name,
            release_config=args.release_config,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
