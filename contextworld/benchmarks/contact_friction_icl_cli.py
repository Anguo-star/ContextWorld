from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMContactFrictionAdapter,
    StableWorldModelPLDMContactFrictionAdapter,
)
from contextworld.benchmarks.contact_friction_icl_data import (
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    audit_contact_friction_icl_release,
    load_contact_friction_icl_release,
)
from contextworld.benchmarks.contact_friction_icl_score import (
    evaluate_contact_friction_icl_development_model,
    evaluate_contact_friction_icl_model,
    rescore_contact_friction_icl_development_result,
    score_contact_friction_icl_results,
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
    release = load_contact_friction_icl_release(args.release_config)
    runtime = release["runtime"]["stable_worldmodel"]
    normalization = release["evaluation"]["action_normalization"]
    adapter_class = (
        StableWorldModelLeWMContactFrictionAdapter
        if args.adapter == "lewm"
        else StableWorldModelPLDMContactFrictionAdapter
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


def _add_model_args(
    parser: argparse.ArgumentParser,
    *,
    allow_without_records: bool = True,
) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", choices=("lewm", "pldm"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--training-recipe", default="external_method")
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stablewm-repo", default=None)
    parser.add_argument("--stablewm-ref", default=None)
    if allow_without_records:
        parser.add_argument("--without-records", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-contact-friction",
        description=(
            "Data audit and frozen evaluation for PushT History=3 "
            "Contact Friction ICL"
        ),
    )
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info")
    info.add_argument("--output", type=Path, default=None)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--full", action="store_true")
    audit.add_argument("--output", type=Path, default=None)

    evaluate = subparsers.add_parser("eval")
    _add_model_args(evaluate)
    evaluate.add_argument("--output", type=Path, required=True)

    evaluate_development = subparsers.add_parser(
        "eval-development",
        help=(
            "Score the pinned Loader Validation split only; Public Test is "
            "not read or scored."
        ),
    )
    _add_model_args(evaluate_development, allow_without_records=False)
    evaluate_development.add_argument("--output", type=Path, required=True)

    rescore_development = subparsers.add_parser(
        "score-development",
        aliases=("rescore-development",),
        help="Independently recompute a Development-only result from records.",
    )
    rescore_development.add_argument("--input", type=Path, required=True)
    rescore_development.add_argument("--output", type=Path, required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--input", type=Path, action="append", required=True)
    score.add_argument("--method-name", required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "info":
        release = load_contact_friction_icl_release(args.release_config)
        payload = {
            key: value
            for key, value in release.items()
            if not key.startswith("_")
        }
    elif args.command == "audit":
        payload = audit_contact_friction_icl_release(
            release_config=args.release_config,
            repo_root=ROOT,
            full=args.full,
        )
    elif args.command == "eval":
        payload = evaluate_contact_friction_icl_model(
            adapter=_adapter(args),
            model_name=args.model_name,
            training_recipe=args.training_recipe,
            training_seed=args.training_seed,
            release_config=args.release_config,
            repo_root=ROOT,
            batch_size=args.batch_size,
            include_records=not args.without_records,
        )
    elif args.command == "eval-development":
        payload = evaluate_contact_friction_icl_development_model(
            adapter=_adapter(args),
            model_name=args.model_name,
            training_recipe=args.training_recipe,
            training_seed=args.training_seed,
            release_config=args.release_config,
            repo_root=ROOT,
            batch_size=args.batch_size,
        )
    elif args.command in {"score-development", "rescore-development"}:
        payload = rescore_contact_friction_icl_development_result(
            args.input,
            release_config=args.release_config,
        )
    elif args.command == "score":
        payload = score_contact_friction_icl_results(
            result_paths=args.input,
            method_name=args.method_name,
            release_config=args.release_config,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
