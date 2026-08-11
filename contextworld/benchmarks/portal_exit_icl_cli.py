from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMPortalExitAdapter,
    StableWorldModelPLDMPortalExitAdapter,
)
from contextworld.benchmarks.portal_exit_icl_data import (
    DEFAULT_PORTAL_EXIT_RELEASE_CONFIG,
    audit_portal_exit_icl_release,
    load_portal_exit_icl_release,
)
from contextworld.benchmarks.portal_exit_icl_score import (
    evaluate_portal_exit_icl_model,
    score_portal_exit_icl_results,
)
from contextworld.paths import repository_root
from contextworld.synthesis.manifest import write_json


ROOT = repository_root()


def _emit(payload: dict, output: Path | None) -> None:
    if output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        target = output.expanduser().resolve()
        write_json(target, payload)
        print(json.dumps({"status": payload.get("status"), "output": str(target)}))


def _adapter(args: argparse.Namespace):
    release = load_portal_exit_icl_release(args.release_config)
    runtime = release["runtime"]["stable_worldmodel"]
    normalization = release["evaluation"]["action_normalization"]
    cls = (
        StableWorldModelLeWMPortalExitAdapter
        if args.adapter == "lewm"
        else StableWorldModelPLDMPortalExitAdapter
    )
    return cls.from_checkpoint(
        args.checkpoint,
        action_mean=normalization["mean"],
        action_std=normalization["std_unbiased"],
        repo_root=ROOT,
        stablewm_repo=args.stablewm_repo or runtime["repo"],
        stablewm_ref=args.stablewm_ref or runtime.get("expected_ref", ""),
        device=args.device,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-portal-exit",
        description="Audit and score TwoRoom History=3 Portal Exit ICL",
    )
    parser.add_argument(
        "--release-config", type=Path, default=DEFAULT_PORTAL_EXIT_RELEASE_CONFIG
    )
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info")
    info.add_argument("--output", type=Path)
    audit = commands.add_parser("audit")
    audit.add_argument("--full", action="store_true")
    audit.add_argument("--output", type=Path)
    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--adapter", choices=("lewm", "pldm"), required=True)
    evaluate.add_argument("--model-name", required=True)
    evaluate.add_argument("--training-recipe", default="external_method")
    evaluate.add_argument("--training-seed", type=int)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--stablewm-repo")
    evaluate.add_argument("--stablewm-ref")
    evaluate.add_argument("--without-records", action="store_true")
    evaluate.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--input", type=Path, action="append", required=True)
    score.add_argument("--method-name", required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "info":
        release = load_portal_exit_icl_release(args.release_config)
        payload = {key: value for key, value in release.items() if not key.startswith("_")}
    elif args.command == "audit":
        payload = audit_portal_exit_icl_release(
            release_config=args.release_config, repo_root=ROOT, full=args.full
        )
    elif args.command == "eval":
        payload = evaluate_portal_exit_icl_model(
            adapter=_adapter(args),
            model_name=args.model_name,
            training_recipe=args.training_recipe,
            training_seed=args.training_seed,
            release_config=args.release_config,
            repo_root=ROOT,
            batch_size=args.batch_size,
            include_records=not args.without_records,
        )
    else:
        payload = score_portal_exit_icl_results(
            result_paths=args.input,
            method_name=args.method_name,
            release_config=args.release_config,
        )
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
