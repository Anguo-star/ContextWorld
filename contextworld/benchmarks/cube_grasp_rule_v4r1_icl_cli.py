from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMCubeGraspRuleAdapter,
    StableWorldModelPLDMCubeGraspRuleAdapter,
)
from contextworld.benchmarks.cube_grasp_rule_v4r1_icl_data import (
    DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
    audit_cube_grasp_rule_v4r1_icl_release,
    file_sha256,
    load_cube_grasp_rule_v4r1_icl_release,
)
from contextworld.benchmarks.cube_grasp_rule_v4r1_icl_score import (
    evaluate_cube_grasp_rule_v4r1_icl_model,
    score_cube_grasp_rule_v4r1_icl_results,
    validate_cube_grasp_rule_v4r1_external_checkpoint_identity,
    validate_cube_grasp_rule_v4r1_external_evaluation_policy,
)
from contextworld.paths import repository_root, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


ROOT = repository_root()


def _emit(payload: dict, output: Path | None) -> None:
    if output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    target = output.expanduser().resolve()
    write_json(target, payload)
    print(json.dumps({"status": payload.get("status"), "output": str(target)}))


def _assert_external_output(
    output: Path, release: dict, *, repo_root: Path = ROOT
) -> None:
    target = output.expanduser().resolve()
    for logical in release["claim_boundary"]["protected_paths"]:
        protected = resolve_contextworld_path(logical, repo_root=repo_root)
        if target == protected or protected in target.parents:
            raise RuntimeError(
                "External Cube results cannot be written into frozen release evidence"
            )
    protected_repository_paths = [
        repo_root / name
        for name in (
            "configs",
            "contextworld",
            "scripts",
            "tests",
            "docs",
            "pyproject.toml",
            "README.md",
        )
    ]
    if any(
        target == protected.resolve() or protected.resolve() in target.parents
        for protected in protected_repository_paths
    ):
        raise RuntimeError(
            "External Cube results cannot overwrite repository source or documentation"
        )


def _adapter(args: argparse.Namespace, *, repo_root: Path):
    release = load_cube_grasp_rule_v4r1_icl_release(args.release_config)
    runtime = release["runtime"]["stable_worldmodel"]
    normalization = release["evaluation"]["action_normalization"]
    cls = (
        StableWorldModelLeWMCubeGraspRuleAdapter
        if args.adapter == "lewm"
        else StableWorldModelPLDMCubeGraspRuleAdapter
    )
    return cls.from_checkpoint(
        args.checkpoint,
        action_mean=normalization["mean"],
        action_std=normalization["std_population"],
        repo_root=repo_root,
        stablewm_repo=args.stablewm_repo or runtime["repo"],
        stablewm_ref=args.stablewm_ref or runtime.get("expected_ref", ""),
        device=args.device,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-cube-gripper-carry",
        description=(
            "Audit and score the History=3 Cube hidden gripper-carry rule"
        ),
    )
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="ContextWorld source or exported benchmark root",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info")
    info.add_argument("--output", type=Path)
    audit = commands.add_parser("audit")
    audit.add_argument("--full", action="store_true")
    audit.add_argument("--layout", choices=("auto", "source", "bundle"), default="auto")
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
    evaluate.add_argument("--layout", choices=("auto", "source", "bundle"), default="auto")
    evaluate.add_argument("--without-records", action="store_true")
    evaluate.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--input", type=Path, action="append", required=True)
    score.add_argument("--method-name", required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    release = load_cube_grasp_rule_v4r1_icl_release(args.release_config)
    if args.output is not None:
        _assert_external_output(args.output, release, repo_root=repo_root)
    if args.command == "eval":
        validate_cube_grasp_rule_v4r1_external_evaluation_policy(release)
        validate_cube_grasp_rule_v4r1_external_checkpoint_identity(
            release,
            checkpoint_sha256=file_sha256(args.checkpoint),
            checkpoint_path=str(args.checkpoint),
        )
    if args.command == "info":
        payload = {
            key: value for key, value in release.items() if not key.startswith("_")
        }
    elif args.command == "audit":
        payload = audit_cube_grasp_rule_v4r1_icl_release(
            release_config=args.release_config,
            repo_root=repo_root,
            full=args.full,
            layout=args.layout,
        )
    elif args.command == "eval":
        payload = evaluate_cube_grasp_rule_v4r1_icl_model(
            adapter=_adapter(args, repo_root=repo_root),
            model_name=args.model_name,
            training_recipe=args.training_recipe,
            training_seed=args.training_seed,
            release_config=args.release_config,
            repo_root=repo_root,
            batch_size=args.batch_size,
            include_records=not args.without_records,
            layout=args.layout,
        )
    else:
        payload = score_cube_grasp_rule_v4r1_icl_results(
            result_paths=args.input,
            method_name=args.method_name,
            release_config=args.release_config,
        )
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
