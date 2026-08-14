from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMCubeGraspRuleAdapter,
    StableWorldModelPLDMCubeGraspRuleAdapter,
)
from contextworld.benchmarks.cube_grasp_rule_reference_score import (
    evaluate_cube_reference_development_checkpoint,
    score_cube_reference_development_results,
)
from contextworld.benchmarks.cube_grasp_rule_reference_training import (
    DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
    expected_cube_reference_training_cell,
    load_cube_reference_training_prereg,
)
from contextworld.paths import repository_root, resolve_contextworld_path


ROOT = repository_root()


def _emit(payload: dict, output: Path) -> None:
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": payload.get("status"), "output": str(target)}))


def _adapter(args: argparse.Namespace, prereg: dict):
    cell = expected_cube_reference_training_cell(
        prereg,
        model_family=args.model_family,
        training_seed=args.training_seed,
        repo_root=ROOT,
    )
    if args.checkpoint.expanduser().resolve() != Path(cell["checkpoint"]):
        raise RuntimeError("Cube Development CLI checkpoint is not its frozen cell")
    runtime = prereg["runtime"]["stable_worldmodel"]
    normalization = prereg["evaluation"]["action_normalization"]
    cls = (
        StableWorldModelLeWMCubeGraspRuleAdapter
        if args.model_family == "lewm"
        else StableWorldModelPLDMCubeGraspRuleAdapter
    )
    return cls.from_checkpoint(
        args.checkpoint,
        action_mean=normalization["mean"],
        action_std=normalization["std_population"],
        repo_root=ROOT,
        stablewm_repo=runtime["repo"],
        stablewm_ref=runtime["expected_ref"],
        device=args.device,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score frozen Cube v4r1 checkpoints on Development only"
    )
    parser.add_argument(
        "--prereg", type=Path, default=DEFAULT_CUBE_REFERENCE_TRAINING_PREREG
    )
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--model-family", choices=("lewm", "pldm"), required=True)
    evaluate.add_argument("--model-name", required=True)
    evaluate.add_argument("--training-recipe", required=True)
    evaluate.add_argument("--training-seed", type=int, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--without-records", action="store_true")
    evaluate.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--input", type=Path, action="append", required=True)
    score.add_argument("--model-family", choices=("lewm", "pldm"), required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prereg = load_cube_reference_training_prereg(args.prereg, require_freeze=True)
    score_root = resolve_contextworld_path(
        prereg["planned_artifacts"]["development_score_root"], repo_root=ROOT
    )
    if args.command == "eval":
        expected_name = f"cube_gripper_carry_{args.model_family}_seed{args.training_seed}"
        expected_output = (
            score_root / "checkpoints" / f"{args.model_family}_seed{args.training_seed}.json"
        )
        expected_recipe = prereg["training"]["reference_matrix"]["models"][
            args.model_family
        ]["variant"]
        if (
            args.output.expanduser().resolve() != expected_output
            or args.model_name != expected_name
            or args.training_recipe != expected_recipe
            or int(args.batch_size) != int(prereg["evaluation"]["inference_batch_size"])
        ):
            raise RuntimeError("Cube Development CLI invocation drifted from matrix")
        payload = evaluate_cube_reference_development_checkpoint(
            adapter=_adapter(args, prereg),
            model_family=args.model_family,
            model_name=args.model_name,
            training_recipe=args.training_recipe,
            training_seed=args.training_seed,
            prereg_config=args.prereg,
            repo_root=ROOT,
            batch_size=args.batch_size,
            include_records=not args.without_records,
            loaded_prereg=prereg,
        )
    else:
        expected_output = score_root / f"{args.model_family}_three_seed_score.json"
        seeds = prereg["training"]["reference_matrix"]["training_seeds"]
        expected_inputs = [
            score_root / "checkpoints" / f"{args.model_family}_seed{seed}.json"
            for seed in seeds
        ]
        if (
            args.output.expanduser().resolve() != expected_output
            or [path.expanduser().resolve() for path in args.input] != expected_inputs
        ):
            raise RuntimeError("Cube Development method-score invocation drifted")
        payload = score_cube_reference_development_results(
            result_paths=args.input,
            model_family=args.model_family,
            prereg_config=args.prereg,
            loaded_prereg=prereg,
        )
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
