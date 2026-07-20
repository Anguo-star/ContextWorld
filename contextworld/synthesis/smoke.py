from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from contextworld.paths import resolve_contextworld_path

from .collector import collect_scenario
from .config import build_compiler, load_config, scenario_requests
from .manifest import write_catalog, write_json, write_manifest
from .stablewm import load_stable_worldmodel
from .validator import (
    run_required_atom_oracles,
    validate_loader_mix,
    validate_independent_seed_assignment,
    validate_minimum_episode_start_oracle,
    validate_numeric_atom_isolation,
    validate_paired_seed_crossing,
    validate_reset_coverage,
    validate_scenario,
    validate_split_isolation,
    validate_training_unseen_combinations,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_VALIDATION_SWM = None


def _initialize_validation_worker(
    repo_root: str, stablewm_repo: str, expected_ref: str | None
) -> None:
    """Load StableWM independently in a spawned validation worker."""

    global _VALIDATION_SWM
    _VALIDATION_SWM, _, _ = load_stable_worldmodel(
        Path(repo_root), stablewm_repo, expected_ref
    )


def _validate_scenario_worker(scenario):
    if _VALIDATION_SWM is None:
        raise RuntimeError("Scenario validation worker was not initialized")
    return validate_scenario(_VALIDATION_SWM, scenario)


def _projection(scenarios, config: dict) -> dict:
    split_counts = Counter(scenario.split for scenario in scenarios)
    regime_counts = Counter(
        scenario.regime or scenario.split for scenario in scenarios
    )
    episodes = sum(scenario.episodes for scenario in scenarios)
    maximum_rows = sum(
        scenario.episodes * scenario.max_episode_steps for scenario in scenarios
    )
    height, width = scenarios[0].image_shape
    raw_pixel_bytes = maximum_rows * height * width * 3
    estimated_bytes_per_row = config.get("planning", {}).get(
        "estimated_lance_bytes_per_row"
    )
    estimated_lance_bytes = (
        None
        if estimated_bytes_per_row is None
        else int(maximum_rows * estimated_bytes_per_row)
    )
    return {
        "scenarios": len(scenarios),
        "scenarios_by_split": dict(sorted(split_counts.items())),
        "scenarios_by_regime": dict(sorted(regime_counts.items())),
        "episodes": episodes,
        "maximum_rows": maximum_rows,
        "raw_pixel_bytes": raw_pixel_bytes,
        "estimated_lance_bytes": estimated_lance_bytes,
        "estimated_lance_bytes_per_row": estimated_bytes_per_row,
    }


def run(
    config_path: Path,
    *,
    compile_only: bool = False,
    resume: bool = False,
    skip_loader_check: bool = False,
) -> dict:
    repo_root = _repo_root()
    config = load_config(config_path)
    swm_config = config["stable_worldmodel"]
    swm, stable_repo, commit = load_stable_worldmodel(
        repo_root,
        swm_config["repo"],
        swm_config.get("expected_ref"),
    )

    compiler = build_compiler(config, repo_root)
    scenarios = compiler.compile_all(scenario_requests(config))
    validation_config = config.get("validation", {})
    scenario_validation_workers = int(
        validation_config.get("scenario_validation_workers", 1)
    )
    if scenario_validation_workers < 1:
        raise ValueError("scenario_validation_workers must be >= 1")
    atom_oracles = run_required_atom_oracles(
        scenarios, compiler.registry, validation_config
    )
    oracle_coverage = atom_oracles["coverage"]
    output = config["output"]
    manifest_path = resolve_contextworld_path(
        output["manifest"], repo_root=repo_root
    )
    report_path = resolve_contextworld_path(output["report"], repo_root=repo_root)
    catalog_path = resolve_contextworld_path(
        output["catalog"], repo_root=repo_root
    )
    original_dataset = resolve_contextworld_path(
        config["original_dataset"]["path"], repo_root=repo_root
    )

    # All scenario-level and atom-specific semantic checks are preflight
    # gates: no manifest/catalog or trajectory is published before they pass.
    split_check = validate_split_isolation(scenarios)
    numeric_check_config = validation_config.get("numeric_atom_isolation")
    numeric_check = {"passed": True, "skipped": True}
    if numeric_check_config is not None:
        numeric_check = validate_numeric_atom_isolation(
            scenarios,
            atom_kind=numeric_check_config["atom"],
            minimum_cross_split_gap=float(
                numeric_check_config["minimum_cross_split_gap"]
            ),
        )
    minimum_start_config = validation_config.get(
        "minimum_episode_start_oracle"
    )
    minimum_start_check = {"passed": True, "skipped": True}
    if minimum_start_config is not None:
        try:
            minimum_start_check = validate_minimum_episode_start_oracle(
                scenarios, minimum_start_config
            )
        except Exception as exc:
            minimum_start_check = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    combination_config = validation_config.get(
        "training_unseen_combinations"
    )
    combination_check = {"passed": True, "skipped": True}
    if combination_config is not None:
        try:
            combination_check = validate_training_unseen_combinations(
                scenarios, combination_config
            )
        except Exception as exc:
            combination_check = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    paired_seed_config = validation_config.get("paired_seed_crossing")
    paired_seed_check = {"passed": True, "skipped": True}
    if paired_seed_config is not None:
        try:
            paired_seed_check = validate_paired_seed_crossing(
                scenarios, paired_seed_config
            )
        except Exception as exc:
            paired_seed_check = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    independent_seed_config = validation_config.get(
        "independent_seed_assignment"
    )
    independent_seed_check = {"passed": True, "skipped": True}
    if independent_seed_config is not None:
        try:
            independent_seed_check = validate_independent_seed_assignment(
                scenarios, independent_seed_config
            )
        except Exception as exc:
            independent_seed_check = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    reset_coverage_config = validation_config.get("reset_coverage")
    reset_coverage_check = {"passed": True, "skipped": True}
    if reset_coverage_config is not None:
        try:
            reset_coverage_check = validate_reset_coverage(
                scenarios, reset_coverage_config
            )
        except Exception as exc:
            reset_coverage_check = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    frame_skip_check = atom_oracles["checks"].get(
        "speed_frame_skip_oracle", {"passed": True, "skipped": True}
    )
    door_oracle_check = atom_oracles["checks"].get(
        "door_position_pixel_oracle", {"passed": True, "skipped": True}
    )
    door_passage_check = atom_oracles["checks"].get(
        "door_position_passage_oracle",
        {"passed": True, "skipped": True},
    )
    preflight_passed = all(
        (
            atom_oracles["passed"],
            split_check["passed"],
            numeric_check["passed"],
            minimum_start_check["passed"],
            combination_check["passed"],
            paired_seed_check["passed"],
            independent_seed_check["passed"],
            reset_coverage_check["passed"],
            frame_skip_check["passed"],
            door_oracle_check["passed"],
            door_passage_check["passed"],
        )
    )
    if not preflight_passed:
        report = {
            "schema_version": 1,
            "experiment": config["experiment"],
            "passed": False,
            "compile_only": compile_only,
            "preflight_passed": False,
            "stable_worldmodel": {
                "repo": str(stable_repo),
                "commit": commit,
            },
            "manifest": None,
            "catalog": None,
            "collection_status": {},
            "projection": _projection(scenarios, config),
            "atom_oracles": atom_oracles,
            "atom_oracle_coverage": oracle_coverage,
            "split_isolation": split_check,
            "numeric_atom_isolation": numeric_check,
            "minimum_episode_start_oracle": minimum_start_check,
            "training_unseen_combinations": combination_check,
            "paired_seed_crossing": paired_seed_check,
            "independent_seed_assignment": independent_seed_check,
            "reset_coverage": reset_coverage_check,
            "speed_frame_skip_oracle": frame_skip_check,
            "door_position_pixel_oracle": door_oracle_check,
            "door_position_passage_oracle": door_passage_check,
            "scenarios": [],
            "loader_compatibility": {"passed": False, "skipped": True},
        }
        write_json(report_path, report)
        return report

    write_manifest(
        manifest_path,
        scenarios,
        repo_root=repo_root,
        stable_worldmodel_commit=commit,
    )
    write_catalog(
        catalog_path,
        scenarios,
        original_dataset=original_dataset,
        repo_root=repo_root,
    )

    collection_status: dict[str, str] = {}
    if not compile_only:
        for scenario in scenarios:
            collection_status[scenario.scenario_id] = collect_scenario(
                swm,
                scenario,
                config["collection"],
                resume=resume,
            )
        write_manifest(
            manifest_path,
            scenarios,
            repo_root=repo_root,
            stable_worldmodel_commit=commit,
        )

    if compile_only:
        scenario_reports = []
    elif scenario_validation_workers == 1:
        scenario_reports = [
            validate_scenario(swm, scenario) for scenario in scenarios
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=min(scenario_validation_workers, len(scenarios)),
            mp_context=get_context("spawn"),
            initializer=_initialize_validation_worker,
            initargs=(
                str(repo_root),
                str(swm_config["repo"]),
                swm_config.get("expected_ref"),
            ),
        ) as executor:
            scenario_reports = list(
                executor.map(_validate_scenario_worker, scenarios, chunksize=1)
            )
    loader_check: dict = {"passed": True, "skipped": True}
    if not compile_only and not skip_loader_check:
        train_scenario = next(s for s in scenarios if s.split == "train")
        loader_check = validate_loader_mix(
            swm,
            original_dataset=original_dataset,
            synthetic_dataset=train_scenario.output_path,
            cache_dir=Path("/tmp/contextworld-smoke-cache"),
        )

    passed = (
        atom_oracles["passed"]
        and split_check["passed"]
        and numeric_check["passed"]
        and minimum_start_check["passed"]
        and combination_check["passed"]
        and paired_seed_check["passed"]
        and independent_seed_check["passed"]
        and reset_coverage_check["passed"]
        and frame_skip_check["passed"]
        and door_oracle_check["passed"]
        and door_passage_check["passed"]
        and all(item["passed"] for item in scenario_reports)
        and loader_check["passed"]
    )
    report = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "passed": passed,
        "compile_only": compile_only,
        "preflight_passed": preflight_passed,
        "scenario_validation_workers": scenario_validation_workers,
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": commit,
        },
        "manifest": str(manifest_path),
        "catalog": str(catalog_path),
        "collection_status": collection_status,
        "projection": _projection(scenarios, config),
        "atom_oracles": atom_oracles,
        "atom_oracle_coverage": oracle_coverage,
        "split_isolation": split_check,
        "numeric_atom_isolation": numeric_check,
        "minimum_episode_start_oracle": minimum_start_check,
        "training_unseen_combinations": combination_check,
        "paired_seed_crossing": paired_seed_check,
        "independent_seed_assignment": independent_seed_check,
        "reset_coverage": reset_coverage_check,
        "speed_frame_skip_oracle": frame_skip_check,
        "door_position_pixel_oracle": door_oracle_check,
        "door_position_passage_oracle": door_passage_check,
        "scenarios": scenario_reports,
        "loader_compatibility": loader_check,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile, collect, and validate a ContextWorld synthesis config"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root()
        / "configs/synthesis/tworoom_composable_smoke.yaml",
    )
    parser.add_argument(
        "--compile-only", action="store_true", help="Write manifest/catalog only"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse immutable scenario tables that already exist",
    )
    parser.add_argument(
        "--skip-loader-check",
        action="store_true",
        help="Skip the original-H5 plus synthetic-Lance concat check",
    )
    args = parser.parse_args()
    report = run(
        args.config,
        compile_only=args.compile_only,
        resume=args.resume,
        skip_loader_check=args.skip_loader_check,
    )
    summary = {
        "experiment": report["experiment"],
        "passed": report["passed"],
        "scenarios": report["projection"]["scenarios"],
        "manifest": report["manifest"],
        "catalog": report["catalog"],
    }
    print(json.dumps(summary, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
