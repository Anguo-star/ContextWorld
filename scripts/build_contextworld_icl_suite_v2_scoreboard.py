from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextworld.benchmarks.cube_grasp_rule_v4r1_icl_data import (
    DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
    load_cube_grasp_rule_v4r1_icl_release,
    recompute_cube_grasp_rule_v4r1_public_reference,
)
from contextworld.benchmarks.public_score import make_public_scoreboard_from_spec
from contextworld.paths import repository_root, resolve_contextworld_path


ROOT = repository_root()
DEFAULT_V1_SPEC = ROOT / (
    "artifacts/evaluation/contextworld_icl_suite_v1_release/"
    "public_scoreboard_spec.json"
)
DEFAULT_OUTPUT = resolve_contextworld_path(
    "artifacts/evaluation/contextworld_icl_suite_v2_release", repo_root=ROOT
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def canonical_cube_scoreboard_spec_row(reference: dict) -> dict:
    if (
        reference.get("passed") is not True
        or reference.get("external_result") is not False
        or reference.get("model_family") != "lewm"
        or reference.get("formal_reference_source")
        != "canonical_frozen_cube_public_recovery_v1"
        or reference.get("training_seeds") != [17321, 17322, 17323]
    ):
        raise RuntimeError("Cube canonical frozen reference is not eligible")
    seeds = reference["training_seeds"]
    comparisons = reference["original_task_retention"]["comparisons"]
    return {
        "component_id": "cube_gripper_carry",
        "component_name": "Cube 夹爪携带规则",
        "method_name": "LeWM（固定图像编码器，拟合配对真实未来）",
        "primary_metric": {
            "id": "correct_future_rate",
            "label": "真实携带规则下一状态选择正确率",
            "per_seed_values": [
                reference["per_seed"][str(seed)]["metrics"][
                    "correct_future_rate"
                ]
                for seed in seeds
            ],
        },
        "per_seed_gate_passes": [
            reference["per_seed"][str(seed)]["gate"]["passed"]
            for seed in seeds
        ],
        "ability_passed": True,
        "required_training_seeds": 3,
        "evidence_scope": "behavioral",
        "original_task_retention": {
            "result": "PASS",
            "metric_id": "standard_cube_cem_success_rate",
            "metric_label": "标准 Cube CEM 成功率",
            "per_seed_values": [
                row["candidate_successes"] / row["evaluation_count"]
                for row in comparisons
            ],
            "baseline_value": 198 / 300,
        },
    }


def build_suite_v2_scoreboard(
    *,
    v1_spec: Path = DEFAULT_V1_SPEC,
    cube_release_config: Path = DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
    output: Path = DEFAULT_OUTPUT,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Suite v2 scoreboard namespace is consumed: {output}")
    source = _read_json(v1_spec)
    if (
        set(source) != {"schema_version", "result_kind", "components"}
        or source.get("schema_version") != 1
        or source.get("result_kind") != "contextworld_public_scoreboard_spec"
        or not isinstance(source.get("components"), list)
        or len(source["components"]) != 10
        or any(row.get("component_id") == "cube_gripper_carry" for row in source["components"])
    ):
        raise RuntimeError("Suite v1 scoreboard specification drifted")

    release = load_cube_grasp_rule_v4r1_icl_release(cube_release_config)
    reference = recompute_cube_grasp_rule_v4r1_public_reference(
        release, layout="source"
    )
    cube = canonical_cube_scoreboard_spec_row(reference)
    spec = {**source, "components": [*source["components"], cube]}
    scoreboard = make_public_scoreboard_from_spec(spec)
    results = scoreboard.get("component_results", [])
    cube_rows = [
        row for row in results if row.get("component_id") == "cube_gripper_carry"
    ]
    if (
        len(results) != 11
        or len(cube_rows) != 1
        or cube_rows[0].get("icl_ability", {}).get("result") != "PASS"
        or "pldm" in cube_rows[0].get("method_name", "").lower()
    ):
        raise RuntimeError("Suite v2 scoreboard did not produce one LeWM-only Cube row")
    output.mkdir(parents=True)
    spec_path = output / "public_scoreboard_spec.json"
    scoreboard_path = output / "public_scoreboard.json"
    _write_json(spec_path, spec)
    _write_json(scoreboard_path, scoreboard)
    return {
        "schema_version": 1,
        "status": "passed",
        "formal_reference_rows": 11,
        "formal_reference_components": 7,
        "cube_formal_rows": 1,
        "cube_reference_family": "lewm",
        "external_results_included": False,
        "pldm_cube_result_included": False,
        "specification": str(spec_path),
        "scoreboard": str(scoreboard_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-spec", type=Path, default=DEFAULT_V1_SPEC)
    parser.add_argument(
        "--cube-release-config",
        type=Path,
        default=DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_suite_v2_scoreboard(
        v1_spec=args.v1_spec,
        cube_release_config=args.cube_release_config,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
