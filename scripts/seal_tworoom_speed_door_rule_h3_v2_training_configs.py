#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)


BASE = (
    ROOT
    / "configs/benchmark/tworoom_speed_door_rule_h3_training_v1.yaml"
)
V2_CONFIG = (
    ROOT / "configs/benchmark/tworoom_speed_door_rule_h3_v2.yaml"
)
MAIN_REPORT = (
    "artifacts/synthesis/speed_door_rule_h3_v2/formal/build_report.json"
)
ANCHOR_REPORT = (
    "artifacts/synthesis/speed_door_rule_h3_v2_door_anchor/formal/"
    "build_report.json"
)
OUTPUT_ROOT = "artifacts/protocols/speed_door_rule_h3_v2"
STABLE_COMMIT = "4b6f5d94693631ce64ed1f561dd0bc1d23ca38fa"
CATALOG_SHA256 = (
    "d5744dbb2c20b8161648625e0fc012d12d653c517bfbc7ac39380dce573f17e8"
)
EXCLUSION_SHA256 = (
    "cd2181dcc6a89fd059779869625bb585367b4c38da7cd9372911dce9a727ce50"
)
CONTENT_SHA256 = (
    "0ae2586f7c147117121cd8b97283018b6e931f476566c108e0ed69c6bc25420a"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _load_report(logical_path: str) -> tuple[Path, dict[str, Any]]:
    path = resolve_contextworld_path(logical_path, repo_root=ROOT)
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("passed") is not True or report.get("scale") != "formal":
        raise RuntimeError(f"Formal data report did not pass: {path}")
    return path, report


def _group_quality(
    report: dict[str, Any],
    *,
    group: str,
    speeds: list[float],
) -> dict[str, Any]:
    artifact = report["artifacts_by_group"][group]
    counts = artifact["counts"]
    train = counts["train"]
    val = counts["val"]
    test = counts["test"]
    raw_train = int(train["clips"])
    return {
        "require_complete_synthesis_report": True,
        "required_pixel_codec": {
            "format": "png",
            "compress_level": 1,
            "lossless": True,
        },
        "balance_by_factor": "passage.open",
        "exact_train_scenarios": int(train["shards"]),
        "exact_validation_scenarios": int(val["shards"]),
        "exact_test_scenarios": int(test["shards"]),
        "exact_train_clips": int(train["clips"]),
        "exact_validation_clips": int(val["clips"]),
        "exact_test_clips": int(test["clips"]),
        "factor_support_contract": {
            "factor": "agent.speed",
            "expected_by_split": {
                "train": [speeds],
                "validation": [speeds],
            },
        },
        "minimum_raw_train_clips": raw_train,
        "maximum_formal_mean_draws_per_raw_clip": float(
            1_048_576 / raw_train + 1.0
        ),
        "minimum_train_scenarios": int(train["shards"]),
        "minimum_validation_scenarios": int(val["shards"]),
        "required_catalog_sha256": str(artifact["catalog_sha256"]),
        "required_manifest_sha256": str(artifact["manifest_sha256"]),
        "required_synthesis_report_sha256": str(
            artifact["synthesis_report_sha256"]
        ),
    }


def _derive(
    *,
    base: dict[str, Any],
    report_path: Path,
    report: dict[str, Any],
    release_root: str,
    model_id: str,
    display_name: str,
    training_group: str,
    speeds: list[float],
    benchmark: str,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.update(
        {
            "schema_version": 2,
            "benchmark": benchmark,
            "status": "sealed_after_data_before_training_or_model_scoring",
            "preregistered_date": "2026-07-28",
            "question": (
                "在墙边相同 query 几何下，History=3 是否足以提供"
                "当前模型所需的隐藏速度或门规则信息？"
            ),
        }
    )
    config["stable_worldmodel"]["commit"] = STABLE_COMMIT
    config["data"]["training_exclusion_manifest"] = {
        "path": (
            "artifacts/evaluation/history3/"
            "speed_door_rule_composition_v2/"
            "training_exclusion_manifest.json"
        ),
        "sha256": EXCLUSION_SHA256,
        "content_sha256": CONTENT_SHA256,
        "query_count": 300,
    }
    config["data"]["formal_build_report"] = {
        "path": portable_contextworld_path(
            report_path, repo_root=ROOT
        ),
        "sha256": _sha256(report_path),
        "benchmark": str(report["benchmark"]),
        "scale": "formal",
    }
    prefix = str(report["artifacts_by_group"]["passage_passable"][
        "catalog"
    ])
    if not prefix.startswith("artifacts/"):
        raise RuntimeError("Formal catalog path is not portable")
    config["data"]["catalogs"] = {
        group: str(report["artifacts_by_group"][group]["catalog"])
        for group in (
            "passage_passable",
            "passage_blocked",
            "passage_mixed",
        )
    }
    config["passage_support"]["training_speeds"] = list(speeds)
    config["passage_support"]["eval_speeds"] = [3.1, 5.1, 7.0]
    config["passage_support"]["requirements"] = [
        "training_and_loader_validation_exclude_every_eval_only_door",
        "selected_validation_query_pixels_excluded_from_training",
        "same_query_geometry_as_speed_door_rule_v2_validation",
    ]
    config["data_quality"]["groups"] = {
        group: _group_quality(
            report, group=group, speeds=list(speeds)
        )
        for group in (
            "passage_passable",
            "passage_blocked",
            "passage_mixed",
        )
    }
    config["training_protocol"]["paired_training_seeds"] = [
        3072,
        4096,
        5120,
    ]
    config["training_protocol"]["group_sampling"] = {
        model_id: {training_group: 1.0}
    }
    config["models"] = [
        {
            "model_id": model_id,
            "display_name": display_name,
            "training_groups": [training_group],
        }
    ]
    config["comparison_controls"] = {}
    config["evaluation_gate"] = {
        "validation_catalog": (
            "artifacts/evaluation/history3/"
            "speed_door_rule_composition_v2/catalog.json"
        ),
        "validation_catalog_sha256": CATALOG_SHA256,
        "role": (
            "door_only"
            if training_group == "passage_mixed"
            and speeds == [5.1]
            else (
                "joint"
                if training_group == "passage_mixed"
                else "speed_only"
            )
        ),
    }
    config["release_identity"] = {
        "release_root": release_root,
        "data_build_report_sha256": _sha256(report_path),
        "validation_catalog_sha256": CATALOG_SHA256,
        "validation_exclusion_sha256": EXCLUSION_SHA256,
        "validation_content_sha256": CONTENT_SHA256,
        "v2_design_config_sha256": _sha256(V2_CONFIG),
    }
    return config


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(
        value,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    base = _load_yaml(BASE)
    main_path, main_report = _load_report(MAIN_REPORT)
    anchor_path, anchor_report = _load_report(ANCHOR_REPORT)
    output_root = resolve_contextworld_path(
        OUTPUT_ROOT, repo_root=ROOT
    )
    outputs = {
        "speed_only": output_root / "speed_only_training.yaml",
        "door_only": output_root / "door_only_training.yaml",
        "joint": output_root / "joint_training.yaml",
    }
    speed = _derive(
        base=base,
        report_path=main_path,
        report=main_report,
        release_root=(
            "artifacts/synthesis/speed_door_rule_h3_v2/formal"
        ),
        model_id="H3_SpeedDoorV2_SpeedOnly_PLDM",
        display_name="只见过多速度、门始终可通过的 PLDM",
        training_group="passage_passable",
        speeds=[2.7, 4.3, 6.1, 7.7],
        benchmark="tworoom_speed_door_rule_h3_v2_speed_training",
    )
    door = _derive(
        base=base,
        report_path=anchor_path,
        report=anchor_report,
        release_root=(
            "artifacts/synthesis/"
            "speed_door_rule_h3_v2_door_anchor/formal"
        ),
        model_id="H3_SpeedDoorV2_DoorOnly_PLDM",
        display_name="只见过速度 5.1、两种门规则的 PLDM",
        training_group="passage_mixed",
        speeds=[5.1],
        benchmark="tworoom_speed_door_rule_h3_v2_door_training",
    )
    joint = _derive(
        base=base,
        report_path=main_path,
        report=main_report,
        release_root=(
            "artifacts/synthesis/speed_door_rule_h3_v2/formal"
        ),
        model_id="H3_SpeedDoorV2_Joint_PLDM",
        display_name="同时见过多速度与两种门规则的 PLDM",
        training_group="passage_mixed",
        speeds=[2.7, 4.3, 6.1, 7.7],
        benchmark="tworoom_speed_door_rule_h3_v2_joint_training",
    )
    for role, value in (
        ("speed_only", speed),
        ("door_only", door),
        ("joint", joint),
    ):
        _write(outputs[role], value)
    manifest = {
        "schema_version": 2,
        "status": "sealed_before_training_or_model_scoring",
        "design_config": portable_contextworld_path(
            V2_CONFIG, repo_root=ROOT
        ),
        "design_config_sha256": _sha256(V2_CONFIG),
        "configs": {
            role: {
                "path": portable_contextworld_path(
                    path, repo_root=ROOT
                ),
                "sha256": _sha256(path),
            }
            for role, path in outputs.items()
        },
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
