#!/usr/bin/env python3
"""Aggregate the frozen one-step Action Delay ICL core evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from contextworld.evaluation.action_delay_h7_core import (
    summarize_action_delay_h1_physical as summarize_h1_physical,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3072, 4096, 5120)
FAMILIES = ("pldm", "lewm")
PHYSICAL_GROUPS = tuple(range(6))
DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_core_icl_v2.yaml"
)


def _artifact_root() -> Path:
    configured = os.environ.get("CONTEXTWORLD_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (ROOT.parents[1] / "data/world_model/context_world").resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def _stats(values: Iterable[float]) -> dict[str, float]:
    rows = [float(value) for value in values]
    _require(bool(rows), "Cannot summarize an empty metric")
    return {
        "mean": float(statistics.fmean(rows)),
        "sample_std": (
            float(statistics.stdev(rows)) if len(rows) > 1 else 0.0
        ),
        "minimum": float(min(rows)),
        "maximum": float(max(rows)),
    }


def _slug(family: str, seed: int, *, run_stem: str) -> str:
    return f"{run_stem}_{family}_formal_s{seed}"


def _model_result(
    payload: dict[str, Any],
    *,
    config: dict[str, Any],
    family: str,
    seed: int,
    slug: str,
) -> dict[str, Any]:
    _require(payload.get("label") == slug, f"Wrong label for {slug}")
    _require(
        payload.get("model_family") == family,
        f"Wrong family for {slug}",
    )
    _require(
        payload.get("status") == "completed_post_hoc_diagnostic",
        f"Validation did not complete for {slug}",
    )
    _require(
        payload.get("training_receipt", {}).get("passed") is True,
        f"Training receipt failed for {slug}",
    )
    _require(
        payload.get("model_state_sha256_before")
        == payload.get("model_state_sha256_after"),
        f"Model state changed during Eval for {slug}",
    )
    audit = payload["score_audit"]
    _require(
        audit["queries"] == 300
        and audit["model_predictions"] == 3300
        and audit["target_encodings"] == 9900
        and audit["horizon_loss_records"] == 108900
        and audit["online_environment_calls"] == 0,
        f"Frozen Eval count audit failed for {slug}: {audit}",
    )
    query_metrics = payload["summary"]["by_horizon"]["1"][
        "query_metrics"
    ]
    uncertainty = config["core_metric"]["uncertainty"]
    core = summarize_h1_physical(
        query_metrics,
        bootstrap_resamples=int(uncertainty["resamples"]),
        bootstrap_seed=int(uncertainty["random_seed"]),
    )
    _require(
        core["eval_seed_query_counts"]
        == {str(value): 50 for value in (42, 43, 44, 45, 46, 47)},
        f"Eval seed/query counts changed for {slug}",
    )
    gate = config["primary_gate"]
    checks = {
        "physical_group_macro_accuracy": (
            core["physical_group_macro_accuracy"]
            >= float(gate["physical_group_macro_accuracy_minimum"])
        ),
        "minimum_physical_group_accuracy": (
            core["minimum_physical_group_accuracy"]
            >= float(gate["minimum_physical_group_accuracy"])
        ),
        "bootstrap_lower_bound": (
            core["paired_query_bootstrap_95_percent_interval"]["lower"]
            >= float(
                gate[
                    "paired_query_bootstrap_95_percent_lower_bound_minimum"
                ]
            )
        ),
    }
    summary = payload["summary"]
    diagnostics = {
        f"h{horizon}": {
            "exact_target_selection_rate": float(
                summary["by_horizon"][str(horizon)]["overall"][
                    "exact_target_selection_rate"
                ]
            ),
            "physical_target_group_selection_rate": float(
                summary["by_horizon"][str(horizon)]["overall"][
                    "physical_target_group_selection_rate"
                ]
            ),
        }
        for horizon in (2, 3)
    }
    return {
        "label": slug,
        "model_family": family,
        "training_seed": int(seed),
        "core_h1": core,
        "diagnostic_only": diagnostics,
        "gate": {
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def _family_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "models": len(rows),
        "training_seeds": [row["training_seed"] for row in rows],
        "core_h1": {
            metric: _stats(row["core_h1"][metric] for row in rows)
            for metric in (
                "physical_group_macro_accuracy",
                "minimum_physical_group_accuracy",
            )
        },
        "bootstrap_lower_bound": _stats(
            row["core_h1"][
                "paired_query_bootstrap_95_percent_interval"
            ]["lower"]
            for row in rows
        ),
        "by_physical_group": {
            str(group): _stats(
                row["core_h1"]["by_physical_group"][str(group)][
                    "accuracy"
                ]
                for row in rows
            )
            for group in PHYSICAL_GROUPS
        },
        "diagnostic_only": {
            f"h{horizon}": {
                metric: _stats(
                    row["diagnostic_only"][f"h{horizon}"][metric]
                    for row in rows
                )
                for metric in (
                    "exact_target_selection_rate",
                    "physical_target_group_selection_rate",
                )
            }
            for horizon in (2, 3)
        },
        "passed_seeds": sum(row["gate"]["passed"] for row in rows),
    }


def _historical_reference(
    *,
    artifact_root: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = (
        artifact_root
        / "evaluation/history7/action_delay_paired_repair_v1/model_results"
    )
    rows = []
    artifacts = {}
    uncertainty = config["core_metric"]["uncertainty"]
    for seed in SEEDS:
        for family in FAMILIES:
            slug = f"h7_action_delay_paired_{family}_formal_s{seed}"
            path = root / f"{slug}_validation.json"
            _require(path.is_file(), f"Missing historical result: {path}")
            payload = _load_json(path)
            _require(
                payload.get("label") == slug
                and payload.get("model_family") == family
                and payload.get("score_audit", {}).get("queries") == 300,
                f"Historical result identity failed: {path}",
            )
            core = summarize_h1_physical(
                payload["summary"]["by_horizon"]["1"][
                    "query_metrics"
                ],
                bootstrap_resamples=int(uncertainty["resamples"]),
                bootstrap_seed=int(uncertainty["random_seed"]),
            )
            rows.append(
                {
                    "label": slug,
                    "model_family": family,
                    "training_seed": seed,
                    "core_h1": core,
                }
            )
            artifacts[slug] = {
                "path": str(path),
                "sha256": _sha256(path),
            }
    summary = {
        family: {
            "models": 3,
            "physical_group_macro_accuracy": _stats(
                row["core_h1"]["physical_group_macro_accuracy"]
                for row in rows
                if row["model_family"] == family
            ),
            "minimum_physical_group_accuracy": _stats(
                row["core_h1"]["minimum_physical_group_accuracy"]
                for row in rows
                if row["model_family"] == family
            ),
            "by_physical_group": {
                str(group): _stats(
                    row["core_h1"]["by_physical_group"][str(group)][
                        "accuracy"
                    ]
                    for row in rows
                    if row["model_family"] == family
                )
                for group in PHYSICAL_GROUPS
            },
        }
        for family in FAMILIES
    }
    return {
        "recipe": "same_query_delays_0_4_8_v1",
        "models": rows,
        "by_family": summary,
    }, artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-stem",
        default="h7_action_delay_full_range",
    )
    parser.add_argument(
        "--recipe-name",
        default="full_range_delay_balanced_v2",
    )
    parser.add_argument("--predecessor-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = _artifact_root()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(
        config["benchmark"]
        == "tworoom_action_delay_history7_core_icl_v2",
        "Wrong core evaluation configuration",
    )
    catalog_spec = config["frozen_validation"]["catalog"]
    catalog = (
        artifact_root
        / Path(catalog_spec["path"]).relative_to("artifacts")
    )
    _require(catalog.is_file(), f"Missing frozen catalog: {catalog}")
    _require(
        _sha256(catalog) == catalog_spec["sha256"],
        "Frozen Validation catalog hash changed",
    )
    results_root = (
        args.results_root.expanduser().resolve()
        if args.results_root
        else artifact_root
        / "evaluation/history7/action_delay_full_range_v2/model_results"
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else artifact_root
        / "evaluation/history7/action_delay_full_range_v2/"
        "core_summary.json"
    )

    model_rows = []
    artifacts = {}
    for seed in SEEDS:
        for family in FAMILIES:
            slug = _slug(
                family,
                seed,
                run_stem=str(args.run_stem),
            )
            path = results_root / f"{slug}_validation.json"
            _require(path.is_file(), f"Missing Eval result: {path}")
            payload = _load_json(path)
            model_rows.append(
                _model_result(
                    payload,
                    config=config,
                    family=family,
                    seed=seed,
                    slug=slug,
                )
            )
            artifacts[slug] = {
                "validation_result": str(path),
                "validation_result_sha256": _sha256(path),
                "checkpoint": payload["identity"]["checkpoint"],
                "checkpoint_sha256": payload["identity"][
                    "checkpoint_sha256"
                ],
                "training_report": payload["training_receipt"]["path"],
                "training_report_sha256": payload["training_receipt"][
                    "sha256"
                ],
            }

    by_family = {
        family: _family_summary(
            [row for row in model_rows if row["model_family"] == family]
        )
        for family in FAMILIES
    }
    historical, historical_artifacts = _historical_reference(
        artifact_root=artifact_root,
        config=config,
    )
    recipe_delta = {
        family: {
            "physical_group_macro_accuracy": (
                by_family[family]["core_h1"][
                    "physical_group_macro_accuracy"
                ]["mean"]
                - historical["by_family"][family][
                    "physical_group_macro_accuracy"
                ]["mean"]
            ),
            "minimum_physical_group_accuracy": (
                by_family[family]["core_h1"][
                    "minimum_physical_group_accuracy"
                ]["mean"]
                - historical["by_family"][family][
                    "minimum_physical_group_accuracy"
                ]["mean"]
            ),
        }
        for family in FAMILIES
    }
    predecessor = None
    if args.predecessor_summary is not None:
        predecessor_path = args.predecessor_summary.expanduser().resolve()
        predecessor_payload = _load_json(predecessor_path)
        _require(
            predecessor_payload.get("benchmark") == config["benchmark"]
            and predecessor_payload.get("status") == "completed",
            f"Invalid predecessor core summary: {predecessor_path}",
        )
        predecessor = {
            "path": str(predecessor_path),
            "sha256": _sha256(predecessor_path),
            "recipe": predecessor_payload.get("recipe"),
            "passed": predecessor_payload.get("passed"),
            "by_family": predecessor_payload["by_family"],
            "current_minus_predecessor": {
                family: {
                    metric: (
                        by_family[family]["core_h1"][metric]["mean"]
                        - predecessor_payload["by_family"][family][
                            "core_h1"
                        ][metric]["mean"]
                    )
                    for metric in (
                        "physical_group_macro_accuracy",
                        "minimum_physical_group_accuracy",
                    )
                }
                for family in FAMILIES
            },
        }
    required = len(config["primary_gate"]["required_training_seeds"])
    pldm_passed = by_family["pldm"]["passed_seeds"] == required
    conclusion = (
        "PLDM 的三个训练种子都通过了冻结的单步真实下一帧门槛；"
        "本训练集与 Eval 已证明可测量 History=7 Action Delay ICL。"
        if pldm_passed
        else "PLDM 未在三个训练种子上全部通过，当前训练配方尚未达到发布门槛。"
    )
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "recipe": str(args.recipe_name),
        "status": "completed",
        "passed": pldm_passed,
        "core_claim": conclusion,
        "identity": {
            "core_config": str(config_path),
            "core_config_sha256": _sha256(config_path),
            "frozen_validation_catalog": str(catalog),
            "frozen_validation_catalog_sha256": _sha256(catalog),
            "analyzer": str(Path(__file__).resolve()),
            "analyzer_sha256": _sha256(Path(__file__).resolve()),
        },
        "gate": config["primary_gate"],
        "models": model_rows,
        "by_family": by_family,
        "reference_comparison": {
            "historical_three_delay_recipe": historical,
            "current_minus_three_delay": recipe_delta,
            "predecessor": predecessor,
            "comparison_scope": (
                "same frozen 300-query Validation and same core metric"
            ),
        },
        "artifacts": artifacts,
        "historical_artifacts": historical_artifacts,
        "claim_boundary": config["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "passed": payload["passed"],
                "core_claim": payload["core_claim"],
                "by_family": by_family,
                "output": str(output),
                "output_sha256": _sha256(output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
