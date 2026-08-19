#!/usr/bin/env python3
"""Build the frozen 18-row complete LeWM/PLDM comparison table.

Reads only already-frozen inputs:
  * the immutable 13-row public scoreboard + its per-seed spec;
  * the base complete-comparison freeze (prereg v1 + receipt) and its newly
    executed ICL receipts, method scores, and CEM aggregates;
  * the execution amendment v2 freeze and its newly executed CEM aggregates;
  * the six reused motion-damping CEM results (identity-verified here).

Thresholds decide PASS/FAIL only; they never decide execution.  Failed scores
are reported unchanged.  The historical scoreboard file is never modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "artifacts/evaluation/complete_reference_comparison_v1"
PREREG = ROOT / "configs/benchmark/contextworld_complete_reference_comparison_prereg_v1.yaml"
RECEIPT = OUT_ROOT / "freeze_receipt.json"
AMEND = (
    ROOT
    / "configs/benchmark/"
    "contextworld_complete_reference_comparison_execution_amendment_v2.yaml"
)
AMEND_RECEIPT = OUT_ROOT / "execution_amendment_v2_freeze_receipt.json"
SCOREBOARD = (
    ROOT
    / "artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1/"
    "public_scoreboard.json"
)
SPEC = (
    ROOT
    / "artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1/"
    "public_scoreboard_spec.json"
)
OUTPUT = OUT_ROOT / "complete_comparison.json"

EXPECTED_SCOREBOARD_SHA256 = (
    "78dec56c735bf94c452a0a0e85a3d25619cee27c8179aafd613f79345f9e62f8"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, float]:
    rows = [float(value) for value in values]
    stats = {
        "mean": float(statistics.fmean(rows)),
        "minimum": float(min(rows)),
        "maximum": float(max(rows)),
    }
    if len(rows) > 1:
        stats["sample_std"] = float(statistics.stdev(rows))
        stats["sample_variance"] = float(statistics.variance(rows))
    return stats


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _standard_cem(path: Path) -> dict[str, Any]:
    """Read one aggregate.json from the standard/cube frozen CEM runners."""

    payload = _load_json(path)
    model = payload["model"] if "model" in payload else payload["models"][0]
    seeds = {
        int(row["eval_seed"]): int(row["success_count"]) for row in model["seeds"]
    }
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": file_sha256(path),
        "checkpoint_sha256": (
            model["checkpoint_sha256"]
            if "checkpoint_sha256" in model
            else model["checkpoint"]["sha256"]
        ),
        "successes": int(model["aggregate"]["success_count"]),
        "evaluations": int(model["aggregate"]["evaluation_count"]),
        "per_eval_seed_successes": seeds,
    }


def _tworoom_cem(directory: Path) -> dict[str, Any]:
    """Fold the six per-seed TwoRoom receipts of one checkpoint."""

    seeds: dict[int, int] = {}
    receipts = []
    checkpoint_hashes = set()
    for eval_seed in range(42, 48):
        path = directory / f"seed{eval_seed}.json"
        payload = _load_json(path)
        aggregate = payload["aggregate"]
        if int(aggregate["evaluations"]) != 50:
            raise RuntimeError(f"TwoRoom receipt is not a 50-episode cell: {path}")
        seeds[eval_seed] = int(aggregate["successes"])
        checkpoint_hashes.add(
            payload["frozen_input_preflight"]["checkpoint"]["sha256"]
        )
        receipts.append(
            {
                "eval_seed": eval_seed,
                "successes": int(aggregate["successes"]),
                "path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
            }
        )
    if len(checkpoint_hashes) != 1:
        raise RuntimeError(f"TwoRoom chain mixes checkpoints: {directory}")
    return {
        "checkpoint_sha256": checkpoint_hashes.pop(),
        "successes": sum(seeds.values()),
        "evaluations": 300,
        "per_eval_seed_successes": seeds,
        "receipts": receipts,
    }


def _icl_from_method_score(
    path: Path, *, metric: str, benchmark_label: str
) -> dict[str, Any]:
    payload = _load_json(path)
    rows = payload.get("checkpoints") or payload.get("checkpoint_results")
    per_seed: list[dict[str, Any]] = []
    for row in rows:
        seed = row.get("training_seed")
        value = row.get(metric)
        passed = row.get("passed")
        if value is None:  # cube layout nests metrics/gate
            seed = row["model"]["training_seed"]
            value = row["metrics"][metric]
            passed = row["gate"]["passed"]
        per_seed.append(
            {"training_seed": int(seed), "value": float(value), "passed": bool(passed)}
        )
    per_seed.sort(key=lambda row: row["training_seed"])
    passed = payload.get("decision", {}).get("passed")
    if passed is None:
        passed = payload.get("passed")
    return {
        "source": {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)},
        "primary_metric_id": metric,
        "primary_metric_label": benchmark_label,
        "per_seed": per_seed,
        **_stats([row["value"] for row in per_seed]),
        "passed_checkpoints": sum(row["passed"] for row in per_seed),
        "result": "PASS" if bool(passed) else "FAIL",
    }


def _cem_block(
    cells: list[dict[str, Any]],
    *,
    floor: int,
    baseline_successes: int,
    baseline_label: str,
    evidence_limit: str | None = None,
) -> dict[str, Any]:
    per_seed = [
        {
            "training_seed": cell["training_seed"],
            "successes": cell["successes"],
            "evaluations": cell["evaluations"],
            "success_rate": cell["successes"] / cell["evaluations"],
            "passed": cell["successes"] >= floor,
            "source": {
                key: cell[key]
                for key in ("path", "sha256")
                if key in cell
            }
            or cell.get("source"),
            "per_eval_seed_successes": cell["per_eval_seed_successes"],
            "per_eval_seed_rate_stats": _stats(
                [
                    successes / (cell["evaluations"] / len(cell["per_eval_seed_successes"]))
                    for successes in cell["per_eval_seed_successes"].values()
                ]
            ),
        }
        for cell in cells
    ]
    per_seed.sort(key=lambda row: row["training_seed"])
    block = {
        "protocol": "original_task_real_environment_cem_300_episodes_per_checkpoint",
        "metric_definitions": {
            "what_is_scored": (
                "冻结的对照 checkpoint（context 混合配方，3 个训练 seed）"
                "在原始任务真实环境中用冻结 CEM 规划协议执行的成功率"
            ),
            "family_baseline": (
                "原始数据配方训练的同家族模型，在同一原始任务、"
                "同一 CEM 协议、同一 eval seed 集合下的成功率（此前已冻结）"
            ),
            "per_seed.success_rate": "单个训练 seed 的 checkpoint 汇总成功率（全部 eval seed 合并）",
            "per_seed.per_eval_seed_successes": "该 checkpoint 在每个 eval seed 上的成功数（多 seed 采样）",
            "per_seed.per_eval_seed_rate_stats": "该 checkpoint 跨 eval seed 的成功率均值/方差",
            "mean|sample_std|sample_variance|minimum|maximum": (
                "跨 3 个训练 seed 的 checkpoint 汇总成功率统计"
            ),
        },
        "family_baseline": {
            "label": baseline_label,
            "successes": baseline_successes,
            "evaluations": 300,
            "success_rate": baseline_successes / 300,
        },
        "noninferiority_floor_successes": floor,
        "per_seed": per_seed,
        **_stats([row["success_rate"] for row in per_seed]),
        "passed_checkpoints": sum(row["passed"] for row in per_seed),
        "result": "PASS" if all(row["passed"] for row in per_seed) else "FAIL",
    }
    if evidence_limit:
        block["evidence_limit"] = evidence_limit
    return block


def main() -> None:
    observed_scoreboard = file_sha256(SCOREBOARD)
    if observed_scoreboard != EXPECTED_SCOREBOARD_SHA256:
        raise RuntimeError("Historical scoreboard drifted; refusing to build")
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    amend = yaml.safe_load(AMEND.read_text(encoding="utf-8"))
    receipt = _load_json(RECEIPT)
    spec = _load_json(SPEC)

    rows: list[dict[str, Any]] = []

    # ---- 13 historical rows: values carried verbatim from the frozen spec ----
    amendment_cells = {
        ("action_strength", "PLDM"): "action_strength_pldm",
        ("robot_arm_mass", "PLDM"): "reacher_arm_mass_pldm",
        ("portal_exit", "PLDM"): "portal_exit_pldm",
    }
    amendment_floors = {
        "action_strength_pldm": 218,
        "reacher_arm_mass_pldm": 233,
        "portal_exit_pldm": 263,
    }
    amendment_baselines = {
        "action_strength_pldm": ("原始数据配方 PushT PLDM 基线（同协议原任务 CEM）", 233),
        "reacher_arm_mass_pldm": ("原始数据配方 Reacher PLDM 基线（同协议原任务 CEM）", 248),
        "portal_exit_pldm": ("原始数据配方 TwoRoom PLDM 基线（同协议原任务 CEM）", 278),
    }
    for component in spec["components"]:
        family = "PLDM" if "PLDM" in component["method_name"] else "LeWM"
        icl = {
            "primary_metric_id": component["primary_metric"]["id"],
            "primary_metric_label": component["primary_metric"]["label"],
            "per_seed": [
                {"value": float(value), "passed": bool(passed)}
                for value, passed in zip(
                    component["primary_metric"]["per_seed_values"],
                    component["per_seed_gate_passes"],
                    strict=True,
                )
            ],
            **_stats(component["primary_metric"]["per_seed_values"]),
            "passed_checkpoints": sum(component["per_seed_gate_passes"]),
            "result": "PASS" if component["ability_passed"] else "FAIL",
            "source": {
                "path": str(SPEC.relative_to(ROOT)),
                "sha256": file_sha256(SPEC),
            },
        }
        retention = component["original_task_retention"]
        key = (component["component_id"], family)
        if key in amendment_cells:
            cell_name = amendment_cells[key]
            cell = amend["execution_cells"][cell_name]
            cells = []
            for checkpoint in cell["checkpoints"]:
                output = ROOT / checkpoint["output"]
                if cell_name == "portal_exit_pldm":
                    folded = _tworoom_cem(output)
                else:
                    folded = _standard_cem(output / "aggregate.json")
                if folded["checkpoint_sha256"] != checkpoint["sha256"]:
                    raise RuntimeError(
                        f"CEM result checkpoint hash drifted: {output}"
                    )
                folded["training_seed"] = int(checkpoint["seed"])
                if "receipts" in folded:
                    folded["path"] = str(output.relative_to(ROOT))
                    folded["sha256"] = None
                cells.append(folded)
            label, baseline = amendment_baselines[cell_name]
            cem = _cem_block(
                cells,
                floor=amendment_floors[cell_name],
                baseline_successes=baseline,
                baseline_label=label,
            )
            cem["completion"] = {
                "historical_status": "NOT_EVALUATED",
                "completed_by": amend["amendment_id"],
                "policy": "threshold_controls_verdict_only",
            }
        else:
            cem = {
                "carried_from_historical_scoreboard": True,
                "result": retention["result"],
            }
            if "per_seed_values" in retention:
                cem.update(
                    {
                        "primary_metric_id": retention["metric_id"],
                        "primary_metric_label": retention["metric_label"],
                        "family_baseline_rate": retention["baseline_value"],
                        "per_seed": [
                            {"success_rate": float(value)}
                            for value in retention["per_seed_values"]
                        ],
                        **_stats(retention["per_seed_values"]),
                    }
                )
        rows.append(
            {
                "row_kind": "historical_scoreboard_row",
                "component_id": component["component_id"],
                "component_name": component["component_name"],
                "family": family,
                "method_name": component["method_name"],
                "icl": icl,
                "original_task_cem": cem,
            }
        )

    # ---- 5 new rows from the base freeze ----
    icl_root = OUT_ROOT / "icl"
    cem_root = OUT_ROOT / "cem"
    pusht_baselines = prereg["cem_inputs"]["pusht"]["frozen_baselines"]
    pusht_floor = {
        "lewm": max(210, int(pusht_baselines["lewm"]["successes"]) - 15),
        "pldm": max(210, int(pusht_baselines["pldm"]["successes"]) - 15),
    }
    legacy_note = prereg["claim_boundary"]["contact_friction_and_motion_damping"][
        "disclosure"
    ]

    def new_cem_cells(
        component: str,
        family: str,
        seeds: list[int],
        *,
        receipt_component: str | None = None,
    ) -> list[dict]:
        cells = []
        for seed in seeds:
            folded = _standard_cem(
                cem_root / component / family / f"seed_{seed}" / "aggregate.json"
            )
            receipt_key = f"{receipt_component or component}/{family}/seed{seed}"
            expected = receipt["checkpoints"][receipt_key]["checkpoint"]["sha256"]
            if folded["checkpoint_sha256"] != expected:
                raise RuntimeError(
                    f"CEM checkpoint hash drifted: {component}/{family}/{seed}"
                )
            folded["training_seed"] = seed
            cells.append(folded)
        return cells

    # contact friction (both families, newly executed CEM)
    for family, score_path in (
        ("LeWM", icl_root / "contact_friction/lewm_method_score.json"),
        (
            "PLDM",
            icl_root
            / "contact_friction/pldm_method_score_float32_recovery_v1.json",
        ),
    ):
        fam = family.lower()
        icl = _icl_from_method_score(
            score_path,
            metric="correct_future_rate",
            benchmark_label="真实摩擦下一状态选择正确率",
        )
        cem = _cem_block(
            new_cem_cells("contact_friction", fam, [13313, 13314, 13315]),
            floor=pusht_floor[fam],
            baseline_successes=int(pusht_baselines[fam]["successes"]),
            baseline_label=f"原始数据配方 PushT {family} 基线（同协议原任务 CEM）",
        )
        rows.append(
            {
                "row_kind": "comparison_addendum_row",
                "component_id": "contact_friction",
                "component_name": "PushT 接触摩擦",
                "family": family,
                "method_name": f"{family}（legacy 2048 对 4096 步配方，rescored）",
                "comparator_kind": "fixed_legacy_comparator",
                "training_disclosure": legacy_note,
                "icl": icl,
                "original_task_cem": cem,
            }
        )

    # motion damping (both families, reused CEM evidence)
    reused = receipt["reused_motion_damping_cem"]
    for family in ("LeWM", "PLDM"):
        fam = family.lower()
        icl = _icl_from_method_score(
            icl_root / f"motion_damping/{fam}_method_score.json",
            metric="correct_future_rate",
            benchmark_label="真实阻尼下一状态选择正确率",
        )
        cells = []
        for row in reused:
            if row["family"] != fam:
                continue
            payload = _load_json(Path(row["resolved_path"]))
            model = payload["models"][0] if "models" in payload else payload["model"]
            cells.append(
                {
                    "training_seed": int(row["seed"]),
                    "successes": int(row["successes"]),
                    "evaluations": int(row["evaluations"]),
                    "checkpoint_sha256": model["checkpoint_sha256"],
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "per_eval_seed_successes": {
                        int(seed_row["eval_seed"]): int(seed_row["success_count"])
                        for seed_row in model["seeds"]
                    },
                }
            )
        cem = _cem_block(
            cells,
            floor=pusht_floor[fam],
            baseline_successes=int(pusht_baselines[fam]["successes"]),
            baseline_label=f"原始数据配方 PushT {family} 基线（同协议原任务 CEM）",
            evidence_limit=prereg["reused_motion_damping_cem"]["evidence_limit"],
        )
        rows.append(
            {
                "row_kind": "comparison_addendum_row",
                "component_id": "motion_damping",
                "component_name": "PushT 运动阻尼",
                "family": family,
                "method_name": f"{family}（legacy 2048 对 4096 步配方，rescored）",
                "comparator_kind": "fixed_legacy_comparator",
                "training_disclosure": legacy_note,
                "icl": icl,
                "original_task_cem": cem,
            }
        )

    # cube pldm (current v4r1 checkpoint, newly executed CEM)
    cube_rule = prereg["decision_rules"]["cube_cem"]
    icl = _icl_from_method_score(
        icl_root / "cube_pldm/pldm_method_score.json",
        metric="correct_future_rate",
        benchmark_label="真实携带规则下一状态选择正确率",
    )
    cem = _cem_block(
        new_cem_cells("cube", "pldm", [17321, 17322, 17323], receipt_component="cube_pldm"),
        floor=int(cube_rule["minimum_successes"]),
        baseline_successes=int(cube_rule["baseline_successes"]),
        baseline_label="原始数据配方 Cube PLDM 基线（同协议原任务 CEM）",
    )
    rows.append(
        {
            "row_kind": "comparison_addendum_row",
            "component_id": "cube_gripper_carry",
            "component_name": "Cube 夹爪携带规则",
            "family": "PLDM",
            "method_name": "PLDM（当前 v4r1 配方，补充公开结果）",
            "comparator_kind": "frozen_current_v4r1_trained_checkpoint",
            "formal_scoreboard_eligible": False,
            "icl": icl,
            "original_task_cem": cem,
        }
    )

    if len(rows) != 18:
        raise RuntimeError(f"Expected 18 rows, built {len(rows)}")

    payload = {
        "schema_version": 1,
        "result_kind": "contextworld_complete_reference_comparison",
        "comparison_id": "contextworld_complete_reference_comparison_v1",
        "report_all_policy": amend["report_all_policy"],
        "preregistration": {
            "base": {
                "path": str(PREREG.relative_to(ROOT)),
                "sha256": file_sha256(PREREG),
            },
            "base_freeze_receipt": {
                "path": str(RECEIPT.relative_to(ROOT)),
                "sha256": file_sha256(RECEIPT),
            },
            "execution_amendment_v2": {
                "path": str(AMEND.relative_to(ROOT)),
                "sha256": file_sha256(AMEND),
            },
            "execution_amendment_v2_receipt": {
                "path": str(AMEND_RECEIPT.relative_to(ROOT)),
                "sha256": file_sha256(AMEND_RECEIPT),
            },
        },
        "historical_scoreboard": {
            "path": str(SCOREBOARD.relative_to(ROOT)),
            "sha256": observed_scoreboard,
            "row_count": 13,
            "modified_by_this_comparison": False,
        },
        "claim_boundary": {
            "comparison_addendum_is_a_formal_scoreboard_rewrite": False,
            "negative_results_are_reported": True,
            "historical_scoreboard_rows_unchanged": True,
        },
        "row_count": len(rows),
        "rows": rows,
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": file_sha256(OUTPUT),
                "rows": len(rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
