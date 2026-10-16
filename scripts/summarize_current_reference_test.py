#!/usr/bin/env python3
"""Summarize the frozen current-reference cohort's offline Public Test results.

Only receipts for runs declared in the frozen plan count. An incomplete cell
keeps its observed runs in JSON, but has no publishable aggregate score.
This report never incorporates the separate Development CEM evaluations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.freeze_current_reference_baseline import _extract_score
from scripts.render_training_comparison import (MODEL_ORDER, MODEL_ZH, REGIME_ORDER,
                                                 REGIME_ZH, TASK_ORDER, TASK_ZH, stats)
from scripts.run_current_reference_test import SCHEMA as PLAN_SCHEMA, ADAPTER, canonical, sha256

SCHEMA = "contextworld.current-reference-public-test-summary.v1"


def _validate_native_result(payload: dict, run: dict) -> None:
    """Check identities actually emitted by the public task scorers."""
    task, family, regime, seed = (run[k] for k in
                                  ("task", "family", "regime", "training_seed"))
    expected_name = f"{family}-{regime}-s{seed}"
    if (payload.get("evaluation_split") != "test" or
            payload.get("result_kind") != "public_test_offline_final_report_v1" or
            payload.get("task") != task or payload.get("diagnostic") or
            payload.get("adapter_spec") != ADAPTER[family] or
            payload.get("model_name") != expected_name):
        raise ValueError("result is not a matching native Public Test report")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("Test result lacks native task payload")
    model = result.get("model")
    if not isinstance(model, dict):
        raise ValueError("Test result lacks native model identity")
    adapter = model.get("adapter", model)  # Speed flattens adapter metadata.
    if not isinstance(adapter, dict):
        raise ValueError("Test result lacks adapter identity")
    recipe = model.get("training_role" if task == "speed" else "training_recipe")
    if (model.get("name") != expected_name or recipe != regime or
            model.get("training_seed") != seed or
            adapter.get("checkpoint_sha256") != run["checkpoint_sha256"] or
            not isinstance(adapter.get("checkpoint"), str)):
        # SHA256 fixes checkpoint content. Absolute source paths are provenance,
        # and may resolve differently when the result archive is downloaded.
        raise ValueError("native model, seed, or checkpoint differs from frozen plan")
    protocol = adapter.get("protocol")
    adapter_id = adapter.get("adapter_id")
    native_family = "prejepa" if family == "dinowm" else family
    if (not isinstance(protocol, dict) or
            not isinstance(adapter_id, str) or
            not adapter_id.startswith(f"stable_worldmodel_{native_family}_") or
            "h3_tail_projection" in adapter_id or "diagnostic" in adapter_id or
            protocol.get("history_tokens") != (7 if task == "action_delay" else 3) or
            adapter.get("diagnostic") or adapter.get("missing_context_strategy") or
            adapter.get("prejepa_missing_context_policy")):
        raise ValueError("native history mode is unresolved or diagnostic")
    if task == "speed":
        audit = result.get("frozen_weight_audit")
        if (not isinstance(audit, dict) or audit.get("passed") is not True or
                not audit.get("state_hash_before") or
                audit.get("state_hash_before") != audit.get("state_hash_after")):
            raise ValueError("model state changed or audit is missing")
    elif task == "door":
        audit = result.get("score_audit")
        if (not isinstance(audit, dict) or audit.get("passed") is not True or
                not audit.get("frozen_state_hash_before") or
                audit.get("frozen_state_hash_before") !=
                audit.get("frozen_state_hash_after")):
            raise ValueError("model state changed or audit is missing")
    elif (not model.get("state_sha256_before") or
          model.get("state_sha256_before") != model.get("state_sha256_after")):
        raise ValueError("model state changed or audit is missing")


def summarize(plan_path: Path, results_root: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != PLAN_SCHEMA or plan.get("evaluation_split") != "test":
        raise ValueError("expected a frozen current-reference Public Test plan")
    runs = plan.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("plan has no runs")
    ids = [run["id"] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run ID in frozen plan")
    plan_sha = hashlib.sha256(canonical(plan)).hexdigest()
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    failures = []
    for run in runs:
        task, family, regime, seed = (run[k] for k in
                                      ("task", "family", "regime", "training_seed"))
        run_id = f"{task}/{family}/{regime}/s{seed}"
        if (run["id"] != run_id or task not in TASK_ORDER or family not in MODEL_ORDER
                or regime not in REGIME_ORDER or isinstance(seed, bool)
                or not isinstance(seed, int)):
            raise ValueError(f"invalid frozen run identity: {run['id']}")
        record = {"id": run_id, "training_seed": seed,
                  "checkpoint_sha256": run["checkpoint_sha256"], "status": "pending",
                  "main_score": None, "scores": {}, "result": None,
                  "result_sha256": None}
        receipt_path = results_root / run_id / "receipt.json"
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text())
                if (receipt.get("id") != run_id or
                        receipt.get("plan_sha256") != plan_sha or
                        receipt.get("checkpoint_sha256") != run["checkpoint_sha256"]):
                    raise ValueError("receipt identity differs from frozen plan")
                result_path = results_root / run_id / "test.json"
                if receipt.get("returncode") != 0:
                    raise ValueError(f"evaluator exit code {receipt.get('returncode')}")
                recorded = receipt.get("result")
                suffix = (*run_id.split("/"), "test.json")
                if (not isinstance(recorded, str) or
                        tuple(Path(recorded).parts[-5:]) != suffix or
                        not result_path.is_file()):
                    raise ValueError("successful receipt lacks expected Test result")
                digest = sha256(result_path)
                if receipt.get("result_sha256") != digest:
                    raise ValueError("Test result hash differs from receipt")
                payload = json.loads(result_path.read_text())
                _validate_native_result(payload, run)
                rate = _extract_score(payload, task, "test")
                if not 0 <= rate <= 1:
                    raise ValueError("native main score outside unit interval")
                score = 100 * rate
                record.update(status="completed", main_score=score, scores={"main": score},
                              result=str(result_path), result_sha256=digest)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record.update(status="failed", error=str(exc))
                failures.append({"id": run_id, "error": str(exc),
                                 "receipt": str(receipt_path)})
        cells[(task, family, regime)].append(record)
    groups = []
    for (task, family, regime), records in sorted(cells.items(), key=lambda item:
            (TASK_ORDER.index(item[0][0]), MODEL_ORDER.index(item[0][1]),
             REGIME_ORDER.index(item[0][2]))):
        completed = [r for r in records if r["status"] == "completed"]
        expected = len(records)
        score_stats = stats([r["main_score"] for r in completed])
        complete = len(completed) == expected
        groups.append({"task": task, "family": family, "regime": regime,
                       "expected": expected, "completed": len(completed),
                       "failed": sum(r["status"] == "failed" for r in records),
                       "status": "complete" if complete else "incomplete",
                       "observed_main": score_stats,
                       "main": score_stats if complete else None,
                       "runs": sorted(records, key=lambda r: r["training_seed"])})
    return {"schema": SCHEMA, "plan": str(plan_path), "plan_sha256": plan_sha,
            "results_root": str(results_root), "evaluation_split": "test",
            "scope": "native offline Public Test; main task scores only; CEM remains Development provenance",
            "expected_runs": len(runs), "completed_runs": sum(g["completed"] for g in groups),
            "complete_cells": sum(g["status"] == "complete" for g in groups),
            "expected_cells": len(groups), "cells": groups, "failures": failures}


def _cell_text(cell: dict | None) -> str:
    if cell is None:
        return "—"
    coverage = f"{cell['completed']}/{cell['expected']}"
    if cell["status"] != "complete":
        return f"待完成 ({coverage})"
    score = cell["main"]
    value = f"{score['mean']:.2f}"
    if score["std"] is not None:
        value += f" ± {score['std']:.2f}"
    return f"{value} (n={cell['completed']})"


def render_markdown(summary: dict, task: str | None = None) -> str:
    by_key = {(c["task"], c["family"], c["regime"]): c for c in summary["cells"]}
    lines = ["# ContextWorld Public Test 结果", "",
             "各任务主指标均值（%，越高越好），相同配置的训练重复等权汇总。"
             "± 为训练重复的样本标准差，n 为已评测训练重复数；只有一次训练时不显示标准差。"
             "每项任务分别使用对应检查点，一行不代表同一份权重同时掌握九项任务。", "",
             "训练方案、任务定义及缺项说明见[技术报告](../../../ContextWorld_ICL_Benchmark.md#training-comparison)。"
             "原环境 CEM 与预测诊断见[结果附录](../../../reference/ContextWorld_Expanded_Training_Results.md)。"
             "仅对预先确定的全部重复均完成评测的组合报告均值，未完成的组合标明覆盖数；未设可比检查点的单元记为 —。", "",
             f"已评测 {summary['completed_runs']} 个检查点，覆盖 {summary['complete_cells']} 个完整模型／方案／任务组合。"
             "逐次结果、统计量与评测身份见 [JSON](summary.json)、[CSV](summary.csv) 和[检查点清单](plan.json)。", ""]
    if task is None:
        lines += ["| 模型 | 方案 | " + " | ".join(TASK_ZH[t] for t in TASK_ORDER) + " |",
                  "|---|---|" + "---:|" * len(TASK_ORDER)]
        for family in MODEL_ORDER:
            for regime in REGIME_ORDER:
                if not any((t, family, regime) in by_key for t in TASK_ORDER):
                    continue
                lines.append("| " + " | ".join(
                    [MODEL_ZH[family], REGIME_ZH[regime]] +
                    [_cell_text(by_key.get((t, family, regime))) for t in TASK_ORDER]) + " |")
    else:
        if task not in TASK_ORDER:
            raise ValueError(f"unknown task: {task}")
        lines += [f"## {TASK_ZH[task]}", "", "| Model | Regime | Main score ↑ | Coverage | Failed runs |",
                  "|---|---|---:|---:|---:|"]
        for family in MODEL_ORDER:
            for regime in REGIME_ORDER:
                cell = by_key.get((task, family, regime))
                if cell is None:
                    continue
                lines.append("| " + " | ".join([
                    MODEL_ZH[family], REGIME_ZH[regime], _cell_text(cell),
                    f"{cell['completed']}/{cell['expected']}", str(cell["failed"])]) + " |")
    return "\n".join(lines) + "\n"


def render_public_table(summary: dict) -> str:
    """Compact Test table for the public report; full SD stays in JSON/CSV."""
    by_key = {(c["task"], c["family"], c["regime"]): c for c in summary["cells"]}
    lines = ["| 模型 | 方案 | " + " | ".join(TASK_ZH[t] for t in TASK_ORDER) + " |",
             "|---|---|" + "---:|" * len(TASK_ORDER)]
    for family in MODEL_ORDER:
        for regime in REGIME_ORDER:
            if not any((t, family, regime) in by_key for t in TASK_ORDER):
                continue
            values = []
            for task in TASK_ORDER:
                cell = by_key.get((task, family, regime))
                if cell is None:
                    values.append("—")
                elif cell["status"] != "complete":
                    raise ValueError("public table requires complete frozen cells")
                else:
                    values.append(f"{cell['main']['mean']:.2f} (n={cell['completed']})")
            lines.append("| " + " | ".join([MODEL_ZH[family], REGIME_ZH[regime], *values]) + " |")
    return "\n".join(lines)


def sync_report(summary: dict, path: Path, *, check: bool = False) -> None:
    if (summary["completed_runs"] != summary["expected_runs"] or summary["failures"]):
        raise ValueError("public report requires every frozen Test run")
    begin, end = "<!-- BEGIN CURRENT_REFERENCE_TEST -->", "<!-- END CURRENT_REFERENCE_TEST -->"
    text = path.read_text()
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("expected one Test report marker pair")
    start, stop = text.index(begin) + len(begin), text.index(end)
    if stop < start:
        raise ValueError("Test report markers are reversed")
    rendered = text[:start] + "\n" + render_public_table(summary) + "\n" + text[stop:]
    if check:
        if rendered != text:
            raise ValueError("public Test table is stale")
    else:
        path.write_text(rendered)


def write_outputs(summary: dict, output_dir: Path, task: str | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=["task", "family", "regime", "completed",
                          "expected", "failed", "status", "main_mean", "main_std"])
        writer.writeheader()
        for cell in summary["cells"]:
            score = cell["main"] or {}
            writer.writerow({k: cell[k] for k in ("task", "family", "regime", "completed",
                             "expected", "failed", "status")} |
                            {"main_mean": score.get("mean"), "main_std": score.get("std")})
    (output_dir / (f"task_{task}.md" if task else "overview.md")).write_text(
        render_markdown(summary, task))


def archive_complete(plan_path: Path, results_root: Path, archive_dir: Path,
                     *, expected_runs: int = 136) -> dict:
    """Copy a verified complete cohort, then regenerate its report from the copy."""
    source = summarize(plan_path, results_root)
    if (source["expected_runs"] != expected_runs or
            source["completed_runs"] != expected_runs or
            source["complete_cells"] != source["expected_cells"] or
            source["failures"]):
        raise ValueError("archive requires every frozen Test run to be complete")
    if archive_dir.exists():
        raise ValueError("archive directory already exists")
    archive_dir.mkdir(parents=True)
    shutil.copyfile(plan_path, archive_dir / "plan.json")
    archived_root = archive_dir / "results"
    for cell in source["cells"]:
        for run in cell["runs"]:
            run_id = run["id"]
            dest = archived_root / run_id
            dest.mkdir(parents=True)
            for name in ("test.json", "receipt.json"):
                shutil.copyfile(results_root / run_id / name, dest / name)
    archived = summarize(archive_dir / "plan.json", archived_root)
    if archived["completed_runs"] != expected_runs or archived["failures"]:
        raise ValueError("copied archive failed verification")
    write_outputs(archived, archive_dir)
    return archived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=TASK_ORDER)
    parser.add_argument("--archive-dir", type=Path,
                        help="copy a complete frozen 136-run cohort and regenerate its report")
    parser.add_argument("--report-doc", type=Path,
                        help="update the public report's generated Test table")
    parser.add_argument("--check-doc", action="store_true",
                        help="check the public table without writing any files")
    args = parser.parse_args()
    if args.check_doc and args.report_doc is None:
        parser.error("--check-doc requires --report-doc")
    if args.check_doc and args.archive_dir is not None:
        parser.error("--check-doc cannot create an archive")
    summary = summarize(args.plan, args.results_root)
    if not args.check_doc:
        write_outputs(summary, args.output_dir, args.task)
    if args.report_doc is not None:
        sync_report(summary, args.report_doc, check=args.check_doc)
    if args.archive_dir is not None:
        archive_complete(args.plan, args.results_root, args.archive_dir)
    print(json.dumps({"completed_runs": summary["completed_runs"],
                      "expected_runs": summary["expected_runs"],
                      "complete_cells": summary["complete_cells"]}))


if __name__ == "__main__":
    main()
