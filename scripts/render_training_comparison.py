#!/usr/bin/env python3
"""Render training-comparison blocks in docs/ContextWorld_ICL_Benchmark.md from the
published docs/research/data/icl_training_study_v2.json (and validate its CSV twin).

Usage:
  python3 scripts/render_training_comparison.py           # regenerate CSV + update the unified table
  python3 scripts/render_training_comparison.py --check   # validate only, never mutate

Errors (non-zero exit) on: missing/duplicate markers, duplicate row ids, id/task/model/regime
contradictions, nonfinite scores, percent metrics outside [0,100], unexpected CEM seed budgets,
or a mismatch between per-seed CEM means and the aggregate CEM score.
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "ContextWorld_ICL_Benchmark.md"
JSON_PATH = REPO / "docs" / "research" / "data" / "icl_training_study_v2.json"
CSV_PATH = REPO / "docs" / "research" / "data" / "icl_training_study_v2.csv"

TASK_ORDER = ["speed", "action_strength", "robot_arm_mass", "action_delay",
              "contact_friction", "motion_damping", "cube_gripper_carry", "door", "portal_exit"]
TASK_ZH = {"action_delay": "动作延迟", "action_strength": "推手移动幅度", "contact_friction": "接触摩擦",
           "motion_damping": "运动阻尼", "cube_gripper_carry": "Cube 夹爪携带", "door": "门通行规则",
           "portal_exit": "传送门出口", "robot_arm_mass": "机械臂质量", "speed": "速度"}
MODEL_ORDER = ["lewm", "pldm", "dinowm"]
MODEL_ZH = {"lewm": "LeWM", "pldm": "PLDM", "dinowm": "DINO-WM"}
REGIME_ZH = {"original": "原始模型", "scratch": "ICL 从头", "joint": "二阶段联合", "frozen": "二阶段冻结",
             "projected": "历史转换初始化"}
PCT = ["main", "worst", "history", "switch", "joint", "calibrated", "cem"]
RATIO = ["gain", "alignment", "nre"]
ALL = ["main", "worst", "history", "switch", "joint", "gain", "alignment", "nre", "calibrated", "cem"]
SPEED_TRACKS = ["seen_for_multi", "unseen_interpolation", "extrapolation_low", "extrapolation_high"]
TRACK_ZH = {"seen_for_multi": "训练中已见速度", "unseen_interpolation": "未见速度插值",
            "extrapolation_low": "低端外推", "extrapolation_high": "高端外推"}
MARKERS = ["FULL"]
COLS = (["id", "task", "task_zh", "model", "regime", "measurement_status", "evidence_kind",
         "training_seed", "training_epochs", "training_dataset_id",
         "training_dataset_manifest_sha256", "training_data_version", "training_history_length", "evaluation_history_tokens",
         "frozen_encoder_semantics", "initialization_mode", "init_weights_sha256",
         "checkpoint_sha256", "result_sha256", "metric_name", "evaluation_split",
         "cem_reference_scope", "cem_evaluation_count", "cem_budget", "comparison_variant", "training_pair_count"] + ["score_" + k for k in ALL]
        + [f"cem_seed{s}_percent" for s in range(42, 48)] + ["speed_tracks_present"])


class Fail(Exception):
    pass


def f2(v):
    return "—" if v is None else f"{v:.2f}"


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def validate(doc):
    if doc.get("schema_version") != "contextworld.icl-training-study.v2":
        raise Fail("unexpected schema_version")
    rows = doc["rows"]
    seen = set()
    for r in rows:
        rid = r["id"]
        if r["task"] not in TASK_ORDER or r["model"] not in MODEL_ORDER or r["regime"] not in REGIME_ZH:
            raise Fail(f"unknown task/model/regime in {rid}")
        if (r["model"] == "dinowm" and r["regime"] == "joint") or (r["model"] != "dinowm" and r["regime"] == "projected"):
            raise Fail(f"incompatible model/regime in {rid}")
        if rid in seen:
            raise Fail(f"duplicate row id {rid}")
        seen.add(rid)
        expected_id = f"{r['task']}/{r['model']}/{r['regime']}"
        if r.get("comparison_variant"):
            expected_id += "/" + r["comparison_variant"]
        if rid != expected_id:
            raise Fail(f"id contradicts task/model/regime: {rid}")
        variant = r.get("comparison_variant")
        if variant and (variant not in ("scale_2k", "scale_8k") or
                        r.get("training_pair_count") != {"scale_2k": 2048, "scale_8k": 8192}[variant]):
            raise Fail(f"data scale contradicts comparison variant: {rid}")
        if r["measurement_status"] not in ("available", "not_measured"):
            raise Fail(f"bad measurement_status {rid}")
        for k, v in r["scores"].items():
            if v is None:
                continue
            if not math.isfinite(v):
                raise Fail(f"nonfinite score {k} in {rid}")
            if k in PCT and not (0.0 <= v <= 100.0):
                raise Fail(f"percent metric out of range {k}={v} in {rid}")
        seeds = r.get("cem_per_seed") or []
        if seeds:
            seed_ids = {s["eval_seed"] for s in seeds}
            ok6 = len(seeds) == 6 and seed_ids == set(range(42, 48)) and all(s["budget_episodes"] == 50 for s in seeds)
            ok3 = (r["regime"] == "original" and r["model"] in ("lewm", "pldm") and len(seeds) == 3 and seed_ids == {42, 43, 44} and all(s["budget_episodes"] == 100 for s in seeds))
            if not (ok6 or ok3):
                raise Fail(f"unexpected CEM seed/budget layout in {rid}")
            agg = r["scores"]["cem"]
            if agg is not None:
                tot = sum(s["budget_episodes"] for s in seeds)
                m = sum(s["success_rate_percent"] * s["budget_episodes"] for s in seeds) / tot
                if abs(m - agg) > 1e-6:
                    raise Fail(f"CEM per-seed mean != aggregate in {rid}: {m} vs {agg}")
        if r["measurement_status"] == "not_measured" and r["scores"]["main"] is not None:
            raise Fail(f"not_measured row carries main score: {rid}")
    expected = {f"{t}/{m}/{reg}" for t in TASK_ORDER for m in MODEL_ORDER
                for reg in (("original", "scratch", "frozen") if m == "dinowm"
                            else ("original", "scratch", "joint", "frozen"))}
    if not expected <= seen:
        raise Fail(f"missing declared configurations: {sorted(expected - seen)}")
    return rows


def validate_csv(doc, rows):
    if not CSV_PATH.exists():
        raise Fail("CSV twin missing")
    with open(CSV_PATH, newline="") as f:
        rd = list(csv.reader(f))
    if not rd or rd[0] != COLS:
        raise Fail("CSV header mismatch")
    body = rd[1:]
    if len(body) != len(rows):
        raise Fail(f"CSV row count {len(body)} != JSON rows {len(rows)}")
    for r, c in zip(rows, body):
        if c[0] != r["id"] or c[1] != r["task"] or c[3] != r["model"] or c[4] != r["regime"]:
            raise Fail(f"CSV row identity mismatch near {c[0]}")
        for key in ("comparison_variant", "training_pair_count"):
            expected_value = "" if r.get(key) is None else str(r[key])
            if c[COLS.index(key)] != expected_value:
                raise Fail(f"CSV/JSON mismatch {key} in {r['id']}")
        for i, k in enumerate(ALL):
            jv, cv = r["scores"][k], c[COLS.index("score_" + k)]
            if jv is None:
                if cv != "":
                    raise Fail(f"CSV/JSON mismatch score_{k} in {r['id']}")
            elif abs(float(cv) - jv) > 1e-9:
                raise Fail(f"CSV/JSON mismatch score_{k} in {r['id']}")


def write_csv(rows):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLS)
        for r in rows:
            seeds = {s["eval_seed"]: s["success_rate_percent"] for s in (r.get("cem_per_seed") or [])}
            w.writerow([
                r["id"], r["task"], r["task_zh"], r["model"], r["regime"], r["measurement_status"],
                r["evidence_kind"], r["training_seed"], r["training_epochs"], r["training_dataset_id"],
                r["training_dataset_manifest_sha256"], r["training_data_version"], r["training_history_length"],
                r["evaluation_history_tokens"],
                (r.get("frozen_encoder_semantics") or {}).get("semantics"),
                r["initialization_mode"], r["init_weights_sha256"], r["checkpoint_sha256"],
                r["result_sha256"], r["metric_name"], r["evaluation_split"], r["cem_reference_scope"],
                r["cem_evaluation_count"], r["cem_budget"], r.get("comparison_variant"), r.get("training_pair_count"),
            ] + [r["scores"][k] for k in ALL]
              + [seeds.get(s) for s in range(42, 48)]
              + ["yes" if r.get("speed_tracks") else "no"])



def table_records(rows):
    """Every declared configuration appears, with Speed distributions kept distinct."""
    selected = sorted(rows, key=lambda r: (TASK_ORDER.index(r["task"]),
        MODEL_ORDER.index(r["model"]), tuple(REGIME_ZH).index(r["regime"]),
        r.get("training_pair_count") or 0))
    records = []
    for r in selected:
        tracks = r.get("speed_tracks") or {}
        if tracks:
            if set(tracks) != set(SPEED_TRACKS):
                raise Fail(f"incomplete Speed distributions in {r['id']}")
            for track in SPEED_TRACKS:
                records.append((r, TRACK_ZH[track], dict(tracks[track], cem=r["scores"]["cem"])))
        else:
            condition = "六个响应组" if r["task"] == "action_delay" else "两种隐藏条件"
            if r["task"] == "speed":
                condition = "未报告"
            if r["task"] == "action_delay" and r["training_history_length"] == 3:
                condition = "H3 参考†"
            records.append((r, condition, r["scores"]))
    if {r["id"] for r, _, _ in records} != {r["id"] for r in rows}:
        raise Fail("unified table omitted configurations")
    return records


def data_label(r):
    if r["regime"] == "original":
        return "原环境"
    if not r.get("training_data_version"):
        return "—"
    count = r.get("training_pair_count")
    if count is not None:
        label = {2048: "2k", 8192: "8k", 10000: "10k 独立来源", 32768: "32k"}.get(count, f"{count:,}")
        if r["task"] == "portal_exit" and "coverage-v2" in r["training_data_version"]:
            label += " 覆盖扩展"
        return "混合 / " + label
    return "混合 / 基础版"


def metric_cell(r, key, value):
    if value is not None:
        return f3(value) if key in RATIO else f2(value)
    if key == "joint" and r["task"] in ("speed", "action_delay", "door"):
        return "N/A"
    return "—"


def render_full(rows):
    hdr = ["任务", "模型", "种子", "训练数据", "训练方案", "评测条件", "主分↑", "最弱条件↑",
           "History↑", "Switch↑", "Joint↑", "Gain≈1", "Alignment↑", "NRE↓", "CalResp↑", "CEM↑"]
    out = ["| " + " | ".join(hdr) + " |",
           "|" + "---|" * 6 + "---:|" * len(ALL)]
    for r, condition, scores in table_records(rows):
        label = REGIME_ZH[r["regime"]]
        if r["regime"] == "projected":
            label += "‡"
        if r["measurement_status"] == "not_measured":
            label += "（仅 CEM）" if r["scores"]["cem"] is not None else "（未报告）"
        out.append("| " + " | ".join([TASK_ZH[r["task"]], MODEL_ZH[r["model"]], str(r["training_seed"]) if r["training_seed"] is not None else "—",
            data_label(r), label, condition] + [metric_cell(r, k, scores[k]) for k in ALL]) + " |")
    return "\n".join(out)


def splice(text, marker, block):
    begin, end = f"<!-- BEGIN TRAINING_COMPARISON_{marker} -->", f"<!-- END TRAINING_COMPARISON_{marker} -->"
    nb, ne = text.count(begin), text.count(end)
    if nb != 1 or ne != 1:
        raise Fail(f"marker {marker}: found {nb} begin / {ne} end tags (need exactly 1 each)")
    i, j = text.index(begin) + len(begin), text.index(end)
    if j < i:
        raise Fail(f"marker {marker}: end occurs before begin")
    return text[:i] + "\n" + block + "\n" + text[j:]


def extract(text, marker):
    begin, end = f"<!-- BEGIN TRAINING_COMPARISON_{marker} -->", f"<!-- END TRAINING_COMPARISON_{marker} -->"
    i, j = text.index(begin) + len(begin), text.index(end)
    return text[i:j].strip("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate only; never modify files")
    args = ap.parse_args()

    doc = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = validate(doc)
    blocks = {"FULL": render_full(rows)}

    if not args.check:
        write_csv(rows)

    if not DOC.exists():
        raise Fail(f"benchmark doc missing: {DOC}")
    text = DOC.read_text(encoding="utf-8")
    for m in MARKERS:
        begin, end = f"<!-- BEGIN TRAINING_COMPARISON_{m} -->", f"<!-- END TRAINING_COMPARISON_{m} -->"
        if text.count(begin) != 1 or text.count(end) != 1:
            raise Fail(f"marker {m}: found {text.count(begin)} begin / {text.count(end)} end tags")
    # CSV must be validated in both modes (generated above in run mode)
    validate_csv(doc, rows)

    if args.check:
        for m in MARKERS:
            if extract(text, m).strip() != blocks[m].strip():
                raise Fail(f"--check: block {m} is stale or does not match published JSON")
        print(f"OK: {len(rows)} rows validated; JSON/CSV consistent; the unified table is complete and up to date")
        return

    new = text
    for m in MARKERS:
        new = splice(new, m, blocks[m])
    if new != text:
        DOC.write_text(new, encoding="utf-8")
        print("updated unified table in", DOC)
    else:
        print("no changes needed")
    print(f"CSV written: {CSV_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Fail as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
