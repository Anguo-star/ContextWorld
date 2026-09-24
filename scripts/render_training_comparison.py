#!/usr/bin/env python3
"""Render training-comparison blocks in docs/ContextWorld_ICL_Benchmark.md from the
published docs/research/data/icl_training_study_v2.json (and validate its CSV twin).

Usage:
  python3 scripts/render_training_comparison.py           # regenerate CSV + update the four blocks
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
MARKERS = ["OVERVIEW", "METRICS", "SPEED", "LEGACY"]
COLS = (["id", "task", "task_zh", "model", "regime", "measurement_status", "evidence_kind",
         "training_seed", "training_epochs", "training_dataset_id",
         "training_dataset_manifest_sha256", "training_data_version", "training_history_length", "evaluation_history_tokens",
         "frozen_encoder_semantics", "initialization_mode", "init_weights_sha256",
         "checkpoint_sha256", "result_sha256", "metric_name", "evaluation_split",
         "cem_reference_scope", "cem_evaluation_count", "cem_budget"] + ["score_" + k for k in ALL]
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
        if rid != f"{r['task']}/{r['model']}/{r['regime']}":
            raise Fail(f"id contradicts task/model/regime: {rid}")
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
        for i, k in enumerate(ALL):
            jv, cv = r["scores"][k], c[COLS.index("score_" + k)]
            if jv is None:
                if cv != "":
                    raise Fail(f"CSV/JSON mismatch score_{k} in {r['id']}")
            elif abs(float(cv) - jv) > 1e-9:
                raise Fail(f"CSV/JSON mismatch score_{k} in {r['id']}")


def write_csv(rows):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
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
                r["cem_evaluation_count"], r["cem_budget"],
            ] + [r["scores"][k] for k in ALL]
              + [seeds.get(s) for s in range(42, 48)]
              + ["yes" if r.get("speed_tracks") else "no"])


def cell(r, regime):
    """overview cell: main / CEM, 2 decimals; — when absent; † marks H3 delay references."""
    if r["model"] == "dinowm" and regime == "joint":
        return "N/A"
    s = r["scores"]
    if s["main"] is None and s["cem"] is None:
        return "—"
    dagger = "†" if (r["task"] == "action_delay" and regime == "original") else ""
    return f"{f2(s['main'])} / {f2(s['cem'])}{dagger}"


def render_overview(rows):
    idx = {(r["task"], r["model"], r["regime"]): r for r in rows}
    out = ["| 任务 | 模型 | 原始模型 | ICL 从头 | 二阶段联合 | 二阶段冻结 Encoder | DINO 历史转换 |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for t in TASK_ORDER:
        for m in MODEL_ORDER:
            cells = []
            for reg in ("original", "scratch", "joint", "frozen", "projected"):
                r = idx.get((t, m, reg))
                cells.append(cell(r, reg) if r else ("N/A" if (m == "dinowm" and reg == "joint") else "—"))
            out.append(f"| {TASK_ZH[t]} | {MODEL_ZH[m]} | " + " | ".join(cells) + " |")
    out += ["",
            "单元格为主指标 / CEM（百分比，2 位小数）；未评测为 —。DINO-WM 无二阶段联合训练（N/A）；"
            "DINO-WM 从新的纯图像与动作原始权重开始的完整二阶段结果尚未报告（—）。",
            "†：动作延迟的原始参考来自 H3 原始模型（DINO 为原生 CEM 参考，LeWM/PLDM 为尾部投影诊断），"
            "与原生 H7 评测不可直接比较。旧 DINO 投影列为历史参考，非 pixels+action 原始权重完整 warmstart。"]
    return "\n".join(out)


def render_metrics(rows):
    hdr = ["任务", "模型", "训练设定", "主分↑", "最弱条件↑", "History↑", "Switch↑", "Joint↑",
           "Gain≈1", "Alignment↑", "NRE↓", "CalResp↑", "CEM↑"]
    out = ["| " + " | ".join(hdr) + " |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    selected = [
        r for r in rows
        if r["regime"] != "projected"
        and (r["measurement_status"] == "available"
             or (r["task"] == "action_delay" and r["model"] == "dinowm"
                 and r["regime"] == "original"))  # CEM-only H3 reference row
    ]
    selected.sort(key=lambda r: (TASK_ORDER.index(r["task"]), MODEL_ORDER.index(r["model"]),
                                 ("original", "scratch", "joint", "frozen").index(r["regime"])))
    if len(selected) != 78:
        raise Fail(f"metrics table expected 78 rows, got {len(selected)}")
    for r in selected:
        s = r["scores"]
        if r["task"] == "action_delay" and r["regime"] == "original":
            label = "原始（H3 尾部投影†）" if r["model"] != "dinowm" else "原始（H3，仅 CEM†）"
        else:
            label = REGIME_ZH[r["regime"]]
        out.append("| " + " | ".join([
            TASK_ZH[r["task"]], MODEL_ZH[r["model"]], label,
            f2(s["main"]), f2(s["worst"]), f2(s["history"]), f2(s["switch"]), f2(s["joint"]),
            f3(s["gain"]), f3(s["alignment"]), f3(s["nre"]), f2(s["calibrated"]), f2(s["cem"]),
        ]) + " |")
    out += ["",
            "百分比指标保留 2 位小数；增益/对齐/NRE 为无量纲比值，保留 3 位小数；— 表示该行未产出该项。"
            "仅收录已测得主 ICL 分数的行（77 行）加 1 行 DINO-WM 动作延迟 H3 原始模型 CEM 参考；"
            "两行 H3 尾部投影诊断行以 † 标注，与原生 H7 行不可直接比较。ICL 使用 Development；CEM 是原环境规划结果。"]
    return "\n".join(out)


def render_speed(rows):
    out = ["| 模型 | 训练设定 | 速度分布 | 主分↑ | 最弱条件↑ | History↑ | Switch↑ | Gain≈1 | Alignment↑ | NRE↓ | CalResp↑ |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    n = 0
    for m in MODEL_ORDER:
        for reg in ("original", "scratch"):
            r = next(x for x in rows if (x["task"], x["model"], x["regime"]) == ("speed", m, reg))
            tracks = r.get("speed_tracks") or {}
            for tr in SPEED_TRACKS:
                if tr not in tracks:
                    raise Fail(f"missing speed track {tr} for {r['id']}")
                t = tracks[tr]
                out.append("| " + " | ".join([
                    MODEL_ZH[m], REGIME_ZH[reg], TRACK_ZH[tr],
                    f2(t["main"]), f2(t["worst"]), f2(t["history"]), f2(t["switch"]),
                    f3(t["gain"]), f3(t["alignment"]), f3(t["nre"]), f2(t["calibrated"]),
                ]) + " |")
                n += 1
    if n != 24:
        raise Fail(f"speed table expected 24 rows, got {n}")
    out += ["",
            "速度任务按四个分轨分别报告（6 个已观测的 模型×训练设定 × 4 轨 = 24 行）；"
            "不同速度分布与预测 horizon 不合并、不平均。"]
    return "\n".join(out)


def render_legacy(rows):
    out = ["| 任务 | 模型 | 主分↑ | 最弱条件↑ | History↑ | Switch↑ | Joint↑ | Gain≈1 | Alignment↑ | NRE↓ | CalResp↑ | CEM↑ |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    legacy = [r for r in rows if r["regime"] == "projected"]
    if len(legacy) != 6:
        raise Fail(f"legacy table expected 6 rows, got {len(legacy)}")
    for r in sorted(legacy, key=lambda x: TASK_ORDER.index(x["task"])):
        s = r["scores"]
        out.append("| " + " | ".join([
            TASK_ZH[r["task"]], MODEL_ZH[r["model"]],
            f2(s["main"]), f2(s["worst"]), f2(s["history"]), f2(s["switch"]), f2(s["joint"]),
            f3(s["gain"]), f3(s["alignment"]), f3(s["nre"]), f2(s["calibrated"]), f2(s["cem"]),
        ]) + " |")
    out += ["",
            "以上 6 行为历史来源：旧 DINO-WM 投影初始化参考（仅覆盖 6 个任务），"
            "不是新的 pixels+action 原始权重完整 warmstart，不得当作 DINO 原始权重迁移成绩引用。"]
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
    blocks = {"OVERVIEW": render_overview(rows), "METRICS": render_metrics(rows),
              "SPEED": render_speed(rows), "LEGACY": render_legacy(rows)}

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
        print(f"OK: {len(rows)} rows validated; JSON/CSV consistent; all 4 blocks up to date")
        return

    new = text
    for m in MARKERS:
        new = splice(new, m, blocks[m])
    if new != text:
        DOC.write_text(new, encoding="utf-8")
        print("updated 4 blocks in", DOC)
    else:
        print("no changes needed")
    print(f"CSV written: {CSV_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Fail as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
