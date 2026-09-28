#!/usr/bin/env python3
"""Render aggregated training-comparison blocks in docs/ContextWorld_ICL_Benchmark.md
(and the historical appendix in docs/reference/Benchmark_Result_Provenance.md) from the
published docs/research/data/icl_training_study_v2.json, and maintain its aggregated CSV twin.

Every published row stays a separate record: distinct ids, training recipes, data versions,
or data scales never merge.  When a row carries ``replicates`` -- per-training-run dicts that
share the row's recipe/data/protocol -- the renderer aggregates them into per-metric mean,
sample SD (ddof=1; None when n=1) and count, computed independently for each metric.  Missing
metrics are never filled with zeros, the six CEM evaluation seeds are never counted as
training replicates, and the published representative scores are never overwritten.  Rows
without ``replicates`` fall back to the published representative scores with n=1.

Usage:
  python3 scripts/render_training_comparison.py           # regenerate CSV + update doc blocks
  python3 scripts/render_training_comparison.py --check   # validate only, never mutate

Errors (non-zero exit) on: missing/duplicate markers, duplicate row ids, id/task/model/regime
contradictions, nonfinite or out-of-range scores (published or per-replica), duplicate
training seeds or checkpoints inside one row's replicates, unexpected CEM seed budgets, a
mismatch between per-seed CEM means and aggregate CEM scores, or a stale CSV twin / doc block.
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
APPENDIX_DOC = REPO / "docs" / "reference" / "Benchmark_Result_Provenance.md"
JSON_PATH = REPO / "docs" / "research" / "data" / "icl_training_study_v2.json"
CSV_PATH = REPO / "docs" / "research" / "data" / "icl_training_study_v2.csv"
DECISION_PATH = REPO / "docs" / "research" / "data" / "icl_action_selection_v1.json"
CURVES_PATH = REPO / "docs" / "research" / "data" / "icl_training_dynamics_v1.json"

TASK_ORDER = ["speed", "action_strength", "robot_arm_mass", "action_delay",
              "contact_friction", "motion_damping", "cube_gripper_carry", "door", "portal_exit"]
TASK_ZH = {"action_delay": "动作延迟", "action_strength": "推手移动幅度", "contact_friction": "接触摩擦",
           "motion_damping": "运动阻尼", "cube_gripper_carry": "Cube 夹爪携带", "door": "门通行规则",
           "portal_exit": "传送门出口", "robot_arm_mass": "机械臂质量", "speed": "速度"}
MODEL_ORDER = ["lewm", "pldm", "dinowm"]
MODEL_ZH = {"lewm": "LeWM", "pldm": "PLDM", "dinowm": "DINO-WM"}
REGIME_ORDER = ["original", "scratch", "joint", "frozen"]
REGIME_ZH = {"original": "原始模型", "scratch": "ICL 从头", "joint": "二阶段联合", "frozen": "二阶段冻结",
             "projected": "历史转换初始化"}
PCT = ["main", "worst", "history", "switch", "joint", "calibrated", "cem"]
RATIO = ["gain", "alignment", "nre"]
ALL = ["main", "worst", "history", "switch", "joint", "gain", "alignment", "nre", "calibrated", "cem"]
# Response is a display transform, not an additional evaluator output or gate.
DISPLAY = ["main", "response"] + [k for k in ALL[1:] if k != "nre"]
EXPORTED = ALL + ["response"]
SPEED_TRACKS = ["seen_for_multi", "unseen_interpolation", "extrapolation_low", "extrapolation_high"]
SPEED_MAIN_TRACK = "unseen_interpolation"
TRACK_ZH = {"seen_for_multi": "训练中已见速度", "unseen_interpolation": "未见速度插值",
            "extrapolation_low": "低端外推", "extrapolation_high": "高端外推"}
# Tasks that never define the Joint contrast.
JOINT_NA_TASKS = ("speed", "action_delay", "door")
SCALING_TASKS = ["action_strength", "contact_friction", "motion_damping",
                 "robot_arm_mass", "cube_gripper_carry", "portal_exit"]
LEWM_STRENGTH_JOINT_IDS = ["action_strength/lewm/joint/scale_2k",
                           "action_strength/lewm/joint/scale_8k",
                           "action_strength/lewm/joint"]

DETAIL_MARKERS = {task: "DETAIL_" + task.upper() for task in TASK_ORDER}
MAIN_MARKERS = ["OVERVIEW", "DECISION", "CURVES"] + [DETAIL_MARKERS[t] for t in TASK_ORDER] + ["SCALING"]
APPENDIX_MARKERS = ["HISTORICAL"]
MARKERS = MAIN_MARKERS + APPENDIX_MARKERS


def marker_doc(marker):
    """Path of the document that owns a marker (HISTORICAL lives in the appendix)."""
    return APPENDIX_DOC if marker in APPENDIX_MARKERS else DOC

CSV_COLS = (["id", "task", "model", "regime", "training_data_version", "training_pair_count",
             "n_icl", "n_cem", "training_seeds"]
            + [f"score_{m}_{k}" for m in EXPORTED for k in ("mean", "std", "n")]
            + ["speed_tracks_json"])

OVERVIEW_GROUPS = [(model, regime) for model in MODEL_ORDER for regime in REGIME_ORDER
                   if not (model == "dinowm" and regime == "joint")]
METRIC_HDR = ["主分↑", "最弱条件↑", "History↑", "Switch↑", "Joint↑", "Gain≈1",
              "Alignment↑", "NRE↓", "CalResp↑", "CEM↑"]
DISPLAY_LABEL = dict(zip(ALL, METRIC_HDR)) | {"response": "响应分↑", "calibrated": "优于零响应↑"}


class Fail(Exception):
    pass


def f2(v):
    return "—" if v is None else f"{v:.2f}"


def f3(v):
    return "—" if v is None else f"{v:.3f}"


# ---------------------------------------------------------------- aggregation

def stats(values):
    """Mean / sample SD (ddof=1, None for n<=1) / count over the non-None values."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if n > 1 else None,
            "n": n}


def aggregate(row):
    """Aggregate one published row across its training replicates.

    Returns ``{"scores": {metric: stats}, "speed_tracks": {track: {metric: stats}},
    "n_replicates": int, "seeds": [...], "from_replicates": bool}``.  Replicates are
    used when present; otherwise the published representative scores act as a
    single (n=1) run and are never mutated.
    """
    reps = row.get("replicates")
    if reps is None:
        score_sets = [row.get("scores") or {}]
        track_sets = [row.get("speed_tracks") or {}]
        seeds = [row.get("training_seed")]
        from_reps = False
    else:
        score_sets = [rep.get("scores") or {} for rep in reps]
        track_sets = [rep.get("speed_tracks") or {} for rep in reps]
        seeds = [rep.get("training_seed") for rep in reps]
        from_reps = True
    scores = {m: stats([s.get(m) for s in score_sets]) for m in ALL}
    speed_tracks = {}
    for track in SPEED_TRACKS:
        carrying = [t for t in track_sets if track in t]
        if carrying:
            speed_tracks[track] = {m: stats([t[track].get(m) for t in carrying]) for m in ALL}
    return {"scores": scores, "speed_tracks": speed_tracks,
            "n_replicates": len(score_sets),
            "seeds": [s for s in seeds if s is not None],
            "from_replicates": from_reps}


def with_response(metrics):
    """Add 100*(1-NRE) without clipping or changing the source statistics.

    This affine transform preserves the contributing runs and scales sample SD
    by 100. NRE is already a squared error ratio; do not square it again.
    """
    nre = metrics.get("nre") or {"mean": None, "std": None, "n": 0}
    return {**metrics, "response": {
        "mean": None if nre["mean"] is None else 100.0 * (1.0 - nre["mean"]),
        "std": None if nre["std"] is None else 100.0 * nre["std"],
        "n": nre["n"],
    }}


def display_stats(row):
    """Per-metric stats used for the main report; Speed reports unseen_interpolation.

    Returns ``(metrics, agg)`` where ``metrics`` maps the source metrics plus the derived response score to stats
    (the Speed row borrows them from the unseen-interpolation track plus row CEM).
    """
    agg = aggregate(row)
    if row.get("task") == "speed":
        track = agg["speed_tracks"].get(SPEED_MAIN_TRACK)
        if track:
            metrics = dict(track)
            metrics["cem"] = agg["scores"]["cem"]
            return with_response(metrics), agg
    return with_response(agg["scores"]), agg


def is_current_ordinary(row):
    """Current ordinary configuration: not a legacy projection, not a scale variant."""
    return row["regime"] != "projected" and not row.get("comparison_variant")


def ordered_current(rows, task=None):
    selected = [r for r in rows if is_current_ordinary(r) and (task is None or r["task"] == task)]
    return sorted(selected, key=lambda r: (MODEL_ORDER.index(r["model"]),
                                           REGIME_ORDER.index(r["regime"]),
                                           r.get("training_pair_count") or 0))


def overview_groups(rows):
    """The eleven (model, regime) groups, each mapping every task to its ordinary row."""
    groups = {}
    for r in rows:
        if is_current_ordinary(r):
            groups.setdefault((r["model"], r["regime"]), {})[r["task"]] = r
    if set(groups) != set(OVERVIEW_GROUPS):
        raise Fail(f"unexpected overview groups: {sorted(set(groups) ^ set(OVERVIEW_GROUPS))}")
    for key, tasks in groups.items():
        if set(tasks) != set(TASK_ORDER):
            raise Fail(f"overview group {key} lacks ordinary rows for {sorted(set(TASK_ORDER) - set(tasks))}")
    return [(key, groups[key]) for key in OVERVIEW_GROUPS]


# ---------------------------------------------------------------- validation

def _check_scores(tag, scores):
    for k, v in scores.items():
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise Fail(f"non-numeric score {k} in {tag}")
        if not math.isfinite(v):
            raise Fail(f"nonfinite score {k} in {tag}")
        if k in PCT and not (0.0 <= v <= 100.0):
            raise Fail(f"percent metric out of range {k}={v} in {tag}")


def _check_cem_per_seed(tag, scores, seeds_list, regime, model):
    if not seeds_list:
        return
    for entry in seeds_list:
        value = entry["success_rate_percent"]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise Fail(f"invalid per-evaluation CEM rate in {tag}")
    seed_ids = [s["eval_seed"] for s in seeds_list]
    if len(set(seed_ids)) != len(seed_ids):
        raise Fail(f"duplicate CEM eval seeds in {tag}")
    ok6 = (len(seeds_list) == 6 and set(seed_ids) == set(range(42, 48))
           and all(s["budget_episodes"] == 50 for s in seeds_list))
    ok3 = (regime == "original" and model in ("lewm", "pldm") and len(seeds_list) == 3
           and set(seed_ids) == {42, 43, 44} and all(s["budget_episodes"] == 100 for s in seeds_list))
    if not (ok6 or ok3):
        raise Fail(f"unexpected CEM seed/budget layout in {tag}")
    agg = scores.get("cem")
    if agg is not None:
        tot = sum(s["budget_episodes"] for s in seeds_list)
        m = sum(s["success_rate_percent"] * s["budget_episodes"] for s in seeds_list) / tot
        if abs(m - agg) > 1e-6:
            raise Fail(f"CEM per-seed mean != aggregate in {tag}: {m} vs {agg}")


def validate_replicates(row):
    """Structural checks on one row's per-training-run replicates."""
    reps = row.get("replicates")
    if reps is None:
        return
    if not isinstance(reps, list):
        raise Fail(f"malformed replicates in {row['id']}")
    if not reps:
        if any(v is not None for v in row.get("scores", {}).values()):
            raise Fail(f"empty replicates would hide measured scores in {row['id']}")
        return
    seeds, checkpoints = [], []
    for rep in reps:
        if not isinstance(rep, dict):
            raise Fail(f"non-dict replica in {row['id']}")
        seed, ckpt = rep.get("training_seed"), rep.get("checkpoint_sha256")
        if seed is None or not ckpt:
            raise Fail(f"replica missing training_seed/checkpoint_sha256 in {row['id']}")
        seeds.append(seed)
        checkpoints.append(ckpt)
        tag = f"{row['id']}[training_seed {seed}]"
        scores = rep.get("scores")
        if not isinstance(scores, dict):
            raise Fail(f"replica without scores in {tag}")
        _check_scores(tag, scores)
        if any(scores.get(k) is not None for k in ALL if k != "cem"):
            split = rep.get("evaluation_split", row.get("evaluation_split"))
            if split is not None and split != "development":
                raise Fail(f"non-Development ICL replica in {tag}")
        tracks = rep.get("speed_tracks")
        if tracks:
            if set(tracks) != set(SPEED_TRACKS):
                raise Fail(f"incomplete Speed distributions in {tag}")
            for track, ts in tracks.items():
                _check_scores(f"{tag} {track}", ts)
        _check_cem_per_seed(tag, scores, rep.get("cem_per_seed") or [],
                            row["regime"], row["model"])
    if len(set(seeds)) != len(seeds):
        raise Fail(f"duplicate training_seed in replicates of {row['id']}")
    if len(set(checkpoints)) != len(checkpoints):
        raise Fail(f"duplicate checkpoint_sha256 in replicates of {row['id']}")


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
        _check_scores(rid, r["scores"])
        _check_cem_per_seed(rid, r["scores"], r.get("cem_per_seed") or [], r["regime"], r["model"])
        if r["measurement_status"] == "not_measured" and r["scores"]["main"] is not None:
            raise Fail(f"not_measured row carries main score: {rid}")
        validate_replicates(r)
        if "statistics" in r and r["statistics"] != aggregate(r):
            raise Fail(f"stored statistics do not match training replicates in {rid}")
    expected = {f"{t}/{m}/{reg}" for t in TASK_ORDER for m in MODEL_ORDER
                for reg in (("original", "scratch", "frozen") if m == "dinowm"
                            else ("original", "scratch", "joint", "frozen"))}
    if not expected <= seen:
        raise Fail(f"missing declared configurations: {sorted(expected - seen)}")
    return rows


# ---------------------------------------------------------------- rendering

def scheme_label(row):
    label = REGIME_ZH[row["regime"]]
    if row["regime"] == "projected":
        label += "‡"
    if row["measurement_status"] == "not_measured":
        label += "（仅 CEM）" if row["scores"]["cem"] is not None else "（未报告）"
    if row["task"] == "action_delay" and row["regime"] == "original":
        label += "†"  # H3 original Delay reference, not a native H7 counterpart
    return label


def stat_cell(row, key, st):
    if st is None or st["mean"] is None:
        if key == "joint" and row["task"] in JOINT_NA_TASKS:
            return "N/A"
        return "—"
    fmt = f3 if key in RATIO else f2
    text = fmt(st["mean"])
    if st["n"] > 1 and st["std"] is not None:
        text += " ± " + fmt(st["std"])
    return text


def plain_stat(key, st):
    if st is None or st["mean"] is None:
        return "—"
    return f3(st["mean"]) if key in RATIO else f2(st["mean"])


def _markdown_table(header, body):
    align = ["---" if h in ("模型", "方案", "任务", "训练数据", "评测条件") else "---:" for h in header]
    return ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"] + body


def render_overview(rows):
    """Eleven model/scheme rows, with nine task cells (mean ICL / mean CEM)."""
    header = ["模型", "方案"] + [f"[{TASK_ZH[t]}](#task-{t.replace('_', '-')})" for t in TASK_ORDER]
    body = []
    for (model, regime), tasks in overview_groups(rows):
        cells = []
        for t in TASK_ORDER:
            metrics, agg = display_stats(tasks[t])
            icl = f2(metrics["main"]["mean"])
            if t == "action_delay" and regime == "original" and icl != "—":
                icl += "†"
            cem = f2(agg["scores"]["cem"]["mean"])
            cells.append("—" if icl == "—" and cem == "—" else f"{icl} / {cem}")
        body.append("| " + " | ".join([MODEL_ZH[model], REGIME_ZH[regime]] + cells) + " |")
    return "\n".join(_markdown_table(header, body))


def render_detail(task, rows):
    """One table per task; Speed conditions share the same original-environment CEM."""
    task_rows = ordered_current(rows, task)
    keys = [k for k in DISPLAY if not (k == "joint" and task in JOINT_NA_TASKS)
            and not (k == "history" and task == "speed")]
    header = (["模型", "方案"] + (["评测条件"] if task == "speed" else [])
              + ["n(ICL)", "n(CEM)"] + [DISPLAY_LABEL[k] for k in keys])
    body = []
    for r in task_rows:
        metrics, agg = display_stats(r)
        if not any(st["n"] for st in metrics.values()):
            continue
        if task == "speed":
            # The overview uses unseen interpolation. Put it first and attach the
            # shared CEM result to that row only; other conditions are not new CEM runs.
            tracks = [SPEED_MAIN_TRACK] + [t for t in SPEED_TRACKS if t != SPEED_MAIN_TRACK]
            for track in tracks:
                st = with_response(agg["speed_tracks"].get(track) or {})
                st["cem"] = agg["scores"]["cem"]
                shared_cem = track != SPEED_MAIN_TRACK
                body.append("| " + " | ".join(
                    [MODEL_ZH[r["model"]], scheme_label(r), TRACK_ZH[track],
                     str(st.get("main", {"n": 0})["n"]),
                     "同组" if shared_cem else str(st["cem"]["n"])]
                    + ["同组" if k == "cem" and shared_cem else stat_cell(r, k, st.get(k))
                       for k in keys]) + " |")
            continue
        body.append("| " + " | ".join(
            [MODEL_ZH[r["model"]], scheme_label(r), str(metrics["main"]["n"]),
             str(agg["scores"]["cem"]["n"])]
            + [stat_cell(r, k, metrics.get(k)) for k in keys]) + " |")
    return "\n".join(_markdown_table(header, body))


def scale_label(row):
    count = row.get("training_pair_count")
    label = {2048: "2k", 8192: "8k", 32768: "32k", 10000: "10k 独立来源"}.get(
        count, f"{count:,}" if count is not None else "—")
    if row["task"] == "portal_exit" and count == 32768 \
            and "coverage" in (row.get("training_data_version") or ""):
        label = "32k 覆盖扩展"
    return label


def scaling_rows(rows):
    """Small-vs-large Scratch pairs for six tasks plus the LeWM strength Joint series."""
    by_id = {r["id"]: r for r in rows}
    selected = []
    for t in SCALING_TASKS:
        for model in MODEL_ORDER:
            small = [r for r in rows if r["task"] == t and r["model"] == model
                     and r["regime"] == "scratch" and r.get("comparison_variant")]
            large = [by_id.get(f"{t}/{model}/scratch")]
            if len(small) != 1 or large != [r for r in large if r]:
                raise Fail(f"expected exactly one scale variant and one ordinary row for {t}/{model}/scratch")
            selected.extend(sorted(small + large, key=lambda r: r["training_pair_count"]))
    for rid in LEWM_STRENGTH_JOINT_IDS:
        if rid not in by_id:
            raise Fail(f"missing LeWM strength Joint scale row {rid}")
        selected.append(by_id[rid])
    return selected


def render_scaling(rows):
    header = ["任务", "模型", "方案", "训练数据", "n", "ICL 主分↑", "响应分↑", "Joint↑", "Gain≈1", "CEM↑"]
    body = []
    for r in scaling_rows(rows):
        metrics, agg = display_stats(r)
        body.append("| " + " | ".join(
            [TASK_ZH[r["task"]], MODEL_ZH[r["model"]], REGIME_ZH[r["regime"]], scale_label(r),
             f"{metrics['main']['n']}/{agg['scores']['cem']['n']}", stat_cell(r, "main", metrics["main"])]
            + [stat_cell(r, k, metrics[k]) for k in ("response", "joint", "gain", "cem")]) + " |")
    return "\n".join(_markdown_table(header, body))


def render_historical(rows):
    """Appendix-only legacy DINO-WM projected-initialization rows with full statistics."""
    projected = sorted((r for r in rows if r["regime"] == "projected"),
                       key=lambda r: TASK_ORDER.index(r["task"]))
    header = ["任务", "模型", "方案", "n(ICL)", "n(CEM)"] + METRIC_HDR
    body = []
    for r in projected:
        metrics, agg = display_stats(r)
        body.append("| " + " | ".join(
            [TASK_ZH[r["task"]], MODEL_ZH[r["model"]], scheme_label(r),
             str(metrics["main"]["n"]), str(agg["scores"]["cem"]["n"])]
            + [stat_cell(r, k, metrics.get(k)) for k in ALL]) + " |")
    return "\n".join(_markdown_table(header, body))


def render_decision():
    """Simulator-based action selection is reported separately from latent scores."""
    evidence = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    header = ["模型", "方案", "动作 regret↓", "正确历史收益↑（95%区间）", "真实未来编码 regret↓"]
    body = []
    rows = sorted(evidence["rows"], key=lambda r: (
        MODEL_ORDER.index(r["training_comparison_id"].split("/")[1]),
        REGIME_ORDER.index(r["training_comparison_id"].split("/")[2])))
    for row in rows:
        _, model, regime = row["training_comparison_id"].split("/")
        lo, hi = row["history_benefit_ci95"]
        benefit = f"{row['history_benefit']:.3f} [{lo:.3f}, {hi:.3f}]"
        body.append("| " + " | ".join([
            MODEL_ZH[model], REGIME_ZH[regime], f3(row["correct_regret"]),
            benefit, f3(row["encoded_true_future_regret"])]) + " |")
    return "\n".join(_markdown_table(header, body))


def render_curves():
    """Keep diagnostic re-evaluations separate from archived benchmark scores."""
    evidence = json.loads(CURVES_PATH.read_text(encoding="utf-8"))
    indexed = {(r["training_comparison_id"], r["epoch"], r["split"]): r
               for r in evidence["rows"]}
    header = ["任务", "方案", "Epoch", "训练主分↑", "Dev 主分↑",
              "训练响应分↑", "Dev 响应分↑", "Dev Gain≈1"]
    body = []
    for task in ("action_strength", "motion_damping"):
        for regime in ("joint", "frozen"):
            for epoch in evidence["epochs"]:
                rid = f"{task}/pldm/{regime}"
                train, dev = (indexed[rid, epoch, split] for split in ("training", "development"))
                body.append("| " + " | ".join([
                    TASK_ZH[task], REGIME_ZH[regime], str(epoch),
                    f2(train["main_percent"]), f2(dev["main_percent"]),
                    f2(train["response_score"]), f2(dev["response_score"]),
                    f3(dev["gain"])]) + " |")
    return "\n".join(_markdown_table(header, body))


def build_blocks(rows):
    blocks = {"OVERVIEW": render_overview(rows),
              "DECISION": render_decision(),
              "CURVES": render_curves(),
              "SCALING": render_scaling(rows),
              "HISTORICAL": render_historical(rows)}
    for task, marker in DETAIL_MARKERS.items():
        blocks[marker] = render_detail(task, rows)
    return blocks


# ---------------------------------------------------------------- CSV twin

def _seeds_field(agg):
    return ";".join(str(s) for s in agg["seeds"])


def write_csv(rows):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CSV_COLS)
        for r in rows:
            agg = aggregate(r)
            metrics, _ = display_stats(r)
            cells = [r["id"], r["task"], r["model"], r["regime"],
                     r.get("training_data_version"), r.get("training_pair_count"),
                     metrics["main"]["n"], agg["scores"]["cem"]["n"], _seeds_field(agg)]
            exported = with_response(agg["scores"])
            for m in EXPORTED:
                st = exported[m]
                cells += [st["mean"], st["std"], st["n"]]
            cells.append(json.dumps(agg["speed_tracks"], sort_keys=True,
                                    ensure_ascii=False, separators=(",", ":"))
                         if agg["speed_tracks"] else "")
            w.writerow(cells)


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _compare_stat(parsed, expected, tag):
    if not isinstance(parsed, dict) or set(parsed) != {"mean", "std", "n"}:
        raise Fail(f"CSV speed-track stats shape mismatch in {tag}")
    if parsed["n"] != expected["n"] or not _close(parsed["mean"], expected["mean"]) \
            or not _close(parsed["std"], expected["std"]):
        raise Fail(f"CSV/JSON mismatch {tag}: {parsed} vs {expected}")


def _compare_tracks(parsed, expected, rid):
    if not isinstance(parsed, dict) or set(parsed) != set(expected):
        raise Fail(f"CSV speed-track set mismatch in {rid}")
    for track in expected:
        if set(parsed[track]) != set(expected[track]):
            raise Fail(f"CSV speed-track metrics mismatch in {rid} {track}")
        for m, st in expected[track].items():
            _compare_stat(parsed[track][m], st, f"{rid} {track} {m}")


def validate_csv(doc, rows):
    if not CSV_PATH.exists():
        raise Fail("CSV twin missing")
    with open(CSV_PATH, newline="") as f:
        rd = list(csv.reader(f))
    if not rd or rd[0] != CSV_COLS:
        raise Fail("CSV header mismatch")
    body = rd[1:]
    if len(body) != len(rows):
        raise Fail(f"CSV row count {len(body)} != JSON rows {len(rows)}")
    for r, c in zip(rows, body):
        rid = r["id"]
        if c[0] != rid or c[1] != r["task"] or c[2] != r["model"] or c[3] != r["regime"]:
            raise Fail(f"CSV row identity mismatch near {c[0]}")
        for key in ("training_data_version", "training_pair_count"):
            expected_value = "" if r.get(key) is None else str(r[key])
            if c[CSV_COLS.index(key)] != expected_value:
                raise Fail(f"CSV/JSON mismatch {key} in {rid}")
        agg = aggregate(r)
        metrics, _ = display_stats(r)
        if c[CSV_COLS.index("n_icl")] != str(metrics["main"]["n"]) \
                or c[CSV_COLS.index("n_cem")] != str(agg["scores"]["cem"]["n"]):
            raise Fail(f"CSV/JSON mismatch replicate counts in {rid}")
        if c[CSV_COLS.index("training_seeds")] != _seeds_field(agg):
            raise Fail(f"CSV/JSON mismatch training_seeds in {rid}")
        exported = with_response(agg["scores"])
        for m in EXPORTED:
            st = exported[m]
            mean_c, std_c, n_c = (c[CSV_COLS.index(f"score_{m}_{k}")] for k in ("mean", "std", "n"))
            if n_c != str(st["n"]):
                raise Fail(f"CSV/JSON mismatch score_{m}_n in {rid}")
            if st["mean"] is None:
                if mean_c != "" or std_c != "":
                    raise Fail(f"CSV/JSON mismatch score_{m} in {rid}")
            elif not _close(float(mean_c), st["mean"]):
                raise Fail(f"CSV/JSON mismatch score_{m}_mean in {rid}")
            if st["std"] is None:
                if std_c != "":
                    raise Fail(f"CSV/JSON mismatch score_{m}_std in {rid}")
            elif not _close(float(std_c), st["std"]):
                raise Fail(f"CSV/JSON mismatch score_{m}_std in {rid}")
        track_cell = c[CSV_COLS.index("speed_tracks_json")]
        if track_cell == "":
            if agg["speed_tracks"]:
                raise Fail(f"CSV missing aggregated Speed tracks in {rid}")
        else:
            try:
                parsed = json.loads(track_cell)
            except json.JSONDecodeError:
                raise Fail(f"CSV speed_tracks_json is not valid JSON in {rid}")
            _compare_tracks(parsed, agg["speed_tracks"], rid)


# ---------------------------------------------------------------- doc splicing

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
    blocks = build_blocks(rows)

    texts = {}
    for path in sorted({DOC, APPENDIX_DOC}):
        if not path.exists():
            raise Fail(f"benchmark doc missing: {path}")
        text = path.read_text(encoding="utf-8")
        for m in MARKERS:
            if marker_doc(m) != path:
                continue
            begin, end = f"<!-- BEGIN TRAINING_COMPARISON_{m} -->", f"<!-- END TRAINING_COMPARISON_{m} -->"
            if text.count(begin) != 1 or text.count(end) != 1:
                raise Fail(f"marker {m}: found {text.count(begin)} begin / {text.count(end)} end tags")
        texts[path] = text

    if not args.check:
        write_csv(rows)
    # CSV must be validated in both modes (generated above in run mode)
    validate_csv(doc, rows)

    if args.check:
        for m in MARKERS:
            if extract(texts[marker_doc(m)], m).strip() != blocks[m].strip():
                raise Fail(f"--check: block {m} is stale or does not match published JSON")
        print(f"OK: {len(rows)} rows aggregated; JSON/CSV consistent; "
              f"all {len(MARKERS)} doc blocks are complete and up to date")
        return

    for path, text in texts.items():
        new = text
        for m in MARKERS:
            if marker_doc(m) == path:
                new = splice(new, m, blocks[m])
        if new != text:
            path.write_text(new, encoding="utf-8")
            print("updated comparison blocks in", path)
    print(f"CSV written: {CSV_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Fail as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
