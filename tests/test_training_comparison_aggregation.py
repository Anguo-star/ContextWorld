"""Aggregation semantics of scripts/render_training_comparison.py.

The published study rows now carry per-training-run ``replicates``; the renderer
must aggregate them (mean, sample SD with ddof=1, per-metric counts), keep
distinct recipes / data scales separate, never count CEM evaluation seeds as
training replicates, and keep the aggregated CSV twin honest.  These tests run
against synthetic replicate sets plus the real published JSON; the docs
themselves are only touched through throwaway copies in tmp_path.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "scripts" / "render_training_comparison.py"
SOURCE = ROOT / "docs" / "research" / "data" / "icl_training_study_v2.json"


def _load_renderer():
    spec = importlib.util.spec_from_file_location("render_training_comparison_aggregation", RENDERER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RTC = _load_renderer()


@pytest.fixture(autouse=True)
def _restore_renderer_paths():
    saved = (RTC.DOC, RTC.APPENDIX_DOC, RTC.JSON_PATH, RTC.CSV_PATH)
    yield
    RTC.DOC, RTC.APPENDIX_DOC, RTC.JSON_PATH, RTC.CSV_PATH = saved



@pytest.fixture(scope="module")
def rtc():
    return RTC


@pytest.fixture(scope="module")
def published_rows(rtc):
    doc = json.loads(SOURCE.read_text(encoding="utf-8"))
    return rtc.validate(doc)


# ----------------------------------------------------------------- test data

def _scores(**overrides):
    scores = {k: None for k in RTC.ALL}
    scores.update({"main": 50.0, "worst": 25.0, "history": 40.0, "switch": 60.0,
                   "gain": 0.25, "alignment": 0.5, "nre": 0.75, "calibrated": 55.0,
                   "cem": 80.0})
    scores.update(overrides)
    return scores


def _track_scores(main):
    scores = {k: None for k in RTC.ALL}
    scores.update({"main": main, "worst": main / 2.0})
    return scores


def _speed_tracks(unseen_main=50.0, others=(20.0, 10.0, 12.0)):
    mains = dict(zip([t for t in RTC.SPEED_TRACKS if t != "unseen_interpolation"], others))
    mains["unseen_interpolation"] = unseen_main
    return {track: _track_scores(main) for track, main in mains.items()}


def _cem_per_seed(cem, eval_seeds=range(42, 48), budget=50):
    return [{"eval_seed": s, "budget_episodes": budget, "success_rate_percent": cem}
            for s in eval_seeds]


def _replica(seed, scores, tracks=None, checkpoint=None, with_cem_seeds=True):
    return {
        "training_seed": seed,
        "checkpoint_sha256": checkpoint or f"ckpt-{seed}",
        "result_sha256": f"result-{seed}",
        "evaluation_split": "development",
        "scores": scores,
        "speed_tracks": tracks,
        "cem_per_seed": (_cem_per_seed(scores["cem"]) if with_cem_seeds and scores.get("cem") is not None else []),
        "cem_budget": "6x50",
        "cem_evaluation_count": 300,
    }


def _row(row_id, *, task="contact_friction", model="lewm", regime="scratch",
         scores=None, replicates=None, **extra):
    row = {
        "id": row_id,
        "task": task,
        "model": model,
        "regime": regime,
        "measurement_status": "available",
        "scores": scores if scores is not None else _scores(),
        "training_seed": 3072,
        "training_pair_count": None,
        "training_data_version": "ContextWorld-v1",
        "comparison_variant": None,
    }
    if replicates is not None:
        row["replicates"] = replicates
    row.update(extra)
    return row


# ------------------------------------------------------- aggregation semantics

def test_three_replicates_produce_mean_and_sample_sd(rtc):
    row = _row("contact_friction/lewm/scratch", replicates=[
        _replica(3072, _scores(main=50.0, cem=80.0)),
        _replica(3073, _scores(main=60.0, cem=90.0)),
        _replica(3074, _scores(main=70.0, cem=100.0)),
    ])
    agg = rtc.aggregate(row)
    main = agg["scores"]["main"]
    assert main["n"] == 3
    assert main["mean"] == pytest.approx(60.0)
    assert main["std"] == pytest.approx(statistics.stdev([50.0, 60.0, 70.0]))  # ddof=1 -> 10.0
    assert agg["scores"]["cem"]["std"] == pytest.approx(10.0)
    assert agg["from_replicates"] is True
    assert agg["seeds"] == [3072, 3073, 3074]


def test_single_replicate_and_fallback_rows_have_no_sd(rtc):
    one = rtc.aggregate(_row("t/lewm/scratch", replicates=[_replica(3072, _scores(main=42.0))]))
    assert one["scores"]["main"] == {"mean": 42.0, "std": None, "n": 1}

    fallback = rtc.aggregate(_row("t/lewm/joint", scores=_scores(main=97.5, cem=88.0)))
    assert fallback["from_replicates"] is False
    assert fallback["scores"]["main"] == {"mean": 97.5, "std": None, "n": 1}
    assert fallback["scores"]["cem"]["n"] == 1
    assert fallback["seeds"] == [3072]

    empty = rtc.aggregate(_row("t/lewm/frozen", scores=_scores(main=None, cem=None), replicates=[]))
    assert empty["scores"]["main"] == {"mean": None, "std": None, "n": 0}


def test_cem_counts_training_replicates_not_evaluation_seeds(rtc):
    # One training run evaluated on six evaluation seeds is n=1, not n=6.
    single = rtc.aggregate(_row("t/lewm/scratch", replicates=[
        _replica(3072, _scores(cem=84.0))]))
    assert single["scores"]["cem"] == {"mean": 84.0, "std": None, "n": 1}

    # Three training runs: the mean averages the three per-checkpoint aggregates,
    # and the SD is across training runs -- never across the 18 evaluation seeds.
    row = _row("t/lewm/scratch", replicates=[
        _replica(3072, _scores(cem=80.0)),
        _replica(3073, _scores(cem=86.0)),
        _replica(3074, _scores(cem=90.0)),
    ])
    cem = rtc.aggregate(row)["scores"]["cem"]
    assert cem["n"] == 3
    assert cem["mean"] == pytest.approx(statistics.fmean([80.0, 86.0, 90.0]))
    assert cem["std"] == pytest.approx(statistics.stdev([80.0, 86.0, 90.0]))
    pooled = [s["success_rate_percent"] for r in row["replicates"] for s in r["cem_per_seed"]]
    assert len(pooled) == 18
    assert cem["n"] != len(pooled)  # evaluation seeds are not training replicates
    assert cem["std"] != pytest.approx(statistics.stdev(pooled))  # and no pooled-seed SD


def test_missing_metrics_are_counted_independently_and_never_zero_filled(rtc):
    row = _row("t/lewm/scratch", replicates=[
        _replica(3072, _scores(main=40.0, gain=None)),
        _replica(3073, _scores(main=60.0, gain=1.0, cem=None)),
        _replica(3074, _scores(main=80.0, gain=1.2)),
    ])
    agg = rtc.aggregate(row)
    assert agg["scores"]["main"]["mean"] == 60.0
    assert agg["scores"]["main"]["std"] == pytest.approx(20.0)
    assert agg["scores"]["main"]["n"] == 3
    assert agg["scores"]["gain"]["n"] == 2
    assert agg["scores"]["gain"]["mean"] == pytest.approx(1.1)
    assert agg["scores"]["gain"]["std"] == pytest.approx(statistics.stdev([1.0, 1.2]))
    # CEM is missing in one replicate: it is counted independently of main.
    assert agg["scores"]["cem"]["n"] == 2
    assert agg["scores"]["cem"]["mean"] == 80.0
    assert agg["scores"]["cem"]["std"] == 0.0
    assert agg["scores"]["joint"] == {"mean": None, "std": None, "n": 0}  # absent everywhere


def test_duplicate_training_seed_or_checkpoint_rejected(rtc):
    dup_seed = _row("t/lewm/scratch", replicates=[
        _replica(3072, _scores()), _replica(3072, _scores(), checkpoint="other")])
    with pytest.raises(rtc.Fail, match="duplicate training_seed"):
        rtc.validate_replicates(dup_seed)

    dup_ckpt = _row("t/lewm/scratch", replicates=[
        _replica(3072, _scores(), checkpoint="same"),
        _replica(3073, _scores(), checkpoint="same")])
    with pytest.raises(rtc.Fail, match="duplicate checkpoint"):
        rtc.validate_replicates(dup_ckpt)


def test_nonfinite_or_out_of_range_replicate_values_rejected(rtc):
    with pytest.raises(rtc.Fail, match="nonfinite"):
        rtc.validate_replicates(_row("t/lewm/scratch", replicates=[
            _replica(3072, _scores(main=float("nan")))]))
    with pytest.raises(rtc.Fail, match="out of range"):
        rtc.validate_replicates(_row("t/lewm/scratch", replicates=[
            _replica(3072, _scores(main=150.0))]))
    with pytest.raises(rtc.Fail, match="nonfinite"):
        rtc.validate_replicates(_row("t/lewm/scratch", replicates=[
            _replica(3072, _scores(gain=float("inf")))]))
    with pytest.raises(rtc.Fail, match="Speed distributions"):
        rtc.validate_replicates(_row("t/lewm/scratch", task="speed", replicates=[
            _replica(3072, _scores(), tracks={"unseen_interpolation": _track_scores(50.0)})]))


def test_replica_cem_per_seed_mean_must_match_its_aggregate(rtc):
    bad = _replica(3072, _scores(cem=91.0))
    bad["cem_per_seed"] = _cem_per_seed(90.0)
    with pytest.raises(rtc.Fail, match="CEM per-seed mean"):
        rtc.validate_replicates(_row("t/lewm/scratch", replicates=[bad]))
    rtc.validate_replicates(_row("t/lewm/scratch", replicates=[
        _replica(3072, _scores(cem=90.0))]))  # consistent replica passes


# ------------------------------------------------------- separation and averages

def test_distinct_data_scale_rows_stay_separate(rtc):
    small = _row("action_strength/lewm/scratch/scale_2k", task="action_strength",
                 scores=_scores(main=40.0, cem=60.0),
                 comparison_variant="scale_2k", training_pair_count=2048)
    large = _row("action_strength/lewm/scratch", task="action_strength",
                 scores=_scores(main=80.0, cem=70.0), training_pair_count=32768)
    selected = rtc.scaling_rows(_scaling_fixture(small, large))
    ids = [r["id"] for r in selected]
    assert ids.count("action_strength/lewm/scratch/scale_2k") == 1
    assert ids.count("action_strength/lewm/scratch") == 1

    text = rtc.render_scaling(_scaling_fixture(small, large))
    small_line = [l for l in text.splitlines() if "| 2k |" in l][0]
    large_line = [l for l in text.splitlines() if "| 32k |" in l][0]
    assert "40.00" in small_line and "80.00" in large_line  # never pooled into one mean
    assert "60.00" in small_line and "70.00" in large_line


def _scaling_fixture(*replacements):
    """A complete synthetic scaling row set with selected rows replaced."""
    rows = []
    for task in ("action_strength", "contact_friction", "motion_damping",
                 "robot_arm_mass", "cube_gripper_carry", "portal_exit"):
        for model in ("lewm", "pldm", "dinowm"):
            for regime, count, variant in (("scratch", 2048, "scale_2k"),
                                           ("scratch", 32768, None)):
                row_id = f"{task}/{model}/{regime}" + (f"/{variant}" if variant else "")
                rows.append(_row(row_id, task=task, model=model, regime=regime,
                                 scores=_scores(main=50.0, cem=70.0),
                                 training_pair_count=count, comparison_variant=variant))
        if task == "action_strength":
            for count, variant in ((2048, "scale_2k"), (8192, "scale_8k"), (32768, None)):
                row_id = f"action_strength/lewm/joint" + (f"/{variant}" if variant else "")
                rows.append(_row(row_id, task=task, model="lewm", regime="joint",
                                 scores=_scores(main=50.0, cem=70.0),
                                 training_pair_count=count, comparison_variant=variant))
    by_id = {r["id"]: r for r in rows}
    for r in replacements:
        by_id[r["id"]] = r
    return list(by_id.values())


def test_overview_reports_all_nine_tasks_without_a_composite_score(rtc, published_rows):
    text = rtc.render_overview(published_rows)
    lines = [l for l in text.splitlines() if l.startswith("|")][2:]
    assert len(lines) == 11
    assert all(len(l.split("|")) == 13 for l in lines)
    assert "ICL Avg" not in text
    assert "†" in "\n".join(lines)
    assert "历史转换初始化" not in text


def test_speed_four_tracks_are_distributions_not_tasks(rtc):
    row = _row("speed/lewm/scratch", task="speed", replicates=[
        _replica(3072, _scores(main=90.0), tracks=_speed_tracks(unseen_main=90.0, others=(30.0, 10.0, 12.0))),
        _replica(3073, _scores(main=96.0), tracks=_speed_tracks(unseen_main=96.0, others=(32.0, 11.0, 14.0))),
        _replica(3074, _scores(main=99.0), tracks=_speed_tracks(unseen_main=99.0, others=(34.0, 12.0, 16.0))),
    ])
    agg = rtc.aggregate(row)
    assert set(agg["speed_tracks"]) == set(rtc.SPEED_TRACKS)
    unseen = agg["speed_tracks"]["unseen_interpolation"]["main"]
    low = agg["speed_tracks"]["extrapolation_low"]["main"]
    assert unseen["n"] == 3 and unseen["mean"] == pytest.approx(95.0)
    assert low["mean"] == pytest.approx(11.0)  # tracks aggregate independently
    metrics, _ = rtc.display_stats(row)
    assert metrics["main"]["mean"] == pytest.approx(95.0)  # main report = unseen interpolation


def test_speed_detail_reports_tracks_separately_without_repeating_cem(rtc, published_rows):
    text = rtc.render_detail("speed", published_rows)
    lines = [l for l in text.splitlines() if l.startswith("|")]
    body = [l for l in lines[2:] if not l.startswith("| ---") and not l.startswith("| 模型")]
    summary = [l for l in body if len(l.split("|")) == len(lines[0].split("|"))]
    tracks = [l for l in body if l not in summary]
    current = rtc.ordered_current(published_rows, "speed")
    assert len(summary) == len(current) == 11  # one row per scheme, not per distribution
    assert len(tracks) == 4 * len(current)
    assert sum("CEM↑" in l for l in lines) == 1  # CEM appears exactly once
    for label in ("训练中已见速度", "未见速度插值", "低端外推", "高端外推"):
        assert any(label in l for l in tracks)
    # No seed ids leak into the rendered tables.
    assert not any("3072" in l or "3073" in l for l in lines)


def test_each_measured_current_row_renders_in_its_own_task_detail(rtc, published_rows):
    for task in rtc.TASK_ORDER:
        current = rtc.ordered_current(published_rows, task)
        current = [r for r in current if any(st["n"] for st in rtc.display_stats(r)[0].values())]
        text = rtc.render_detail(task, published_rows)
        width = len(text.splitlines()[0].split("|"))
        labels = [(l.split("|")[1].strip(), l.split("|")[2].strip())
                  for l in text.splitlines()
                  if l.startswith("|") and len(l.split("|")) == width and "---" not in l
                  and not l.startswith("| 模型")]
        assert labels == [(rtc.MODEL_ZH[r["model"]], rtc.scheme_label(r)) for r in current]
        assert "历史转换初始化" not in text  # projected rows only live in the appendix
        assert "2k" not in text and "32k" not in text  # scale variants stay in SCALING


def test_historical_projected_rows_are_isolated(rtc, published_rows):
    blocks = rtc.build_blocks(published_rows)
    assert "HISTORICAL" in rtc.APPENDIX_MARKERS
    assert "HISTORICAL" not in rtc.MAIN_MARKERS
    historical = [l for l in blocks["HISTORICAL"].splitlines() if l.startswith("|")][2:]
    assert len(historical) == 6
    assert all("DINO-WM" in l and "历史转换初始化‡" in l for l in historical)
    for marker in rtc.MAIN_MARKERS:
        assert "历史转换初始化" not in blocks[marker]


# ------------------------------------------------------------------- pipeline

def _sandbox(rtc, tmp_path, doc=None):
    """Temp copies of the published docs/JSON; renderer paths are rebound to them."""
    sandbox = tmp_path / "sandbox"
    (sandbox / "docs" / "research" / "data").mkdir(parents=True)
    (sandbox / "docs" / "reference").mkdir(parents=True)
    study = json.loads(SOURCE.read_text(encoding="utf-8"))
    if doc is not None:
        study = doc
        for row in study["rows"]:
            row.pop("statistics", None)  # synthetic replicates replace the stored source summaries
    (sandbox / "docs" / "research" / "data" / "icl_training_study_v2.json").write_text(
        json.dumps(study), encoding="utf-8")
    main_doc = sandbox / "docs" / "ContextWorld_ICL_Benchmark.md"
    appendix_doc = sandbox / "docs" / "reference" / "Benchmark_Result_Provenance.md"
    markers = [f"<!-- BEGIN TRAINING_COMPARISON_{m} -->\nplaceholder {m}\n<!-- END TRAINING_COMPARISON_{m} -->"
               for m in rtc.MAIN_MARKERS]
    appendix_markers = [f"<!-- BEGIN TRAINING_COMPARISON_{m} -->\nplaceholder {m}\n<!-- END TRAINING_COMPARISON_{m} -->"
                        for m in rtc.APPENDIX_MARKERS]
    main_doc.write_text("\n".join(markers) + "\n", encoding="utf-8")
    appendix_doc.write_text("\n".join(appendix_markers) + "\n", encoding="utf-8")
    csv_path = sandbox / "docs" / "research" / "data" / "icl_training_study_v2.csv"
    saved = (rtc.DOC, rtc.APPENDIX_DOC, rtc.JSON_PATH, rtc.CSV_PATH)
    rtc.DOC, rtc.APPENDIX_DOC = main_doc, appendix_doc
    rtc.JSON_PATH = sandbox / "docs" / "research" / "data" / "icl_training_study_v2.json"
    rtc.CSV_PATH = csv_path
    return sandbox, saved


def _run(rtc, *argv):
    saved_argv = sys.argv
    sys.argv = ["render_training_comparison.py", *argv]
    try:
        rtc.main()
    finally:
        sys.argv = saved_argv


def test_pipeline_writes_aggregated_csv_and_check_revalidates(rtc, tmp_path):
    study = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in study["rows"]}
    # Three synthetic replicates whose aggregate must differ from the representative row.
    rows["speed/lewm/scratch"]["replicates"] = [
        _replica(11, _scores(main=90.0, cem=80.0), tracks=_speed_tracks(90.0)),
        _replica(12, _scores(main=96.0, cem=84.0), tracks=_speed_tracks(96.0)),
        _replica(13, _scores(main=99.0, cem=88.0), tracks=_speed_tracks(99.0)),
    ]
    representative_main = rows["speed/lewm/scratch"]["scores"]["main"]
    representative_cem = rows["speed/lewm/scratch"]["scores"]["cem"]
    # A row without replicates keeps its published representative scores (n=1).
    rows["contact_friction/pldm/joint"].pop("replicates", None)

    _sandbox(rtc, tmp_path, doc=study)
    _run(rtc)
    with open(rtc.CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        body = list(reader)
    assert header == rtc.CSV_COLS
    assert len(body) == len(study["rows"])
    by_id = {c[0]: c for c in body}

    speed = by_id["speed/lewm/scratch"]
    ix = {name: header.index(name) for name in
          ("n_icl", "n_cem", "training_seeds", "score_main_mean", "score_main_std", "score_cem_mean")}
    assert speed[ix["score_main_mean"]] == str(statistics.fmean([90.0, 96.0, 99.0]))
    assert float(speed[ix["score_main_std"]]) == pytest.approx(statistics.stdev([90.0, 96.0, 99.0]))
    assert speed[ix["score_cem_mean"]] == str(statistics.fmean([80.0, 84.0, 88.0]))
    assert speed[ix["n_icl"]] == "3" and speed[ix["n_cem"]] == "3"
    assert speed[ix["training_seeds"]] == "11;12;13"
    assert float(speed[ix["score_main_mean"]]) != pytest.approx(representative_main)
    tracks = json.loads(speed[header.index("speed_tracks_json")])
    assert set(tracks) == set(rtc.SPEED_TRACKS)
    assert tracks["unseen_interpolation"]["main"]["mean"] == pytest.approx(95.0)
    assert tracks["extrapolation_low"]["main"]["n"] == 3

    fallback = by_id["contact_friction/pldm/joint"]
    assert fallback[header.index("score_main_mean")] == str(
        rows["contact_friction/pldm/joint"]["scores"]["main"])
    assert fallback[header.index("score_main_std")] == ""
    assert fallback[header.index("n_icl")] == "1"
    assert fallback[header.index("speed_tracks_json")] == ""

    _run(rtc, "--check")  # rendered output validates

    tampered = list(speed)
    tampered[header.index("score_main_mean")] = "1.0"
    body[body.index(speed)] = tampered
    with open(rtc.CSV_PATH, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(body)
    with pytest.raises(rtc.Fail, match="score_main_mean"):
        _run(rtc, "--check")


def test_check_mode_requires_every_marker(rtc, tmp_path):
    _sandbox(rtc, tmp_path)
    _run(rtc)
    text = rtc.DOC.read_text(encoding="utf-8").replace(
        "<!-- BEGIN TRAINING_COMPARISON_SCALING -->", "")
    rtc.DOC.write_text(text, encoding="utf-8")
    with pytest.raises(rtc.Fail, match="marker SCALING"):
        _run(rtc, "--check")


def test_empty_replicates_represent_unmeasured_configuration():
    row = {"id": "action_delay/lewm/joint", "scores": {"main": None, "cem": None}, "replicates": []}
    RTC.validate_replicates(row)
    assert RTC.aggregate(row)["scores"]["main"] == {"mean": None, "std": None, "n": 0}


def test_empty_replicates_cannot_hide_an_observed_score():
    row = {"id": "x", "scores": {"main": 80.0}, "replicates": []}
    with pytest.raises(RTC.Fail, match="hide measured"):
        RTC.validate_replicates(row)


def test_non_development_replica_is_rejected():
    row = {"id": "x", "model": "lewm", "regime": "scratch", "replicates": [
        {"training_seed": 1, "checkpoint_sha256": "abc", "evaluation_split": "test", "scores": {"main": 80.0}}
    ]}
    with pytest.raises(RTC.Fail, match="non-Development"):
        RTC.validate_replicates(row)


def test_nonfinite_cem_evaluation_cannot_pass_mean_validation():
    evaluations = [{"eval_seed": s, "budget_episodes": 50, "success_rate_percent": float("nan")} for s in range(42, 48)]
    with pytest.raises(RTC.Fail, match="per-evaluation CEM"):
        RTC._check_cem_per_seed("example", {"cem": 50}, evaluations, "scratch", "lewm")


def test_stored_statistics_must_agree_with_training_runs(rtc):
    doc = json.loads(SOURCE.read_text())
    doc["rows"][0]["statistics"]["scores"]["main"]["mean"] += 1
    with pytest.raises(rtc.Fail, match="stored statistics"):
        rtc.validate(doc)


def test_response_score_preserves_negative_values_missingness_and_repeat_spread(rtc):
    # These NRE values correspond to perfect, zero, and worse-than-zero responses.
    nre = rtc.stats([0.0, 1.0, 2.0, None])
    transformed = rtc.with_response({"nre": nre})["response"]
    assert transformed == {"mean": 0.0, "std": 100.0, "n": 3}
    assert rtc.with_response({"nre": rtc.stats([1.521])})["response"]["mean"] == pytest.approx(-52.1)
    assert rtc.with_response({})["response"] == {"mean": None, "std": None, "n": 0}
    # NRE already uses squared error; 0.25 must map to 75, not 93.75.
    assert rtc.with_response({"nre": rtc.stats([0.25])})["response"] == {"mean": 75.0, "std": None, "n": 1}
    assert nre == {"mean": 1.0, "std": 1.0, "n": 3}  # original statistics stay unchanged


def test_existing_decision_evidence_matches_comparison_checkpoints(published_rows):
    evidence = json.loads((ROOT / "docs/research/data/icl_action_selection_v1.json").read_text())
    indexed = {r["id"]: r for r in published_rows}
    assert evidence["pair_count"] == 256 and evidence["candidate_count"] == 21
    for row in evidence["rows"]:
        source = indexed[row["training_comparison_id"]]
        rep = next(r for r in source["replicates"] if r["checkpoint_sha256"] == row["checkpoint_sha256"])
        assert row["main_percent"] == pytest.approx(rep["scores"]["main"])
        assert row["history_benefit"] == pytest.approx(row["swapped_regret"] - row["correct_regret"])
        lo, hi = row["history_benefit_ci95"]
        assert lo <= row["history_benefit"] <= hi


def test_diagnostic_curves_keep_score_semantics_and_checkpoint_identity(published_rows):
    evidence = json.loads((ROOT / "docs/research/data/icl_training_dynamics_v1.json").read_text())
    indexed = {r["id"]: r for r in published_rows}
    rows = evidence["rows"]
    assert len(rows) == 16
    assert len({(r["training_comparison_id"], r["epoch"], r["split"]) for r in rows}) == 16
    for row in rows:
        assert row["pair_count"] == (64 if row["split"] == "training" else 256)
        assert row["response_score"] == pytest.approx(100 * (1 - row["nre"]))
        assert row["amplitude_error"] + row["orthogonal_error"] == pytest.approx(row["nre"])
        assert row["orthogonal_error"] >= -1e-10
        if row["epoch"] == 10:
            source = indexed[row["training_comparison_id"]]
            assert row["checkpoint_sha256"] in {r["checkpoint_sha256"] for r in source["replicates"]}
        if row["epoch"] == 10 and row["split"] == "development":
            archived = indexed[row["training_comparison_id"]]["scores"]
            assert row["difference_from_published"]["main_percentage_points"] == pytest.approx(
                row["main_percent"] - archived["main"])
