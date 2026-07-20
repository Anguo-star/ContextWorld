import importlib.util
import json
from pathlib import Path


def _load_runner():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/run_tworoom_history3_eval.py"
    spec = importlib.util.spec_from_file_location("history3_eval_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_planning_seed_aggregation_preserves_50x6_protocol(tmp_path: Path) -> None:
    runner = _load_runner()
    runs = []
    seeds = (42, 43, 44, 45, 46, 47)
    success_counts = (20, 25, 30, 35, 40, 45)
    for seed, successes in zip(seeds, success_counts):
        path = tmp_path / f"raw_s{seed}.json"
        payload = {
            "status": "passed",
            "policy": {"checkpoint_sha256": "frozen"},
            "protocol": {"eval_seed": seed},
            "aggregate": {
                "evaluations": 50,
                "successes": successes,
                "scenario_balanced_success_rate": float(2 * successes),
                "factor_readback_passed": True,
            },
            "scenarios": [
                {
                    "scenario": "speed_a",
                    "evaluations": 50,
                    "successes": [1] * successes + [0] * (50 - successes),
                    "success_rate": float(2 * successes),
                }
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        runs.append({"seed": seed, "output": path})

    summary = tmp_path / "summary.json"
    result = runner._aggregate_planning_runs(
        {
            "experiment_id": "E3",
            "family": "speed",
            "profile": "full",
            "num_eval_per_seed": 50,
            "eval_seeds": list(seeds),
            "runs": runs,
            "summary": summary,
        }
    )

    assert result["protocol"]["num_eval_per_seed"] == 50
    assert result["protocol"]["total_evaluations"] == 300
    assert result["aggregate"]["evaluations"] == 300
    assert result["aggregate"]["success_rate"] == 65.0
    assert result["aggregate"]["pooled_success_rate"] == 65.0
    assert result["aggregate"]["seed_success_rates"] == {
        "42": 40.0,
        "43": 50.0,
        "44": 60.0,
        "45": 70.0,
        "46": 80.0,
        "47": 90.0,
    }
    assert summary.is_file()
