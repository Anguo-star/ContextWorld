from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.compare_tworoom_e4_models import run


def _write_result(root: Path, name: str, successes: dict[tuple[str, str], bool]) -> Path:
    raw = root / f"{name}_raw.json"
    records = []
    for (evaluation_id, condition), success in successes.items():
        records.append(
            {
                "evaluation_id": evaluation_id,
                "eval_seed": 42,
                "condition": condition,
                "query_id": evaluation_id.split("-")[-1],
                "speed": 5.0,
                "template_id": "s0",
                "cem_seed": 7,
                "goal_state": [190.0, 190.0],
                "success": success,
                "final_distance": 10.0 if success else 30.0,
            }
        )
    raw.write_text(json.dumps({"records": records}), encoding="utf-8")
    correct = sum(value for (key, condition), value in successes.items() if condition == "correct")
    wrong = sum(value for (key, condition), value in successes.items() if condition == "wrong")
    summary = root / f"{name}.json"
    summary.write_text(
        json.dumps(
            {
                "protocol": {"raw_results": [str(raw)]},
                "aggregate": {
                    "correct_minus_wrong_success_rate_points": 50.0 * (correct - wrong)
                },
            }
        ),
        encoding="utf-8",
    )
    return summary


def test_cross_model_comparison_counts_paired_successes(tmp_path: Path) -> None:
    reference = _write_result(
        tmp_path,
        "reference",
        {
            ("e0-q0", "correct"): True,
            ("e0-q0", "wrong"): False,
            ("e1-q1", "correct"): False,
            ("e1-q1", "wrong"): False,
        },
    )
    candidate = _write_result(
        tmp_path,
        "candidate",
        {
            ("e0-q0", "correct"): True,
            ("e0-q0", "wrong"): False,
            ("e1-q1", "correct"): True,
            ("e1-q1", "wrong"): False,
        },
    )
    output = tmp_path / "comparison.json"
    result = run(Namespace(reference=reference, candidate=candidate, output=output))
    assert result["conditions"]["correct"]["candidate_only_successes"] == 1
    assert result["conditions"]["correct"]["reference_only_successes"] == 0
    assert result["conditions"]["correct"]["candidate_minus_reference_success_rate_points"] == 50.0
    assert result["conditions"]["correct"]["candidate_lower_final_distance_pairs"] == 1
    assert result["conditions"]["correct"]["by_seed"]["42"]["evaluations"] == 2
    assert (
        result["conditions"]["correct"]["by_paired_success_outcome"]
        ["candidate_only_success"]["evaluations"]
        == 1
    )
    assert result["context_effect"]["difference_in_differences_points"] == 50.0
