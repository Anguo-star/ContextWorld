from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _write_result(root: Path, name: str, counts: tuple[int, int, int]) -> None:
    root.mkdir(parents=True)
    catalog = {
        str(seed): {"row_indices": list(range(100))}
        for seed in (42, 43, 44)
    }
    (root / "query_catalog.json").write_text(
        json.dumps(catalog),
        encoding="utf-8",
    )
    rows = []
    for seed, count in zip((42, 43, 44), counts):
        outcomes = [True] * count + [False] * (100 - count)
        rows.append(
            {
                "eval_seed": seed,
                "episode_successes": outcomes,
            }
        )
    total = sum(counts)
    payload = {
        "task": "reacher",
        "protocol": {"eval_seeds": [42, 43, 44], "num_eval_per_seed": 100},
        "models": [
            {
                "model": name,
                "checkpoint": f"/{name}.pt",
                "checkpoint_sha256": name.ljust(64, "0")[:64],
                "seeds": rows,
                "aggregate": {
                    "success_count": total,
                    "evaluation_count": 300,
                },
            }
        ],
    }
    (root / "aggregate.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_reacher_retention_uses_shared_queries_and_three_candidates(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidates = [tmp_path / f"candidate-{index}" for index in range(3)]
    _write_result(baseline, "baseline", (57, 54, 50))
    _write_result(candidates[0], "candidate-1", (51, 55, 49))
    _write_result(candidates[1], "candidate-2", (55, 54, 48))
    _write_result(candidates[2], "candidate-3", (49, 55, 33))
    output = tmp_path / "summary.json"
    command = [
        sys.executable,
        "scripts/aggregate_reacher_original_task_retention.py",
        "--baseline",
        str(baseline),
    ]
    for candidate in candidates:
        command.extend(["--candidate", str(candidate)])
    command.extend(["--output", str(output)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["query_catalog"]["identical_across_all_results"] is True
    assert [row["success_delta"] for row in payload["comparisons"]] == [
        -6,
        -4,
        -24,
    ]
    assert [row["passed"] for row in payload["comparisons"]] == [
        True,
        True,
        False,
    ]
    assert payload["passed"] is False
