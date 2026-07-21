from pathlib import Path

import pytest
import yaml

from scripts.build_tworoom_speed_next_latent_catalogs import (
    _assign_eval_seeds,
)
from scripts.eval_tworoom_speed_next_latent import (
    CONDITIONS,
    _scheduled_records,
    summarize_records,
)


EVAL_SEEDS = [42, 43, 44, 45, 46, 47]
ROOT = Path(__file__).resolve().parents[1]


def test_v4_protocol_freezes_independent_50_by_6() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs/benchmark/tworoom_speed_next_latent_v4.yaml"
        ).read_text(encoding="utf-8")
    )
    evaluation = config["evaluation"]
    assert config["status"] == (
        "preregistered_before_independent_catalog_generation_and_scoring"
    )
    assert evaluation["eval_seeds"] == EVAL_SEEDS
    assert evaluation["unique_queries_per_reference_speed_per_seed"] == 50
    assert evaluation["unique_queries_per_reference_speed"] == 300
    assert evaluation["expected_records_per_checkpoint_per_track"] == 2700
    assert sum(len(rows) for rows in config["models"].values()) == 7
    assert "inferred_speed" in config["metrics"]["excluded"]
    assert "projected_pixel_position" in config["metrics"]["excluded"]


def _unique_rows() -> list[dict]:
    rows = []
    for speed_index, speed in enumerate((3.0, 5.0, 7.0)):
        matching = CONDITIONS[speed_index]
        for seed_index, eval_seed in enumerate(EVAL_SEEDS):
            for evaluation_index in range(50):
                query_index = seed_index * 50 + evaluation_index
                for condition in CONDITIONS:
                    rows.append(
                        {
                            "query_id": f"q{speed:g}-{query_index:03d}",
                            "static_query_id": f"static-{query_index:03d}",
                            "template_id": f"template-{query_index:03d}",
                            "reference_speed": speed,
                            "matching_condition": matching,
                            "condition": condition,
                            "history_speed": float(
                                CONDITIONS.index(condition)
                            ),
                            "next_frame_latent_mse": (
                                1.0 if condition == matching else 2.0
                            ),
                            "eval_seed": eval_seed,
                            "evaluation_index": evaluation_index,
                        }
                    )
    return rows


def test_catalog_assignment_is_disjoint_and_paired_across_speeds() -> None:
    catalog = {
        "protocol": {},
        "summary": {},
        "bundles": [
            {
                "static_query_id": f"static-{query_index:03d}",
                "query_factors": {"agent.speed": speed},
            }
            for speed in (3.0, 5.0, 7.0)
            for query_index in range(300)
        ],
    }
    assigned = _assign_eval_seeds(
        catalog,
        eval_seeds=EVAL_SEEDS,
        per_seed=50,
        assignment_seed=123,
    )
    for speed in (3.0, 5.0, 7.0):
        rows = [
            row
            for row in assigned["bundles"]
            if row["query_factors"]["agent.speed"] == speed
        ]
        assert len(rows) == 300
        assert {
            seed: sum(row["eval_seed"] == seed for row in rows)
            for seed in EVAL_SEEDS
        } == {seed: 50 for seed in EVAL_SEEDS}
    by_static = {}
    for row in assigned["bundles"]:
        pair = (row["eval_seed"], row["evaluation_index"])
        assert by_static.setdefault(row["static_query_id"], pair) == pair


def test_full_50_by_6_matrix_count_and_decision() -> None:
    records = _scheduled_records(
        _unique_rows(),
        eval_seeds=EVAL_SEEDS,
        evaluations=50,
    )
    summary = summarize_records(records, bootstrap_seed=123)
    assert len(records) == 2700
    assert summary["count_audit"]["passed"]
    assert summary["count_audit"][
        "all_queries_unique_within_eval_seed_cells"
    ]
    assert set(summary["count_audit"]["records_per_cell"].values()) == {
        300
    }
    assert summary["decision"]["passed"]
    assert summary["overall"]["relative_loss_reduction"] == 0.5


def test_deterministic_metric_rejects_repeated_queries() -> None:
    rows = _unique_rows()
    duplicate = dict(rows[0])
    duplicate["evaluation_index"] = 999
    rows.append(duplicate)
    with pytest.raises(RuntimeError, match="Expected 50 rows per seed cell"):
        _scheduled_records(rows, eval_seeds=EVAL_SEEDS, evaluations=50)
