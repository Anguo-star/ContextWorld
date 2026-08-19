from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from contextworld.evaluation.action_delay import (
    DELAY_VALUES,
    MODEL_INPUT_KEYS,
    build_feasibility_catalog,
    make_feasibility_templates,
    model_input_projection,
    simulate_template,
    validate_delay_family,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_action_delay_h3_feasibility_v1.yaml"
)


def test_feasibility_templates_cover_32_unique_two_direction_cases() -> None:
    templates = make_feasibility_templates(catalog_seed=20260726)

    assert len(templates) == 32
    assert len({template.template_id for template in templates}) == 32
    assert {template.direction for template in templates} == {"up", "down"}
    assert sum(template.direction == "up" for template in templates) == 16
    assert sum(template.direction == "down" for template in templates) == 16


def test_one_five_delay_family_is_strictly_paired_and_replayable() -> None:
    template = make_feasibility_templates(catalog_seed=20260726)[0]
    rollouts = {
        delay: simulate_template(
            template,
            delay_steps=delay,
            agent_speed=7.0,
        )
        for delay in DELAY_VALUES
    }
    audit = validate_delay_family(
        template,
        rollouts,
        agent_speed=7.0,
    )

    assert audit["passed"] is True
    assert all(audit["checks"].values())
    assert {
        tuple(rollout["query_state"]) for rollout in rollouts.values()
    } == {(40.0, 60.0)}
    assert {
        tuple(rollout["target_state"]) for rollout in rollouts.values()
    } == {
        (40.0, 95.0),
        (40.0, 88.0),
        (40.0, 81.0),
        (40.0, 74.0),
        (40.0, 67.0),
    }


def test_model_projection_excludes_delay_and_state() -> None:
    template = make_feasibility_templates(catalog_seed=20260726)[0]
    rollout = simulate_template(
        template,
        delay_steps=4,
        agent_speed=7.0,
    )
    projection = model_input_projection(rollout)

    assert tuple(projection) == MODEL_INPUT_KEYS
    assert projection["pixels"].shape == (3, 224, 224, 3)
    assert projection["action"].shape == (3, 5, 2)
    assert np.count_nonzero(projection["action"][1]) == 0


def test_full_feasibility_catalog_has_frozen_counts(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    catalog, report = build_feasibility_catalog(
        config=config,
        repo_root=ROOT,
        output_root=tmp_path,
    )

    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert report["counts"] == {
        "paired_templates": 32,
        "delay_rollouts": 160,
        "unique_query_pixels": 32,
    }
    assert catalog["content_manifest_sha256"] == report[
        "content_manifest_sha256"
    ]
