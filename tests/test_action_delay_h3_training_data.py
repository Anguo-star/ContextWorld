from __future__ import annotations

import copy
from pathlib import Path
import sys

import yaml

from contextworld.evaluation.action_delay_h3_data import (
    audit_shard,
    build_shard_plans,
    collect_shard,
    training_template,
)
from contextworld.synthesis.stablewm import load_stable_worldmodel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_action_delay_h3_training_data_v1.yaml"
)


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_training_raster_coordinates_are_reserved_from_validation_grid() -> None:
    eval_left_x = set(range(25, 94, 3))
    eval_right_x = set(range(133, 201, 3))
    eval_y = set(range(60, 166, 5))

    for shard_index in range(12):
        for episode_index in range(20):
            template = training_template(
                catalog_seed=2026072602,
                split="train",
                shard_index=shard_index,
                episode_index=episode_index,
            )
            query_y = template.reset_state[1] + (
                35.0 if template.direction == "up" else -35.0
            )
            query_x = template.reset_state[0]
            assert int(query_y) not in eval_y
            if query_x < 112.0:
                assert int(query_x) not in eval_left_x
            else:
                assert int(query_x) not in eval_right_x


def test_single_and_multi_shards_share_geometry_but_not_delay_support(
    tmp_path: Path,
) -> None:
    config = _config()
    config["output_root"] = str(tmp_path / "release")
    config["counts"] = {
        "train": {"shards": 3, "episodes_per_shard": 2},
        "val": {"shards": 3, "episodes_per_shard": 2},
    }
    plans = build_shard_plans(config, repo_root=ROOT)

    assert {
        shard.delay_steps
        for shard in plans["action_delay_single"]
    } == {2}
    assert {
        shard.delay_steps
        for shard in plans["action_delay_multi"]
    } == {0, 2, 4}
    for left, right in zip(
        plans["action_delay_single"],
        plans["action_delay_multi"],
        strict=True,
    ):
        assert [
            plan.template for plan in left.episodes
        ] == [
            plan.template for plan in right.episodes
        ]


def test_two_episode_lance_shard_round_trips_exactly(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(_config())
    config["output_root"] = str(tmp_path / "release")
    config["counts"] = {
        "train": {"shards": 3, "episodes_per_shard": 2},
        "val": {"shards": 3, "episodes_per_shard": 2},
    }
    shard = build_shard_plans(config, repo_root=ROOT)[
        "action_delay_multi"
    ][0]
    pinned = Path("/tmp/stable-worldmodel-5864")
    config["stable_worldmodel"]["repo"] = str(
        pinned if pinned.is_dir() else ROOT.parent / "stable-worldmodel"
    )
    for name in tuple(sys.modules):
        if name == "stable_worldmodel" or name.startswith("stable_worldmodel."):
            del sys.modules[name]
    swm, _, _ = load_stable_worldmodel(
        ROOT,
        config["stable_worldmodel"]["repo"],
        config["stable_worldmodel"]["commit"],
    )

    collect_shard(swm, shard=shard, config=config)
    audit = audit_shard(swm, shard=shard, config=config)

    assert audit["passed"] is True
    assert all(audit["checks"].values())
    assert audit["episodes"] == 2
    assert audit["raw_rows"] == 40
    assert audit["model_clips"] == 2
