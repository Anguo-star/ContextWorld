from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from contextworld.evaluation.action_delay_h7_domain_diagnostic import (
    DELAYS,
    DIAGNOSTIC_EVAL_SEEDS,
    QUERIES_PER_TRACK,
    QUERY_COUNT,
    build_domain_asset,
    select_domain_assignments,
)
from contextworld.evaluation.action_delay_h7_domain_score import (
    score_domain_track,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.stablewm import load_stable_worldmodel


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_domain_diagnostic_data_v1.yaml"
)


class _OracleAdapter:
    protocol = SimpleNamespace(
        history_tokens=7,
        action_block_raw_steps=5,
        action_dim=2,
        future_action_blocks=3,
    )

    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        del raw_action_blocks, batch_size
        delay = input_pixels[:, 0, 0, 0, 0].astype(np.float32)
        return np.repeat(delay[:, None, None], 3, axis=1)

    def encode_pixels(
        self,
        pixels: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        del batch_size
        return pixels[:, 0, 0, 0].astype(np.float32)[:, None]


def _inputs() -> tuple[dict, dict, dict]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source = config["source_training_release"]
    training_config = yaml.safe_load(
        resolve_contextworld_path(
            source["config"]["path"],
            repo_root=ROOT,
        ).read_text(encoding="utf-8")
    )
    training_catalog = json.loads(
        resolve_contextworld_path(
            source["multi_delay_catalog"]["path"],
            repo_root=ROOT,
        ).read_text(encoding="utf-8")
    )
    return config, training_config, training_catalog


def test_six_tracks_each_have_independent_50x6_queries() -> None:
    config, training_config, training_catalog = _inputs()
    rows = select_domain_assignments(
        config=config,
        training_config=training_config,
        training_catalog=training_catalog,
        repo_root=ROOT,
    )

    assert len(rows) == QUERY_COUNT
    assert len({row.query_id for row in rows}) == QUERY_COUNT
    assert (
        len({row.template.template_id for row in rows}) == QUERY_COUNT
    )
    counts = Counter(row.track for row in rows)
    assert set(counts.values()) == {QUERIES_PER_TRACK}
    for track in counts:
        by_seed = Counter(
            row.diagnostic_eval_seed
            for row in rows
            if row.track == track
        )
        assert by_seed == Counter(
            {seed: 50 for seed in DIAGNOSTIC_EVAL_SEEDS}
        )
        for seed in DIAGNOSTIC_EVAL_SEEDS:
            selected = [
                row
                for row in rows
                if row.track == track
                and row.diagnostic_eval_seed == seed
            ]
            assert Counter(row.room for row in selected) == {
                "left": 25,
                "right": 25,
            }
            assert Counter(
                row.template.direction for row in selected
            ) == {"up": 25, "down": 25}


def test_source_delay_matches_the_frozen_multi_shard_assignment() -> None:
    config, training_config, training_catalog = _inputs()
    rows = select_domain_assignments(
        config=config,
        training_config=training_config,
        training_catalog=training_catalog,
        repo_root=ROOT,
    )

    assert all(
        DELAYS[row.source_shard_index % len(DELAYS)]
        == row.source_delay
        for row in rows
    )


def test_one_paired_asset_has_three_distinct_physical_futures(monkeypatch) -> None:
    config, training_config, training_catalog = _inputs()
    source = config["source_training_release"]
    monkeypatch.setattr(sys, "path", list(sys.path))
    # Other integration tests intentionally load the frozen 5864 checkout.
    # This diagnostic targets its configured sibling checkout, so clear the
    # process-wide import cache before exercising that explicit repository.
    for name in tuple(sys.modules):
        if name == "stable_worldmodel" or name.startswith("stable_worldmodel."):
            del sys.modules[name]
    load_stable_worldmodel(
        ROOT,
        str(source["stable_worldmodel"]["repo"]),
        None,
    )
    assignment = select_domain_assignments(
        config=config,
        training_config=training_config,
        training_catalog=training_catalog,
        repo_root=ROOT,
    )[0]
    arrays, audit = build_domain_asset(
        assignment,
        agent_speed=7.0,
        action_magnitude=0.5,
        maximum_delay_steps=8,
    )

    assert audit["physical"]["passed"] is True
    assert audit["physical"]["future_state_group_counts"] == {
        "1": 3,
        "2": 3,
        "3": 3,
    }
    assert arrays["history_pixels"].shape[:2] == (3, 7)
    assert arrays["true_future_pixels"].shape[:2] == (3, 3)


def test_domain_oracle_selects_all_three_histories_and_targets() -> None:
    assets = []
    for query_index in range(300):
        history = np.zeros((3, 7, 1, 1, 3), dtype=np.uint8)
        future = np.zeros((3, 3, 1, 1, 3), dtype=np.uint8)
        for delay_index, delay in enumerate(DELAYS):
            history[delay_index, ..., 0] = delay
            future[delay_index, ..., 0] = delay
        assets.append(
            {
                "query_id": f"oracle-{query_index:03d}",
                "track": "training_replay_delay_0",
                "source_split": "train",
                "source_delay": 0,
                "diagnostic_eval_seed": DIAGNOSTIC_EVAL_SEEDS[
                    query_index // 50
                ],
                "evaluation_index": query_index % 50,
                "room": "left",
                "direction": "up",
                "history_pixels": history,
                "action_blocks": np.zeros((9, 5, 2), dtype=np.float32),
                "true_future_pixels": future,
            }
        )

    result = score_domain_track(
        _OracleAdapter(),
        assets,
        batch_size=128,
    )
    source = result["by_horizon"]["1"][
        "source_supervised_target"
    ]["overall"]
    assert source["exact_history_selection_rate"] == 1.0
    assert source["exact_target_selection_rate"] == 1.0
    assert source["matching_history_strict_win_rate"] == 1.0
    assert (
        result["latent_alignment"]["trajectory"][
            "prediction_to_target_pair_magnitude_ratio"
        ]
        == 1.0
    )
