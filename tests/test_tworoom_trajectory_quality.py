from __future__ import annotations

import numpy as np

from scripts.analyze_tworoom_trajectory_quality import (
    EpisodeTable,
    room_relation,
    summarize_episodes,
)


def test_room_relation_uses_opposite_sides_of_vertical_wall() -> None:
    start = np.asarray([[55.0, 70.0], [169.0, 70.0], [169.0, 150.0]])
    goal = np.asarray([[190.0, 190.0], [190.0, 190.0], [55.0, 150.0]])
    assert room_relation(start, goal).tolist() == [True, False, True]


def test_episode_summary_keeps_geometry_distinct_from_episode_count() -> None:
    episodes = EpisodeTable(
        start=np.asarray([[55.0, 70.0], [55.0, 70.0], [169.0, 70.0]]),
        goal=np.asarray([[190.0, 190.0], [190.0, 190.0], [190.0, 190.0]]),
        final=np.asarray([[180.0, 180.0], [100.0, 100.0], [190.0, 190.0]]),
        lengths=np.asarray([50, 100, 20]),
        terminated=np.asarray([True, False, True]),
        truncated=np.asarray([False, True, False]),
        speed=np.asarray([5.0, 7.0, 5.0]),
    )

    summary = summarize_episodes(episodes)

    assert summary["episodes"] == 3
    assert summary["unique_start_goal_pairs"] == 2
    assert summary["cross_room"]["episodes"] == 2
    assert summary["same_room"]["episodes"] == 1
    assert summary["termination_successes"] == 2
