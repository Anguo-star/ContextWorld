"""Respect the paired unit and distinct correct/wrong-history denominators."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_speed_timed_arrival import scene_metrics, summarize


def test_context_gain_averages_two_wrong_histories_per_condition():
    plans = [dict(history_index=h, outcomes=[dict(speed_index=s, success=s == h,
                 terminal_distance=1 if s == h else 7, contact_steps=[])
                 for s in range(3)]) for h in range(3)]
    row = scene_metrics(plans)
    assert row['correct_success_percent'] == 100
    assert row['wrong_success_percent'] == 0
    assert row['history_distance_benefit'] == 6
    assert row['history_success_gain_pp'] == 100


def test_repeated_equal_scenes_have_zero_bootstrap_width():
    plans = [dict(history_index=h, outcomes=[dict(speed_index=s, success=False,
                 terminal_distance=3, contact_steps=[]) for s in range(3)]) for h in range(3)]
    row = scene_metrics(plans)
    result = summarize([dict(query_id=str(i), scheme='T0', **row) for i in range(6)])
    assert result['correct_distance'] == pytest.approx(3)
    assert result['history_distance_benefit_ci95'] == [0, 0]
