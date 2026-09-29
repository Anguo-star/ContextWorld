"""Independent oracle geometry must establish a need for speed-specific plans."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_speed_timed_arrival import speed_blind_bound


def test_registered_speeds_require_distinct_precise_controls():
    result = speed_blind_bound(32, [3.4, 4.8, 6.9])
    assert result['mean_distance'] == pytest.approx(7.7777777778)
    assert result['maximum_success_fraction'] == pytest.approx(1/3)


def test_minimizer_is_weighted_not_ordinary_median():
    result = speed_blind_bound(100, [1, 2, 10])
    assert result['cumulative_action'] == 10
    assert result['mean_distance'] == pytest.approx(170/3)


def test_zero_goal_and_overlapping_intervals_need_no_speed_information():
    result = speed_blind_bound(0, [3.4, 4.8, 6.9])
    assert result['mean_distance'] == 0
    assert result['maximum_success_fraction'] == 1
    assert speed_blind_bound(1, [3.4, 4.8, 6.9], tolerance=2)['maximum_success_fraction'] == 1
