from scripts.analyze_tworoom_synth5_matched import (
    _absolute_gate,
    _minimum_fraction_gate,
    _quantile_gate,
    _total_variation,
)


def test_synth5_distribution_gate_helpers() -> None:
    assert _absolute_gate(0.51, 0.50, 0.02)["passed"]
    assert not _absolute_gate(0.53, 0.50, 0.02)["passed"]
    assert _minimum_fraction_gate(90, 100, 0.9)["passed"]
    assert not _minimum_fraction_gate(89, 100, 0.9)["passed"]
    assert _quantile_gate(
        {"0.1": 10.0, "0.5": 20.0},
        {"0.1": 9.0, "0.5": 18.0},
        2.0,
    )["passed"]


def test_total_variation_uses_union_support() -> None:
    assert _total_variation({"a": 1}, {"a": 1}) == 0.0
    assert _total_variation({"a": 1}, {"b": 1}) == 1.0
