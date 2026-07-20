from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_tworoom_id_retention import (
    _parse_success_rate,
    _prepare_native_eval_home,
    _stablewm_policy_location,
)


def test_stablewm_policy_location_preserves_native_checkpoint_semantics(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs/checkpoints/h3_speedseen/weights.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    home, policy = _stablewm_policy_location(checkpoint)
    assert home == (tmp_path / "runs").resolve()
    assert policy == "h3_speedseen/weights.pt"


def test_stablewm_policy_location_rejects_non_native_layout(tmp_path: Path) -> None:
    checkpoint = tmp_path / "h3_speedseen/weights.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    with pytest.raises(ValueError, match="STABLEWM_HOME"):
        _stablewm_policy_location(checkpoint)


def test_native_eval_home_isolates_seed_outputs_with_symlinked_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "runs/checkpoints/h3_speedseen/weights.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    config = checkpoint.parent / "config.json"
    config.write_text("{}", encoding="utf-8")

    home, policy = _prepare_native_eval_home(checkpoint, tmp_path / "eval", 42)
    native_run = home / "checkpoints/h3_speedseen"
    assert policy == "h3_speedseen/weights.pt"
    assert (native_run / "weights.pt").is_symlink()
    assert (native_run / "weights.pt").resolve() == checkpoint.resolve()
    assert (native_run / "config.json").resolve() == config.resolve()
    assert native_run != checkpoint.parent


def test_parse_stablewm_success_rate(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.txt"
    metrics.write_text(
        "==== RESULTS ====\nmetrics: {'success_rate': 98.0, "
        "'episode_successes': array([ True, False])}\n",
        encoding="utf-8",
    )
    assert _parse_success_rate(metrics) == 98.0
