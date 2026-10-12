"""Scalar cloud architecture settings use the guarded family profile."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_stablewm_train as launcher  # noqa: E402
from test_stablewm_profile_launcher import (  # noqa: E402
    _args,
    _build,
    dataset,
    stablewm_repo,
)


@pytest.mark.parametrize(
    ("name", "env_name", "flag", "environment", "command_line"),
    [
        ("encoder_scale", "CW_ENCODER_SCALE", "--encoder-scale", "base", "large"),
        ("predictor_depth", "CW_PREDICTOR_DEPTH", "--predictor-depth", "3", "5"),
        ("predictor_hidden_dim", "CW_PREDICTOR_HIDDEN_DIM", "--predictor-hidden-dim", "256", "512"),
        ("embed_dim", "CW_EMBED_DIM", "--embed-dim", "128", "256"),
    ],
)
def test_scalar_settings_default_to_none_and_cli_overrides_environment(
    monkeypatch, name, env_name, flag, environment, command_line
):
    monkeypatch.delenv(env_name, raising=False)
    assert getattr(launcher.parse_args([]), name) is None
    monkeypatch.setenv(env_name, environment)
    expected = int(environment) if name != "encoder_scale" else environment
    assert getattr(launcher.parse_args([]), name) == expected
    expected_cli = int(command_line) if name != "encoder_scale" else command_line
    assert getattr(launcher.parse_args([flag, command_line]), name) == expected_cli


@pytest.mark.parametrize("family", ["lewm", "viswm", "pldm"])
def test_scalar_settings_render_supported_family_keys(
    stablewm_repo, dataset, family
):
    args = _args(
        stablewm_repo, dataset, "--family", family,
        "--embed-dim", "128", "--encoder-scale", "base",
        "--predictor-depth", "3", "--predictor-hidden-dim", "256",
    )
    _, pairs, _ = _build(args, stablewm_repo)
    embed_key = "wm.embed_dim" if family == "pldm" else "embed_dim"
    assert pairs[embed_key] == "128"
    assert pairs["encoder_scale"] == "base"
    assert pairs["model.predictor.depth"] == "3"
    assert pairs["model.predictor.hidden_dim"] == "256"


@pytest.mark.parametrize("flag", ["--predictor-depth", "--predictor-hidden-dim"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_predictor_dimensions_must_be_positive(flag, value):
    args = launcher.parse_args(["--family", "lewm", flag, value])
    with pytest.raises(SystemExit, match="must be positive"):
        launcher._validate_args(args, "lewm")


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--embed-dim", "128"),
        ("--encoder-scale", "base"),
        ("--predictor-depth", "3"),
        ("--predictor-hidden-dim", "256"),
    ],
)
def test_prejepa_rejects_explicit_scalar_architecture_settings(flag, value):
    args = launcher.parse_args(["--family", "prejepa", flag, value])
    with pytest.raises(SystemExit, match=flag):
        launcher._validate_args(args, "prejepa")
