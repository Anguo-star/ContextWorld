"""Cloud architecture settings follow the existing guarded Hydra override path."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_stablewm_train as launcher


def test_cloud_model_overrides_and_cli_precedence(monkeypatch):
    monkeypatch.setenv('CW_OVERRIDES', '["encoder_scale=base", "model.predictor.depth=9"]')
    args = launcher.parse_args(['--family', 'lewm', '--override', 'model.predictor.depth=8'])
    assert args.override == ['encoder_scale=base', 'model.predictor.depth=9', 'model.predictor.depth=8']
    launcher._validate_args(args, 'lewm', None)


@pytest.mark.parametrize('value', ['encoder_scale=base', '{}', '[1]', '["encoder_scale"]'])
def test_cloud_model_overrides_reject_malformed_values(monkeypatch, value):
    monkeypatch.setenv('CW_OVERRIDES', value)
    with pytest.raises(SystemExit):
        launcher.parse_args(['--family', 'lewm'])


def test_cloud_overrides_keep_identity_guard(monkeypatch):
    monkeypatch.setenv('CW_OVERRIDES', '["seed=99"]')
    args = launcher.parse_args(['--family', 'lewm'])
    with pytest.raises(SystemExit, match='launcher-owned'):
        launcher._validate_args(args, 'lewm', None)
