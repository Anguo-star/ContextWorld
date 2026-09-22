"""Matched image/action-only original-data training for PreJEPA."""
import h5py
import pytest
from test_stablewm_profile_launcher import launcher, stablewm_repo  # noqa: F401


@pytest.mark.parametrize('environment,action_dim,state_key', [
    ('pusht', 2, 'proprio'), ('tworoom', 2, 'proprio'),
    ('reacher', 2, 'observation'), ('cube', 5, 'observation'),
])
def test_original_prejepa_pixels_action_profile(
    stablewm_repo, tmp_path, monkeypatch, environment, action_dim, state_key,
):
    """An original-data run can use exactly the component input contract."""
    dataset = tmp_path / 'original.h5'
    with h5py.File(dataset, 'w') as handle:
        handle.create_dataset('pixels', shape=(4, 8, 8, 3), dtype='uint8')
        handle.create_dataset('action', shape=(4, action_dim), dtype='float32')
    monkeypatch.setenv('CW_MODEL_INPUTS', 'pixels_action')
    args = launcher.parse_args([
        '--family', 'prejepa', '--original-env', environment,
        '--dataset', str(dataset), '--stablewm-repo', str(stablewm_repo),
        '--checkpoint-root', str(tmp_path / 'ckpt'),
    ])
    contract = launcher.load_profile_contract()
    target = launcher.resolve_target(args, contract)
    launcher.validate_training_dataset_schema(
        target=target, family='prejepa', stablewm_repo=stablewm_repo,
    )
    entries = launcher.build_overrides(args, contract, target, run_name='run', seed=3073, stablewm_repo=stablewm_repo)
    assert '~wm.encoding.proprio' in entries
    assert not any('wm.encoding.observation' in entry for entry in entries)
    assert target.encoding_key is None
    args.override = [f'+wm.encoding.{state_key}=10']
    with pytest.raises(SystemExit, match='fixes model inputs'):
        launcher.build_overrides(args, contract, target, run_name='run', seed=3073, stablewm_repo=stablewm_repo)
    args.override = []
    args.model_inputs = 'native'
    native = launcher.resolve_target(args, contract)
    assert native.encoding_key == state_key
    with pytest.raises(SystemExit, match='Missing columns'):
        launcher.validate_training_dataset_schema(target=native, family='prejepa', stablewm_repo=stablewm_repo)
