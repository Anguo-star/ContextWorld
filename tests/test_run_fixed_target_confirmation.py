"""CPU checks for safe reuse of protocol-matched confirmation runs."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_fixed_target_confirmation.py'
SPEC = importlib.util.spec_from_file_location('fixed_target_confirmation', SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def save(path, value):
    path.write_text(json.dumps(value))


def completed_run(tmp_path):
    actual = tmp_path / 'actual'
    alias = tmp_path / 'alias'
    actual.mkdir()
    alias.symlink_to(actual, target_is_directory=True)
    run = tmp_path / 'run'
    run.mkdir()
    frozen = {'encoder': {'sha256': 'enc', 'tensors': 198},
              'projector': {'sha256': 'proj', 'tensors': 9}}
    save(run / 'recipe.json', {
        'full_t0_sha256': 't0-sha', 'full_t0_checkpoint': str(actual / 't0.ckpt'),
        'freeze_scope': 'visual', 'frozen_modules': ['encoder', 'projector'],
        'training_seed': 3073, 'reference_training_seed': 3072,
        'reference_run': str(actual / 'reference'), 'native_config_sha256': 'config-sha',
        'stablewm_repo': str(actual / 'stablewm'),
        'smoke_max_steps': None, 'devices_override': None})
    save(run / 'loader_actual.json', {
        'world_size': 8, 'max_epochs': 10, 'max_steps': -1,
        'accumulate_grad_batches': 1, 'estimated_optimizer_steps': 12})
    save(run / 'loader_constructed.json', {'batch_size_per_rank': 128})
    save(run / 'initial_frozen_hashes.json', frozen)
    save(run / 'epoch_10_audit.json', {
        'epoch': 10, 'global_step': 12, 'frozen_state_unchanged': True,
        'frozen_hashes': frozen})
    save(run / 'train_end_audit.json', {
        'global_step': 12, 'frozen_state_unchanged': True, 'frozen_hashes': frozen})
    checkpoint = run / 'weights_epoch_10.pt'
    checkpoint.write_bytes(b'checkpoint')
    task = {'baseline_sha256': 't0-sha', 'baseline_checkpoint': str(alias / 't0.ckpt'),
            'reference_run': str(alias / 'reference'), 'reference_config_sha256': 'config-sha',
            'stablewm_repo': str(alias / 'stablewm'), 'expected_optimizer_steps': 12}
    training = {'seeds': [3072, 3073, 3074], 'world_size': 8, 'max_epochs': 10,
                'accumulate_grad_batches': 1, 'batch_size_per_rank': 128}
    return run, checkpoint, task, training


def test_completed_run_accepts_resolved_path_aliases(tmp_path):
    run, checkpoint, task, training = completed_run(tmp_path)
    assert RUNNER.verified_existing_training(run, task, 3073, training) == checkpoint


def test_partial_existing_run_is_refused(tmp_path):
    run, _, task, training = completed_run(tmp_path)
    (run / 'train_end_audit.json').unlink()
    with pytest.raises(RuntimeError, match='never reset it'):
        RUNNER.verified_existing_training(run, task, 3073, training)
