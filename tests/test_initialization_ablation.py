"""Initialization intervention, resume precedence and launcher scope."""
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from contextworld.training.initialization import apply_initialization, validate_initialization
from scripts import run_stablewm_family_entry as bridge
from scripts import run_stablewm_train as launcher


def checkpoint(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.7)
        model[1].running_mean.fill_(0.4)
    path = tmp_path / 'original.ckpt'
    torch.save({'state_dict': {**{'model.' + k: v for k, v in model.state_dict().items()},
                              'sigreg.t': torch.tensor([0.0])},
                'optimizer_states': [{'sentinel': 'must not restore'}], 'epoch': 99}, path)
    return model, path, hashlib.sha256(path.read_bytes()).hexdigest()


def args(**changes):
    result = dict(training_track='joint_init_ablation_v1',
                  init_checkpoint=None,
                  family='lewm', component='action_strength', original_env=None,
                  method='native', rollout_steps=1, dataset=None, component_payload=None,
                  mix_original_weight=None, mix_synthetic_weight=None, component_epoch_size=None,
                  run_name='action_strength_init_ablation_scratch_s3072', resume='auto',
                  eval_icl_only=True)
    result.update(changes)
    return SimpleNamespace(**result)


def test_strict_full_model_load_keeps_rng_and_trainability(tmp_path):
    source, path, sha = checkpoint(tmp_path)
    target = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    before_rng = torch.random.get_rng_state().clone()
    spec = validate_initialization(str(path))
    receipt = apply_initialization(target, spec)
    assert receipt['initialization'] == spec
    for key, tensor in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], tensor)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert all(p.requires_grad for p in target.parameters())
    assert receipt['model_state_sha256_before'] != receipt['model_state_sha256_after']


def test_scratch_is_exact_noop():
    target = torch.nn.Linear(2, 2)
    rng = torch.random.get_rng_state().clone()
    receipt = apply_initialization(target, validate_initialization())
    assert receipt['model_state_sha256_before'] == receipt['model_state_sha256_after']
    assert torch.equal(rng, torch.random.get_rng_state())


def test_raw_model_state_supported(tmp_path):
    source = torch.nn.Linear(2, 2)
    path = tmp_path / 'raw.pt'
    torch.save(source.state_dict(), path)
    spec = validate_initialization(str(path))
    target = torch.nn.Linear(2, 2)
    apply_initialization(target, spec)
    assert torch.equal(target.weight, source.weight)


def test_changed_file_and_incompatible_model_fail(tmp_path):
    _, path, sha = checkpoint(tmp_path)
    spec = validate_initialization(str(path))
    with pytest.raises(RuntimeError):
        apply_initialization(torch.nn.Linear(3, 3), spec)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        apply_initialization(torch.nn.Linear(2, 2), spec)


@pytest.mark.parametrize('changes', [dict(method='coja_v1'), dict(component=None),
    dict(family='unknown'), dict(rollout_steps=3)])
def test_research_scope_fails_closed(changes, tmp_path):
    _, path, _ = checkpoint(tmp_path)
    with pytest.raises(SystemExit):
        launcher._validate_training_track(args(init_checkpoint=str(path), **changes))


def test_historical_track_rejects_initializer_but_current_default_remains_valid():
    with pytest.raises(SystemExit, match='frozen historical_release'):
        launcher._validate_training_track(args(training_track='historical_release', init_checkpoint='/some/model.ckpt'))
    launcher._validate_training_track(args(training_track='joint_scratch_v1'))


def test_initializer_identity_is_explicit_and_bound(tmp_path):
    _, path, sha = checkpoint(tmp_path)
    scratch = args()
    warm = args(init_checkpoint=str(path))
    launcher._validate_training_track(scratch)
    launcher._validate_training_track(warm)
    assert scratch.initialization_identity != warm.initialization_identity
    assert warm.initialization_identity['sha256'] == sha
    with pytest.raises(ValueError):
        apply_initialization(torch.nn.Linear(2, 2),
                             {**warm.initialization_identity, 'sha256': '0' * 64})


def test_full_state_resume_overrides_initializer(tmp_path, monkeypatch):
    _, path, sha = checkpoint(tmp_path)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    observed = {}
    class Manager:
        def __init__(self, *, module, ckpt_path=None, weights_only=True):
            observed.update(initial_weight=module.model[0].weight.detach().clone(),
                            ckpt_path=ckpt_path, weights_only=weights_only)
            self._resolve_run_dir()
            if ckpt_path is not None:
                with torch.no_grad():
                    module.model[0].weight.fill_(0.9)  # simulate subsequent full restoration
        def _resolve_run_dir(self):
            return tmp_path / 'run'
    monkeypatch.setitem(sys.modules, 'stable_pretraining', SimpleNamespace(Manager=Manager))
    monkeypatch.setattr(bridge, '_prepare_optional_flash_attention', lambda: False)
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    bridge._install_manager_bridge(run_name='init_ablation', identity_sha256='identity',
        resume_checkpoint=tmp_path / 'last.ckpt',
        initialization=validate_initialization(str(path)))
    sys.modules['stable_pretraining'].Manager(module=SimpleNamespace(model=model))
    assert torch.all(observed['initial_weight'] == .7)
    assert observed['weights_only'] is False
    assert observed['ckpt_path'] == str(tmp_path / 'last.ckpt')
    assert torch.all(model[0].weight == .9)
    receipt = json.loads((tmp_path / 'run/contextworld_initialization_v1.json').read_text())
    assert receipt['initialization']['sha256'] == sha


def test_bootstrap_forwards_same_initializer_for_each_rank(monkeypatch):
    spec = validate_initialization()
    calls = []
    monkeypatch.setitem(sys.modules, 'scripts.run_stablewm_family_entry',
                        SimpleNamespace(_install_manager_bridge=lambda **kwargs: calls.append(kwargs)))
    monkeypatch.setenv('CONTEXTWORLD_SPT_BRIDGE', '1')
    monkeypatch.setenv('CONTEXTWORLD_SPT_RUN_NAME', 'init_ablation')
    monkeypatch.setenv('CONTEXTWORLD_SPT_IDENTITY_SHA256', 'identity')
    monkeypatch.setenv('CONTEXTWORLD_RESEARCH_INITIALIZATION', json.dumps(spec))
    monkeypatch.delenv('CONTEXTWORLD_STABLEWM_BUNDLE', raising=False)
    monkeypatch.delenv('CONTEXTWORLD_SPT_RESUME_CHECKPOINT', raising=False)
    for rank in range(8):
        monkeypatch.setenv('LOCAL_RANK', str(rank))
        runpy.run_path(str(Path(launcher.__file__).parent / 'stablewm_bootstrap/sitecustomize.py'))
    assert len(calls) == 8
    assert all(call['initialization'] == spec for call in calls)


@pytest.mark.parametrize('path', [None, ''])
def test_absent_or_empty_path_preserves_default_initialization(path):
    request = args(init_checkpoint=path)
    launcher._validate_training_track(request)
    assert request.initialization_identity is None
    assert validate_initialization(path)['mode'] == 'scratch'


def test_user_checkpoint_is_automatically_bound(tmp_path):
    _, path, digest = checkpoint(tmp_path)
    request = args(init_checkpoint=str(path))
    launcher._validate_training_track(request)
    assert request.initialization_identity['checkpoint'] == str(path.resolve())
    assert request.initialization_identity['sha256'] == digest
    path.write_bytes(b'changed after planning')
    with pytest.raises(ValueError):
        apply_initialization(torch.nn.Linear(2, 2), request.initialization_identity)


def test_interface_does_not_hardcode_task_or_run_name():
    request = args(component='portal_exit', run_name='my_custom_control')
    launcher._validate_training_track(request)
    assert request.initialization_identity is None
    # Ordinary launcher validation, not this interface, validates custom data.
    launcher._validate_training_track(args(dataset='/tmp/custom', mix_original_weight=.4))


def test_relative_user_path_is_rejected_even_if_file_exists(tmp_path, monkeypatch):
    _, path, digest = checkpoint(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        validate_initialization(path.name)


def test_removed_options_are_not_in_parser_help():
    # The fingerprint is internal provenance, not another operator option.
    import subprocess
    result = subprocess.run([sys.executable, launcher.__file__, '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--init-checkpoint-sha256' not in result.stdout
    assert '--initialization' not in result.stdout
    assert '--init-checkpoint' in result.stdout
    assert '--init-checkpoint ' in result.stdout


@pytest.mark.parametrize('resume', ['auto', 'never', 'required', 'reset'])
@pytest.mark.parametrize('with_checkpoint', [False, True])
@pytest.mark.parametrize('track', ['joint_scratch_v1', 'joint_init_ablation_v1', 'my_run'])
def test_path_alone_selects_initialization_independent_of_resume(tmp_path, monkeypatch, resume, with_checkpoint, track):
    _, path, _ = checkpoint(tmp_path)
    monkeypatch.delenv('CW_SEED', raising=False)
    monkeypatch.delenv('CW_TRAINING_SEED', raising=False)
    monkeypatch.setenv('CW_INIT_CHECKPOINT', str(path) if with_checkpoint else '')
    parsed = launcher.parse_args(['--family', 'lewm', '--component', 'action_strength',
        '--training-track', track, '--method', 'native',
        '--run-name', 'custom', '--resume', resume])
    assert not hasattr(parsed, 'initialization')
    launcher._validate_training_track(parsed)
    if with_checkpoint:
        assert parsed.initialization_identity['mode'] == 'warmstart'
    else:
        assert parsed.initialization_identity is None
    assert parsed.resume == resume


@pytest.mark.parametrize('with_checkpoint', [False, True])
def test_custom_track_names_run_without_run_name_and_binds_weights(tmp_path, monkeypatch, with_checkpoint):
    _, path, digest = checkpoint(tmp_path)
    track = 'init_form_pusht_lewm_baseline_seed3073_weights'
    monkeypatch.setenv('CW_INIT_CHECKPOINT', str(path) if with_checkpoint else '')
    request = launcher.parse_args(['--family', 'lewm', '--component', 'action_strength',
        '--training-track', track, '--method', 'native', '--resume', 'reset'])
    request.run_name = None
    launcher._validate_training_track(request)
    assert launcher._base_training_track(request) == 'joint_scratch_v1'
    target = SimpleNamespace(label='action_strength', original_env=None)
    for seed in (3072, 3073):
        name = launcher._run_name(request, target, seed, (3072, 3073))
        assert name == f'action_strength_lewm_{track}_s{seed}'
        assert launcher._evaluation_checkpoint(tmp_path, name, 10) == tmp_path / 'checkpoints' / name / 'weights_epoch_10.pt'
    if with_checkpoint:
        assert request.initialization_identity['mode'] == 'warmstart'
        assert request.initialization_identity['sha256'] == digest
    else:
        assert request.initialization_identity is None  # Label alone does not load weights.


@pytest.mark.parametrize('track', ['../escape', 'two words', '', '/absolute'])
def test_invalid_track_names_fail(track):
    with pytest.raises(SystemExit, match='CW_TRAINING_TRACK'):
        launcher._validate_training_track(args(training_track=track))


def test_custom_track_uses_current_coja_capability_contract():
    request = args(training_track='my_method_run', method='coja_v1', run_name=None)
    launcher._validate_training_track(request)
    launcher._validate_method(request, launcher.load_profile_contract())


@pytest.mark.parametrize('family', ['lewm','pldm','viswm','prejepa'])
def test_all_builtin_families_accept_initialization(family,tmp_path):
    _, path, _ = checkpoint(tmp_path)
    request = args(family=family, init_checkpoint=str(path))
    launcher._validate_training_track(request)
    assert request.initialization_identity['mode'] == 'warmstart'
