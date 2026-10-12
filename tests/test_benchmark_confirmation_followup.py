import pytest
from scripts.run_benchmark_confirmation_followup import upstream_ready


def test_training_completion_requires_every_locked_job():
    state = {'protocol_sha256': 'p', 'status': 'completed', 'jobs': {
        f'{task}/s{seed}': {'status': 'completed'}
        for task in ('contact_friction', 'robot_arm_mass') for seed in (3073, 3074)}}
    assert upstream_ready(state, 'p')
    state['jobs'].pop('robot_arm_mass/s3074')
    with pytest.raises(ValueError, match='all four'):
        upstream_ready(state, 'p')


def test_changed_or_failed_training_never_launches_test():
    with pytest.raises(ValueError, match='identity'):
        upstream_ready({'protocol_sha256': 'wrong'}, 'p')
    with pytest.raises(RuntimeError, match='training failed'):
        upstream_ready({'protocol_sha256': 'p', 'status': 'failed'}, 'p')


def test_direct_test_does_not_require_or_wait_for_training(tmp_path, monkeypatch):
    import json
    from scripts import run_benchmark_confirmation_followup as followup
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    output = tmp_path / 'output'
    monkeypatch.setattr(followup.sys, 'argv', ['followup', '--plan', str(plan),
                        '--output-root', str(output), '--devices', '0'])
    class FinishedWorker:
        pid = 123
        def poll(self):
            return 0
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        return FinishedWorker()
    monkeypatch.setattr(followup.subprocess, 'Popen', launch)
    monkeypatch.setattr(followup, 'summarize', lambda *args: {
        'completed_runs': 1, 'expected_runs': 1})
    monkeypatch.setattr(followup, 'write_outputs', lambda *args: None)
    def unexpected_wait(*args):
        raise AssertionError('Direct Test must not wait for seed training')
    monkeypatch.setattr(followup.time, 'sleep', unexpected_wait)
    monkeypatch.setattr(followup, 'upstream_ready', unexpected_wait)
    followup.main()
    assert len(commands) == 1
    assert 'execute' in commands[0]
    status = json.loads((output / 'followup_status.json').read_text())
    assert status['status'] == 'completed'
    assert status['upstream_protocol_sha256'] is None
