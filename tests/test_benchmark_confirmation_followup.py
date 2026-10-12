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
