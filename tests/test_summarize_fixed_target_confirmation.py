"""Protocol identity and missing-seed behavior of the Test archive."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/summarize_fixed_target_confirmation.py'
spec = importlib.util.spec_from_file_location('confirmation_summary', SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture(tmp_path):
    task = 'contact_friction'
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(b'candidate weights')
    source = tmp_path / 'source'
    files = [{'path': 'test.lance/data.lance', 'bytes': 5, 'sha256': 'e' * 64}]
    task_spec = {
        'baseline_sha256': 'a' * 64, 'completed_checkpoint': str(checkpoint),
        'completed_checkpoint_sha256': summary.digest(checkpoint), 'stable_ref': 'pinned-ref',
        'test_payload': 'test.lance', 'test_files': files, 'normalization': {'mean': [0.0]},
        'normalization_manifest': '/normalization.json', 'normalization_manifest_sha256': 'f' * 64,
    }
    panels = tmp_path / 'panels'
    manifest_path = panels / task / 'manifest.json'
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        'schema': 'contextworld.fixed_target_test_panel.v1', 'task': task,
        'source_root': str(source), 'source_manifest_sha256': 'b' * 64,
        'source_registry_sha256': 'c' * 64, 'source_lance': str(source / 'test.lance'),
        'verified_source_files': files, 'normalization': {'mean': [0.0]},
        'normalization_manifest': '/normalization.json', 'normalization_manifest_sha256': 'f' * 64,
        'splits': {'test': {'path': 'test.npz', 'queries': 256, 'rows': 512, 'sha256': 'd' * 64}},
    }
    manifest_path.write_text(json.dumps(manifest))
    protocol = {
        'source': {'root': str(source), 'manifest_sha256': 'b' * 64, 'registry_sha256': 'c' * 64},
        'paths': {'panels': str(panels)}, 'tasks': {task: task_spec},
        'training': {'seeds': [3072, 3073, 3074], 'repeat_seeds': [3073, 3074]},
    }
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    result = {
        'schema': 'contextworld.fixed_target_control.v1', 'task': task,
        't0_checkpoint_sha256': 'a' * 64, 'new_checkpoint_sha256': summary.digest(checkpoint),
        'stable_ref': 'pinned-ref', 'visual_identity': {'tensor_and_buffer_count': 207, 'bitwise_equal': True},
        'no_fitting': True, 'no_test_read': False,
        'panel_manifest_sha256': summary.digest(manifest_path),
        'model_state_unchanged': {'t0': {'before': 'x', 'after': 'x'}, 'new': {'before': 'y', 'after': 'y'}},
    }
    panel = {
        'pairs': 256, 'rows': 512, 'action_blocks_equal_per_pair': True, 'panel_sha256': 'd' * 64,
        'old': {'response_nre': 1.0, 'full_error_over_B': 2.0, 'history_benefit_over_B': 0.0},
        'new': {'response_nre': 0.6, 'full_error_over_B': 1.8, 'history_benefit_over_B': 0.2},
    }
    for key in ('delta_response_nre', 'delta_full_error_over_B', 'delta_history_benefit_over_B'):
        panel[key] = {'delta': -0.2, 'ci95': [-0.3, -0.1], 'source_groups': 256, 'resamples': 2000}
    result['splits'] = {'test': panel}
    result_path = tmp_path / 'results' / task / 's3072' / 'result.json'
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(result))
    return protocol_path, result_path, tmp_path / 'results', tmp_path / 'absent-queue.json'


def test_available_seed_and_missing_repeats(tmp_path):
    protocol, _, results, queue = fixture(tmp_path)
    archive, rows = summary.build(protocol, results, queue)
    task = archive['tasks']['contact_friction']
    assert task['n'] == 1 and task['available_seeds'] == [3072]
    assert [x['seed'] for x in task['missing']] == [3073, 3074]
    assert len(rows) == 1 and rows[0]['nre_after'] == 0.6
    assert task['aggregate']['nre_t0'] == {'mean': 1.0, 'seed_std': None}
    assert task['aggregate']['nre_after'] == {'mean': 0.6, 'seed_std': None}


@pytest.mark.parametrize('mutation', ['wrong_panel', 'wrong_split', 'wrong_source'])
def test_rejects_mismatched_result_or_panel(tmp_path, mutation):
    protocol, result_path, results, queue = fixture(tmp_path)
    if mutation == 'wrong_source':
        spec = json.loads(protocol.read_text())
        spec['source']['manifest_sha256'] = '0' * 64
        protocol.write_text(json.dumps(spec))
    else:
        result = json.loads(result_path.read_text())
        if mutation == 'wrong_panel':
            result['splits']['test']['panel_sha256'] = '0' * 64
        else:
            result['splits']['development'] = result['splits'].pop('test')
        result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        summary.build(protocol, results, queue)
