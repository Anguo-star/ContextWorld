#!/usr/bin/env python3
"""Run the protocol-locked LeWM fixed-target seed confirmations sequentially."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'scripts/run_lewm_fixed_visual_native_v1.py'
EVALUATOR = ROOT / 'scripts/evaluate_fixed_target_control.py'
TASKS = ('contact_friction', 'robot_arm_mass')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, status):
    status['updated_utc'] = now()
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_protocol(path, expected_hash, protocol):
    require(digest(path) == expected_hash, 'protocol changed during confirmation')
    for relative, expected in protocol['code_sha256'].items():
        source = ROOT / relative
        require(source.is_file() and digest(source) == expected, f'protocol code hash mismatch: {relative}')


def read_json(path):
    return json.loads(Path(path).read_text())


def same_path(left, right):
    return Path(left).resolve() == Path(right).resolve()


def check_training(run_dir, task_spec, seed, training):
    recipe = read_json(run_dir / 'recipe.json')
    loader = read_json(run_dir / 'loader_actual.json')
    audit = read_json(run_dir / 'train_end_audit.json')
    epoch = read_json(run_dir / f"epoch_{training['max_epochs']:02d}_audit.json")
    checkpoint = run_dir / f"weights_epoch_{training['max_epochs']}.pt"
    require(checkpoint.is_file(), f'missing final checkpoint: {checkpoint}')
    require(recipe['full_t0_sha256'] == task_spec['baseline_sha256'], 'T0 hash differs from protocol')
    require(same_path(recipe['full_t0_checkpoint'], task_spec['baseline_checkpoint']), 'T0 path differs from protocol')
    require(recipe['freeze_scope'] == 'visual' and recipe['frozen_modules'] == ['encoder', 'projector'], 'visual freeze differs')
    require(int(recipe['training_seed']) == seed and int(recipe['reference_training_seed']) == training['seeds'][0], 'training/reference seed differs')
    require(same_path(recipe['reference_run'], task_spec['reference_run']), 'reference run differs')
    require(recipe['native_config_sha256'] == task_spec['reference_config_sha256'], 'reference config differs')
    require(same_path(recipe['stablewm_repo'], task_spec['stablewm_repo']), 'pinned StableWM source differs')
    require(recipe['smoke_max_steps'] is None and recipe['devices_override'] is None, 'smoke/device override present')
    require(int(loader['world_size']) == training['world_size'], 'world size differs')
    require(int(loader['max_epochs']) == training['max_epochs'], 'epoch count differs')
    require(int(loader['max_steps']) == -1, 'step cap differs')
    require(int(loader['accumulate_grad_batches']) == training['accumulate_grad_batches'], 'gradient accumulation differs')
    require(int(loader['estimated_optimizer_steps']) == task_spec['expected_optimizer_steps'], 'optimizer steps differ')
    constructed = read_json(run_dir / 'loader_constructed.json')
    require(int(constructed['batch_size_per_rank']) == training['batch_size_per_rank'], 'batch size differs')
    require(int(audit['global_step']) == task_spec['expected_optimizer_steps'], 'training did not finish all steps')
    require(int(epoch['epoch']) == training['max_epochs'] and int(epoch['global_step']) == audit['global_step'], 'final epoch audit differs')
    require(audit['frozen_state_unchanged'] is True and epoch['frozen_state_unchanged'] is True, 'frozen state changed')
    require(audit['frozen_hashes'] == epoch['frozen_hashes'], 'final visual hash differs')
    require(audit['frozen_hashes'] == read_json(run_dir / 'initial_frozen_hashes.json'), 'visual state differs from initialization')
    return checkpoint


def verified_existing_training(run_dir, task_spec, seed, training):
    """Reuse only a complete, protocol-matching run; never reset a partial run."""
    try:
        return check_training(run_dir, task_spec, seed, training)
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f'existing run is incomplete or mismatched; inspect it manually, never reset it: {run_dir}: {exc}') from exc


def check_evaluation(result_path, task, task_spec, checkpoint, panels):
    result = read_json(result_path)
    require(result['task'] == task and set(result['splits']) == {'test'}, 'Test evaluation task/split differs')
    require(result['new_checkpoint_sha256'] == digest(checkpoint), 'evaluated checkpoint differs')
    require(result['t0_checkpoint_sha256'] == task_spec['baseline_sha256'], 'evaluation T0 differs')
    require(result['visual_identity']['bitwise_equal'] is True, 'visual identity failed')
    require(result['stable_ref'] == task_spec['stable_ref'], 'StableWM reference differs')
    require(all(value['before'] == value['after'] for value in result['model_state_unchanged'].values()), 'evaluation changed model state')
    require(result['no_fitting'] is True and result['no_test_read'] is False, 'evaluation protocol flags differ')
    manifest = Path(panels) / task / 'manifest.json'
    require(result['panel_manifest_sha256'] == digest(manifest), 'panel manifest differs')
    require(result['splits']['test']['panel_sha256'] == read_json(manifest)['splits']['test']['sha256'], 'Test panel differs')


def invoke(command, log_path):
    with log_path.open('a') as stream:
        stream.write(f'[{now()}] command: {json.dumps(command)}\n')
        stream.flush()
        return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol_hash = digest(protocol_path)
    protocol = read_json(protocol_path)
    paths = protocol['paths']
    root = (args.output_root or Path(paths['repeat_root'])).resolve()
    training = protocol['training']
    require(training['seeds'] == [3072, 3073, 3074] and training['repeat_seeds'] == [3073, 3074], 'unexpected confirmation seeds')
    require(training['max_epochs'] == 10 and training['world_size'] == 8 and training['batch_size_per_rank'] == 128, 'unexpected training budget')
    require(set(protocol['tasks']) == set(TASKS), 'unexpected task set')
    verify_protocol(protocol_path, protocol_hash, protocol)
    for task in TASKS:
        spec = protocol['tasks'][task]
        for path_key, hash_key in (('baseline_checkpoint', 'baseline_sha256'),
                                   ('completed_checkpoint', 'completed_checkpoint_sha256')):
            source = Path(spec[path_key])
            require(source.is_file() and digest(source) == spec[hash_key], f'{task}: {path_key} hash mismatch')
        reference_config = Path(spec['reference_run']) / 'config.json'
        require(reference_config.is_file() and digest(reference_config) == spec['reference_config_sha256'], f'{task}: reference config hash mismatch')
        normalization = Path(spec['normalization_manifest'])
        require(normalization.is_file() and digest(normalization) == spec['normalization_manifest_sha256'], f'{task}: normalization manifest hash mismatch')
        require(Path(spec['stablewm_repo']).is_dir(), f'{task}: missing StableWM source')
        require((Path(paths['panels']) / task / 'manifest.json').is_file(), f'{task}: missing panel manifest')
    require(Path(paths['models']).is_file() and digest(paths['models']) == protocol['models_sha256'], 'models manifest hash mismatch')
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / 'queue_status.json'
    if status_path.exists():
        status = read_json(status_path)
        require(status['protocol_sha256'] == protocol_hash, 'existing queue belongs to another protocol')
    else:
        status = {'protocol': str(protocol_path), 'protocol_sha256': protocol_hash,
                  'output_root': str(root), 'pid': os.getpid(), 'started_utc': now(), 'status': 'started', 'jobs': {}}
        write_status(status_path, status)
    status['pid'] = os.getpid()
    try:
        for seed in training['repeat_seeds']:
            for task in TASKS:
                verify_protocol(protocol_path, protocol_hash, protocol)
                spec = protocol['tasks'][task]
                name = f'{task}_lewm_fixed_target_v1_s{seed}'
                key = f'{task}/s{seed}'
                run_dir = root / 'checkpoints' / name
                eval_dir = run_dir / 'fixed_target_eval' / 'epoch_10_test'
                result_path = eval_dir / 'result.json'
                log_path = root / f'{name}.train.log'
                eval_log = root / f'{name}.test.log'
                job = status['jobs'].setdefault(key, {'run_dir': str(run_dir), 'train_log': str(log_path),
                                                       'eval_log': str(eval_log), 'result': str(result_path)})
                if run_dir.exists():
                    checkpoint = verified_existing_training(run_dir, spec, seed, training)
                else:
                    require(job.get('status') not in ('running', 'training', 'evaluating', 'completed'), f'queue says job already started but output is absent: {key}')
                    job.update(status='training', started_utc=now(), pid=os.getpid())
                    status['status'] = 'running'
                    write_status(status_path, status)
                    command = [args.python, str(TRAINER), '--reference-run', spec['reference_run'],
                               '--stablewm-repo', spec['stablewm_repo'], '--output-root', str(root),
                               '--run-name', name, '--freeze-scope', 'visual', '--seed', str(seed)]
                    code = invoke(command, log_path)
                    require(code == 0, f'training failed for {key} (exit {code}); see {log_path}')
                    checkpoint = check_training(run_dir, spec, seed, training)
                job['checkpoint'] = str(checkpoint)
                job['checkpoint_sha256'] = digest(checkpoint)
                if result_path.exists():
                    check_evaluation(result_path, task, spec, checkpoint, paths['panels'])
                else:
                    require(not eval_dir.exists(), f'partial evaluation exists; inspect before retry: {eval_dir}')
                    job.update(status='evaluating', pid=os.getpid())
                    write_status(status_path, status)
                    command = [args.python, str(EVALUATOR), '--task', task, '--checkpoint', str(checkpoint),
                               '--stablewm-repo', spec['stablewm_repo'], '--stable-ref', spec['stable_ref'],
                               '--output', str(eval_dir), '--models', paths['models'], '--panels', paths['panels'],
                               '--device', 'cuda:0', '--splits', 'test']
                    code = invoke(command, eval_log)
                    require(code == 0, f'Test evaluation failed for {key} (exit {code}); see {eval_log}')
                    check_evaluation(result_path, task, spec, checkpoint, paths['panels'])
                job.update(status='completed', completed_utc=now(), pid=os.getpid())
                write_status(status_path, status)
        status['status'] = 'completed'
        write_status(status_path, status)
    except Exception as exc:
        status['status'] = 'failed'
        status['error'] = str(exc)
        if 'job' in locals():
            job['status'] = 'failed'
            job['error'] = str(exc)
        write_status(status_path, status)
        raise


if __name__ == '__main__':
    main()
