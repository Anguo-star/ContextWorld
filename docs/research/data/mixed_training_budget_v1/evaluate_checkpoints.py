#!/usr/bin/env python3
"""Reuse epoch-10 Test and score full-state continuation epochs 15/20; no selection."""
import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path
import yaml

ROOT = Path('/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld')
HERE = Path(__file__).resolve().parent
PYTHON = Path('/tmp/cw-strength-ablation-20260922/venv/bin/python')
PLAN = ROOT / 'docs/research/data/current_reference_test_v1/plan.json'
MAIN = ROOT / 'docs/research/data/current_reference_test_v1/metrics.json'
CONTROL_PLAN = HERE / 'plan.json'
EPOCHS = (10, 15, 20)
NEW_EPOCHS = (15, 20)
STABLE = {
    'lewm': 'c3dfc4431bfdba39ad05955752feca7939773103',
    'pldm': 'c34a107e787133468369e08f8f3efa2a43db67a2',
}
BUNDLE_KEYS = ('bundle_root', 'artifact_root', 'manifest_sha256', 'task_registry_sha256')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(cond, message):
    if not cond:
        raise ValueError(message)


def write_status(path, state, **details):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps({'state': state, **details}, indent=2) + '\n')
    os.replace(temp, path)


def dataset_uri(uri):
    require(uri.startswith('contextworld://v1/'), 'Unexpected Training dataset URI')
    text = uri.removeprefix('contextworld://v1/')
    return json.loads(base64.urlsafe_b64decode(text + '=' * (-len(text) % 4)))


def local_dataset_root(root):
    old = '/opt/huawei/dataset/ag_data/'
    new = '/opt/huawei/explorer-env/dataset/ag_data/'
    if root.startswith(old):
        return new + root[len(old):]
    require(root.startswith(new), 'Unexpected Training dataset mount prefix')
    return root


def sources(family):
    plan = read(PLAN)
    baseline = next(x for x in plan['runs'] if x['id'] == f'action_strength/{family}/scratch/s3072')
    stable_ref = STABLE[family]
    require(baseline['adapter'] == family and baseline['stable_ref'] == stable_ref, 'Pinned Test adapter/stable ref differs')
    main_run = next(run for cell in read(MAIN)['cells']
                    if cell['task'] == 'action_strength' and cell['family'] == family
                    and cell['regime'] == 'scratch' for run in cell['runs'] if run['seed'] == 3072)
    ref_path = ROOT / main_run['result']
    require(sha(ref_path) == main_run['result_sha256'], 'Archived epoch-10 Test raw SHA differs')
    ref_report = read(ref_path)
    require(ref_report['evaluation_split'] == 'test' and ref_report['result']['data']['pair_count'] == 256,
            'Pinned reference is not 256-pair Test')
    require(ref_report['bundle']['manifest_sha256'] == plan['pinned_files']['bundle/manifest.jsonl']
            and ref_report['bundle']['task_registry_sha256'] == plan['pinned_files']['bundle/task_registry.json'],
            'Reference Test bundle differs from plan')
    model = ref_report['result']['model']['adapter']
    require(model['checkpoint_sha256'] == baseline['checkpoint_sha256']
            and model['stable_worldmodel_commit'] == stable_ref
            and model['stable_worldmodel_repo'] == baseline['stable_repo']
            and model['model_class'].split('.')[-1].lower() == family
            and model['parameters'] > 0, 'Baseline Test model parameters differ from plan')
    marker = Path(baseline['stable_repo']) / '.contextworld_snapshot_complete_v1'
    require(marker.read_text().splitlines()[0] == stable_ref, 'Stable repo marker differs from pinned commit')
    return plan, baseline, ref_report


def parent_source(family, family_plan, baseline, reference):
    """Prove the parent trainer checkpoint is the scored epoch-10 model."""
    import torch
    root = Path(family_plan['checkpoint_dir']).parents[1]
    parent_name = Path(baseline['checkpoint']).parent.name
    parent_identity_file = Path(family_plan['reference_identity'])
    parent_identity = read(parent_identity_file)
    require(parent_identity['identity']['run_name'] == parent_name, 'Parent training run differs from Test weights')
    candidates = []
    for marker_path in (root / 'runs').rglob('contextworld_run_identity_v1.json'):
        marker = read(marker_path)
        candidate = marker_path.parent / 'checkpoints/last.ckpt'
        if (marker.get('run_name') == parent_name
                and marker.get('training_identity_sha256') == parent_identity['identity_sha256']
                and candidate.is_file()):
            candidates.append(candidate)
    require(len(candidates) == 1, f'Expected one identity-bound parent full-state checkpoint, found {len(candidates)}')
    checkpoint = candidates[0]
    require(Path(family_plan['parent_full_state_checkpoint']).resolve() == checkpoint.resolve()
            and sha(checkpoint) == family_plan['parent_full_state_sha256']
            and family_plan['parent_run_name'] == parent_name
            and family_plan['parent_identity_sha256'] == parent_identity['identity_sha256'],
            'Plan parent checkpoint/identity differs')
    trainer = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = torch.load(Path(baseline['checkpoint']), map_location='cpu', weights_only=False)
    require(trainer['epoch'] == 9 and trainer['global_step'] == 34830,
            'Parent full-state is not the completed 10-epoch endpoint')
    require(family_plan['parent_epoch'] == trainer['epoch']
            and family_plan['parent_global_step'] == trainer['global_step']
            and family_plan['parent_completed_epochs'] == 10,
            'Plan parent progress differs')
    require(trainer['loops']['fit_loop']['epoch_progress']['total']['processed'] == 10
            and trainer['loops']['fit_loop']['epoch_loop.batch_progress']['total']['completed'] == 34830,
            'Parent fit-loop state differs')
    require(len(trainer['optimizer_states']) >= 1 and all(x['state'] for x in trainer['optimizer_states'])
            and len(trainer['lr_schedulers']) == len(trainer['optimizer_states']),
            'Parent optimizer/scheduler state missing')
    require(all('model.' + key in trainer['state_dict']
                and torch.equal(value, trainer['state_dict']['model.' + key])
                for key, value in weights.items()), 'Parent full-state model differs from scored epoch-10 weights')
    require(reference['result']['model']['adapter']['checkpoint_sha256'] == sha(Path(baseline['checkpoint'])),
            'Epoch-10 Test report checkpoint differs from parent weights')
    return {'parent_checkpoint': str(checkpoint), 'parent_checkpoint_sha256': sha(checkpoint),
            'parent_identity_file': str(parent_identity_file),
            'parent_identity_sha256': parent_identity['identity_sha256'],
            'parent_run_name': parent_name, 'parent_epoch_index': trainer['epoch'],
            'parent_processed_epochs': 10, 'parent_global_step': trainer['global_step'],
            'parent_optimizer_count': len(trainer['optimizer_states']),
            'parent_scheduler_max_steps': [x.get('max_steps') for x in trainer['lr_schedulers']],
            'epoch10_weights_sha256': sha(Path(baseline['checkpoint']))}


def training_identity(run_dir, family, family_plan):
    receipt_file = HERE / family / 'continuation_receipt.json'
    receipt = read(receipt_file)
    digest = receipt.pop('continuation_identity_sha256')
    payload = json.dumps(receipt, sort_keys=True, separators=(',', ':')).encode()
    require(hashlib.sha256(payload).hexdigest() == digest, 'Continuation receipt digest differs')
    baseline_identity = read(Path(family_plan['reference_identity']))['identity']
    require(receipt['schema_version'] == 'contextworld.mixed20.continuation.v1'
            and receipt['family'] == family and receipt['run_name'] == run_dir.name
            and receipt['parent_run_name'] == family_plan['parent_run_name']
            and receipt['parent_identity_sha256'] == family_plan['parent_identity_sha256']
            and receipt['parent_full_state_checkpoint'] == family_plan['parent_full_state_checkpoint']
            and receipt['parent_full_state_sha256'] == family_plan['parent_full_state_sha256']
            and receipt['parent_epoch'] == family_plan['parent_epoch']
            and receipt['parent_completed_epochs'] == family_plan['parent_completed_epochs']
            and receipt['parent_global_step'] == family_plan['parent_global_step']
            and receipt['optimizer_state_count'] == (1 if family == 'lewm' else 2)
            and receipt['scheduler_state_count'] == receipt['optimizer_state_count']
            and receipt['source_ref'] == family_plan['source_ref']
            and receipt['stablewm_source'] == baseline_identity['stablewm_source']
            and receipt['gpu_ids'] == family_plan['gpu_ids']
            and receipt['world_size'] == family_plan['world_size'] == 4
            and receipt['per_rank_batch'] == family_plan['local_batch_size'] == 256
            and receipt['accumulate_grad_batches'] == 1
            and receipt['global_batch'] == family_plan['global_batch_size'] == 1024
            and receipt['new_max_epochs'] == 20
            and receipt['scheduler_policy'] == 'restore_parent_state_without_horizon_change',
            'Continuation receipt differs from plan or parent')
    training_repo = Path(family_plan['stable_repo'])
    require((training_repo / '.contextworld_snapshot_complete_v1').read_text().splitlines()[0]
            == family_plan['source_ref'], 'Training snapshot commit differs from control plan')
    env_text = Path(family_plan['env_path']).read_text()
    require('CW_DEVICES=4\n' in env_text and 'CW_BATCH_SIZE=256\n' in env_text
            and 'CW_ACCUMULATE=1\n' in env_text, '4-GPU global-batch-1024 continuation configuration differs')
    proof = {'receipt_file': str(receipt_file), 'receipt_sha256': sha(receipt_file),
             'identity_sha256': digest, 'run_name': run_dir.name,
             'stablewm_source_sha256': receipt['stablewm_source']['sha256'],
             'training_stable_ref': family_plan['source_ref']}
    if not run_dir.is_dir():
        return proof
    config_file = run_dir / 'config.json'
    config_yaml = run_dir / 'config.yaml'
    require(config_file.is_file() or config_yaml.is_file(), 'Missing exported training config')
    cfg_path = config_file if config_file.is_file() else config_yaml
    config = read(cfg_path) if cfg_path == config_file else yaml.safe_load(cfg_path.read_text())
    require(config['output_model_name'] == run_dir.name and config['seed'] == 3072
            and config['trainer']['max_epochs'] == 20 and config['loader']['batch_size'] == 256
            and config['wm']['history_size'] == 3 and config['wm']['num_preds'] == 1
            and config['data']['dataset']['frameskip'] == 5, 'Training duration/architecture differs')
    baseline_path = Path(family_plan['reference_config'])
    baseline_path = baseline_path.with_suffix(cfg_path.suffix)
    baseline_config = read(baseline_path) if cfg_path.suffix == '.json' else yaml.safe_load(baseline_path.read_text())
    for key in ('model', 'wm', 'encoder_scale', 'img_size', 'patch_size',
                'projector_loss_weight', 'loss', 'optimizer'):
        require(config[key] == baseline_config[key], f'Model parameter differs from pinned {family} scratch: {key}')
    data = dataset_uri(config['data']['dataset']['name'])
    require(data['component'] == 'action_strength' and data['split'] == 'training'
            and data['payload_id'] == 'data' and data['weights'] == {'original': 0.5, 'synthetic': 0.5}
            and Path(data['root']).name == 'ContextWorld-action-strength-32k-v1',
            '32k Mixed Training source differs')
    contract = baseline_identity['target']['dataset']['contract']
    require(local_dataset_root(contract['root']) == local_dataset_root(data['root'])
            and contract['weights'] == data['weights']
            and contract['component'] == data['component']
            and contract['split'] == data['split'] == 'training'
            and contract['payload_id'] == data['payload_id'] == 'data',
            'Training dataset contract differs')
    bundle_root = Path(local_dataset_root(data['root']))
    require(sha(bundle_root / 'manifest.jsonl') == contract['manifest_sha256']
            and sha(bundle_root / 'task_registry.json') == contract['task_registry_sha256'],
            'Training bundle manifest or registry differs from parent contract')
    matches = [p for p in (run_dir.parents[1] / 'runs').rglob('contextworld_run_identity_v1.json')
               if (lambda m: m.get('run_name') == run_dir.name
                   and m.get('training_identity_sha256') == digest)(read(p))]
    require(len(matches) == 1, f'Expected one receipt-bound SPT run identity, found {len(matches)}')
    proof.update({'run_identity_file': str(matches[0]), 'run_identity_sha256': sha(matches[0]),
                  'config_file': str(cfg_path), 'config_sha256': sha(cfg_path),
                  'dataset_manifest_sha256': contract['manifest_sha256']})
    return proof


def completion(path, family, family_plan):
    require(path is not None and path.is_file(), 'Missing durable training completion marker')
    marker = read(path)
    require(marker.get('schema_version') == 'contextworld.mixed20.train_exit.v1'
            and marker.get('family') == family and marker.get('run_name') == family_plan['run_name']
            and marker.get('checkpoint_dir') == family_plan['checkpoint_dir']
            and marker.get('parent_full_state_sha256') == family_plan['parent_full_state_sha256']
            and marker.get('continuation_identity_sha256') == read(HERE / family / 'continuation_receipt.json')['continuation_identity_sha256']
            and marker.get('mode') == 'execute' and marker.get('exit_code') == 0
            and marker.get('training_process_exit_unix', 0) > marker.get('start_unix', 0),
            'Training supervisor did not report successful completion')
    return marker


def validate_test(path, expected_sha, baseline, reference):
    report = read(path)
    result = report['result']
    adapter = result['model']['adapter']
    require(report['evaluation_split'] == 'test' and result['status'] == 'completed'
            and result['data']['pair_count'] == 256 and result['metrics']['pair_count'] == 256
            and adapter['checkpoint_sha256'] == expected_sha
            and adapter['stable_worldmodel_commit'] == baseline['stable_ref']
            and result['model']['training_recipe'] == 'scratch' and result['model']['training_seed'] == 3072
            and result['model']['state_sha256_before'] == result['model']['state_sha256_after']
            and report['bundle'].get('test_payload_sha256_verified', True), f'Invalid frozen Test report: {path}')
    require(all(report['bundle'][key] == reference['bundle'][key] for key in BUNDLE_KEYS),
            f'Test bundle identity differs from existing 32k Test: {path}')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=('lewm', 'pldm'), required=True)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--completion-marker', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    args = p.parse_args()
    plan, baseline, reference = sources(args.family)
    control = read(CONTROL_PLAN)
    require(control['schema_version'] == 'contextworld.mixed20.continuation.plan.v1', 'Continuation plan schema differs')
    family_plan = control['families'][args.family]
    require(family_plan['gpu_ids'] == ([0, 1, 2, 3] if args.family == 'lewm' else [4, 5, 6, 7]),
            'Unexpected family GPU group')
    require(family_plan['checkpoint_epochs'] == list(EPOCHS)
            and family_plan['initialization'] == 'full_state_resume'
            and family_plan['epoch10_source'] == 'parent_existing'
            and family_plan['epoch15_source'] == family_plan['epoch20_source'] == 'continuation',
            'Plan does not specify parent 10/new 15/20 full-state continuation')
    run_dir = args.run_dir or Path(family_plan['checkpoint_dir'])
    completion_path = args.completion_marker or Path(family_plan['completion_marker'])
    require(run_dir.name == family_plan['run_name'], 'Run directory differs from control plan')
    parent = parent_source(args.family, family_plan, baseline, reference)
    if args.preflight:
        old_run = next(run for cell in read(MAIN)['cells']
                       if cell['task'] == 'action_strength' and cell['family'] == args.family
                       and cell['regime'] == 'scratch' for run in cell['runs'] if run['seed'] == 3072)
        validate_test(ROOT / old_run['result'], parent['epoch10_weights_sha256'], baseline, reference)
        proof = training_identity(run_dir, args.family, family_plan)
        print(json.dumps({'status': 'preflight_ok', 'family': args.family, 'test_pairs': 256,
                          'plan_sha256': sha(PLAN), 'stable_ref': baseline['stable_ref'],
                          'stable_repo': baseline['stable_repo'],
                          'training_source_ref': family_plan['source_ref'],
                          'training_gpu_ids': family_plan['gpu_ids'],
                          'training_source_identity_sha256': read(Path(family_plan['reference_identity']))['identity']['stablewm_source']['sha256'],
                          'baseline_checkpoint_sha256': baseline['checkpoint_sha256'],
                          'parent': parent,
                          'reference_model': reference['result']['model']['adapter'],
                          'training': proof, 'training_state': 'present' if run_dir.is_dir() else 'pending'}, indent=2))
        return
    proof = training_identity(run_dir, args.family, family_plan)
    marker = completion(completion_path, args.family, family_plan)
    old_report = ROOT / next(r['result'] for c in read(MAIN)['cells']
                             if c['task'] == 'action_strength' and c['family'] == args.family
                             and c['regime'] == 'scratch' for r in c['runs'] if r['seed'] == 3072)
    main_run = next(r for c in read(MAIN)['cells'] if c['task'] == 'action_strength'
                    and c['family'] == args.family and c['regime'] == 'scratch'
                    for r in c['runs'] if r['seed'] == 3072)
    require(sha(old_report) == main_run['result_sha256'], 'Archived epoch-10 Test raw SHA differs')
    validate_test(old_report, parent['epoch10_weights_sha256'], baseline, reference)
    original_metrics = reference['result']['metrics']
    rows = [{'task': 'action_strength', 'family': args.family, 'regime': 'scratch',
             'epoch': 10, 'split': 'test', 'training_seed': 3072, 'endpoint_source': 'parent_existing',
             'accuracy': 100 * original_metrics['correct_future_rate'],
             'history': 100 * original_metrics['correct_history_rate'],
             'NRE': original_metrics['latent_response']['normalized_response_error'],
             'nre': original_metrics['latent_response']['normalized_response_error'],
             'checkpoint': baseline['checkpoint'], 'checkpoint_sha256': baseline['checkpoint_sha256'],
             'source_raw_result': str(old_report), 'source_raw_result_sha256': sha(old_report),
             'model_state_sha256': reference['result']['model']['state_sha256_after'],
             'training_identity_sha256': parent['parent_identity_sha256'],
             'test_bundle': {k: reference['bundle'][k] for k in BUNDLE_KEYS}}]
    failures = []
    for epoch in NEW_EPOCHS:
        endpoint = HERE / 'results' / args.family / f'epoch{epoch}'
        status = endpoint / 'status.json'
        try:
            checkpoint = run_dir / f'weights_epoch_{epoch}.pt'
            require(checkpoint.is_file() and checkpoint.stat().st_size > 1024, f'Missing fixed endpoint: {checkpoint}')
            checkpoint_sha = sha(checkpoint)
            output = endpoint / 'test.json'
            reused = output.exists()
            if not output.exists() and not args.summarize_only:
                endpoint.mkdir(parents=True, exist_ok=True)
                cmd = [str(PYTHON), '-m', 'scripts.run_current_reference_test_compat',
                       '--task', 'action_strength', '--adapter', args.family,
                       '--checkpoint', str(checkpoint), '--model-name', f'{args.family}-mixed20-epoch{epoch}-s3072',
                       '--training-recipe', 'scratch', '--training-seed', '3072',
                       '--benchmark-root', plan['benchmark_root'], '--stablewm-repo', baseline['stable_repo'],
                       '--stablewm-ref', baseline['stable_ref'], '--evaluation-split', 'test',
                       '--history-adapter', 'native', '--prejepa-missing-context-policy', 'reject',
                       '--batch-size', '32', '--device', args.device, '--output', str(output)]
                with (endpoint / 'runner.log').open('w') as log:
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, family_plan['gpu_ids'])))
                    subprocess.run(cmd, cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            report = validate_test(output, checkpoint_sha, baseline, reference)
            m = report['result']['metrics']
            rows.append({'task': 'action_strength', 'family': args.family, 'regime': 'scratch',
                         'epoch': epoch, 'split': 'test', 'training_seed': 3072,
                         'endpoint_source': 'continuation',
                         'accuracy': 100 * m['correct_future_rate'],
                         'history': 100 * m['correct_history_rate'],
                         'NRE': m['latent_response']['normalized_response_error'],
                         'nre': m['latent_response']['normalized_response_error'],
                         'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
                         'source_raw_result': str(output), 'source_raw_result_sha256': sha(output),
                         'model_state_sha256': report['result']['model']['state_sha256_after'],
                         'training_identity_sha256': proof['identity_sha256'],
                         'test_bundle': {k: report['bundle'][k] for k in BUNDLE_KEYS}})
            write_status(status, 'completed', checkpoint_sha256=checkpoint_sha,
                         report_sha256=sha(output), reused=reused)
        except Exception as exc:
            failures.append({'epoch': epoch, 'error': str(exc)})
            write_status(status, 'failed', error=str(exc))
    summary = {'schema': 'contextworld.mixed20-fixed-endpoint-test.v1', 'evaluation_split': 'test',
               'selection': 'fixed_epochs_10_15_20_no_test_selection',
               'status': 'complete' if not failures else 'partial', 'failures': failures,
               'protocol': {'plan': str(PLAN), 'plan_sha256': sha(PLAN), 'benchmark_root': plan['benchmark_root'],
                            'test_pairs': 256, 'history_adapter': 'native', 'batch_size': 32,
                            'gpu_ids': family_plan['gpu_ids'], 'logical_eval_device': args.device,
                            'runner': 'scripts.run_current_reference_test_compat',
                            'reference_test_report_sha256': sha(ROOT / next(r['result'] for c in read(MAIN)['cells']
                                if c['task'] == 'action_strength' and c['family'] == args.family and c['regime'] == 'scratch'
                                for r in c['runs'] if r['seed'] == 3072))},
               'training': proof, 'parent': parent,
               'control_plan': str(CONTROL_PLAN), 'control_plan_sha256': sha(CONTROL_PLAN),
               'completion_marker': str(completion_path),
               'completion_marker_sha256': sha(completion_path), 'rows': rows}
    output = HERE / f'{args.family}_summary.json'
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Wrote {output}: fixed epochs {[r["epoch"] for r in rows]}')
    if failures:
        raise RuntimeError(f'{args.family}: {len(failures)} fixed Test endpoint(s) failed; see endpoint status.json')


if __name__ == '__main__':
    main()
