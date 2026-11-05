#!/usr/bin/env python3
"""Dispatch frozen Test prediction jobs after each task panel is finalized."""
import argparse
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld')
EVALUATOR = REPO / 'scripts/diagnose_cross_task_decisions.py'
PYTHON = '/tmp/cw-strength-ablation-20260922/venv/bin/python'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def validated_receipt(path, spec, panel_sha, expected, *, shard=None):
    if not path.is_file():
        return False
    receipt = read(path)
    if receipt.get('model_id') != spec['id'] or receipt.get('evaluation_split') != 'test' or receipt.get('data_version') != 'multistep_test_v1':
        raise RuntimeError(f'Invalid Test provenance: {path}')
    if receipt.get('panel_sha256') != panel_sha or receipt.get('scenes') != expected:
        raise RuntimeError(f'Invalid panel/count receipt: {path}')
    if set(receipt.get('modes', [])) != {'free', 'full'} or not receipt.get('no_training'):
        raise RuntimeError(f'Invalid modes or training state: {path}')
    if receipt.get('state_hash_before') != receipt.get('state_hash_after'):
        raise RuntimeError(f'Model state changed: {path}')
    if len(receipt.get('entry_hashes', {})) != expected:
        raise RuntimeError(f'Incomplete receipt: {path}')
    for name, digest in receipt['entry_hashes'].items():
        source = path.parent / name
        if sha(source) != digest:
            raise RuntimeError(f'Prediction hash mismatch: {source}')
        row = read(source)
        if row.get('model_id') != spec['id'] or row.get('checkpoint_sha256') != spec['checkpoint_sha256']:
            raise RuntimeError(f'Checkpoint mismatch: {source}')
        if set(row.get('modes', {})) != {'free', 'full'} or any('wrong_history_prediction_error_sum' not in row['modes'][mode] for mode in ('free', 'full')):
            raise RuntimeError(f'Incomplete diagnostic modes: {source}')
        if sha(source.with_suffix('.npz')) != row.get('array_sha256'):
            raise RuntimeError(f'Array hash mismatch: {source}')
    return True


def merge(spec, panel_sha, scene_count):
    destination = Path(spec['result_dir'])
    final = destination / 'receipt.json'
    if validated_receipt(final, spec, panel_sha, scene_count):
        return
    receipts = []
    entries = {}
    for shard in range(4):
        base = destination / 'shards' / str(shard)
        path = base / f'shard{shard}_receipt.json'
        count = len([i for i in range(scene_count) if i % 4 == shard])
        if not validated_receipt(path, spec, panel_sha, count, shard=shard):
            raise RuntimeError(f'Missing shard receipt: {path}')
        receipt = read(path)
        receipts.append(receipt)
        for name, digest in receipt['entry_hashes'].items():
            if name in entries:
                raise RuntimeError(f'Duplicate shard entry: {name}')
            entries[name] = digest
            for source in (base / name, (base / name).with_suffix('.npz')):
                target = destination / source.name
                if target.exists():
                    if sha(target) != sha(source):
                        raise RuntimeError(f'Existing partial output differs: {target}')
                else:
                    os.link(source, target)
    if len(entries) != scene_count or len({r['state_hash_before'] for r in receipts}) != 1:
        raise RuntimeError(f'Shard merge identity/count mismatch: {spec["id"]}')
    merged = dict(receipts[0], entry_hashes=entries, scenes=scene_count, shards=4)
    atomic(final, merged)
    validated_receipt(final, spec, panel_sha, scene_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', type=int, default=[6, 7])
    parser.add_argument('--workers-per-gpu', type=int, choices=(1, 2), default=1)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--ready-timeout', type=int, default=3600)
    args = parser.parse_args()
    if not args.gpus or set(args.gpus) - {6, 7} or len(set(args.gpus)) != len(args.gpus):
        parser.error('Only distinct GPU IDs 6 and 7 are allowed')
    protocol = read(ROOT / 'protocol.json')
    models_path = ROOT / 'models.json'
    if sha(models_path) != protocol['models_sha256']:
        raise RuntimeError('Frozen models manifest SHA mismatch')
    models = read(models_path)
    tasks = args.tasks or list(protocol['scenes_by_task'])
    if len(set(tasks)) != len(tasks) or set(tasks) - set(protocol['scenes_by_task']):
        parser.error('Unknown or duplicate task')
    evaluator_sha = sha(EVALUATOR)
    metadata = {'evaluator': str(EVALUATOR), 'evaluator_sha256': evaluator_sha,
                'models_sha256': sha(models_path), 'protocol_sha256': sha(ROOT / 'protocol.json'),
                'gpus': args.gpus, 'workers_per_gpu': args.workers_per_gpu, 'tasks': tasks,
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if args.dry_run:
        print(json.dumps({**metadata, 'jobs': sum(m['task'] in tasks for m in models)}, indent=2))
        return 0
    ROOT.joinpath('logs_gpu').mkdir(exist_ok=True)
    metadata_path = ROOT / 'execution_metadata.json'
    if metadata_path.exists() and read(metadata_path)['evaluator_sha256'] != evaluator_sha:
        raise RuntimeError('Evaluator code differs from earlier suite execution')
    if not metadata_path.exists():
        atomic(metadata_path, metadata)
    jobs = queue.Queue()
    records = []
    lock = threading.Lock()
    stop = threading.Event()
    failures = []
    panels = {}

    def record(item):
        with lock:
            records.append(item)
            atomic(ROOT / 'execution_progress.json', records)
            print(json.dumps(item), flush=True)

    def worker(gpu):
        while True:
            try:
                spec, shard, shards = jobs.get(timeout=1)
            except queue.Empty:
                if stop.is_set():
                    return
                continue
            task = spec['task']
            panel_sha, count = panels[task]
            destination = Path(spec['result_dir'])
            final = destination / 'receipt.json'
            try:
                if validated_receipt(final, spec, panel_sha, count):
                    record({'id': spec['id'], 'shard': shard, 'reused': True, 'exit_code': 0})
                    continue
                if shards > 1:
                    destination = destination / 'shards' / str(shard)
                receipt = destination / (f'shard{shard}_receipt.json' if shards > 1 else 'receipt.json')
                expected = len(range(shard, count, shards))
                if validated_receipt(receipt, spec, panel_sha, expected):
                    record({'id': spec['id'], 'shard': shard, 'reused': True, 'exit_code': 0})
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                command = [PYTHON, str(EVALUATOR), '--models', str(models_path), '--id', spec['id'],
                           '--panel', str(ROOT / 'panels' / task), '--output', str(destination),
                           '--device', 'cuda', '--modes', 'free', 'full', '--evaluation-split', 'test',
                           '--shard', str(shard), '--shards', str(shards)]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                                   HF_HUB_CACHE='/opt/huawei/explorer-env/dataset/ag_data/models',
                                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                                   MPLCONFIGDIR=str(ROOT / 'mpl'))
                log = ROOT / 'logs_gpu' / (spec['id'].replace('/', '_') + (f'_shard{shard}' if shards > 1 else '') + '.log')
                started = time.monotonic()
                if sha(EVALUATOR) != evaluator_sha:
                    raise RuntimeError('Evaluator code changed mid-run')
                with log.open('w') as stream:
                    completed = subprocess.run(command, env=environment, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
                elapsed = round(time.monotonic() - started, 1)
                if completed.returncode:
                    raise RuntimeError(f'Evaluator exit {completed.returncode}; see {log}')
                validated_receipt(receipt, spec, panel_sha, expected)
                record({'id': spec['id'], 'shard': shard, 'shards': shards, 'gpu': gpu,
                        'seconds': elapsed, 'log': str(log), 'exit_code': 0})
            except Exception as exc:
                error = {'id': spec['id'], 'shard': shard, 'gpu': gpu, 'exit_code': 1, 'error': str(exc)}
                failures.append(error)
                record(error)
            finally:
                jobs.task_done()

    deadline = time.monotonic() + args.ready_timeout
    with ThreadPoolExecutor(max_workers=len(args.gpus) * args.workers_per_gpu) as pool:
        futures = [pool.submit(worker, gpu) for gpu in args.gpus for _ in range(args.workers_per_gpu)]
        pending = set(tasks)
        try:
            while pending and time.monotonic() < deadline:
                for task in list(pending):
                    marker = ROOT / 'checks' / f'{task}.json'
                    if not marker.exists():
                        continue
                    check = read(marker)
                    if not (check.get('ready') and check.get('root_verified')):
                        continue
                    manifest = ROOT / 'panels' / task / 'manifest.json'
                    count = protocol['scenes_by_task'][task]
                    if check.get('scenes') != count or check.get('manifest_sha256') != sha(manifest):
                        raise RuntimeError(f'Readiness marker identity mismatch: {task}')
                    panel = read(manifest)
                    if (panel.get('evaluation_split') != 'test' or panel.get('data_version') != protocol['data_version']
                            or len(panel.get('scenes', [])) != count or not panel.get('root_verified')):
                        raise RuntimeError(f'Panel protocol mismatch: {task}')
                    panels[task] = (check['manifest_sha256'], count)
                    group = [m for m in models if m['task'] == task]
                    for spec in group:
                        shards = 4 if task == 'action_delay' and spec['family'] == 'dinowm' else 1
                        for shard in range(shards):
                            jobs.put((spec, shard, shards))
                    pending.remove(task)
                    print(f'dispatched {task}: {len(group)} models', flush=True)
                if pending:
                    time.sleep(5)
            if pending:
                failures.append({'error': f'Readiness timeout after {args.ready_timeout}s: {sorted(pending)}'})
        except Exception as exc:
            failures.append({'error': f'Fatal dispatch/provenance error: {exc}'})
        finally:
            stop.set()
            jobs.join()
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    failures.append({'error': f'Worker failure: {exc}'})
    for spec in models:
        if spec['task'] in tasks and spec['task'] == 'action_delay' and spec['family'] == 'dinowm' and spec['task'] in panels:
            try:
                merge(spec, *panels[spec['task']])
            except Exception as exc:
                failures.append({'id': spec['id'], 'error': f'Merge failed: {exc}'})
    if sha(EVALUATOR) != evaluator_sha:
        failures.append({'error': 'Evaluator code changed during execution'})
    atomic(ROOT / 'execution.json', {'metadata': metadata, 'records': records, 'failures': failures})
    for failure in failures:
        print(json.dumps(failure), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
