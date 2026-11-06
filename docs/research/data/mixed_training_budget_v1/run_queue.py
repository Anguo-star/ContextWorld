#!/usr/bin/env python3
"""Continue two approved Mixed runs on separate four-GPU groups."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = Path('/tmp/cw-pldm-2k-joint-20261105')
PYTHON = Path('/tmp/cw-strength-ablation-20260922/venv/bin/python')
STATUS = ROOT / 'queue_status.json'
STATE = {'state': 'prepared', 'jobs': {name: {'state': 'queued'} for name in ('lewm', 'pldm')}}
LOCK = threading.Lock()


def save(state, **details):
    with LOCK:
        STATE.update(state=state, updated_unix=time.time(), **details)
        write_state()


def write_state():
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(STATE, indent=2) + '\n')
    os.replace(temporary, STATUS)


def job_status(family, state, **details):
    with LOCK:
        STATE['jobs'][family].update(state=state, updated_unix=time.time(), **details)
        write_state()


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def wait_for_existing_job():
    save('waiting_for_existing_pldm_training_and_test')
    while True:
        training = read(PREVIOUS / 'launcher_exit.json')
        testing = read(PREVIOUS / 'test_watcher_status.json')
        if training and testing and testing.get('state') in {'test_completed', 'failed'}:
            STATE['previous_job'] = {'training': training, 'testing': testing}
            if training.get('exit_code') != 0 or testing.get('state') != 'test_completed':
                raise RuntimeError('The preceding PLDM training/Test did not complete successfully; no continuation was started')
            return
        time.sleep(30)


def wait_for_free_gpus(family, gpu_ids):
    selected = subprocess.run(
        ['nvidia-smi', '-i', ','.join(map(str, gpu_ids)), '--query-gpu=uuid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True)
    uuids = set(selected.stdout.split())
    if len(uuids) != 4:
        raise RuntimeError(f'{family}: expected four available GPU identities')
    while True:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True)
        pids = [line.strip() for line in result.stdout.splitlines()
                if line.split(',')[0].strip() in uuids]
        if not pids:
            return
        job_status(family, 'waiting_for_assigned_gpus', occupied_process_count=len(pids))
        time.sleep(30)


def execute(family, stage, command, gpu_ids):
    receipt = ROOT / family / f'queue_{stage}_exit.json'
    previous = read(receipt)
    if previous and previous.get('exit_code') == 0 and previous.get('command') == command:
        job_status(family, f'{stage}_already_completed', receipt=str(receipt))
        return True
    wait_for_free_gpus(family, gpu_ids)
    job_status(family, f'{stage}_running', command=command, gpu_ids=gpu_ids)
    record = {'family': family, 'stage': stage, 'command': command, 'gpu_ids': gpu_ids,
              'started_unix': time.time()}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with (receipt.parent / f'queue_{stage}.log').open('a') as log:
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, finished_unix=time.time())
    temporary = receipt.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    os.replace(temporary, receipt)
    job_status(family, f'{stage}_finished', exit_code=result.returncode, receipt=str(receipt))
    return result.returncode == 0


def run_family(family, plan):
    gpu_ids = plan['gpu_ids']
    try:
        if not execute(family, 'training', [str(PYTHON), str(ROOT / 'launch.py'),
                       '--family', family, '--execute'], gpu_ids):
            job_status(family, 'training_failed')
            return False
        tested = execute(family, 'test', [str(PYTHON), str(ROOT / 'evaluate_checkpoints.py'),
                         '--family', family], gpu_ids)
        job_status(family, 'completed' if tested else 'test_failed')
        return tested
    except Exception as exc:
        job_status(family, 'failed', error=str(exc))
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(read(STATUS) or STATE, indent=2))
        return
    with (ROOT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (read(STATUS) or {}).get('state') == 'completed':
            return
        for name in ['launch.py', 'evaluate_checkpoints.py', 'plan.json']:
            if not (ROOT / name).is_file():
                raise FileNotFoundError(ROOT / name)
        try:
            plans = read(ROOT / 'plan.json')['families']
            assert plans['lewm']['gpu_ids'] == [0, 1, 2, 3]
            assert plans['pldm']['gpu_ids'] == [4, 5, 6, 7]
            assert all(plans[name]['initialization'] == 'full_state_resume' for name in plans)
            for family in ('lewm', 'pldm'):
                job_status(family, 'queued', gpu_ids=plans[family]['gpu_ids'],
                           start_epoch=10, target_epoch=20, global_batch_size=1024)
            wait_for_existing_job()
            save('parallel_continuation_and_test')
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run_family, family, plans[family]) for family in ('lewm', 'pldm')]
                all_ok = all([future.result() for future in futures])
            save('completed' if all_ok else 'completed_with_failures')
            if not all_ok:
                raise SystemExit(1)
        except Exception as exc:
            save('failed', error=str(exc))
            raise


if __name__ == '__main__':
    main()
