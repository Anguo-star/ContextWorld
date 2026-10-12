#!/usr/bin/env python3
"""Wait for fixed-target confirmations, then evaluate the locked nine-task Test cohort."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_current_reference_test import sha256
from scripts.summarize_current_reference_test import summarize, write_outputs


def atomic_json(path, value):
    value['updated_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def upstream_ready(status, protocol_sha):
    if status.get('protocol_sha256') != protocol_sha:
        raise ValueError('upstream training protocol identity changed')
    if status.get('status') == 'failed':
        raise RuntimeError('upstream training failed; inspect queue before Test')
    if status.get('status') == 'completed':
        expected = {f'{task}/s{seed}' for task in ('contact_friction', 'robot_arm_mass')
                    for seed in (3073, 3074)}
        jobs = status.get('jobs', {})
        if set(jobs) != expected or any(job.get('status') != 'completed' for job in jobs.values()):
            raise ValueError('upstream completion lacks all four confirmations')
        return True
    pid = status.get('pid')
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError('upstream training has no valid process ID')
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        raise RuntimeError('upstream training process stopped before completion') from exc
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--upstream-status', type=Path, required=True)
    parser.add_argument('--upstream-protocol', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    args = parser.parse_args()
    devices = [int(d) for d in args.devices.split(',')]
    if len(devices) != len(set(devices)) or not devices or min(devices) < 0:
        parser.error('devices must be distinct non-negative GPU indices')
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'followup.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_sha = sha256(args.upstream_protocol)
    plan_sha = sha256(args.plan)
    status = dict(status='waiting_for_training', pid=os.getpid(), plan=str(args.plan),
                  plan_sha256=plan_sha, upstream_protocol_sha256=protocol_sha, devices=devices)
    state_path = root / 'followup_status.json'
    atomic_json(state_path, status)
    children = []
    logs = []
    try:
        while True:
            if sha256(args.plan) != plan_sha or sha256(args.upstream_protocol) != protocol_sha:
                raise ValueError('locked plan or training protocol changed')
            upstream = json.loads(args.upstream_status.read_text())
            if upstream_ready(upstream, protocol_sha):
                break
            status['upstream_status'] = upstream['status']
            atomic_json(state_path, status)
            time.sleep(60)
        status['status'] = 'evaluating'
        atomic_json(state_path, status)
        for shard, device in enumerate(devices):
            command = [sys.executable, str(ROOT / 'scripts/run_current_reference_test.py'),
                       'execute', '--plan', str(args.plan.resolve()),
                       '--output-root', str(root / 'results'), '--num-shards', str(len(devices)),
                       '--shard-index', str(shard), '--device', f'cuda:{device}']
            log = (root / f'shard-{shard}.log').open('a')
            logs.append(log)
            children.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        while True:
            codes = [child.poll() for child in children]
            report = summarize(args.plan, root / 'results')
            write_outputs(report, root / 'report')
            status.update(worker_pids=[child.pid for child in children], worker_returncodes=codes,
                          completed_runs=report['completed_runs'], expected_runs=report['expected_runs'])
            atomic_json(state_path, status)
            if all(code is not None for code in codes):
                break
            time.sleep(60)
        if any(code != 0 for code in codes) or report['completed_runs'] != report['expected_runs']:
            raise RuntimeError('Test report is incomplete; see per-run receipts and worker logs')
        status['status'] = 'completed'
        atomic_json(state_path, status)
    except Exception as exc:
        status.update(status='failed', error=str(exc))
        atomic_json(state_path, status)
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
