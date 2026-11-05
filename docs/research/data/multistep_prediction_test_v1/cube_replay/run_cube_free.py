#!/usr/bin/env python3
"""Run the 15 registered Cube Test checkpoints on a verified panel; no training."""
import argparse, concurrent.futures, hashlib, json, os, subprocess, time
from pathlib import Path

ROOT = Path('/tmp/cw-test-multistep-20261028')
REPO = Path('/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld')
PYTHON = '/tmp/cw-strength-ablation-20260922/venv/bin/python'
EVALUATOR = REPO / 'scripts/diagnose_cross_task_decisions.py'
REFERENCE = REPO / 'docs/research/data/current_reference_test_v1/summary.json'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def validate_receipt(path, spec, panel_sha):
    if not path.is_file(): return False
    r = read(path)
    assert r['model_id'] == spec['id'] and r['evaluation_split'] == 'test'
    assert r['panel_sha256'] == panel_sha and r['scenes'] == 256
    assert r['state_hash_before'] == r['state_hash_after'] and r['no_training']
    assert r['data_version'] == 'multistep_test_v1' and 'free' in r['modes']
    assert len(r['entry_hashes']) == 256
    return True

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', nargs='+', type=int, default=[0,1,2,3,4])
    p.add_argument('--panel', type=Path, default=ROOT/'panels/cube_gripper_carry')
    p.add_argument('--results-root', type=Path, default=ROOT/'results')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    assert a.gpus and len(set(a.gpus)) == len(a.gpus) and set(a.gpus) <= set(range(8))
    models_path = ROOT/'models.json'; protocol = read(ROOT/'protocol.json')
    assert sha(models_path) == protocol['models_sha256']
    models = [m for m in read(models_path) if m['task'] == 'cube_gripper_carry']
    assert len(models) == 15 and len({m['id'] for m in models}) == 15
    refs = {r['id']:r['checkpoint_sha256'] for cell in read(REFERENCE)['cells'] for r in cell['runs']}
    assert all(refs[m['id']] == m['checkpoint_sha256'] and Path(m['checkpoint']).is_file() for m in models)
    manifest_path = a.panel/'manifest.json'; check_path = ROOT/'checks/cube_gripper_carry.json'
    if not manifest_path.is_file() or not check_path.is_file():
        raise RuntimeError('Verified Cube Test panel is not ready')
    manifest, check = read(manifest_path), read(check_path)
    panel_sha = sha(manifest_path)
    assert check.get('ready') is True and check.get('root_verified') is True
    assert check['manifest_sha256'] == panel_sha and check['scenes'] == 256
    assert manifest['evaluation_split'] == 'test' and manifest['data_version'] == 'multistep_test_v1'
    assert len(manifest['scenes']) == 256 and len({s['scene_id'] for s in manifest['scenes']}) == 256
    assert all((a.panel/s['path']).is_file() for s in manifest['scenes'])
    for m in models:
        assert not m.get('result_dir') or Path(m['result_dir']) == a.results_root/m['id']
    if a.dry_run:
        print(json.dumps({'panel':str(manifest_path),'panel_sha256':panel_sha,'jobs':len(models),'gpus':a.gpus,'modes':['free']},indent=2))
        return
    query = subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
    state = {int(v[0]):(int(v[1]),int(v[2])) for line in query.stdout.splitlines() if (v:=[x.strip() for x in line.split(',')]) and len(v)==3}
    assert all(state[g][0] <= 10 and state[g][1] <= 1024 for g in a.gpus), f'Selected GPU is busy: {state}'
    logs = ROOT/'logs_cube_free'; logs.mkdir(exist_ok=True)
    def run_one(gpu, spec):
        dest = a.results_root/spec['id']; receipt = dest/'receipt.json'
        if validate_receipt(receipt,spec,panel_sha): return {'id':spec['id'],'reused':True}
        dest.mkdir(parents=True,exist_ok=True)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_CACHE='/opt/huawei/explorer-env/dataset/ag_data/models',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLCONFIGDIR=str(ROOT/'mpl'))
        command = [PYTHON,str(EVALUATOR),'--models',str(models_path),'--id',spec['id'],'--panel',str(a.panel),'--output',str(dest),'--device','cuda','--modes','free','--evaluation-split','test']
        log = logs/(spec['id'].replace('/','_')+'.log'); started = time.monotonic()
        with log.open('w') as f: completed = subprocess.run(command,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        if completed.returncode: raise RuntimeError(f'{spec["id"]} failed ({completed.returncode}): {log}')
        assert validate_receipt(receipt,spec,panel_sha)
        return {'id':spec['id'],'gpu':gpu,'seconds':round(time.monotonic()-started,1),'receipt':str(receipt)}
    # One thread owns each GPU and runs its assigned checkpoints sequentially.
    def gpu_worker(index, gpu):
        return [run_one(gpu, spec) for spec in models[index::len(a.gpus)]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        futures = [pool.submit(gpu_worker,i,gpu) for i,gpu in enumerate(a.gpus)]
        for f in concurrent.futures.as_completed(futures):
            for result in f.result(): print(json.dumps(result),flush=True)

if __name__ == '__main__': main()
