import concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path
R=Path('/tmp/cw-test-multistep-20261028'); REPO=Path('/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld')
PYTHON='/tmp/cw-strength-ablation-20260922/venv/bin/python'; E=REPO/'scripts/diagnose_cross_task_decisions.py'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
p=json.loads((R/'protocol.json').read_text()); models=json.loads((R/'models.json').read_text());assert sha(R/'models.json')==p['models_sha256']
for task in ('speed','robot_arm_mass'):
 m=json.loads((R/'panels'/task/'manifest.json').read_text());c=json.loads((R/'checks'/f'{task}.json').read_text())
 assert m['evaluation_split']=='test' and m['data_version']=='multistep_test_v1' and len(m['scenes'])==p['scenes_by_task'][task]
 assert c['ready'] and c['root_verified'] and sha(R/'panels'/task/'manifest.json')==c['manifest_sha256']
jobs=[x for x in models if x['task'] in ('speed','robot_arm_mass')];print('jobs',len(jobs),flush=True)
def run(pair):
 gpu,s=pair;d=Path(s['result_dir']);d.mkdir(parents=True,exist_ok=True);log=R/'logs_gpu'/(s['id'].replace('/','_')+'_gpu05.log')
 if (d/'receipt.json').exists():return {'id':s['id'],'reused':True}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_CACHE='/opt/huawei/explorer-env/dataset/ag_data/models',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLCONFIGDIR=str(R/'mpl'))
 cmd=[PYTHON,str(E),'--models',str(R/'models.json'),'--id',s['id'],'--panel',str(R/'panels'/s['task']),'--output',str(d),'--device','cuda','--modes','free','--evaluation-split','test']
 t=time.monotonic()
 with log.open('w') as f: x=subprocess.run(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
 if x.returncode:raise RuntimeError(f'{s["id"]} exit {x.returncode}; {log}')
 receipt=json.loads((d/'receipt.json').read_text());assert receipt['evaluation_split']=='test' and receipt['state_hash_before']==receipt['state_hash_after'] and receipt['scenes']==p['scenes_by_task'][s['task']] and len(receipt['entry_hashes'])==receipt['scenes']
 return {'id':s['id'],'gpu':gpu,'seconds':round(time.monotonic()-t,1),'receipt':str(d/'receipt.json')}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 fs=[pool.submit(run,(i%5,s)) for i,s in enumerate(jobs)]
 for f in concurrent.futures.as_completed(fs):print(json.dumps(f.result()),flush=True)
