#!/usr/bin/env python3
"""Continue one archived 32k Mixed run from its full epoch-10 trainer state to epoch 20. Default: print-only."""
import argparse,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path('/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld')
BOOTSTRAP=REPO/'scripts/stablewm_bootstrap'
ENV_KEYS={'PATH','HOME','USER','LANG','LC_ALL','LD_LIBRARY_PATH','LD_PRELOAD','TMPDIR'}
def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def read_env(path):
 out={}
 for line in path.read_text().splitlines():
  if not line.strip() or line.lstrip().startswith('#'):continue
  k,v=line.split('=',1)
  if k in out:raise RuntimeError(f'duplicate env key: {k}')
  out[k]=v
 return out
def source_identity(source,family):
 paths={source/'scripts/train'/f'{family}.py'}
 for relative,pattern in [('stable_worldmodel','*.py'),('scripts/train/config','*.yaml')]:
  p=source/relative
  if p.is_dir():paths.update(p.rglob(pattern))
 if (source/'pyproject.toml').is_file():paths.add(source/'pyproject.toml')
 h=hashlib.sha256()
 for p in sorted(paths):
  h.update(p.relative_to(source).as_posix().encode());h.update(b'\0');h.update(bytes.fromhex(file_sha(p)))
 return {'sha256':h.hexdigest(),'file_count':len(paths)}
def training_command(family,plan):
 line=[x for x in (ROOT/family/'print_only.log').read_text().splitlines() if x.startswith('[stablewm-train] train: ')]
 assert len(line)==1,'Missing fixed print-only trainer command'
 cmd=shlex.split(line[0].split('train: ',1)[1]);source=Path(plan['stable_repo'])
 assert Path(cmd[0]).resolve()==Path(read_env(Path(plan['env_path']))['PYTHON_BIN']).resolve()
 assert Path(cmd[1]).resolve()==(source/'scripts/train'/f'{family}.py').resolve()
 overrides={x.split('=',1)[0]:x.split('=',1)[1] for x in cmd[2:] if '=' in x}
 assert overrides['trainer.max_epochs']=='20' and overrides['trainer.devices']=='4' and overrides['loader.batch_size']=='256'
 assert overrides['seed']=='3072' and overrides['wm.history_size']=='3' and overrides['wm.num_preds']=='1' and overrides['data.dataset.frameskip']=='5'
 assert overrides['output_model_name']==plan['run_name'] and overrides['subdir']==plan['run_name']
 return cmd
def preflight(family,plan):
 vals=read_env(Path(plan['env_path']));source=Path(plan['stable_repo']);parent=Path(plan['parent_full_state_checkpoint'])
 assert vals['CW_FAMILY']==family and vals['CW_MAX_EPOCHS']=='20' and vals['CW_BATCH_SIZE']=='256' and vals['CW_DEVICES']=='4'
 assert vals['CW_METHOD']=='native' and vals['CW_FREEZE_ENCODER']=='0' and vals['CW_POST_TRAIN_EVAL']=='0'
 assert not any(k in vals for k in ('CW_INIT_CHECKPOINT','CW_PREDICTION_TARGET_STOP_GRAD','CW_COMPONENT_EPOCH_SIZE','CW_CHECKPOINT_EPOCH_INTERVAL','CW_OVERRIDES'))
 assert plan['gpu_ids']==([0,1,2,3] if family=='lewm' else [4,5,6,7])
 assert source.name==plan['source_ref']==vals['CW_STABLEWM_REF']
 assert (source/'.contextworld_snapshot_complete_v1').is_file()
 old=json.loads(Path(plan['reference_identity']).read_text());identity=old['identity']
 assert old['identity_sha256']==plan['parent_identity_sha256']
 assert identity['run_name']==plan['parent_run_name'] and identity['family']==family and identity['seed']==3072
 assert identity['base_training_recipe']=='joint_scratch_v1' and 'initialization' not in identity
 assert identity['stablewm_source']==source_identity(source,family)
 assert identity['target']['dataset']['contract']['split']=='training'
 assert identity['target']['dataset']['contract']['weights']=={'original':0.5,'synthetic':0.5}
 marker=json.loads((parent.parent.parent/'contextworld_run_identity_v1.json').read_text())
 assert marker['run_name']==plan['parent_run_name'] and marker['training_identity_sha256']==plan['parent_identity_sha256']
 assert parent.is_file() and not parent.is_symlink() and file_sha(parent)==plan['parent_full_state_sha256']
 import torch
 ck=torch.load(str(parent),map_location='cpu',weights_only=False,mmap=True)
 assert ck['epoch']==plan['parent_epoch']==9 and ck['global_step']==plan['parent_global_step']==34830
 assert ck.get('state_dict') and ck.get('optimizer_states') and ck.get('lr_schedulers') and ck.get('loops')
 assert len(ck['optimizer_states'])==(1 if family=='lewm' else 2)
 assert not Path(plan['checkpoint_dir']).exists(),'Continuation output already exists; never overwrite or silently resume'
 receipt={'schema_version':'contextworld.mixed20.continuation.v1','family':family,'run_name':plan['run_name'],'parent_run_name':plan['parent_run_name'],'parent_identity_sha256':plan['parent_identity_sha256'],'parent_full_state_checkpoint':str(parent),'parent_full_state_sha256':plan['parent_full_state_sha256'],'parent_epoch':ck['epoch'],'parent_completed_epochs':10,'parent_global_step':ck['global_step'],'optimizer_state_count':len(ck['optimizer_states']),'scheduler_state_count':len(ck['lr_schedulers']),'source_ref':plan['source_ref'],'stablewm_source':identity['stablewm_source'],'gpu_ids':plan['gpu_ids'],'world_size':4,'per_rank_batch':256,'accumulate_grad_batches':1,'global_batch':1024,'new_max_epochs':20,'scheduler_policy':'restore_parent_state_without_horizon_change'}
 payload=json.dumps(receipt,sort_keys=True,separators=(',',':')).encode();receipt['continuation_identity_sha256']=hashlib.sha256(payload).hexdigest()
 receipt_path=ROOT/family/'continuation_receipt.json'
 if receipt_path.exists():assert json.loads(receipt_path.read_text())==receipt
 else:receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
 return vals,receipt
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--family',required=True,choices=['lewm','pldm']);p.add_argument('--execute',action='store_true');a=p.parse_args()
 plan=json.loads((ROOT/'plan.json').read_text())['families'][a.family]
 vals,receipt=preflight(a.family,plan);cmd=training_command(a.family,plan)
 env={k:v for k,v in os.environ.items() if k in ENV_KEYS};env.update(vals)
 env['PATH']=str(Path(vals['PYTHON_BIN']).parent)+os.pathsep+env.get('PATH','')
 env['PYTHONPATH']=os.pathsep.join([str(BOOTSTRAP),str(REPO),plan['stable_repo']])
 env.update({'CUDA_VISIBLE_DEVICES':','.join(map(str,plan['gpu_ids'])),'MASTER_ADDR':'127.0.0.1','MASTER_PORT':str(plan['master_port']),'SPT_CACHE_DIR':str(Path(plan['checkpoint_dir']).parent.parent),'STABLEWM_HOME':str(Path(plan['checkpoint_dir']).parent.parent),'CONTEXTWORLD_SPT_BRIDGE':'1','CONTEXTWORLD_STABLEWM_BUNDLE':'1','CONTEXTWORLD_DATALOADER_START_METHOD':'spawn','CONTEXTWORLD_SPT_RUN_NAME':plan['run_name'],'CONTEXTWORLD_SPT_IDENTITY_SHA256':receipt['continuation_identity_sha256'],'CONTEXTWORLD_SPT_RESUME_CHECKPOINT':plan['parent_full_state_checkpoint'],'CW_FAMILY':a.family,'HYDRA_FULL_ERROR':'1'})
 assert Path(env['STABLEWM_HOME'],'checkpoints',plan['run_name'])==Path(plan['checkpoint_dir'])
 assert 'CONTEXTWORLD_RESEARCH_INITIALIZATION' not in env and 'CW_INIT_CHECKPOINT' not in env
 marker=Path(plan['completion_marker']);assert not marker.exists(),'Completion marker already exists'
 if not a.execute:
  print(json.dumps({'mode':'print-only','family':a.family,'run_name':plan['run_name'],'cmd':cmd,'parent_checkpoint':plan['parent_full_state_checkpoint'],'parent_sha256':plan['parent_full_state_sha256'],'resume_identity':receipt['continuation_identity_sha256'],'cuda_visible_devices':env['CUDA_VISIBLE_DEVICES'],'master_port':env['MASTER_PORT'],'checkpoint_dir':plan['checkpoint_dir']},indent=2),flush=True);return
 log=Path(plan['train_log']);start=time.time()
 with log.open('w') as f:result=subprocess.run(cmd,cwd=plan['stable_repo'],env=env,stdout=f,stderr=subprocess.STDOUT)
 end=time.time()
 record={'schema_version':'contextworld.mixed20.train_exit.v1','mode':'execute','family':a.family,'run_name':plan['run_name'],'exit_code':result.returncode,'launcher':str(Path(__file__).resolve()),'launcher_pid':os.getpid(),'start_unix':start,'training_process_exit_unix':end,'train_log':str(log),'checkpoint_dir':plan['checkpoint_dir'],'parent_full_state_sha256':plan['parent_full_state_sha256'],'continuation_identity_sha256':receipt['continuation_identity_sha256']}
 tmp=marker.with_suffix('.json.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');os.replace(tmp,marker)
 print(json.dumps(record),flush=True);raise SystemExit(result.returncode)
if __name__=='__main__':main()
