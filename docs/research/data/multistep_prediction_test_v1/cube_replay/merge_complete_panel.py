#!/usr/bin/env python3
"""Stage a complete coherent Cube Test panel; never mark it ready or replace old panels."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
import numpy as np
ROOT=Path('/tmp/cw-cube-replay-20261105')
BUNDLE=Path('/opt/huawei/explorer-env/dataset/ag_data/data/world_model/ContextWorld-v1-full')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--shards',type=Path,default=ROOT/'panel_shards_fast')
 p.add_argument('--output',type=Path,default=ROOT/'complete_panel_staging')
 a=p.parse_args();assert not a.output.exists(),f'output already exists: {a.output}'
 source=read(ROOT/'results.json');src=source['scenes'];assert len(src)==256 and source['summary']['scenes']==256 and source['summary']['error_count']==0
 ids=[x['pair_id'] for x in src];assert len(set(ids))==256 and ids==sorted(ids)
 registry_path=BUNDLE/'task_registry.json';registry=read(registry_path)
 component=next(x for x in registry['components'] if x['component_id']=='cube_gripper_carry')
 payload=component['public_test_evaluation']['payloads'][0]['public_path'];assert payload=='artifacts/synthesis/cube_gripper_carry_rule_h3_v4r1_release_projection_v1/validation.lance'
 bundle_manifest=BUNDLE/'manifest.jsonl';members=[json.loads(line) for line in bundle_manifest.read_text().splitlines() if line.strip()]
 members=[x for x in members if x.get('role')=='dataset_payload' and x.get('component')=='cube_gripper_carry' and x.get('split')=='test' and x['path'].startswith(payload)]
 assert members and all((BUNDLE/x['path']).is_file() and
                        (BUNDLE/x['path']).stat().st_size==x['bytes'] and
                        sha(BUNDLE/x['path'])==x['sha256'] for x in members)
 receipts=[]
 for index,identity in enumerate(src):
  scene_id=identity['pair_id'];base=a.shards/f'{index//8:02d}'
  receipt_path=base/(scene_id+'.receipt.json');npz_path=base/(scene_id+'.npz')
  assert receipt_path.is_file() and npz_path.is_file(),scene_id
  row=read(receipt_path);assert row['index']==index and row['pair_id']==scene_id and row['status']=='complete'
  assert row['source_row']==identity['source_row'] and row['render_mode']=='coherent_test_source_replay'
  si=row['source_identity'];assert si['qpos_float64_equal'] and si['control_float64_equal'] and si['lance_action_equal'] and si['action_profile_id']==si['action_profile_sha256']
  aud=row['source_audits'];assert aud['paired_audit_passed'] and aud['current_raw_pixel_equal'] and aud['candidate_raw_prefix_equal']
  assert aud['current_full_simulator_state_max']<=1e-10 and row['native_goal_equal'] and row['current_pixel_equal'] and row['candidate_prefix_equal']
  assert row['npz_sha256']==sha(npz_path) and row['npz_bytes']==npz_path.stat().st_size
  receipts.append((row,receipt_path,npz_path))
 a.output.mkdir(parents=True)
 (a.output/'receipts').mkdir()
 scenes=[];candidate_ids=None
 for index,(row,receipt_path,npz_path) in enumerate(receipts):
  scene_id=row['pair_id']
  with np.load(npz_path,allow_pickle=False) as arrays:
   assert len(arrays.files)==24,(scene_id,len(arrays.files))
   assert arrays['history_pixels'].shape==(2,3,224,224,3)
   assert arrays['context_actions'].shape==(2,2,5,5)
   assert arrays['candidate_actions'].shape==(14,5,5,5)
   assert arrays['future_pixels'].shape==(2,14,5,224,224,3)
   assert arrays['future_states'].shape==(2,14,5,7)
   assert arrays['physical_cost'].shape==(2,14,5)
   assert np.array_equal(arrays['history_pixels'][0,-1],arrays['history_pixels'][1,-1])
   assert bool(arrays['source_exact_replay_passed'])==row['source_exact_replay_passed']
   ids_here=arrays['candidate_ids'].tolist()
   if candidate_ids is None:candidate_ids=ids_here
   assert ids_here==candidate_ids and 'amp_+1.0_repeat' in ids_here
   native=ids_here.index('amp_+1.0_repeat')
   assert np.array_equal(arrays['goal_pixels'],arrays['future_pixels'][0,native,0])
   assert np.array_equal(arrays['goal_state'],arrays['future_states'][0,native,0])
  os.link(npz_path,a.output/npz_path.name)
  shutil.copy2(receipt_path,a.output/'receipts'/receipt_path.name)
  scenes.append({'scene_id':scene_id,'path':npz_path.name,'sha256':row['npz_sha256'],'bootstrap_cluster':f"episode:{row['source_episode']}",'source_row':row['source_row'],'source_episode':row['source_episode'],'source_step':row['source_step'],'catalog_index':row['catalog_index'],'render_mode':'coherent_test_source_replay','source_exact_replay_passed':row['source_exact_replay_passed'],'source_controlled_position_max':row['source_controlled_position_max'],'source_history_pixel_max':row['source_history_pixel_max'],'paired_audit_passed':True,'current_raw_pixel_equal':True,'candidate_raw_prefix_equal':True,'current_full_simulator_state_max':row['source_audits']['current_full_simulator_state_max'],'receipt':f'receipts/{receipt_path.name}','receipt_sha256':sha(receipt_path)})
 manifest={'schema':'contextworld.multistep_test_panel.v1','task':'cube_gripper_carry','evaluation_split':'test','data_version':'multistep_test_v1','render_mode':'coherent_test_source_replay','source_exact_replay_criterion_unchanged':True,'source_exact_replay_is_required_for_panel_validity':False,'source_exact_replay_passed_count':sum(x['source_exact_replay_passed'] for x in scenes),'scene_count':256,'scene_ids':ids,'scenes':scenes,'bootstrap_clusters':{x['scene_id']:x['bootstrap_cluster'] for x in scenes},'normalization':component['development_evaluation']['action_normalization'],'physical_steps':[5,10,15,20,25],'candidate_ids':candidate_ids,'candidate_count':14,'source_measurement':{'path':str(ROOT/'results.json'),'sha256':sha(ROOT/'results.json'),'summary_path':str(ROOT/'summary.json'),'summary_sha256':sha(ROOT/'summary.json')},'test_source':{'bundle':str(BUNDLE),'registry':str(registry_path),'registry_sha256':sha(registry_path),'manifest':str(bundle_manifest),'manifest_sha256':sha(bundle_manifest),'registered_payload':payload,'manifest_members':members,'manifest_member_hashes_verified':True},'shard_source':str(a.shards),'builder':str(ROOT/'build_panels_fast.py'),'builder_sha256':sha(ROOT/'build_panels_fast.py'),'physics_builder':str(ROOT/'coherent_builder.py'),'physics_builder_sha256':sha(ROOT/'coherent_builder.py'),'root_verified':False,'review_required':True}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'staging':str(a.output),'scenes':len(scenes),'source_exact_replay_passed':manifest['source_exact_replay_passed_count'],'manifest_sha256':sha(a.output/'manifest.json'),'ready':False},indent=2))
if __name__=='__main__':main()
