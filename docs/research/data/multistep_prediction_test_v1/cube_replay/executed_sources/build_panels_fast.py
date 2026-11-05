#!/usr/bin/env python3
"""Build all 256 coherent Cube Test panels, preserving every registered source identity."""
import os,sys,json,hashlib,time,multiprocessing as mp,traceback
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa')
sys.path[:0]=['/tmp/cw-cube-replay-20261105','/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld','/tmp/cw-test-multistep-20261028/panels/motion_damping/.swm_875e607fc08a']
import numpy as np,scipy,mujoco
import coherent_builder as b
from contextworld.evaluation.cube_grasp_rule_h3_v4 import CubeGraspRuleV4Simulator,action_blocks
ROOT=Path('/tmp/cw-cube-replay-20261105/panel_shards_fast');ROOT.mkdir(exist_ok=True)
SCENES=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def worker(shard):
 out=ROOT/f'{shard:02d}';out.mkdir(exist_ok=True);sim=CubeGraspRuleV4Simulator();rows=[]
 for idx in range(shard*8,(shard+1)*8):
  scene=SCENES[idx];rec={'index':idx,'pair_id':scene['pair_id'],'source_row':scene['source_row'],'source_episode':scene['source_episode'],'source_step':scene['source_step'],'catalog_index':scene['catalog_index'],'status':'started'}
  try:
   candidate=b.cube_candidate(scene);blocks=action_blocks(candidate.action_profile)
   rec['source_identity']={'qpos_float64_equal':bool(np.array_equal(np.asarray(candidate.qpos,np.float64),np.asarray(scene['qpos'],np.float64))),'control_float64_equal':bool(np.array_equal(np.asarray(candidate.control,np.float64),np.asarray(scene['control'],np.float64))),'action_profile_id':scene['expected_profile_id'],'action_profile_sha256':hashlib.sha256(np.ascontiguousarray(blocks).tobytes()).hexdigest(),'lance_action_equal':bool(all(np.array_equal(scene['episodes'][mode]['blocks'],blocks) for mode in scene['modes']))}
   if not all((rec['source_identity']['qpos_float64_equal'],rec['source_identity']['control_float64_equal'],rec['source_identity']['action_profile_id']==rec['source_identity']['action_profile_sha256'],rec['source_identity']['lance_action_equal'])):raise RuntimeError('source identity mismatch')
   arr=b.run_cube_scene(scene,sim,rec);native=list(arr['candidate_ids']).index('amp_+1.0_repeat')
   assert np.array_equal(arr['goal_pixels'],arr['future_pixels'][0,native,0]);assert np.array_equal(arr['goal_state'],arr['future_states'][0,native,0]);assert np.array_equal(arr['history_pixels'][0,-1],arr['history_pixels'][1,-1]);assert arr['future_pixels'].shape==(2,14,5,224,224,3)
   path=out/(scene['pair_id']+'.npz');np.savez_compressed(path,**arr)
   rec.update(status='complete',npz=path.name,npz_sha256=sha(path),npz_bytes=path.stat().st_size,source_exact_replay_passed=bool(scene['audits']['source_exact_replay_passed']),source_controlled_position_max=float(arr['source_controlled_position_max']),source_history_pixel_max=int(arr['source_history_pixel_max']),native_goal_equal=True,current_pixel_equal=True,current_full_simulator_state_max=float(scene['audits']['current_full_simulator_state_max']),paired_audit_passed=bool(scene['audits']['paired_audit_passed']),candidate_raw_prefix_equal=bool(scene['audits']['candidate_raw_prefix_equal']),candidate_prefix_equal=True,current_full_state_tolerance=1e-10,render_mode='coherent_test_source_replay',source_audits=scene['audits'])
  except Exception as e:rec.update(status='failed',error=repr(e),traceback=traceback.format_exc())
  rows.append(rec)
  receipt_path=out/(scene['pair_id']+'.receipt.json')
  temporary=receipt_path.with_suffix('.json.tmp')
  temporary.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
  temporary.replace(receipt_path)
  print('shard',shard,'scene',idx,rec['status'],flush=True)
 sim.close();(out/'receipt.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n');return str(out/'receipt.json')

if __name__=='__main__':
 started=time.time();SCENES=b.load_cube_scenes(b.DEFAULT_BUNDLE,256,b.DEFAULT_CUBE_H5)
 assert len(SCENES)==256 and len({s['pair_id'] for s in SCENES})==256
 with mp.get_context('fork').Pool(32) as pool:receipts=pool.map(worker,range(32))
 rows=[r for p in receipts for r in json.loads(Path(p).read_text())]
 report={'schema':'contextworld.cube-coherent-test-panel.v1','render_mode':'coherent_test_source_replay','source_exact_replay_criterion_unchanged':True,'source_exact_replay_is_required_for_panel_validity':False,'source_measurement_archive':'/tmp/cw-cube-replay-20261105/results.json','source_bundle':str(b.DEFAULT_BUNDLE),'source_lance':b.CUBE_DATA,'source_h5':str(b.DEFAULT_CUBE_H5),'stable_repo':str(b.DEFAULT_STABLE_REPO),'stable_ref':b.DEFAULT_STABLE_REF,'physics_module':'contextworld/evaluation/cube_grasp_rule_h3_v4.py','runtime':{'numpy':np.__version__,'scipy':scipy.__version__,'mujoco':mujoco.__version__},'count':len(rows),'complete':sum(r['status']=='complete' for r in rows),'failed':sum(r['status']=='failed' for r in rows),'source_exact_replay_passed':sum(r.get('source_exact_replay_passed') is True for r in rows),'elapsed_seconds':time.time()-started,'scenes':rows}
 (ROOT/'manifest.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({k:report[k] for k in ('count','complete','failed','source_exact_replay_passed','elapsed_seconds')},indent=2),flush=True)
 raise SystemExit(report['failed']!=0 or report['complete']!=256)
