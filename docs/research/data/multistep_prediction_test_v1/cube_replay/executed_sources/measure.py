#!/usr/bin/env python3
"""Exhaustive source-only Cube public Test replay measurement; no threshold gate or model."""
import os,sys,importlib.util,json,hashlib,time,multiprocessing as mp
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa')
sys.path[:0]=['/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld','/tmp/cw-test-multistep-20261028/panels/motion_damping/.swm_875e607fc08a']
import numpy as np, scipy, mujoco
spec=importlib.util.spec_from_file_location('rb','/tmp/cw-test-multistep-20261028/builders/physical/robot_test_coherent.py');rb=importlib.util.module_from_spec(spec);sys.modules['rb']=rb;spec.loader.exec_module(rb)
from contextworld.evaluation.cube_grasp_rule_h3_v4 import CubeGraspRuleV4Simulator,action_blocks
OUT=Path('/tmp/cw-cube-replay-20261105')
SCENES=None

def run_shard(shard):
 sim=CubeGraspRuleV4Simulator();result=[]
 for idx in range(shard*32,(shard+1)*32):
  s=SCENES[idx];entry={'index':idx,'pair_id':s['pair_id'],'source_row':s['source_row'],'modes':{},'errors':[]}
  try:
   c=rb.cube_candidate(s);blocks=action_blocks(c.action_profile)
   entry['source_identity']={'qpos_exact_float64':np.array_equal(np.asarray(c.qpos,np.float64),np.asarray(s['qpos'],np.float64)),'control_exact_float64':np.array_equal(np.asarray(c.control,np.float64),np.asarray(s['control'],np.float64)),'action_profile_sha256':hashlib.sha256(np.ascontiguousarray(blocks).tobytes()).hexdigest(),'profile_id_equal':hashlib.sha256(np.ascontiguousarray(blocks).tobytes()).hexdigest()==s['expected_profile_id'],'lance_actions_exact_both_modes':all(np.array_equal(s['episodes'][m]['blocks'],blocks) for m in s['modes'])}
   outputs={}
   for mode in s['modes']:
    o=sim._run_mode(c,mode=mode,blocks=blocks);outputs[mode]=o;src=s['episodes'][mode];frames=[]
    for i in range(4):
     diff=np.abs(o['physical_state'][i].astype(np.float64)-src['physical_state'][i].astype(np.float64));render=rb.jpeg_roundtrip(o['pixels'][i]);stored=rb.jpeg_decode(src['pixel_bytes'][i]);pixdiff=np.abs(render.astype(np.int16)-stored.astype(np.int16))
     frames.append({'frame':i,'controlled_position_max':float(diff[[0,1,2,3,4,6]].max()),'full_physical_state_max':float(diff.max()),'gripper_opening_abs':float(diff[5]),'jpeg_pixel_max':int(pixdiff.max()),'jpeg_pixel_mean':float(pixdiff.mean()),'jpeg_changed_fraction':float(np.count_nonzero(pixdiff)/pixdiff.size)})
    entry['modes'][mode]=frames
   a,b=(outputs[m] for m in s['modes']);entry['current_mode_comparison']={'raw_pixel_exact':bool(np.array_equal(a['pixels'][2],b['pixels'][2])),'jpeg_pixel_exact':bool(np.array_equal(rb.jpeg_roundtrip(a['pixels'][2]),rb.jpeg_roundtrip(b['pixels'][2]))),'full_simulator_state_max':float(np.max(np.abs(a['simulator_state'][2]-b['simulator_state'][2]))),'physical_state_max':float(np.max(np.abs(a['physical_state'][2]-b['physical_state'][2])))}
  except Exception as exc:entry['errors'].append(repr(exc))
  result.append(entry)
  if (idx+1)%8==0: print('shard',shard,'done',idx+1,flush=True)
 sim.close();p=OUT/f'shard_{shard:02d}.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');return str(p)

if __name__=='__main__':
 start=time.time();SCENES=rb.load_cube_scenes(rb.DEFAULT_BUNDLE,256,rb.DEFAULT_CUBE_H5)
 with mp.get_context('fork').Pool(8) as pool:paths=pool.map(run_shard,range(8))
 rows=[r for p in paths for r in json.loads(Path(p).read_text())];assert len(rows)==256
 frames=[f for r in rows for m in r['modes'].values() for f in m];errs=[r for r in rows if r['errors']]
 summary={'schema':'contextworld.cube-public-test-source-replay-measurement.v1','split':'test','scenes':len(rows),'modes_per_scene':2,'frames_per_mode':4,'runtime':{'python':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,'mujoco':mujoco.__version__,'stable_repo':str(rb.DEFAULT_STABLE_REPO),'stable_ref':rb.DEFAULT_STABLE_REF},'elapsed_seconds':time.time()-start,'error_count':len(errs),'identity_all':{k:all(r['source_identity'][k] for r in rows if not r['errors']) for k in ('qpos_exact_float64','control_exact_float64','profile_id_equal','lance_actions_exact_both_modes')},'current_mode_raw_pixel_exact_count':sum(r.get('current_mode_comparison',{}).get('raw_pixel_exact') is True for r in rows),'current_mode_full_simulator_state_max':max((r.get('current_mode_comparison',{}).get('full_simulator_state_max',0) for r in rows),default=0),'maxima':{k:max(f[k] for f in frames) for k in ('controlled_position_max','full_physical_state_max','gripper_opening_abs','jpeg_pixel_max','jpeg_pixel_mean','jpeg_changed_fraction')},'exceedance_counts':{'controlled_over_1e-6_frames':sum(f['controlled_position_max']>1e-6 for f in frames),'controlled_over_1e-6_scenes':sum(any(f['controlled_position_max']>1e-6 for mode in r['modes'].values() for f in mode) for r in rows),'jpeg_over_16_frames':sum(f['jpeg_pixel_max']>16 for f in frames),'jpeg_over_16_scenes':sum(any(f['jpeg_pixel_max']>16 for mode in r['modes'].values() for f in mode) for r in rows)}}
 (OUT/'results.json').write_text(json.dumps({'summary':summary,'scenes':rows},indent=2,sort_keys=True)+'\n');(OUT/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2),flush=True)
