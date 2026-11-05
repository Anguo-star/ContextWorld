import sys,importlib.util,json,numpy as np
sys.path[:0]=['/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld','/tmp/cw-test-multistep-20261028/panels/motion_damping/.swm_875e607fc08a']
spec=importlib.util.spec_from_file_location('rb','/tmp/cw-test-multistep-20261028/builders/physical/robot_test_coherent.py');rb=importlib.util.module_from_spec(spec);sys.modules['rb']=rb;spec.loader.exec_module(rb)
scenes=rb.load_cube_scenes(rb.DEFAULT_BUNDLE,256,rb.DEFAULT_CUBE_H5)
rows=[]
for s in scenes:
 a=s['episodes']['cannot_hold']['physical_state'];b=s['episodes']['can_hold']['physical_state'];d=b-a
 rows.append({'pair_id':s['pair_id'],'frame3_cube_response_norm_m':float(np.linalg.norm(d[3,2:5])),'frame3_cube_response_max_m':float(np.max(np.abs(d[3,2:5]))),'frame2_full_physical_response_max':float(np.max(np.abs(d[2]))),'frame3_full_physical_response_max':float(np.max(np.abs(d[3])))})
x=np.array([r['frame3_cube_response_norm_m'] for r in rows]);out={'count':len(rows),'frame3_cube_response_norm_m':{'min':float(x.min()),'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'max':float(x.max()),'mean':float(x.mean()),'nonzero_count':int((x>0).sum())},'frame2_full_physical_response_max':max(r['frame2_full_physical_response_max'] for r in rows),'frame3_full_physical_response_max':max(r['frame3_full_physical_response_max'] for r in rows)}
from pathlib import Path
p=Path('/tmp/cw-cube-replay-20261105');(p/'source_response.json').write_text(json.dumps({'summary':out,'scenes':rows},indent=2)+'\n');print(json.dumps(out,indent=2))
