import os,sys,numpy as np
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa')
sys.path[:0]=['/tmp/cw-cube-replay-20261105','/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld','/tmp/cw-test-multistep-20261028/panels/motion_damping/.swm_875e607fc08a']
import coherent_builder as b
from contextworld.evaluation.cube_grasp_rule_h3_v4 import CubeGraspRuleV4Simulator
s=b.load_cube_scenes(b.DEFAULT_BUNDLE,256,b.DEFAULT_CUBE_H5)[0];sim=CubeGraspRuleV4Simulator();a=b.run_cube_scene(s,sim,{});sim.close()
with np.load('/tmp/cw-cube-replay-20261105/panel_shards/00/cube-carry-v4r1-public-000000.npz',allow_pickle=False) as old:
 for k,v in a.items():
  if not np.array_equal(v,old[k]):raise RuntimeError(f'{k} differs: {v.shape}')
 print('ALL 24 ARRAYS EXACT',flush=True)
