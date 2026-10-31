"""Re-render the selected pair for the paper; never changes evaluation observations."""
import os,sys,json
os.environ['MUJOCO_GL']='osmesa';os.environ['PYOPENGL_PLATFORM']='osmesa'
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
if os.environ.get('CONTEXTWORLD_STABLE_WORLDMODEL_REPO'):
    sys.path.insert(0,os.environ['CONTEXTWORLD_STABLE_WORLDMODEL_REPO'])
ASSETS=HERE/'assets/cube_pair_hd'
import numpy as np,mujoco
from PIL import Image
from contextworld.evaluation.cube_grasp_rule_h3 import CubeGraspRuleCandidate
from contextworld.evaluation.cube_grasp_rule_h3_v4 import CubeGraspRuleV4Simulator,make_v4_candidate
c=json.loads((ASSETS/'candidate.json').read_text());c.pop('action_profile');c=make_v4_candidate(CubeGraspRuleCandidate(**c))
sim=CubeGraspRuleV4Simulator(resolution=224);sim._install_candidate(c)
m=sim.base._model;d=sim.base._data
# Visualization only: common colors and lighting; no dynamic parameters change.
m.vis.global_.offwidth=1800;m.vis.global_.offheight=1100;m.vis.quality.offsamples=8;m.vis.quality.shadowsize=4096
for i in range(m.nmat):
    if m.mat_rgba[i,3]>0:
        m.mat_rgba[i,:3]=[.9,.91,.93]
        m.mat_specular[i]=.18;m.mat_shininess[i]=.2
for i in range(m.ngeom):
    name=m.geom(i).name
    if name=='floor':m.geom_matid[i]=-1;m.geom_rgba[i]=[.58,.60,.63,1]
    elif name.startswith('wall_') or name.startswith('target_'):m.geom_rgba[i,3]=0
    elif name=='object_0':m.geom_rgba[i]=[.025,.22,.72,1]
    elif m.geom_matid[i]<0 and m.geom_rgba[i,3]>0:m.geom_rgba[i,:3]=[.45,.46,.49]
for i in range(m.ntex):
    if int(m.tex_type[i])==2:
        start=int(m.tex_adr[i]);n=int(m.tex_width[i]*m.tex_height[i]*m.tex_nchannel[i]);m.tex_data[start:start+n]=225
m.vis.headlight.ambient=[.24,.24,.24];m.vis.headlight.diffuse=[.55,.55,.55];m.vis.headlight.specular=[.15,.15,.15]
for i in range(m.nlight):m.light_ambient[i]=[.07,.07,.07];m.light_diffuse[i]=[.4,.4,.4]
r=mujoco.Renderer(m,height=900,width=1600);data=np.load(ASSETS/'replay_states.npz')
source=json.loads((HERE/'assets/cube_pair/provenance.json').read_text())
errors=[]
for mode in ['cannot_hold','can_hold']:
    original=np.array([source['frames'][mode][str(t)]['physical_state'] for t in range(4)])
    errors.append(np.abs(data[mode+'_physical_state']-original))
    actions=np.array([source['frames'][mode][str(t)]['action_block'] for t in range(4)]).reshape(4,5,5)
    assert np.array_equal(data[mode+'_action_blocks'],actions)
errors=np.concatenate(errors)
assert errors[:,2:5].max()<1e-6, 'Cube replay differs from the recorded trajectory'

def restore(mode,t):
 a=data[mode+'_simulator_state'][t];off=1;d.time=a[0]
 for field in ['qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','qacc_warmstart']:
  arr=getattr(d,field);n=arr.size;arr[:]=a[off:off+n].reshape(arr.shape);off+=n
 mujoco.mj_forward(m,d)
out=ASSETS
cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
cam.lookat[:]=[c.qpos[14],c.qpos[15],.065]
cam.distance=.26;cam.azimuth=135;cam.elevation=-30
frames={}
for mode in ['cannot_hold','can_hold']:
    frames[mode]={}
    for t in range(4):
        restore(mode,t)
        r.update_scene(d,camera=cam,scene_option=sim.base._scene_option)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        im=r.render()
        name=f'{mode}_t{t}.png';Image.fromarray(im).save(out/name)
        # Locate the actual visible cube, for accurate baseline/arrow placement.
        mask=(im[:,:,2].astype(float)>im[:,:,0]*1.5)&(im[:,:,2].astype(float)>im[:,:,1]*1.3)&(im[:,:,2]>50)
        yy,xx=np.where(mask)
        frames[mode][str(t)]={'file':name,'cube_bbox':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]}
    print('Rendered',mode,flush=True)
assert (out/'cannot_hold_t2.png').read_bytes()==(out/'can_hold_t2.png').read_bytes()
info={'pair_id':c.candidate_id,'render_resolution':[1600,900],
      'source_observations':'../cube_pair/provenance.json',
      'purpose':'Display rendering of a replay; not the pixels supplied to evaluated models.',
      'shared_display_changes':['camera','neutral robot and table palette','blue cube','lighting','shadow-map rendering disabled'],
      'camera':{'azimuth':135,'elevation':-30,'distance':.26,'lookat':cam.lookat.tolist()},
      'cube_state_max_abs_replay_error_m':float(errors[:,2:5].max()),
      'robot_position_max_abs_replay_error_m':float(errors[:,:2].max()),
      'gripper_opening_max_abs_replay_error':float(errors[:,6].max()),
      'cube_lift_m':float(data['can_hold_physical_state'][3,4]-data['cannot_hold_physical_state'][3,4]),
      'current_render_identical':True,'frames':frames}
(out/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
r.close();sim.close()
