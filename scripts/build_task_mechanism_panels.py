#!/usr/bin/env python3
"""Build fixed native-query Training/Development caches from registered releases."""
from __future__ import annotations
import argparse, hashlib, io, json, shutil
from pathlib import Path
import numpy as np
import lance
from PIL import Image

BASE=Path('/opt/huawei/explorer-env/dataset/ag_data/data/world_model')
SPECS={
 'action_strength':('ContextWorld-action-strength-32k-v1','pusht-action-strength','hidden_action_scale',['low_gain','high_gain']),
 'contact_friction':('ContextWorld-contact-friction-32k-v1','pusht-contact-friction','hidden_contact_friction',['low_friction','high_friction']),
 'motion_damping':('ContextWorld-motion-damping-32k-v1','pusht-motion-damping','hidden_motion_damping',['faster_decay','slower_decay']),
 'robot_arm_mass':('ContextWorld-robot-arm-mass-32k-v1','reacher-arm-mass','hidden_arm_density',['lighter','heavier']),
 'portal_exit':('ContextWorld-portal-exit-32k-v1','tworoom-portal-exit','hidden_portal_exit',['near_border','farther_from_border']),
 'cube_gripper_carry':('ContextWorld-cube-gripper-carry-10k-independent-v2','cube-gripper-carry','hidden_grasp_enabled',['cannot_hold','can_hold']),
}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def digest(s):return hashlib.sha256(str(s).encode()).hexdigest()
def frame(data):
 with Image.open(io.BytesIO(data)) as x:return np.asarray(x.convert('RGB'),dtype=np.uint8)
def group(task,r):
 if task=='action_strength':return 'strength-source-episode:'+str(int(r['source_episode_index'][0]))
 if task=='motion_damping':return 'damping-adjacent-catalog:'+str(int(r['catalog_index'][0])//2)
 if task=='cube_gripper_carry':return 'cube-source-episode:'+str(r['source_episode'])
 return task+'-catalog:'+str(int(r['catalog_index'][0]))
def selected_groups(rows,target,seed=20261010):
 groups={}
 for p,rr in rows.items():groups.setdefault(group(TASK,rr[0]),[]).append(p)
 order=sorted(groups,key=lambda x:digest(f'{seed}:{x}'))
 out=[]
 for g in order:
  out.extend(groups[g])
  if len(out)>=target:break
 return sorted(out)
def build_paired(task,out):
 global TASK;TASK=task
 if task=='action_strength':
  canonical=Path('/tmp/cw-strength-mechanism-20261010/panel')
  for name in ('training.npz','development.npz','manifest.json'):
   shutil.copy2(canonical/name,out/name)
  print('action_strength canonical panel reused',flush=True)
  return
 root,comp,param,preferred=SPECS[task];base=BASE/root;component=base/'components'/comp/'v1';reg=json.load(open(base/'task_registry.json'));entry=next(x for x in reg['components'] if x['dataset_id']==comp)
 result={'schema':'contextworld.cross_task_native_panel.v1','task':task,'release':root,'root':str(base),'normalization':entry['development_evaluation']['action_normalization'],'source_group_rule':'strength source episode; damping adjacent catalog mirror; cube source episode; others catalog index','splits':{}}
 for split in ('training','development'):
  if task=='action_strength':
   p=Path('/tmp/cw-strength-mechanism-20261010/panel')/(split+'.npz');z=np.load(p);data={k:z[k] for k in z.files};n=len(data['labels']);data['query_ids']=data['pair_ids'].copy();data['physical_target']=np.where(data['labels']==0,60.,120.).astype(np.float32);data['conditions']=data['labels'];data['future_pixels']=data['queryfuture_pixels'][:,None];data['action_blocks']=data['rawactions'].reshape(n,3,5,2);data['source_groups']=data['source_groups'].astype(str);outfile=out/(split+'.npz');np.savez(outfile,**data);result['splits'][split]={'path':str(outfile),'sha256':sha(outfile),'rows':n,'queries':n//2,'source_group_count':len(set(data['source_groups'])),'query_ids_sha256':digest(data['query_ids'].tolist()),'source':'/tmp/cw-strength-mechanism-20261010/panel/'+split+'.npz'};print(task,split,n,flush=True);continue
  lp=component/split/'data.lance';ds=lance.dataset(lp);idx='model_step_idx' if task=='cube_gripper_carry' else 'step_idx';cols=['episode_idx','pair_id','hidden_mode',param,'catalog_index'];
  if task=='cube_gripper_carry':cols+=['source_episode']
  if task=='action_strength':cols+=['source_episode_index']
  meta=ds.to_table(filter=f'{idx} = 0',columns=cols).to_pylist();pairs={}
  for r in meta:pairs.setdefault(r['pair_id'],{})[r['hidden_mode']]=r
  order=preferred if all(set(x)==set(preferred) for x in pairs.values()) else sorted(next(iter(pairs.values())))
  by={p:[rr[m] for m in order] for p,rr in pairs.items()};selected=selected_groups(by,512) if split=='training' else sorted(by)
  records=[r for p in selected for r in by[p]];ids=[int(r['episode_idx']) for r in records];rowmap={e:{} for e in ids};positions=[0,1,2,3] if task=='cube_gripper_carry' else [0,5,10,15];end=3 if task=='cube_gripper_carry' else 15;act='action_block' if task=='cube_gripper_carry' else 'action'
  for st in range(0,len(ids),100):
   batch=ids[st:st+100];expr='episode_idx IN ('+','.join(map(str,batch))+')';
   for r in ds.to_table(filter=f'{expr} AND {idx} IN ('+','.join(map(str,positions))+')',columns=['episode_idx',idx,'pixels']).to_pylist():rowmap[r['episode_idx']].setdefault(r[idx],{})['pixels']=r['pixels']
   for r in ds.to_table(filter=f'{expr} AND {idx} < {end}',columns=['episode_idx',idx,act]).to_pylist():rowmap[r['episode_idx']].setdefault(r[idx],{})[act]=r[act]
  history=[];future=[];actions=[]
  for r in records:
   ep=rowmap[int(r['episode_idx'])];ff=[frame(ep[i]['pixels']) for i in positions];history.append(np.stack(ff[:3]));future.append(ff[3]);a=np.asarray([ep[i][act] for i in range(end)],np.float32);actions.append(a.reshape(15,-1))
  h=np.stack(history);f=np.stack(future);a=np.stack(actions);labels=np.tile(np.arange(2),len(selected));physical=np.asarray([r[param][0] for r in records],np.float32);pair_ids=np.asarray([r['pair_id'] for r in records]);groups=np.asarray([group(task,r) for r in records]);modes=np.asarray([r['hidden_mode'] for r in records]);
  assert np.array_equal(h[0::2,2],h[1::2,2]),(task,split,'current RGB unequal')
  assert np.array_equal(a[0::2],a[1::2]),(task,split,'raw actions unequal')
  outfile=out/(split+'.npz');np.savez(outfile,history_pixels=h,queryfuture_pixels=f,future_pixels=f[:,None],rawactions=a,action_blocks=a.reshape(len(records),3,5,-1),labels=labels,conditions=labels,physical_target=physical,pair_ids=pair_ids,query_ids=pair_ids,source_groups=groups,modes=modes,episode_ids=np.asarray(ids));result['splits'][split]={'path':str(outfile),'sha256':sha(outfile),'rows':len(records),'queries':len(selected),'source_group_count':len(set(groups)),'query_ids':selected,'query_ids_sha256':digest(pair_ids.tolist()),'query_ids_sha256_algorithm':'sha256(str(full rowwise query_ids.tolist()).encode())','matched_control':{'matched_group_key':'query_ids','current_action_matched':True},'lance_path':str(lp),'lance_version':ds.version,'condition_order':order,'current_rgb_equal_pairs':len(selected),'actions_equal_pairs':len(selected)};print(task,split,len(records),flush=True)
 (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
def build_delay(out):
 src=Path('/tmp/cw-delay-train-dev-20261009');m=json.load(open(src/'manifest.json'));result={'schema':'contextworld.cross_task_native_panel.v1','task':'action_delay','normalization':m['normalization'],'source_manifest':str(src/'manifest.json'),'splits':{}}
 for split in ('training','development'):
  hh=[];ff=[];aa=[];ids=[];groups=[];labels=[];modes=[];epids=[]
  if split=='development':
   catalog=Path('/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld/artifacts/evaluation/history7/action_delay_dev_structural_parity_v1/catalog.json')
   scenes=json.load(open(catalog))['queries'];base=catalog.parents[4]
   for scene in scenes:
    z=np.load(base/scene['asset']);h=z['history_pixels'];f=z['true_future_pixels'][:,0];a=np.broadcast_to(z['action_blocks'][:7],(11,7,5,2));n=len(h);assert n==11
    hh.extend(h);ff.extend(f);aa.extend(a);ids.extend([scene['query_id']]*n);groups.extend([scene['query_id']]*n);labels.extend(range(n));modes.extend([str(i) for i in range(n)]);epids.extend([scene['query_id']]*n)
  else:
   scenes=m['splits'][split]['scenes']
   for scene in scenes:
    z=np.load(src/scene['path']);h=z['history_pixels'];f=z['future_pixels'];a=z['action_blocks'];n=len(h);assert n==11
    hh.extend(h);ff.extend(f[:,0]);aa.extend(a);ids.extend([scene['scene_id']]*n);groups.extend([scene['scene_id']]*n);labels.extend(range(n));modes.extend([str(i) for i in range(n)]);epids.extend([scene['scene_id']]*n)
  h=np.stack(hh);f=np.stack(ff);a=np.stack(aa);l=np.asarray(labels);p=out/(split+'.npz');np.savez(p,history_pixels=h,queryfuture_pixels=f,future_pixels=f[:,None],action_blocks=a,rawactions=a.reshape(len(a),35,2),labels=l,conditions=l,physical_target=l.astype(np.float32),physical_group_labels=np.minimum(l,5),pair_ids=np.asarray(ids),query_ids=np.asarray(ids),source_groups=np.asarray(groups),modes=np.asarray(modes),episode_ids=np.asarray(epids));result['splits'][split]={'path':str(p),'sha256':sha(p),'rows':len(a),'queries':len(scenes),'source_group_count':len(set(groups)),'query_ids_sha256':digest(ids),'matched_control':{'matched_group_key':'query_ids','current_action_matched':True}};print('action_delay',split,len(a),flush=True)
 (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='/tmp/cw-cross-task-mechanism-20261010/native');ap.add_argument('--tasks',nargs='+',default=list(SPECS)+['action_delay','speed','door']);args=ap.parse_args();base=Path(args.output)
 for task in args.tasks:
  out=base/task;out.mkdir(parents=True,exist_ok=True)
  if task=='action_delay':build_delay(out)
  elif task=='speed':build_speed(out)
  elif task=='door':build_door(out)
  else:build_paired(task,out)

def _save_tworoom(task,out,split,records,normalization,matched):
 h=np.stack([x['history'] for x in records]);f=np.stack([x['future'] for x in records]);a=np.stack([x['actions'] for x in records]);labels=np.asarray([x['label'] for x in records]);phys=np.asarray([x['physical'] for x in records],np.float32);ids=np.asarray([x['query_id'] for x in records]);groups=np.asarray([x['source_group'] for x in records]);modes=np.asarray([x['mode'] for x in records]);epids=np.asarray([x['episode_id'] for x in records]);
 p=out/(split+'.npz');np.savez(p,history_pixels=h,queryfuture_pixels=f,future_pixels=f[:,None],rawactions=a,action_blocks=a.reshape(len(a),3,5,2),labels=labels,conditions=labels,physical_target=phys,pair_ids=ids,query_ids=ids,source_groups=groups,modes=modes,episode_ids=epids)
 return {'path':str(p),'sha256':sha(p),'rows':len(records),'queries':len(set(ids)),'source_group_count':len(set(groups)),'query_ids':list(dict.fromkeys(map(str,ids))),'query_ids_sha256':digest(ids.tolist()),'matched_control':{'matched_group_key':'query_ids','current_action_matched':matched}}

def assert_door_label_mapping(records):
 for record in records:
  mode=str(record['mode'])
  assert mode in ('blocked','passable','rule_blocked','rule_passable'), mode
  expected=int(mode.endswith('passable'))
  assert int(record['label'])==expected and int(record['physical'])==expected, (mode,record['label'],record['physical'])

def build_door(out):
 root=BASE/'ContextWorld-v1';reg=json.load(open(root/'task_registry.json'));c=next(x for x in reg['components'] if x['component_id']=='door');norm=c['development_evaluation']['action_normalization'];res={'schema':'contextworld.cross_task_native_panel.v1','task':'door','release':root.name,'normalization':norm,'splits':{}}
 # Training: native start-0 windows, metadata-selected across distinct member/episode sources.
 members=c['payloads'][0]['members'];members=[m for m in members if '/training/' in m];chosen=sorted(members,key=lambda m:digest('20261010:'+m))[:128];recs=[]
 for m in chosen:
  ds=lance.dataset(root/m);meta=ds.to_table(filter='step_idx = 0',columns=['episode_idx','variation_passage_open']).to_pylist();eps=sorted(meta,key=lambda x:digest(m+':'+str(x['episode_idx'])))[:4];ids=[x['episode_idx'] for x in eps];rows={i:{} for i in ids};expr='episode_idx IN ('+','.join(map(str,ids))+')';
  for x in ds.to_table(filter=f'{expr} AND step_idx IN (0,5,10,15)',columns=['episode_idx','step_idx','pixels']).to_pylist():rows[x['episode_idx']].setdefault(x['step_idx'],{})['pixels']=x['pixels']
  for x in ds.to_table(filter=f'{expr} AND step_idx < 15',columns=['episode_idx','step_idx','action']).to_pylist():rows[x['episode_idx']].setdefault(x['step_idx'],{})['action']=x['action']
  for x in eps:
   e=x['episode_idx'];r=rows[e];v=int(x['variation_passage_open'][0]);recs.append(dict(history=np.stack([frame(r[i]['pixels']) for i in (0,5,10)]),future=frame(r[15]['pixels']),actions=np.asarray([r[i]['action'] for i in range(15)],np.float32),label=v,physical=v,query_id=f'{m}:{e}',source_group=f'{m}:{e}',mode='passable' if v else 'blocked',episode_id=f'{m}:{e}'))
 assert_door_label_mapping(recs)
 res['splits']['training']=_save_tworoom('door',out,'training',recs,norm,False);print('door training',len(recs),flush=True)
 panel=Path('/tmp/cw-multistep-coverage-20260930/panels/door');manifest=json.load(open(panel/'manifest.json'));recs=[]
 for scene in manifest['scenes']:
  z=np.load(panel/scene['path']);names=list(map(str,z['aux_candidate_names']));native=names.index('native_amp_1');h=z['history_pixels'];a=np.concatenate([z['context_actions'].reshape(2,10,2),np.broadcast_to(z['aux_native_query_action'],(2,5,2))],axis=1);assert np.array_equal(h[0,2],h[1,2]) and np.array_equal(a[0],a[1]);
  for j in range(2):recs.append(dict(history=h[j],future=z['future_pixels'][j,native,0],actions=a[j],label=int(z['aux_rule_values'][j]),physical=int(z['aux_rule_values'][j]),query_id=scene['query_id'],source_group=scene['query_id'],mode=str(z['conditions'][j]),episode_id=scene['query_id']+':'+str(j)))
 assert_door_label_mapping(recs)
 res['splits']['development']=_save_tworoom('door',out,'development',recs,norm,True);res['development_panel_manifest']=str(panel/'manifest.json');(out/'manifest.json').write_text(json.dumps(res,indent=2)+'\n');print('door development',len(recs),flush=True)

def build_speed(out):
 root=BASE/'ContextWorld-v1';reg=json.load(open(root/'task_registry.json'));c=next(x for x in reg['components'] if x['component_id']=='speed');norm=c['development_evaluation']['action_normalization'];res={'schema':'contextworld.cross_task_native_panel.v1','task':'speed','release':root.name,'normalization':norm,'label_rule':'continuous physical regression only; unseen Development speeds','splits':{}}
 members=next(x for x in c['payloads'] if x.get('split')=='training')['members'];chosen=sorted(members,key=lambda m:digest('20261010:'+m))[:128];recs=[]
 for m in chosen:
  ds=lance.dataset(root/m);meta=ds.to_table(filter='step_idx = 0',columns=['episode_idx','variation_agent_speed']).to_pylist();eligible={x['episode_idx'] for x in ds.to_table(filter='step_idx = 15',columns=['episode_idx']).to_pylist()};eps=sorted((x for x in meta if x['episode_idx'] in eligible),key=lambda x:digest(m+':'+str(x['episode_idx'])))[:4];ids=[x['episode_idx'] for x in eps];rows={i:{} for i in ids};expr='episode_idx IN ('+','.join(map(str,ids))+')';
  for x in ds.to_table(filter=f'{expr} AND step_idx IN (0,5,10,15)',columns=['episode_idx','step_idx','pixels']).to_pylist():rows[x['episode_idx']].setdefault(x['step_idx'],{})['pixels']=x['pixels']
  for x in ds.to_table(filter=f'{expr} AND step_idx < 15',columns=['episode_idx','step_idx','action']).to_pylist():rows[x['episode_idx']].setdefault(x['step_idx'],{})['action']=x['action']
  for x in eps:
   e=x['episode_idx'];r=rows[e];v=float(x['variation_agent_speed'][0]);recs.append(dict(history=np.stack([frame(r[i]['pixels']) for i in (0,5,10)]),future=frame(r[15]['pixels']),actions=np.asarray([r[i]['action'] for i in range(15)],np.float32),label=v,physical=v,query_id=f'{m}:{e}',source_group=f'{m}:{e}',mode=str(v),episode_id=f'{m}:{e}'))
 res['splits']['training']=_save_tworoom('speed',out,'training',recs,norm,False);print('speed training',len(recs),flush=True)
 panel=Path('/tmp/cw-speed-cem-20260929/panel');scenes=json.load(open(panel/'manifest.json'))['queries'];wanted={x['query_id'] for x in scenes};recby={};dev_members=[m for m in c['development_evaluation']['payload']['members'] if 'unseen_interpolation' in m]
 for m in dev_members:
  ds=lance.dataset(root/m);meta=ds.to_table(filter='step_idx = 0',columns=['episode_idx','dev_static_query_id','dev_reference_speed','dev_condition_speed','dev_condition']).to_pylist();chosen=[x for x in meta if x['dev_static_query_id'] in wanted and abs(x['dev_reference_speed'][0]-x['dev_condition_speed'][0])<1e-4];ids=[x['episode_idx'] for x in chosen];rows={i:{} for i in ids}
  for st in range(0,len(ids),100):
   expr='episode_idx IN ('+','.join(map(str,ids[st:st+100]))+')';
   for x in ds.to_table(filter=f'{expr} AND step_idx IN (0,5,10,15)',columns=['episode_idx','step_idx','pixels']).to_pylist():rows[x['episode_idx']].setdefault(x['step_idx'],{})['pixels']=x['pixels']
   for x in ds.to_table(filter=f'{expr} AND step_idx < 15',columns=['episode_idx','step_idx','action']).to_pylist():rows[x['episode_idx']].setdefault(x['step_idx'],{})['action']=x['action']
  for x in chosen:
   e=x['episode_idx'];r=rows[e];v=float(x['dev_condition_speed'][0]);recby[(x['dev_static_query_id'],round(v,1))]=dict(history=np.stack([frame(r[i]['pixels']) for i in (0,5,10)]),future=frame(r[15]['pixels']),actions=np.asarray([r[i]['action'] for i in range(15)],np.float32),label=v,physical=v,query_id=x['dev_static_query_id'],source_group=x['dev_static_query_id'],mode=str(v),episode_id=f'{m}:{e}')
 recs=[]
 for scene in scenes:
  z=np.load(panel/scene['path']);qid=scene['query_id']
  for j,v in enumerate(z['speeds']):
   r=recby[(qid,round(float(v),1))];assert np.array_equal(r['history'],z['history_pixels'][j]),qid;recs.append(r)
 for i in range(0,len(recs),3):
  assert np.array_equal(recs[i]['history'][2],recs[i+1]['history'][2]) and np.array_equal(recs[i]['actions'],recs[i+1]['actions'])
 res['splits']['development']=_save_tworoom('speed',out,'development',recs,norm,True);res['development_panel_manifest']=str(panel/'manifest.json');(out/'manifest.json').write_text(json.dumps(res,indent=2)+'\n');print('speed development',len(recs),flush=True)

if __name__=='__main__':main()
