#!/usr/bin/env python3
"""Publish a checked, compact fixed-checkpoint native mechanism matrix."""
from __future__ import annotations
import argparse, csv, hashlib, io, json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
TMP=Path('/tmp/cw-cross-task-mechanism-20261010')
OUT=REPO/'docs/research/data'
RAW=OUT/'cross_task_mechanism_v1'
TASKS=['speed','action_strength','robot_arm_mass','action_delay','contact_friction','motion_damping','cube_gripper_carry','door','portal_exit']
FAMILIES=['lewm','pldm','dinowm']
TASK_NAMES={'speed':'速度','action_strength':'推手移动幅度','robot_arm_mass':'机械臂质量','action_delay':'动作延迟','contact_friction':'接触摩擦','motion_damping':'运动阻尼','cube_gripper_carry':'Cube 夹爪携带','door':'门通行规则','portal_exit':'传送门出口'}
MODEL_NAMES={'lewm':'LeWM','pldm':'PLDM','dinowm':'DINO-WM'}
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def encoded(x):return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def load(p):return json.loads(Path(p).read_text())
def pointer(data,path):
 require(path.startswith('/'),'invalid JSON pointer')
 for part in path[1:].split('/'):
  part=part.replace('~1','/').replace('~0','~')
  data=data[int(part)] if isinstance(data,list) else data[part]
 return data
def add(outputs,path,value):outputs[Path(path)]=value if isinstance(value,bytes) else encoded(value)
def source_refs():
 idx=load(RAW/'receipts/readout_index.json');gm=load(RAW/'receipts/gradient_manifest.json');require(idx['completed_cells']==27 and gm['model_count']==27,'expected 27 completed cells')
 rr={x['id']:x for x in idx['results']};gg={x['model_id']:x for x in gm['cells']};expected={f'{t}/{f}/scratch/s3072' for t in TASKS for f in FAMILIES};require(set(rr)==set(gg)==expected,'27 unique cell mismatch')
 return rr,gg,gm

def collect():
 """Import immutable source JSON once; later builds read only the public copies."""
 rr=load(TMP/'readout_index.json')['results'];gg=load(TMP/'gradient/manifest.json')['cells'];cross=load(TMP/'full_protocol_nre.json')['rows']
 copies={RAW/'receipts/coverage_summary.json':TMP/'coverage_summary.json',RAW/'receipts/readout_index.json':TMP/'readout_index.json',RAW/'receipts/gradient_manifest.json':TMP/'gradient/manifest.json',RAW/'receipts/door_label_correction.json':TMP/'native/door/label_correction_receipt.json',RAW/'receipts/full_protocol_nre_crosswalk.json':TMP/'full_protocol_nre.json'}
 for task in TASKS:copies[RAW/'panels'/f'{task}.json']=Path('/tmp/cw-strength-mechanism-20261010/panel/manifest.json') if task=='action_strength' else TMP/'native'/task/'manifest.json'
 for x in rr:
  ident=Path(x['id']);copies[RAW/'readout_results'/ident.with_suffix('.json')]=Path(x['result_path'])
  for y in x['validation_receipts']:copies[RAW/'receipts'/ident.parts[0]/ident.parts[1]/Path(y['path']).name]=Path(y['path'])
 for x in gg:copies[RAW/'gradient_results'/Path(x['model_id']).with_suffix('.json')]=Path(x['result'])
 for x in cross:copies[RAW/'original_protocol'/Path(x['model_id']).with_suffix('.json')]=Path(x['original_protocol_source_file'])
 for dst,src in copies.items():
  data=src.read_bytes()
  if dst.exists():require(dst.read_bytes()==data,'collected source changed: '+str(dst))
  else:dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
 print('collected',len(copies),'source JSON records')

def build():
 rr,gg,gm=source_refs();coverage=load(RAW/'receipts/coverage_summary.json');crosswalk_path=RAW/'receipts/full_protocol_nre_crosswalk.json';crosswalk=load(crosswalk_path);full={x['model_id']:x for x in crosswalk['rows']};require(set(full)==set(rr) and len(crosswalk['rows'])==27,'full protocol crosswalk cell mismatch');outputs={};rows=[];source_manifest={};task_selections={}
 for task in TASKS:
  mp=RAW/'panels'/f'{task}.json';m=load(mp);require(coverage['tasks'][task]['manifest_sha256']==digest(mp),'coverage manifest SHA mismatch '+task)
  source_manifest[f'panel/{task}']={'path':str(mp.relative_to(OUT)),'sha256':digest(mp)}
 for task in TASKS:
  for family in FAMILIES:
   ident=f'{task}/{family}/scratch/s3072';ri=rr[ident];gi=gg[ident];relative=Path(ident).with_suffix('.json');rp=RAW/'readout_results'/relative;gp=RAW/'gradient_results'/relative;require(digest(rp)==ri['result_sha256'] and digest(gp)==gi['result_sha256'],'result SHA mismatch '+ident)
   rd=load(rp);gd=load(gp);require(rd['model_id']==gd['model']['id']==ident and rd['checkpoint_sha256']==gd['model']['checkpoint_sha256'] and rd['checkpoint']==gd['model']['checkpoint'],'checkpoint identity mismatch '+ident)
   require(rd['model_state_hash_before']==rd['model_state_hash_after'] and gd['state_hash_before']==gd['state_hash_after'] and rd['no_worldmodel_update'] and gd['optimizer_steps']==0 and rd['no_test_read'] and not gd['public_test_accessed'],'state/scope mismatch '+ident)
   require(ri['manifest_path']==gi['source_manifest'],'panel path mismatch '+ident);mp=RAW/'panels'/f'{task}.json';mh=digest(mp);require(mh==ri['current_manifest_sha256']==gi['current_source_manifest_sha256'],'current panel mismatch '+ident)
   manifest=load(mp);expected_raw=manifest['splits']['development']['sha256'];require(coverage['tasks'][task]['splits']['development']['raw_sha256']==expected_raw,'coverage raw mismatch '+task)
   if rd['panel_manifest_sha256']!=mh:
    require(ri['validation_receipts'],'missing readout rebinding receipt '+ident)
    receipts=[load(RAW/'receipts'/task/family/Path(z['path']).name) for z in ri['validation_receipts']]
    require(any(z.get('old_manifest_sha256')==rd['panel_manifest_sha256'] and z.get('current_manifest_sha256')==mh for z in receipts),'readout old/current manifest rebinding mismatch '+ident)
   if gd['selection']['manifest_sha256']!=mh or gd['selection']['raw_sha256']!=expected_raw:
    require(task=='door','unreceipted gradient source change '+ident);receipt=load(RAW/'receipts/door_label_correction.json');require(gd['selection']['manifest_sha256']==receipt['old_manifest_sha256'] and gd['selection']['raw_sha256']==receipt['old_npz_sha256'] and mh==receipt['new_manifest_sha256'] and expected_raw==receipt['new_npz_sha256'],'Door correction binding mismatch')
   require(gd['query_count']==16 and len(gd['per_query'])==16 and len(set(gd['selection']['source_groups']))==16,'gradient query selection mismatch '+ident)
   selection=(gd['selection']['query_ids'],gd['selection']['source_groups'],expected_raw)
   if task in task_selections:require(selection==task_selections[task],'source identities differ across model families: '+task)
   else:task_selections[task]=selection
   require(ri['development_rows']==coverage['tasks'][task]['splits']['development']['rows'],'readout rows mismatch '+ident)
   for receipt_ref in ri['validation_receipts']:
    p=RAW/'receipts'/task/family/Path(receipt_ref['path']).name;require(digest(p)==receipt_ref['sha256'],'validation receipt SHA mismatch '+ident)
    source_manifest[f'receipt/{ident}/{p.name}']={'path':str(p.relative_to(OUT)),'sha256':digest(p)}
   readout=rd['readouts']['history/ridge']['development'];six=rd['readouts'].get('history/ridge_six_physical_groups',{}).get('development');grad=gd['summary'];mean=gd['mean_gradient'];norm=mean['mean_gradient_norms'];cos=mean['pairwise']['query_response__native_all']['cosine'];ratio=norm['query_response']/norm['native_all'] if norm['native_all'] else None
   target_energy=sum(x['per_position']['target_response_energy_by_position'][-1] for x in gd['per_query']);response_energy=sum(x['per_position']['response_by_position'][-1] for x in gd['per_query']);nre=response_energy/target_energy if target_energy else None;require(abs(nre-grad['energy_pooled_native_query_response_nre'])<1e-9,'NRE recomputation mismatch '+ident)
   fr=full[ident];op=RAW/'original_protocol'/relative;require(digest(op)==fr['original_protocol_source_sha256'] and fr['checkpoint_sha256']==rd['checkpoint_sha256'] and fr['gradient_probe_result_sha256']==digest(gp) and abs(fr['gradient_probe_16_query_native_response_nre']-nre)<1e-9,'full protocol source binding mismatch '+ident)
   observed_full=pointer(load(op),fr['original_protocol_metric_json_pointer']);require(isinstance(observed_full,(int,float)) and abs(observed_full-fr['full_development_original_protocol_response_nre'])<1e-10,'original protocol NRE pointer/value mismatch '+ident)
   require(fr['full_development_query_count']==coverage['tasks'][task]['splits']['development']['queries'],'full protocol query coverage mismatch '+ident)
   source_manifest[f'original_protocol/{ident}']={'path':str(op.relative_to(OUT)),'sha256':digest(op),'metric_json_pointer':fr['original_protocol_metric_json_pointer']}
   o={'id':ident,'task':task,'family':family,'regime':'scratch','seed':3072,'checkpoint_sha256':rd['checkpoint_sha256'],'training_rows':coverage['tasks'][task]['splits']['training']['rows'],'training_queries':coverage['tasks'][task]['splits']['training']['queries'],'training_source_groups':coverage['tasks'][task]['splits']['training']['source_groups'],'development_rows':coverage['tasks'][task]['splits']['development']['rows'],'development_queries':coverage['tasks'][task]['splits']['development']['queries'],'development_source_groups':coverage['tasks'][task]['splits']['development']['source_groups'],'history_readout':{'metric':'mae' if task=='speed' else 'balanced_accuracy','value':readout['mae'] if task=='speed' else readout['balanced_accuracy'],'ci95':readout['source_group_bootstrap']['interval_95'],'baseline_mae':readout.get('baseline_mae'),'r2':readout.get('r2'),'six_physical_group_balanced_accuracy':six['balanced_accuracy'] if six else None,'six_physical_group_ci95':six['source_group_bootstrap']['interval_95'] if six else None},'original_protocol':{'full_development_response_nre':fr['full_development_original_protocol_response_nre'],'query_count':fr['full_development_query_count'],'conditions_per_query':fr['full_development_condition_count_per_query'],'track':fr['original_protocol_track']},'gradient':{'development_query_count':16,'target_response_energy':target_energy,'probe_16_query_response_nre':nre,'aggregate_response_loss_share_percent':100*grad['aggregate_query_response_loss_share'],'mean_response_to_native_gradient_norm_percent':100*ratio if ratio is not None else None,'mean_vector_cosine':cos,'negative_query_cosine_count':grad['negative_query_response_native_cosine_count']},'sources':{'readout_result_sha256':digest(rp),'gradient_result_sha256':digest(gp),'panel_manifest_sha256':mh,'development_raw_sha256':expected_raw,'original_protocol_source_sha256':digest(op)}}
   rows.append(o)
   source_manifest[f'readout/{ident}']={'path':str(rp.relative_to(OUT)),'sha256':digest(rp)};source_manifest[f'gradient/{ident}']={'path':str(gp.relative_to(OUT)),'sha256':digest(gp)}
 # Source receipts and top-level indexes are small and establish stale-manifest/Door bindings.
 for name in ('coverage_summary','readout_index','gradient_manifest','door_label_correction','full_protocol_nre_crosswalk'):
  path=RAW/'receipts'/(name+'.json');source_manifest[name]={'path':str(path.relative_to(OUT)),'sha256':digest(path)}
 require(len(rows)==27,'summary row count')
 summary={'schema':'contextworld.cross_task_mechanism_v1','scope':'Training/Development fixed T1 scratch checkpoints; full Development original-protocol NRE and separate 16-query native gradient probe; no retraining or Test access','units':{'history_balanced_accuracy':'fraction','speed_mae':'speed units; lower is better','full_development_response_nre':'original task-specific Development source NRE; lower is better','probe_16_query_response_nre':'energy-pooled native response error / target response energy on 16 sampled queries; lower is better','response_loss_share_percent':'aggregate 16-query response loss / native loss','gradient_ratio_percent':'100*norm(mean response gradient)/norm(mean native gradient) on 16 queries; not mean per-query ratio','cosine':'cosine of mean gradient vectors on 16 queries'},'source_records':source_manifest,'rows':rows,'limitations':['Door Training readout uses fixed start-0 windows with unverified wall-interaction evidence coverage.','Delay exact 11-class readout and six first-future physical-group readout are distinct targets.','Speed Development speeds are unseen; its readout is continuous regression and MAE is not comparable to classification accuracy.','The full Development original-protocol NRE and the 16-query native gradient probe have different query coverage and weighting; compare them as separate diagnostics.']}
 add(outputs,OUT/'cross_task_mechanism_v1.json',summary)
 fields=['task','family','id','training_rows','training_queries','training_source_groups','development_rows','development_queries','development_source_groups','history_metric','history_value','history_ci95_low','history_ci95_high','speed_baseline_mae','speed_r2','delay_six_group_ba','delay_six_group_ci95_low','delay_six_group_ci95_high','full_development_response_nre','gradient_queries','target_response_energy','probe_16_query_response_nre','response_loss_share_percent','gradient_ratio_percent','mean_vector_cosine','negative_query_cosine_count','checkpoint_sha256','readout_result_sha256','gradient_result_sha256','panel_manifest_sha256','development_raw_sha256','original_protocol_source_sha256']
 sio=io.StringIO();w=csv.DictWriter(sio,fieldnames=fields,lineterminator='\n');w.writeheader()
 for x in rows:
  h=x['history_readout'];g=x['gradient'];w.writerow({'task':x['task'],'family':x['family'],'id':x['id'],**{k:x[k] for k in fields if k in x},'history_metric':h['metric'],'history_value':h['value'],'history_ci95_low':h['ci95'][0],'history_ci95_high':h['ci95'][1],'speed_baseline_mae':h['baseline_mae'],'speed_r2':h['r2'],'delay_six_group_ba':h['six_physical_group_balanced_accuracy'],'delay_six_group_ci95_low':h['six_physical_group_ci95'][0] if h['six_physical_group_ci95'] else None,'delay_six_group_ci95_high':h['six_physical_group_ci95'][1] if h['six_physical_group_ci95'] else None,'full_development_response_nre':x['original_protocol']['full_development_response_nre'],'gradient_queries':g['development_query_count'],'target_response_energy':g['target_response_energy'],'probe_16_query_response_nre':g['probe_16_query_response_nre'],'response_loss_share_percent':g['aggregate_response_loss_share_percent'],'gradient_ratio_percent':g['mean_response_to_native_gradient_norm_percent'],'mean_vector_cosine':g['mean_vector_cosine'],'negative_query_cosine_count':g['negative_query_cosine_count'],**x['sources']})
 add(outputs,OUT/'cross_task_mechanism_v1.csv',sio.getvalue().encode())
 lines=['| 任务 | 模型 | 历史读出 | 全量响应 NRE ↓ | 梯度比 | 平均余弦 |','|---|---|---:|---:|---:|---:|']
 for x in rows:
  h=x['history_readout'];g=x['gradient'];v=(f"{h['value']:.2f}*" if x['task']=='speed' else f"{100*(h['six_physical_group_balanced_accuracy'] if x['task']=='action_delay' else h['value']):.1f}%"+('†' if x['task']=='action_delay' else ''));lines.append(f"| {TASK_NAMES[x['task']]} | {MODEL_NAMES[x['family']]} | {v} | {x['original_protocol']['full_development_response_nre']:.3f} | {g['mean_response_to_native_gradient_norm_percent']:.3g}% | {g['mean_vector_cosine']:+.3f} |")
 lines+=['',r'\* 速度读出为 MAE（越低越好）；其余为平衡识别率（越高越好）。','† 延迟读出采用六个首段物理响应组；精确 11 延迟标签读出见 JSON／CSV。','全量响应 NRE 来自完整 Development 原协议；梯度比与平均余弦只使用每模型 16 个 Development 查询。','']
 add(outputs,RAW/'table.md','\n'.join(lines).encode());return outputs

def marked_document(doc,table):
 begin='<!-- BEGIN CROSS_TASK_MECHANISM -->';end='<!-- END CROSS_TASK_MECHANISM -->'
 require(doc.count(begin)==doc.count(end)==1,'Study marker count mismatch')
 front,rest=doc.split(begin,1);_,back=rest.split(end,1)
 return front+begin+'\n'+table.rstrip()+'\n'+end+back

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--collect',action='store_true');ap.add_argument('--check',action='store_true');a=ap.parse_args()
 if a.collect:collect()
 outputs=build();docpath=REPO/'docs/ICL_Metric_Study.md';doc=docpath.read_text();rendered=marked_document(doc,outputs[RAW/'table.md'].decode())
 if a.check:
  for p,v in outputs.items():require(p.exists() and p.read_bytes()==v,'generated file missing/stale: '+str(p))
  require(doc==rendered,'Study CROSS_TASK_MECHANISM table is stale')
  print('checked',len(outputs),'files and Study marker')
 else:
  for p,v in outputs.items():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
  (TMP/'table.md').parent.mkdir(parents=True,exist_ok=True);(TMP/'table.md').write_bytes(outputs[RAW/'table.md'])
  if doc!=rendered:docpath.write_text(rendered)
  print('wrote',len(outputs),'files')
if __name__=='__main__':main()
