#!/usr/bin/env python3
"""Compare strict paired decisions and full-error S/G from cached distances only.
No model inference. Raw distances are preserved; canonical_strict only fixes
verified identical-target condition columns for exact strict-choice accounting.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

DEFAULT_ROOT=None
DEFAULT_PILOT=Path(__file__).resolve().parents[1]/'docs/research/data/icl_visible_future_pilot_v1.json'
DEFAULT_OUT=None
SEED=20261008
BOOT=1000

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def manifest_map(root,task):
 p=root/'panels'/task/'manifest.json'; raw=p.read_bytes(); j=json.loads(raw)
 entries=j.get('scenes',[])
 out={}
 for r in entries:
  sid=r.get('scene_id'); status=r.get('status','passed' if r.get('ok',True) else 'failed')
  if not sid or status not in ('passed','ok') and r.get('ok') is not True: continue
  group=r.get('source_group') or r.get('bootstrap_cluster') or f'scene:{sid}'
  out[sid]={'path':r.get('path'),'sha256':r.get('sha256'),'group':str(group)}
 return out,hashlib.sha256(raw).hexdigest()

def distances_metric(d,B):
 # d = [history, truth, candidate, time] squared distances.
 K,T=d.shape[2],d.shape[3]
 m=d[0,0]+d[1,1]  # [K,T], matched total over two conditions
 w=d[0,1]+d[1,0]
 future=np.stack([d[0,0]<d[0,1],d[1,1]<d[1,0]],axis=0)
 ftie=np.stack([d[0,0]==d[0,1],d[1,1]==d[1,0]],axis=0)
 hist=np.stack([d[0,0]<d[1,0],d[1,1]<d[0,1]],axis=0)
 htie=np.stack([d[0,0]==d[1,0],d[1,1]==d[0,1]],axis=0)
 def win(arr,t): return float(np.mean(arr[...,t]))
 def tied(arr,t): return float(np.mean(arr[...,t]))
 # Per-scene normalization gives equal scene weight while retaining equal
 # condition/candidate/time weights inside the scene.
 ef=float(np.sum(m[:,0]));wf=float(np.sum(w[:,0]));bf=float(B[0]); denomf=2*K
 ea=float(np.sum(m));wa=float(np.sum(w));ba=float(np.sum(B)); denoma=2*K*T
 return {
  'K':K,'T':T,
  'E_first':ef,'W_first':wf,'B_first':bf,
  'E_all5':ea,'W_all5':wa,'B_all5':ba,
  'Efirst_norm':ef/denomf,'Wfirst_norm':wf/denomf,'Bfirst_norm':bf/denomf,
  'Eall_norm':ea/denoma,'Wall_norm':wa/denoma,'Ball_norm':ba/denoma,
  'S_first':100*(1-ef/bf) if bf else None,'G_first':(wf-ef)/bf if bf else None,
  'S_all5':100*(1-ea/ba) if ba else None,'G_all5':(wa-ea)/ba if ba else None,
  'R_first':win(future,0),'H_first':win(hist,0),'future_tie_first':tied(ftie,0),'history_tie_first':tied(htie,0),
  'R_all5':float(np.mean(future)),'H_all5':float(np.mean(hist)),
  'future_tie_all5':float(np.mean(ftie)),'history_tie_all5':float(np.mean(htie)),
  'future_comparisons':int(future.size),'history_comparisons':int(hist.size),
 }

def maybe_lewm_direct(feat_path,distance):
 with np.load(feat_path,allow_pickle=False) as z:
  p=np.asarray(z['pred']); t=np.asarray(z['target'])
 if p.ndim!=4 or t.shape!=p.shape or p.shape[0]!=2 or p.shape[2]!=5:
  raise ValueError(f'Unexpected native feature shape {p.shape}/{t.shape} at {feat_path}')
 # Exact target equality, no tolerance. Only this mask is eligible for canonicalization.
 same=np.all(t[0]==t[1],axis=-1)
 # Direct distances from native free predictions/targets. Sum over latent dimensions,
 # matching the cache's squared-Euclidean quantity; used only as the diagonal baseline.
 direct=np.zeros_like(distance,dtype=np.float64)
 for h in range(2):
  for c in range(2):
   delta=p[h].astype(np.float64)-t[c].astype(np.float64)
   direct[h,c]=np.square(delta).sum(axis=-1)
 return same,direct,t

def load_models(root,models,groups_by_task):
 loaded=[];checks=[]
 for model in models:
  mid=model['id'];task=model['task'];fam=model['family'];droot=Path(model['result_dir'])
  if not droot.is_dir():raise FileNotFoundError(droot)
  manifest=groups_by_task[task]
  receipts=[]
  for jp in sorted(droot.glob('*.json')):
   try:j=json.loads(jp.read_text())
   except Exception:continue
   if 'distance_file' in j:receipts.append((jp,j))
  if len(receipts)!=256:raise ValueError(f'{mid}: expected 256 scene receipts, got {len(receipts)}')
  recs={}; max_matched=max_wrong=0.; sha_ok=0; energy_ok=0; source_ok=0; feature_target_masks={}; dino_target_hashes={}
  for jp,j in receipts:
   sid=j['scene_id'];
   if sid in recs:raise ValueError(f'{mid}: duplicate {sid}')
   if sid not in manifest:raise ValueError(f'{mid}: scene {sid} missing from panel manifest')
   meta=manifest[sid]
   if j.get('task')!=task or j.get('model_id')!=mid:raise ValueError(f'{jp}: receipt identity mismatch')
   if j.get('source_sha256')!=meta['sha256']:raise ValueError(f'{jp}: source SHA mismatch')
   source_ok+=1
   dp=droot/j['distance_file']
   if sha(dp)!=j.get('distance_file_sha256'):raise ValueError(f'{dp}: distance SHA mismatch')
   sha_ok+=1
   with np.load(dp,allow_pickle=False) as z:
    if 'distance' not in z.files:raise ValueError(f'{dp}: distance key missing')
    dist=np.asarray(z['distance'],dtype=np.float64)
   if dist.ndim!=4 or dist.shape[:2]!=(2,2) or dist.shape[-1]!=5 or not np.isfinite(dist).all():raise ValueError(f'{dp}: bad shape/values {dist.shape}')
   B=np.asarray(j.get('target_separation_energy'),dtype=np.float64)
   if B.shape!=(5,) or not np.isfinite(B).all():raise ValueError(f'{jp}: bad target separation {B.shape}')
   match_by=(dist[0,0]+dist[1,1]).sum(axis=0)
   wrong_by=(dist[0,1]+dist[1,0]).sum(axis=0)
   mrec=np.asarray(j['matched']['energy_sum'],dtype=np.float64)
   wrec=np.asarray(j.get('wrong_allother',j.get('wrong'))['energy_sum'],dtype=np.float64)
   dm=float(np.max(np.abs(match_by-mrec)));dw=float(np.max(np.abs(wrong_by-wrec)))
   max_matched=max(max_matched,dm);max_wrong=max(max_wrong,dw)
   if not(np.allclose(match_by,mrec,rtol=1e-10,atol=1e-8) and np.allclose(wrong_by,wrec,rtol=1e-10,atol=1e-8)):
    raise ValueError(f'{jp}: distance vs receipt matched/wrong energy mismatch {dm}/{dw}')
   energy_ok+=1
   same=None; direct=None; target_hash=None
   fp=droot/j['features_file']
   if fam=='lewm':
    same,direct,t=maybe_lewm_direct(fp,dist)
    target_hash=hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()
   else:
    # Pooled DINO target is used only as an equality *candidate mask*, never for scoring.
    with np.load(fp,allow_pickle=False) as z:
     t=np.asarray(z['target'])
    if t.shape[:3]!=(2,dist.shape[2],5):raise ValueError(f'{fp}: pooled target shape mismatch {t.shape}')
    same=np.all(t[0]==t[1],axis=-1)
    target_hash=hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()
   recs[sid]={'receipt':j,'distance':dist,'B':B,'group':meta['group'],'path':jp,
              'same_mask':same,'direct':direct,'target_hash':target_hash,'canonical_distance':None,
              'pixel_same_mask':None,'canonical_mask':None}
  if len(recs)!=256:raise ValueError(f'{mid}: coverage failure {len(recs)}')
  group_count=len({r['group'] for r in recs.values()})
  loaded.append({'model':model,'records':recs,'group_count':group_count})
  checks.append({'model_id':mid,'scene_count':len(recs),'source_group_count':group_count,
                 'distance_sha_verified':sha_ok,'source_sha_verified':source_ok,
                 'matched_wrong_receipt_energy_verified':energy_ok,
                 'max_abs_matched_energy_delta':max_matched,'max_abs_wrong_energy_delta':max_wrong})
 return loaded,checks

def canonicalize(loaded,root,groups_by_task):
 # DINO: only open panels for positions with exactly equal pooled targets; then
 # require byte-identical true future frames. Nonmatches remain raw and untouched.
 by_task={}
 for item in loaded:
  m=item['model']; task=m['task']; fam=m['family']; recs=item['records']
  for sid,r in recs.items():
   if fam=='dinowm':by_task.setdefault(task,{}).setdefault(sid,[]).append((item,r))
 for task,scenes in by_task.items():
  manifest=groups_by_task[task]
  for sid,entries in scenes.items():
   masks=[r['same_mask'] for _,r in entries]
   union=np.logical_or.reduce(masks)
   pix_same=np.zeros_like(union,dtype=bool)
   positions=np.argwhere(union)
   if len(positions):
    pp=root/'panels'/task/manifest[sid]['path']
    with np.load(pp,allow_pickle=False) as z:
     pix=z['future_pixels']
     for k,h in positions:
      if k>=pix.shape[1] or h>=pix.shape[2]:raise ValueError(f'{pp}: image K/H mismatch')
      pix_same[k,h]=np.array_equal(pix[0,k,h],pix[1,k,h])
   for _,r in entries:r['pixel_same_mask']=pix_same
 # Build raw-preserving canonical matrices.
 for item in loaded:
  fam=item['model']['family']
  for r in item['records'].values():
   d=r['distance'];c=d.copy();same=r['same_mask']
   if fam=='dinowm':same=same & r['pixel_same_mask']
   # LeWM: native exact equal target features. DINO: pooled equality AND panel byte identity.
   for k,h in np.argwhere(same):
    for hist in range(2):
     baseline=(r['direct'][hist,hist,k,h] if fam=='lewm' else d[hist,hist,k,h])
     c[hist,0,k,h]=baseline;c[hist,1,k,h]=baseline
   r['canonical_distance']=c;r['canonical_mask']=same

def row_metrics(item):
 m=item['model']; out=[]
 for sid,r in sorted(item['records'].items()):
  raw=distances_metric(r['distance'],r['B']);can=distances_metric(r['canonical_distance'],r['B'])
  common={'id':m['id'],'task':m['task'],'family':m['family'],'regime':m['regime'],'scene_id':sid,
          'source_group':r['group'],'K':raw['K'],'horizons':5,
          'same_target_candidate_times_verified':int(np.sum(r['canonical_mask']))}
  for variant,x in [('raw',raw),('canonical',can)]:
   for key in ('E_first','W_first','B_first','E_all5','W_all5','B_all5','Efirst_norm','Wfirst_norm','Bfirst_norm','Eall_norm','Wall_norm','Ball_norm','S_first','G_first','S_all5','G_all5','R_first','H_first','future_tie_first','history_tie_first','R_all5','H_all5','future_tie_all5','history_tie_all5'):
    common[f'{variant}_{key}']=x[key]
  # Include auditable target denominator at each horizon.
  common['target_separation_energy_by_horizon']=json.dumps(r['B'].tolist(),separators=(',',':'))
  # Raw-vs-canonical strict labels changed by exact tie correction.
  rawd=r['distance'];cand=r['canonical_distance']
  fraw=np.stack([rawd[0,0]<rawd[0,1],rawd[1,1]<rawd[1,0]])
  fcan=np.stack([cand[0,0]<cand[0,1],cand[1,1]<cand[1,0]])
  hraw=np.stack([rawd[0,0]<rawd[1,0],rawd[1,1]<rawd[0,1]])
  hcan=np.stack([cand[0,0]<cand[1,0],cand[1,1]<cand[0,1]])
  common['future_label_changes_raw_to_canonical']=int(np.sum(fraw!=fcan))
  common['history_label_changes_raw_to_canonical']=int(np.sum(hraw!=hcan))
  out.append(common)
 return out

def metric_point(rows,variant,window):
 suffix='first' if window=='first' else 'all'
 e=np.asarray([r[f'{variant}_E{suffix}_norm'] for r in rows],float)
 w=np.asarray([r[f'{variant}_W{suffix}_norm'] for r in rows],float)
 b=np.asarray([r[f'{variant}_B{suffix}_norm'] for r in rows],float)
 if np.any(b<0) or np.sum(b)<=0: raise ValueError('Nonpositive aggregate denominator')
 return {'S':float(100*(1-e.mean()/b.mean())),'G':float((w.mean()-e.mean())/b.mean()),
         'R':float(np.mean([r[f'{variant}_R_{"first" if window=="first" else "all5"}'] for r in rows])),
         'history_win':float(np.mean([r[f'{variant}_H_{"first" if window=="first" else "all5"}'] for r in rows])),
         'future_tie':float(np.mean([r[f'{variant}_future_tie_{"first" if window=="first" else "all5"}'] for r in rows])),
         'history_tie':float(np.mean([r[f'{variant}_history_tie_{"first" if window=="first" else "all5"}'] for r in rows]))}

def group_stats(rows,variant,window,groups):
 n=len(groups); fields=['E','W','B','R','H','FT','HT','Q']
 arr=np.zeros((n,len(fields)),dtype=np.float64)
 wname='first' if window=='first' else 'all5'
 suffix='first' if window=='first' else 'all'
 for i,g in enumerate(groups):
  rr=[r for r in rows if r['source_group']==g]
  arr[i]=[sum(r[f'{variant}_E{suffix}_norm'] for r in rr),
          sum(r[f'{variant}_W{suffix}_norm'] for r in rr),
          sum(r[f'{variant}_B{suffix}_norm'] for r in rr),
          sum(r[f'{variant}_R_{wname}'] for r in rr),
          sum(r[f'{variant}_H_{wname}'] for r in rr),
          sum(r[f'{variant}_future_tie_{wname}'] for r in rr),
          sum(r[f'{variant}_history_tie_{wname}'] for r in rr),len(rr)]
 return arr

def metrics_from_group_sums(x):
 E,W,B,R,H,FT,HT,Q=np.asarray(x).T
 return {'S':100*(1-E.sum()/B.sum()),'G':(W.sum()-E.sum())/B.sum(),
         'R':R.sum()/Q.sum(),'history_win':H.sum()/Q.sum(),
         'future_tie':FT.sum()/Q.sum(),'history_tie':HT.sum()/Q.sum()}

def ci_from_boot(samples): return [float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]

def bootstrap_model(rows,variant,window,groups,rng,nboot):
 a=group_stats(rows,variant,window,groups);draw=rng.integers(0,len(groups),size=(nboot,len(groups)))
 sums=a[draw].sum(axis=1); out={}
 for m in ('S','G','R','history_win','future_tie','history_tie'):
  if m=='S': vals=100*(1-sums[:,0]/sums[:,2])
  elif m=='G':vals=(sums[:,1]-sums[:,0])/sums[:,2]
  elif m=='R':vals=sums[:,3]/sums[:,7]
  elif m=='history_win':vals=sums[:,4]/sums[:,7]
  elif m=='future_tie':vals=sums[:,5]/sums[:,7]
  else:vals=sums[:,6]/sums[:,7]
  out[m]=ci_from_boot(vals)
 return out,a

def bootstrap_delta(rows0,rows1,variant,window,groups,rng,nboot):
 a0=group_stats(rows0,variant,window,groups);a1=group_stats(rows1,variant,window,groups)
 draw=rng.integers(0,len(groups),size=(nboot,len(groups)))
 s0=a0[draw].sum(axis=1);s1=a1[draw].sum(axis=1);out={}
 for m,ix in [('R',None),('S',None),('G',None)]:
  def val(s):
   if m=='R':return s[:,3]/s[:,7]
   if m=='S':return 100*(1-s[:,0]/s[:,2])
   return (s[:,1]-s[:,0])/s[:,2]
  out[m]=ci_from_boot(val(s1)-val(s0))
 return out

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--root',type=Path,required=True);ap.add_argument('--pilot',type=Path,default=DEFAULT_PILOT)
 ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--bootstrap',type=int,default=BOOT);ap.add_argument('--seed',type=int,default=SEED)
 a=ap.parse_args();root=a.root.resolve();outdir=a.output_dir.resolve();outdir.mkdir(parents=True,exist_ok=True)
 models=json.loads((root/'models.json').read_text()); pilot=json.loads(a.pilot.read_text());pilot_rows={x['id']:x for x in pilot['rows']}
 if len(models)!=8:raise ValueError(f'expected exactly 8 model groups, got {len(models)}')
 groups_by_task={}
 for task in sorted({m['task'] for m in models}):groups_by_task[task]=manifest_map(root,task)[0]
 loaded,checks=load_models(root,models,groups_by_task)
 canonicalize(loaded,root,groups_by_task)
 queries=[]
 for x in loaded:queries.extend(row_metrics(x))
 byid={m['model']['id']:m for m in loaded}; q_byid={m['model']['id']:[r for r in queries if r['id']==m['model']['id']] for m in loaded}
 # Verify T0/T1 denominator energies per scene and horizon, not only total sums.
 dino_align=[]; dino_tasks=sorted({m['task'] for m in models if m['family']=='dinowm'})
 for task in dino_tasks:
  t0=next(x for x in loaded if x['model']['task']==task and x['model']['family']=='dinowm' and x['model']['regime']=='original')
  t1=next(x for x in loaded if x['model']['task']==task and x['model']['family']=='dinowm' and x['model']['regime']=='scratch')
  if set(t0['records'])!=set(t1['records']):raise ValueError(f'{task}: DINO T0/T1 query sets differ')
  for sid in sorted(t0['records']):
   b0=t0['records'][sid]['B'];b1=t1['records'][sid]['B'];diff=np.abs(b1-b0)
   dino_align.append({'task':task,'scene_id':sid,'source_group':t0['records'][sid]['group'],
     'B_T0_by_horizon':json.dumps(b0.tolist(),separators=(',',':')),'B_T1_by_horizon':json.dumps(b1.tolist(),separators=(',',':')),
     'exact_equal_horizon_count':int(np.sum(diff==0)),'max_abs_B_delta':float(np.max(diff)),
     'all5_horizon_energies_exact_equal':bool(np.all(diff==0)),
     'interpretation':'Energy alignment only; scalar equality does not imply target-vector identity.'})
 # Model summaries + bootstrap CIs.
 rng=np.random.default_rng(a.seed); summaries=[]; all_checks=[]; bootcache={}
 for item in loaded:
  m=item['model']; mid=m['id']; rows=q_byid[mid]; groups=sorted({r['source_group'] for r in rows})
  hist=pilot_rows[mid]['history']; point={}; boot={}
  for var in ('raw','canonical'):
   for win in ('first','all5'):
    point[f'{var}_{win}']=metric_point(rows,var,win)
    boot[f'{var}_{win}'],_ = bootstrap_model(rows,var,win,groups,rng,a.bootstrap)
  pscore=float(hist['score']); pgain=float(hist['gain'])
  all_checks.append({'model_id':mid,'computed_raw_S_all5_minus_pilot_score':point['raw_all5']['S']-pscore,
    'computed_raw_G_all5_minus_pilot_gain':point['raw_all5']['G']-pgain,
    'raw_S_all5_matches_pilot':math.isclose(point['raw_all5']['S'],pscore,rel_tol=1e-10,abs_tol=1e-9),
    'raw_G_all5_matches_pilot':math.isclose(point['raw_all5']['G'],pgain,rel_tol=1e-10,abs_tol=1e-10),
    'pilot_score':pscore,'pilot_gain':pgain})
  summaries.append({'model_id':mid,'task':m['task'],'family':m['family'],'regime':m['regime'],
    'n_queries':len(rows),'n_source_groups':len(groups),'candidate_count_range':[min(r['K'] for r in rows),max(r['K'] for r in rows)],
    'point_estimates':point,'source_group_bootstrap_95ci':boot,
    'pilot_comparison':all_checks[-1],
    'same_target_verified_candidate_time_count':sum(r['same_target_candidate_times_verified'] for r in rows),
    'raw_to_canonical_future_label_changes':sum(r['future_label_changes_raw_to_canonical'] for r in rows),
    'raw_to_canonical_history_label_changes':sum(r['history_label_changes_raw_to_canonical'] for r in rows),
    'target_identity_basis':'LeWM: exact equality of native target vectors. DINO: exact equality in pooled target used only as candidate mask, then byte-identical panel future pixels required.'})
 # Paired DINO T1-T0 cluster bootstrap, same sampled groups for both configurations.
 deltas=[]
 for task in dino_tasks:
  t0=next(x for x in loaded if x['model']['task']==task and x['model']['family']=='dinowm' and x['model']['regime']=='original')
  t1=next(x for x in loaded if x['model']['task']==task and x['model']['family']=='dinowm' and x['model']['regime']=='scratch')
  r0=q_byid[t0['model']['id']];r1=q_byid[t1['model']['id']];g0=sorted({r['source_group'] for r in r0});g1=sorted({r['source_group'] for r in r1})
  if g0!=g1:raise ValueError(f'{task}: source-group sets differ for paired bootstrap')
  row={'task':task,'contrast':'T1(scratch)-T0(original)','n_queries_each':256,'n_source_groups':len(g0),'point_deltas':{},'source_group_bootstrap_95ci_all5':{}}
  for var in ('raw','canonical'):
   p0=metric_point(r0,var,'all5');p1=metric_point(r1,var,'all5')
   row['point_deltas'][var]={x:p1[x]-p0[x] for x in ('S','G','R','history_win','future_tie','history_tie')}
   row['source_group_bootstrap_95ci_all5'][var]=bootstrap_delta(r0,r1,var,'all5',g0,rng,a.bootstrap)
  deltas.append(row)
 if not all(c['raw_S_all5_matches_pilot'] and c['raw_G_all5_matches_pilot'] for c in all_checks):
  raise ValueError('Full-error metrics disagree with existing pilot results')
 # Write strict-comparison outputs.
 query_csv=outdir/'metric_comparison_queries.csv'
 with query_csv.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(queries[0]));w.writeheader();w.writerows(queries)
 sum_csv=outdir/'metric_comparison_models.csv'
 simple=[]
 for s in summaries:
  row={'model_id':s['model_id'],'task':s['task'],'family':s['family'],'regime':s['regime'],'n_queries':s['n_queries'],'n_source_groups':s['n_source_groups'],'K_min':s['candidate_count_range'][0],'K_max':s['candidate_count_range'][1],
       'same_target_verified_candidate_time_count':s['same_target_verified_candidate_time_count'],
       'raw_to_canonical_future_label_changes':s['raw_to_canonical_future_label_changes'],'raw_to_canonical_history_label_changes':s['raw_to_canonical_history_label_changes']}
  for var in ('raw','canonical'):
   for win in ('first','all5'):
    for metric in ('R','history_win','future_tie','history_tie','S','G'):
     row[f'{var}_{win}_{metric}']=s['point_estimates'][f'{var}_{win}'][metric]
     if win=='all5':
      ci=s['source_group_bootstrap_95ci'][f'{var}_{win}'][metric]
      row[f'{var}_{win}_{metric}_ci95_low']=ci[0];row[f'{var}_{win}_{metric}_ci95_high']=ci[1]
  row['pilot_raw_S_all5']=s['pilot_comparison']['pilot_score'];row['pilot_raw_G_all5']=s['pilot_comparison']['pilot_gain']
  row['pilot_raw_S_G_match']=s['pilot_comparison']['raw_S_all5_matches_pilot'] and s['pilot_comparison']['raw_G_all5_matches_pilot']
  simple.append(row)
 with sum_csv.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(simple[0]));w.writeheader();w.writerows(simple)
 denom_csv=outdir/'dino_target_energy_alignment.csv'
 with denom_csv.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dino_align[0]));w.writeheader();w.writerows(dino_align)
 result={'schema':'cw_metric_comparison_v1','purpose':'Same cached actions/horizons: compare strict selection/history decisions with complete normalized prediction scores.',
  'no_training_inference_or_simulation':True,'distance_source':'Existing distance_*.npz; no DINO feature-space scoring.',
  'inputs':{'root':str(root),'models_json_sha256':sha(root/'models.json'),'pilot_json':str(a.pilot),'pilot_schema':pilot.get('schema'),'bootstrap_replicates':a.bootstrap,'bootstrap_seed':a.seed},
  'definitions':{'distance_axes':'[history,truth,candidate,time] squared Euclidean distance',
   'future_rate':'strict d[h,h] < d[h,1-h], equally across the two histories, candidates and horizons; exact ties count as false.',
   'history_rate':'strict correct-history prediction error < opposite-history prediction error for each target condition; exact ties count as false.',
   'S':'100*(1-E/B)','G':'(Ewrong-E)/B','first_step':'uses Bfirst only, not the shared all-horizon denominator.',
   'all5':'sum five horizon E/W/B, with equal condition/candidate/time weight within each scene; model score is ratio of mean per-scene normalized sums, matching pilot weighting.',
   'raw':'strict comparisons and E/W from stored distance cache, before same-target numerical-tie canonicalization.',
   'canonical_strict':'only exact same-target candidate/time cells are canonicalized: each history row uses its diagonal distance for both target columns; LeWM baseline is recomputed directly from native pred/target, DINO baseline remains cached distance after byte-identical future-pixel confirmation. No tolerance threshold is used.',
   'all_candidates':'all cached unique candidates retained, including zero-action candidate; no candidates/scenes filtered.'},
  'cache_checks':checks,'models':summaries,'dino_T1_minus_T0_paired_source_group_deltas':deltas,
  'dino_target_separation_energy_alignment':{'rows':len(dino_align),'exact_equal_scene_horizon_entries':sum(r['exact_equal_horizon_count'] for r in dino_align),
     'total_scene_horizon_entries':len(dino_align)*5,'all_exactly_equal':all(r['all5_horizon_energies_exact_equal'] for r in dino_align),
     'max_abs_delta':max(r['max_abs_B_delta'] for r in dino_align) if dino_align else None,
     'interpretation':'Per-scene/per-horizon scalar energy alignment only; this alone does not establish all target vectors are equal.'},
  'pilot_raw_S_G_checks':all_checks,
  'outputs':{'summary_json':str(outdir/'metric_comparison.json'),'models_csv':str(sum_csv),'query_csv':str(query_csv),'dino_target_energy_csv':str(denom_csv)}}
 (outdir/'metric_comparison.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'summary':str(outdir/'metric_comparison.json'),'model_csv':str(sum_csv),'query_csv':str(query_csv),'dino_B_exact_match':result['dino_target_separation_energy_alignment'],
  'pilot_checks':all_checks,'coverage':{x['model_id']:{'queries':x['n_queries'],'groups':x['n_source_groups'],'canonical_same_target':x['same_target_verified_candidate_time_count'],'future_labels_changed':x['raw_to_canonical_future_label_changes']} for x in summaries}},indent=2))

if __name__=='__main__':
 main()
