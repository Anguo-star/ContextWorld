"""Frozen history-encoding diagnostic; no optimizer and no decoder fitting."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--seed',type=int,default=3072);p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
CW=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CW/'scripts'))
from diagnose_cross_task_decisions import load_adapter,sha
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
root=a.run_root;rid=f'action_delay/{a.family}/scratch/s{a.seed}';spec=next(s for s in json.loads((root/'models.json').read_text()) if s['id']==rid);panel=root/'panels/action_delay';manifest=json.loads((panel/'manifest.json').read_text());a.output.mkdir(parents=True,exist_ok=True)
adapter=load_adapter(spec,manifest.get('normalization',manifest.get('protocol',{}).get('normalization')),a.output,a.device);before=adapter.frozen_state_hash()
cache={};rows=[]
for i,entry in enumerate(manifest['scenes']):
 with np.load(panel/entry['path'],allow_pickle=False) as z:
  pixels=z['history_pixels'];actions=z['context_actions'];query=z['aux_native_query_action'];cand=z['candidate_actions']
  native=[j for j,c in enumerate(cand) if np.array_equal(c[0],query)]
  assert np.array_equal(pixels[:,-1],np.repeat(pixels[:1,-1],11,axis=0))
  assert np.array_equal(actions,np.repeat(actions[:1],11,axis=0))
  frames=pixels.reshape(-1,224,224,3);keys=[hashlib.sha256(f.tobytes()).hexdigest() for f in frames];new={k:f for k,f in zip(keys,frames) if k not in cache}
  if new:
   enc=adapter.encode_pixels(np.stack(list(new.values())),batch_size=64)
   cache.update(zip(new,enc))
  h=np.stack([cache[k] for k in keys]).reshape(11,7,-1).astype(np.float64)
  hc=h-h.mean(0,keepdims=True);he=float(np.sum(hc*hc)); hn=float(np.sum(h*h))
  # Explicit temporal differences remove a constant scene appearance; no fitted probe.
  ht=np.diff(h,axis=1); pair=[]; tp=[]; rawpair=[]
  history_keys=[tuple(keys[k*7:(k+1)*7]) for k in range(11)]
  for l in range(11):
   for r in range(l+1,11):
    pair.append(float(np.sum((h[l]-h[r])**2)))
    tp.append(float(np.sum((ht[l]-ht[r])**2)))
    rawpair.append(int(history_keys[l]!=history_keys[r]))
  current=np.max(np.abs(h[:,-1]-h[:1,-1]));assert current==0
  rows.append(dict(scene_id=entry['scene_id'],source_sha256=entry['sha256'],history_shape=list(h.shape),raw_history_distinct_pairs=sum(x>0 for x in rawpair),encoded_history_distinct_pairs=sum(x>0 for x in pair),temporal_difference_distinct_pairs=sum(x>0 for x in tp),pairs=55,min_encoded_pair_squared_distance=min(pair),history_centered_energy=he,history_energy=hn,shared_current_latent_max_difference=float(current),native_query_candidates=native,conditions=list(range(11))))
 if (i+1)%50==0:print(rid,i+1,'unique',len(cache),flush=True)
after=adapter.frozen_state_hash();assert before==after
out=dict(schema='contextworld.delay_history_encoding.v1',model_id=rid,checkpoint_sha256=spec['checkpoint_sha256'],panel_sha256=sha(panel/'manifest.json'),state_hash_before=before,state_hash_after=after,no_training=True,rows=rows,distinct_frames_encoded=len(cache),scope='H7 Development pixels only; per-scene distinctness is NOT a held-out delay decoder or proof of representational sufficiency')
(a.output/'history_encoding.json').write_text(json.dumps(out,indent=2)+'\n');print('finished',rid,flush=True)
