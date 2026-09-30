"""Frozen, task-neutral candidate ranking and autoregressive diagnostics."""
from __future__ import annotations
import argparse, hashlib, importlib, json, os, sys, faulthandler, time
from pathlib import Path
import numpy as np

CW=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CW));sys.modules.setdefault('flash_attn',None)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def load_adapter(spec,normalization,output,device='cpu'):
    import torch
    from contextworld.synthesis.stablewm import load_stable_worldmodel
    from contextworld.benchmarks.external_model_cli import TASKS
    from contextworld.benchmarks.adapters import _load_model
    from contextworld.evaluation.protocol import ColumnStandardizer
    ck=Path(spec['checkpoint']);assert sha(ck)==spec['checkpoint_sha256']
    swm,repo,commit=load_stable_worldmodel(CW,spec['stable_repo'],spec['stable_ref'])
    family='prejepa' if spec['family']=='dinowm' else spec['family']
    name=TASKS[spec['task']].builtins[family];mod,clsname=name.rsplit('.',1)
    cls=getattr(importlib.import_module(mod),clsname)
    if ck.suffix=='.pt':
        model=swm.wm.utils.load_pretrained(str(ck),cache_dir=str(output/'model_cache'))
    else:
        model=_load_model(ck,stable_worldmodel=swm,stable_repo=repo,repo_root=CW,
            model_config_name=family,action_input_dim=cls.action_input_dim)
    return cls(model=model,checkpoint=ck,stable_repo=repo,stable_commit=commit,device=device,
        action_standardizer=ColumnStandardizer(np.asarray(normalization['mean'],np.float32)[None],np.asarray(normalization['std'],np.float32)[None]))

def encode_unique(adapter,*images):
    frames={};inverse=[];shapes=[]
    for group in images:
        shapes.append(group.shape[:-3]);ids=[]
        for f in group.reshape(-1,*group.shape[-3:]):
            h=hashlib.sha256(f.tobytes()).digest()
            if h not in frames:frames[h]=(len(frames),f)
            ids.append(frames[h][0])
        inverse.append(ids)
    z=adapter.encode_pixels(np.stack([v[1] for v in frames.values()]),batch_size=16)
    return [z[ids].reshape(*shape,z.shape[-1]) for ids,shape in zip(inverse,shapes)]

def predict_all(adapter,history,target,raw,family,batch_size=8,modes=('free','full','current','past')):
    """Cache visual encoding; compare free/full/current/past H-length inputs."""
    import torch
    hsize=history.shape[1];nout=target.shape[1];allout={m:[] for m in modes}
    model=adapter.model
    if family=='dinowm':
        assert tuple(model.extra_encoders)==('action',), 'Extra state stream is not allowed'
        channels=model.backbone.config.hidden_size
        patches=history.shape[-1]//channels
        assert patches*channels==history.shape[-1]
    with torch.inference_mode():
        for start in range(0,len(history),batch_size):
            h=torch.from_numpy(history[start:start+batch_size]).to(adapter.device);y=torch.from_numpy(target[start:start+batch_size]).to(adapter.device)
            norm=torch.from_numpy(adapter._normalize_actions(raw[start:start+batch_size])).to(adapter.device)
            actions=model.extra_encoders['action'](norm) if family=='dinowm' else model.action_encoder(norm)
            truth=torch.cat([h,y],1)
            for mode in allout:
                frames=list(h.unbind(1))
                for t in range(nout):
                    win=torch.stack(frames[-hsize:],1).clone();real=truth[:,t:t+hsize]
                    if mode in ('full','current'):win[:,-1]=real[:,-1]
                    if mode in ('full','past'):win[:,:-1]=real[:,:-1]
                    act=actions[:,t:t+hsize]
                    if family=='dinowm':
                        emb=torch.cat([win.reshape(len(win),hsize,patches,channels),act[:,:,None].expand(-1,-1,patches,-1)],-1)
                        z=model.predict(emb)[:,-1,:,:channels].reshape(len(win),-1)
                    else:z=model.predict(win,act)[:,-1]
                    frames.append(z)
                allout[mode].append(torch.stack(frames[hsize:],1).cpu().numpy())
    return {m:np.concatenate(x) for m,x in allout.items()}

def costs_summary(cost,physical):
    # [K,C,T], all histories score the same candidate bank and target.
    K,C,T=cost.shape;choice=cost.argmin(1);oracle=physical.min(1)
    correct=np.stack([physical[k,choice[k],np.arange(T)] for k in range(K)])
    wrong=np.stack([np.mean([physical[k,choice[h],np.arange(T)] for h in range(K) if h!=k],0) for k in range(K)])
    return dict(matched_cost=correct.mean(0).tolist(),matched_regret=(correct-oracle).mean(0).tolist(),
        wrong_history_cost=wrong.mean(0).tolist(),history_benefit=(wrong-correct).mean(0).tolist(),selected_indices=choice.tolist())

def evaluate(adapter,data,family,canonical,modes=('free','full','current','past')):
    pix=data['history_pixels'];K,H=pix.shape[:2];C,T=data['candidate_actions'].shape[:2]
    assert data['future_pixels'].shape[:3]==(K,C,T)
    assert data['context_actions'].shape[:2]==(K,H-1)
    assert np.array_equal(pix[:,-1],np.repeat(pix[:1,-1],K,axis=0)), 'Current image mismatch'
    assert np.allclose(data['context_actions'],data['context_actions'][:1],atol=1e-7,rtol=0), 'Past actions mismatch'
    h,y,g=encode_unique(adapter,pix,data['future_pixels'],data['goal_pixels'][None]);g=g[0]
    hh=np.repeat(h,C,axis=0)
    raw=np.concatenate([np.repeat(data['context_actions'],C,axis=0),np.tile(data['candidate_actions'],(K,1,1,1))],1)
    out=predict_all(adapter,hh,y.reshape(K*C,T,-1),raw,family,modes=modes)
    assert all(np.array_equal(v[:,0],out['free'][:,0]) for v in out.values())
    checks=[]
    if canonical:
        # The H7 public contract exposes 3 future blocks; the diagnostic loop
        # extends to5 without modifying that frozen contract. Check supported depths.
        steps=min(T,adapter.protocol.future_action_blocks)
        indices=np.array([0,K*C-1])
        frames=np.repeat(pix,C,axis=0)[indices]
        native=adapter.rollout_latents(frames,raw[indices,:H-1+steps],batch_size=2)
        z=out['free'][indices,:steps];delta=z-native
        checks.append(dict(mode='free',max_abs=float(abs(delta).max()),relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(native),1e-12))))
        assert checks[-1]['max_abs']<3e-4 and checks[-1]['relative_l2']<3e-5,checks[-1]
        sequence=np.concatenate([np.repeat(pix[:,None],C,axis=1),data['future_pixels']],2).reshape(K*C,H+T,224,224,3)
        for t in ((1,T-1) if 'full' in out else ()):
            native=adapter.rollout_latents(sequence[indices,t:t+H],raw[indices,t:t+H],batch_size=2)[:,0]
            z=out['full'][indices,t];delta=z-native
            check=dict(mode='full',depth=t+1,max_abs=float(abs(delta).max()),relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(native),1e-12)))
            assert check['max_abs']<3e-4 and check['relative_l2']<3e-5,check
            checks.append(check)
    physical=np.asarray(data['physical_cost'],np.float64)
    assert physical.shape==(K,C,T) and np.isfinite(physical).all()
    target_goal=np.mean((y-g)**2,-1)
    baseline=np.sum((y-y.mean(0,keepdims=True))**2,axis=(0,1,3),dtype=np.float64)
    result=dict(conditions=K,candidates=C,horizons=list(range(5,5*T+1,5)),canonical_checks=checks,
        encoded_true=costs_summary(target_goal,physical),modes={},
        target_separation_energy=baseline.tolist())
    arrays=dict(physical_cost=physical,encoded_goal_cost=target_goal)
    for mode,z in out.items():
        z=z.reshape(K,C,T,-1)
        pc=np.mean((z-g)**2,-1);err=np.sum((z-y)**2,-1,dtype=np.float64)
        resp=np.sum(((z-z.mean(0,keepdims=True))-(y-y.mean(0,keepdims=True)))**2,axis=(0,1,3),dtype=np.float64)
        result['modes'][mode]=dict(prediction_error_sum=err.sum((0,1)).tolist(),response_error_sum=resp.tolist(),
            ranking=costs_summary(pc,physical))
        arrays[mode+'_goal_cost']=pc;arrays[mode+'_prediction_error']=err
    return result,arrays

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--id',required=True)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--device',default='cpu')
    p.add_argument('--modes',nargs='+',choices=('free','full','current','past'),default=['free','full','current','past'])
    a=p.parse_args();
    if 'free' not in a.modes:p.error('--modes must include free')
    faulthandler.dump_traceback_later(120,repeat=True);a.output.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(20260930)
    spec=next(x for x in json.loads(a.models.read_text()) if x['id']==a.id)
    manifest=json.loads((a.panel/'manifest.json').read_text());norm=manifest.get('normalization',manifest.get('protocol',{}).get('normalization'))
    assert norm, 'Missing frozen normalization'
    print(a.id,'loading',flush=True);adapter=load_adapter(spec,norm,a.output,a.device);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;before=adapter.frozen_state_hash();print(a.id,'loaded',flush=True)
    entries=manifest.get('scenes',manifest.get('pairs',manifest.get('queries')));assert entries
    rows=[]
    for i,entry in enumerate(entries):
        if i%a.shards!=a.shard:continue
        path=a.panel/entry['path'];assert sha(path)==entry['sha256']
        outpath=a.output/(path.stem+'.json')
        if outpath.exists():
            old=json.loads(outpath.read_text());assert old['checkpoint_sha256']==spec['checkpoint_sha256'] and old['source_sha256']==sha(path)
            if not set(a.modes).issubset(old['modes']):
                raise ValueError('Existing output lacks requested modes; use a separate output directory')
            rows.append(old);continue
        with np.load(path,allow_pickle=False) as f:data={k:f[k] for k in f.files}
        values,arrays=evaluate(adapter,data,spec['family'],i in (0,len(entries)-1),modes=a.modes)
        arraypath=outpath.with_suffix('.npz');np.savez_compressed(arraypath,**arrays)
        row=dict(scene_id=entry.get('scene_id',entry.get('pair_id',entry.get('query_id',path.stem))),model_id=a.id,
            checkpoint_sha256=spec['checkpoint_sha256'],source_sha256=sha(path),array_sha256=sha(arraypath),**values)
        write(outpath,row);rows.append(row);print(a.id,i+1,'/',len(entries),flush=True)
    after=adapter.frozen_state_hash();assert before==after
    write(a.output/('receipt.json' if a.shards==1 else f'shard{a.shard}_receipt.json'),dict(model=adapter.metadata,model_id=a.id,panel_sha256=sha(a.panel/'manifest.json'),
        state_hash_before=before,state_hash_after=after,scenes=len(rows),no_training=True,new_searches=0,modes=a.modes,
        privileged_refresh_is_not_planning=True,entry_hashes={p.name:sha(p) for p in a.output.glob('*.json') if 'receipt' not in p.name}))

if __name__=='__main__':main()
