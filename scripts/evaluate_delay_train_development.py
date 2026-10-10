#!/usr/bin/env python3
"""Frozen native H7 first-endpoint evaluation of real Training/Development windows."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_cross_task_decisions import load_adapter,encode_unique,predict_all,sha,write

def sufficient_statistics(z,y):
    z=z.astype(np.float64);y=y.astype(np.float64);K=len(y)
    assert K==11
    yc=y-y.mean(0);zc=z-z.mean(0)
    B=float(np.sum(yc*yc));R=float(np.sum((zc-yc)**2));E=float(np.sum((z-y)**2));dot=float(np.sum(zc*yc));P=float(np.sum(zc*zc))
    common=float(K*np.sum((z.mean(0)-y.mean(0))**2))
    W=E+2*K/(K-1)*dot
    assert B>0 and np.isfinite([B,R,E,W,dot,P]).all()
    assert np.isclose(E,R+common,rtol=1e-10,atol=1e-8)
    assert np.isclose(R,P+B-2*dot,rtol=1e-10,atol=1e-8)
    # The original query can distinguish six physical response groups, not eleven exact delays.
    assert all(np.array_equal(y[5],y[k]) for k in range(6,11))
    yy=y[:6];loss=np.stack([np.sum((z-yy[g])**2,axis=-1) for g in range(6)],1)
    group=np.minimum(np.arange(K),5);wins=[]
    for k in range(K):wins.append(bool(loss[k,group[k]]<np.min(np.delete(loss[k],group[k]))))
    choice=(sum(wins[:5])+np.mean(wins[5:]))/6
    return {'energy':[B,R,E,W,P,common],'six_group_choice':choice}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--id',required=True)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=True);manifest=json.loads((a.panel/'manifest.json').read_text());spec=next(s for s in json.loads(a.models.read_text()) if s['id']==a.id)
    assert spec['task']=='action_delay' and spec['regime']=='scratch'
    adapter=load_adapter(spec,manifest['normalization'],a.output,a.device);before=adapter.frozen_state_hash();rows=[];t=time.time();canonical=[]
    for split in ['training','development']:
        for index,entry in enumerate(manifest['splits'][split]['scenes']):
            path=a.panel/entry['path'];assert sha(path)==entry['sha256']
            with np.load(path,allow_pickle=False) as data:pix=data['history_pixels'];target=data['future_pixels'];raw=data['action_blocks']
            h,y=encode_unique(adapter,pix,target);z=predict_all(adapter,h,y,raw,spec['family'],modes=('free',))['free'][:,0]
            if index==0:
                native=adapter.rollout_latents(pix,raw,batch_size=8)[:,0];delta=native-z
                check={'split':split,'max_abs':float(abs(delta).max()),'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(native),1e-12))}
                assert check['max_abs']<3e-4 and check['relative_l2']<3e-5,check;canonical.append(check)
            stat=sufficient_statistics(z,y[:,0]);rows.append({'split':split,'scene_id':entry['scene_id'],'source_sha256':entry['sha256'],**stat})
            if (index+1)%32==0:print(a.id,split,index+1,'elapsed',round(time.time()-t,1),flush=True)
    after=adapter.frozen_state_hash();assert before==after
    write(a.output/'result.json',{'schema':'contextworld.delay_train_development.run.v1','model':spec,'panel_sha256':sha(a.panel/'manifest.json'),'no_training':True,'state_hash_before':before,'state_hash_after':after,'canonical_checks':canonical,'rows':rows,'elapsed_seconds':time.time()-t})
    print('done',a.id,round(time.time()-t,1),flush=True)

if __name__=='__main__':main()
