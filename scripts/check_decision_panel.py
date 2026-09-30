"""Model-independent decision relevance, fixed-deadline and overshoot checks."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_panel(root):
    m=json.loads((root/'manifest.json').read_text());entries=m.get('scenes',m.get('pairs',m.get('queries')));assert entries
    rows=[]
    for e in entries:
        p=root/e['path'];assert sha(p)==e['sha256']
        with np.load(p,allow_pickle=False) as f:d=dict(f)
        h=d['history_pixels'];u=d['candidate_actions'];c=np.asarray(d['physical_cost'],np.float64)
        K,C,T=c.shape
        assert h.dtype==np.uint8 and h.shape[0]==K and h.shape[-3:]==(224,224,3)
        assert np.array_equal(h[:,-1],np.repeat(h[:1,-1],K,axis=0)),str(p)+' mismatched current'
        assert np.allclose(d['context_actions'],d['context_actions'][:1],rtol=0,atol=1e-7)
        assert u.shape[:2]==(C,T) and u.shape[2]==5
        assert d['future_pixels'].shape==(K,C,T,224,224,3)
        assert c.min()>=0 and np.isfinite(c).all() and np.isfinite(u).all()
        assert (abs(u)<=1.000001).all(),str(p)+' out of bounds'
        oracle=c.min(1);shared=c.mean(0).min(0);gap=shared-oracle.mean(0)
        assert gap.min()>-1e-9
        opt=c.argmin(1)
        terminal_best=opt[:,-1];chosen=np.stack([c[k,terminal_best[k]] for k in range(K)])
        # Only five observed block endpoints; this is NOT a continuous first-entry metric.
        shortcut=chosen[:,-1]-chosen.min(1)
        rows.append(dict(scene_id=e.get('scene_id',e.get('pair_id',e.get('query_id',p.stem))),
            conditions=K,candidates=C,horizons=d['physical_steps'].tolist(),
            mean_oracle_cost=oracle.mean(0).tolist(),best_shared_cost=shared.tolist(),decision_gap=gap.tolist(),
            condition_optima=opt.tolist(),physical_spread=np.ptp(c,axis=0).mean(0).tolist(),
            terminal_best_plan_earlier_advantage=shortcut.tolist(),
            any_plan_earlier_advantage=float(np.max(c[:,:,-1]-c.min(2))),
            physical_cost_sha256=hashlib.sha256(c.tobytes()).hexdigest()))
    return dict(task=m.get('task',m.get('protocol',{}).get('task')),scenes=len(rows),protocol=m,rows=rows,
        mean_decision_gap=np.mean([r['decision_gap'] for r in rows],0).tolist(),
        mean_oracle_cost=np.mean([r['mean_oracle_cost'] for r in rows],0).tolist(),
        positive_gap_scene_count=np.sum(np.array([r['decision_gap'] for r in rows])>1e-6,0).tolist(),
        scope='Fixed candidate bank, no proof of global reachability; block-endpoint overshoot only')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=check_panel(a.panel);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['task'],'n',result['scenes'],'gap',result['mean_decision_gap'],'oracle',result['mean_oracle_cost'])
