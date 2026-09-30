"""Summarize physical validity and paired frozen-model comparisons by task."""
import json,argparse,hashlib
from pathlib import Path
import numpy as np
from check_decision_panel import check_panel
def mean_ci(values):
    v=np.asarray(values,dtype=np.float64);rng=np.random.default_rng(20260930)
    samples=v[rng.integers(0,len(v),(4000,len(v)))].mean(1)
    return dict(mean=float(v.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),bootstrap_unit='source query, all conditions/candidates together')
def ratio_ci(numerators, denominators):
    n=np.asarray(numerators,dtype=np.float64);d=np.asarray(denominators,dtype=np.float64)
    rng=np.random.default_rng(20260930);ix=rng.integers(0,len(n),(4000,len(n)))
    samples=n[ix].sum(1)/d[ix].sum(1)
    return dict(mean=float(n.sum()/d.sum()),ci95=np.quantile(samples,[.025,.975]).tolist())

def collect(root):
    tasks=[];models=[]
    specs=json.loads((root/'models.json').read_text())
    for task in ('speed','action_strength','robot_arm_mass','action_delay','contact_friction','motion_damping','cube_gripper_carry','door','portal_exit'):
        p=root/'panels'/task/'manifest.json'
        if not p.exists():tasks.append(dict(task=task,status='panel_not_complete'));continue
        phys=check_panel(p.parent);tasks.append(dict(status='complete',**phys))
        for spec in [s for s in specs if s['task']==task]:
            rr=root/'results'/spec['id'];receipt=rr/'receipt.json'
            if not receipt.exists():models.append(dict(id=spec['id'],status='not_complete'));continue
            r=json.loads(receipt.read_text());assert r['panel_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
            queries=[]
            for f, expected_hash in r['entry_hashes'].items():
                source=rr/f;assert hashlib.sha256(source.read_bytes()).hexdigest()==expected_hash
                q=json.loads(source.read_text());array=source.with_suffix('.npz')
                assert hashlib.sha256(array.read_bytes()).hexdigest()==q['array_sha256']
                queries.append(q)
            assert len(queries)==len(phys['rows'])
            # Retain all source queries, including those with a zero physical decision gap.
            summaries=[]
            physical_by_id={q['scene_id']:q for q in phys['rows']}
            for depth in range(5):
                def ranking(mode,key):return [q['modes'][mode]['ranking'][key][depth] for q in queries]
                free=sum(q['modes']['free']['prediction_error_sum'][depth] for q in queries)
                energy=sum(q['target_separation_energy'][depth] for q in queries)
                modes={}
                for mode in ('free','full','current','past'):
                    error=sum(q['modes'][mode]['prediction_error_sum'][depth] for q in queries)
                    modes[mode]=dict(prediction_error_relative_to_free=error/free if free else None,
                        prediction_error_ratio_ci=ratio_ci([q['modes'][mode]['prediction_error_sum'][depth] for q in queries],[q['modes']['free']['prediction_error_sum'][depth] for q in queries]),
                        normalized_full_error=error/energy if energy else None,
                        regret=mean_ci(ranking(mode,'matched_regret')),
                        history_benefit=mean_ci(ranking(mode,'history_benefit')),
                        advantage_over_best_shared_plan=mean_ci([physical_by_id[q['scene_id']]['decision_gap'][depth]-q['modes'][mode]['ranking']['matched_regret'][depth] for q in queries]))
                summaries.append(dict(raw_steps=(depth+1)*5,modes=modes,
                    true_future_encoded_regret=mean_ci([q['encoded_true']['matched_regret'][depth] for q in queries])))
            models.append(dict(id=spec['id'],status='complete',checkpoint_sha256=spec['checkpoint_sha256'],scenes=len(queries),horizons=summaries,
                frozen_state_unchanged=r['state_hash_before']==r['state_hash_after'],native_equivalence_checks=[c for q in queries for c in q['canonical_checks']]))
    return dict(schema='contextworld.cross_task_decision_diagnosis.v1',no_training=True,evaluation_split='development',
        tasks=tasks,models=models,limitations=['fixed candidate bank, not CEM or global optimal control','small deterministic diagnostic panels, not replacement benchmark scores',
        'true-observation refresh supplies privileged future evidence and is not an executable policy','refresh improvement does not distinguish numerical accumulation from renewed dynamics evidence',
        'no-training measurements cannot establish whether more parameters or longer trained context is necessary'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=collect(a.root)
    a.output.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    for t in d['tasks']:
        if t['status']=='complete':print(t['task'],t['scenes'],round(t['mean_decision_gap'][-1],6))
    print('models complete',sum(x['status']=='complete' for x in d['models']))
