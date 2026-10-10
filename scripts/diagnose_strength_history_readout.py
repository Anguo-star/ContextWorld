#!/usr/bin/env python3
"""Fixed full-native Strength history readout; Training/Development only."""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, json, os, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEED = 20261010
ROOT = Path('/opt/huawei/explorer-env/dataset/ag_data/data/world_model/ContextWorld-action-strength-32k-v1')
MODES = ['low_gain', 'high_gain']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)

def helper():
    spec = importlib.util.spec_from_file_location('strength_cross_task_helper', REPO/'scripts/diagnose_cross_task_decisions.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def versions(path):
    return {str(p.relative_to(path)): [p.stat().st_size, p.stat().st_mtime_ns] for p in path.rglob('*') if p.is_file() and '_versions' in p.parts}

def build_panel(output):
    import lance
    from PIL import Image
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists(): raise FileExistsError(output/'manifest.json')
    component = ROOT/'components/pusht-action-strength/v1'
    registry = json.loads((ROOT/'task_registry.json').read_text())
    entry = next(x for x in registry['components'] if x['dataset_id']=='pusht-action-strength')
    provenance_path = REPO/'artifacts/synthesis/pusht_action_strength_h3_release_v1/manifest.json'
    provenance = json.loads(provenance_path.read_text())['source']
    source_identity = provenance['file_sha256']
    result = dict(schema='contextworld.strength_history_panel.v1', seed=SEED, release=ROOT.name,
        component_root=str(component), source_provenance_path=str(provenance_path), source_provenance_sha256=sha(provenance_path),
        source_identity=source_identity, source_group_rule='upstream_source_file_sha256 + source_episode_index; pair IDs are not independent',
        normalization=entry['development_evaluation']['action_normalization'],
        selection='seed20261010 permutation of sorted upstream source episode groups; retain complete groups until >=512 Training pairs; sorted selected pair IDs; all256 Development pairs',
        layout='pair-major, low_gain then high_gain; row axis K=2*pairs; history0/5/10, future15, actions0:15',
        no_test_read=True, release_hashes={n:sha(ROOT/n) for n in ['manifest.jsonl','manifest.sha256','task_registry.json']}, splits={})
    split_groups = {}
    for split, expected in [('training',32768), ('development',256)]:
        path=component/split/'data.lance'; before=versions(path); ds=lance.dataset(path)
        columns=['episode_idx','pair_id','hidden_mode','source_episode_index','source_step_index','source_row_index','split']
        meta=ds.to_table(filter='step_idx = 0', columns=columns).to_pylist()
        by={}
        for row in meta: by.setdefault(row['pair_id'], {})[row['hidden_mode']]=row
        pairs=sorted(by); assert len(pairs)==expected
        assert all(set(v)==set(MODES) for v in by.values())
        selected=pairs
        if split=='training':
            pair_groups={}
            for pair in pairs:
                source=int(by[pair][MODES[0]]['source_episode_index'][0])
                pair_groups.setdefault(source,[]).append(pair)
            selected=[]
            for group_id in np.random.default_rng(SEED).permutation(sorted(pair_groups)):
                selected.extend(pair_groups[int(group_id)])
                if len(selected)>=512:break
            selected=sorted(selected)
        rows=[by[p][mode] for p in selected for mode in MODES]
        episode_ids=[int(r['episode_idx']) for r in rows]; extracted={e:{} for e in episode_ids}
        for start in range(0,len(episode_ids),128):
            ids=','.join(map(str,episode_ids[start:start+128]))
            for row in ds.to_table(filter=f'episode_idx IN ({ids}) AND step_idx < 15',columns=['episode_idx','step_idx','action']).to_pylist():
                extracted[row['episode_idx']].setdefault(row['step_idx'],{})['action']=row['action']
            for row in ds.to_table(filter=f'episode_idx IN ({ids}) AND step_idx IN (0,5,10,15)',columns=['episode_idx','step_idx','pixels']).to_pylist():
                extracted[row['episode_idx']].setdefault(row['step_idx'],{})['pixels']=row['pixels']
        history=[]; future=[]; actions=[]
        for row in rows:
            er=extracted[int(row['episode_idx'])]
            frames=[]
            for pos in [0,5,10,15]:
                with Image.open(io.BytesIO(er[pos]['pixels'])) as image: frames.append(np.asarray(image.convert('RGB'),dtype=np.uint8))
            history.append(np.stack(frames[:3])); future.append(frames[3]); actions.append(np.asarray([er[t]['action'] for t in range(15)],np.float32))
        source_ids=np.asarray([int(r['source_episode_index'][0]) for r in rows])
        groups=np.asarray([f'{source_identity}:episode:{i}' for i in source_ids])
        h=np.stack(history); f=np.stack(future); a=np.stack(actions)
        assert h.shape==(len(rows),3,224,224,3) and f.shape==(len(rows),224,224,3)
        assert np.array_equal(h[0::2,2],h[1::2,2]), 'Current RGB pair inequality'
        assert np.array_equal(a[0::2],a[1::2]), 'Original actions differ across pair'
        for key in ['source_episode_index','source_step_index','source_row_index']:
            assert all(by[p][MODES[0]][key]==by[p][MODES[1]][key] for p in selected)
        outpath=output/f'{split}.npz'
        np.savez(outpath,history_pixels=h,queryfuture_pixels=f,rawactions=a,pair_ids=np.asarray([r['pair_id'] for r in rows]),
            modes=np.asarray([r['hidden_mode'] for r in rows]),labels=np.tile([0,1],len(selected)),episode_ids=np.asarray(episode_ids),
            source_episode_ids=source_ids, source_step_ids=np.asarray([int(r['source_step_index'][0]) for r in rows]),
            source_row_ids=np.asarray([int(r['source_row_index'][0]) for r in rows]),source_groups=groups)
        assert before==versions(path), 'Lance versions changed'
        split_groups[split]=set(int(r['source_episode_index'][0]) for r in meta)
        result['splits'][split]=dict(path=outpath.name,sha256=sha(outpath),pairs=len(selected),pair_ids=selected,
            selected_source_groups=len(set(groups)),metadata_pairs=expected,metadata_source_groups=len(split_groups[split]),
            lance_path=str(path),lance_version=ds.version,lance_versions=before,current_rgb_equal_pairs=len(selected),actions_equal_pairs=len(selected))
        print('panel',split,len(selected),flush=True)
    overlap=split_groups['training']&split_groups['development']; assert not overlap, 'Training/Dev source episode overlap'
    result['full_metadata_source_episode_overlap_count']=len(overlap)
    write(output/'manifest.json',result)

def metrics(labels,pred,groups,pairs):
    from sklearn.metrics import confusion_matrix
    ok=labels==pred; per=[float(ok[labels==i].mean()) for i in range(2)]
    keys=sorted(set(groups)); counts=np.zeros((len(keys),2),int); correct=counts.copy()
    for i,g in enumerate(keys):
        mask=groups==g
        for c in range(2): counts[i,c]=np.sum(mask&(labels==c)); correct[i,c]=np.sum(mask&(labels==c)&ok)
    draw=np.random.default_rng(SEED).integers(len(keys),size=(2000,len(keys)))
    rates=correct[draw].sum(1)/counts[draw].sum(1)
    ba=rates.mean(1); worst=rates.min(1)
    return dict(balanced_accuracy=float(np.mean(per)),worst_condition_accuracy=min(per),condition_accuracy=dict(zip(MODES,per)),
        confusion_matrix=confusion_matrix(labels,pred,labels=[0,1]).tolist(),
        source_group_bootstrap=dict(resamples=2000,seed=SEED,groups=len(keys),balanced_accuracy_95ci=np.quantile(ba,[.025,.975]).tolist(),worst_condition_95ci=np.quantile(worst,[.025,.975]).tolist()),
        predictions=[dict(pair_id=str(p),source_group=str(g),label=int(y),prediction=int(v),correct=bool(y==v)) for p,g,y,v in zip(pairs,groups,labels,pred)])

def run_model(args):
    import torch
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    started=time.time(); torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    manifest_path=args.panel/'manifest.json'; manifest=json.loads(manifest_path.read_text())
    spec=next(r for r in json.loads(args.models.read_text()) if r['id']==args.id)
    assert spec['task']=='action_strength' and spec['family']=='lewm'
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    helpers=helper(); adapter=helpers.load_adapter(spec,manifest['normalization'],out,args.device)
    adapter.model.eval(); before=adapter.frozen_state_hash(); encoded={}; cache_hashes={}; controls={}
    for split in ['training','development']:
        info=manifest['splits'][split]; p=args.panel/info['path']; assert sha(p)==info['sha256']
        with np.load(p) as raw:
            # encode_unique preserves all 192 native coordinates without pooling/projection.
            h,f=helpers.encode_unique(adapter,raw['history_pixels'],raw['queryfuture_pixels'])
            assert h.shape==(len(raw['labels']),3,192) and f.shape==(len(raw['labels']),192)
            assert np.array_equal(h[0::2,2],h[1::2,2]), 'Paired z2 not identical'
            path=out/f'{split}_latents.npz'
            retained={k:raw[k] for k in ['rawactions','pair_ids','modes','labels','episode_ids','source_episode_ids','source_step_ids','source_row_ids','source_groups']}
            np.savez(path,history=h,queryfuture=f,**retained);cache_hashes[split]=dict(path=str(path),sha256=sha(path))
            encoded[split]=dict(history=h,**retained)
            controls[split]=dict(current_latent_equal_pairs=len(raw['labels'])//2,current_rgb_equal_pairs=info['current_rgb_equal_pairs'],
                deterministic_current_only_accuracy_upper_bound=.5)
        print(args.id,split,'encoded',h.shape,flush=True)
    after=adapter.frozen_state_hash(); assert before==after, 'Frozen model state changed'
    result=dict(schema='contextworld.strength_full_native_history_readout.v1',model_id=args.id,checkpoint=spec['checkpoint'],checkpoint_sha256=spec['checkpoint_sha256'],
        panel_manifest_sha256=sha(manifest_path),program_sha256=sha(__file__),model_state_hash_before=before,model_state_hash_after=after,
        no_worldmodel_update=True,no_test_read=True,device=args.device,seed=SEED,latents=cache_hashes,controls=controls,
        feature='concat(z0,z1-z0,z2-z1), full native 192 per-frame coordinates; no actions in classifier; no pooling/projection',
        limitations='Classification success establishes readable condition information, not complete future representation sufficiency. Fixed readout failure does not establish absence of information in encoder.',readouts={})
    for feature in ['history']:
        xs={s:np.concatenate([d['history'][:,0],d['history'][:,1]-d['history'][:,0],d['history'][:,2]-d['history'][:,1]],axis=-1) if feature=='history' else d['history'][:,2] for s,d in encoded.items()}
        for name,model in [('ridge',make_pipeline(StandardScaler(),RidgeClassifier(alpha=1))),('extra_trees',make_pipeline(StandardScaler(),ExtraTreesClassifier(n_estimators=256,random_state=SEED,n_jobs=1)))]:
            model.fit(xs['training'],encoded['training']['labels']); scores={}
            for split,d in encoded.items():
                pred=model.predict(xs[split]);scores[split]=metrics(d['labels'],pred,d['source_groups'],d['pair_ids'])
                if feature=='current_only': assert scores[split]['balanced_accuracy']==.5
            result['readouts'][feature+'/'+name]=scores
            print(args.id,feature,name,{s:v['balanced_accuracy'] for s,v in scores.items()},flush=True)
    result['elapsed_seconds']=time.time()-started; write(out/'result.json',result)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build-panel',action='store_true')
    p.add_argument('--panel',type=Path,default=Path('/tmp/cw-strength-mechanism-20261010/panel'))
    p.add_argument('--models',type=Path,default=Path('/tmp/cw-icl-validity-20261007/models.json'))
    p.add_argument('--id');p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.build_panel: build_panel(args.panel)
    else:
        if not args.id or not args.output:p.error('--id and --output required for readout')
        run_model(args)
if __name__=='__main__':main()
