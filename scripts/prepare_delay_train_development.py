#!/usr/bin/env python3
"""Extract matched H7 first-endpoint queries from registered Lance splits.

No simulation, training, target-based selection, or new synthetic trajectories.
The Training selection must belong to the optimizer split for all checkpoint seeds.
"""
from __future__ import annotations
import argparse, hashlib, io, json, re, sys
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root',type=Path,required=True)
    p.add_argument('--original-h5',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenes-per-split',type=int,default=128)
    a=p.parse_args();assert a.scenes_per_split%4==0
    import h5py,lance,torch,yaml
    from PIL import Image
    cw=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(cw))
    from contextworld.training.groups import LogicalGroupDataset,ScenarioBalancedDataset
    from contextworld.training.stablewm_bundle import _ProjectedLanceSequence,_default_epoch_size
    torch.set_num_threads(2)
    registry=json.loads((a.bundle_root/'task_registry.json').read_text())
    component=next(c for c in registry['components'] if c['component_id']=='action_delay')
    specs=[s for s in json.loads(a.models.read_text()) if s['task']=='action_delay' and s['regime']=='scratch']
    assert len(specs)==9 and sorted(set(s['training_seed'] for s in specs))==[3072,3073,3074]
    payloads={split:next(p for p in component['payloads'] if p['split']==split and p['payload_id']=='full') for split in ['training','development']}
    members=payloads['training']['members']
    identities=[]
    for s in specs:
        ck=Path(s['checkpoint']);identity=json.loads((ck.parent/'contextworld_training_identity_v1.json').read_text())['identity']
        recorded_torch=identity['training_dependencies']['torch']['version'].split('+')[0]
        assert torch.__version__.split('+')[0]==recorded_torch, 'Reconstruct optimizer split with the checkpoint training PyTorch version'
        ds=identity['target']['dataset'];contract=ds['contract']
        assert contract['payload_id']=='full' and contract['split']=='training' and contract['conditional_joint'] is None
        assert contract['history_length']==7 and contract['frameskip']==5 and contract['weights']=={'original':.5,'synthetic':.5}
        # Exact newline-terminated member serialization used by _describe_spec.
        expected=hashlib.sha256(('\n'.join(members)+'\n').encode()).hexdigest()
        assert contract['member_list_sha256']==expected,(contract['member_list_sha256'],expected)
        assert ds['adapter_source_sha256']==sha(cw/'contextworld/training/stablewm_bundle.py')
        cfg=yaml.safe_load((ck.parent/'config.yaml').read_text());assert cfg['train_split']==.9
        identities.append({'id':s['id'],'checkpoint_sha256':s['checkpoint_sha256'],'training_identity_sha256':sha(ck.parent/'contextworld_training_identity_v1.json'),'recorded_manifest_sha256':contract['manifest_sha256'],'recorded_task_registry_sha256':contract['task_registry_sha256'],'adapter_source_sha256':ds['adapter_source_sha256']})
    pattern=re.compile(r'-p(?P<shard>\d+)-d(?P<delay>\d+)-')
    leaves=[];physical={str(i):[] for i in range(5)};physical['5_to_10']=[];location={}
    for name in members:
        m=pattern.search(Path(name).name);shard=int(m['shard']);delay=int(m['delay'])
        leaf=_ProjectedLanceSequence(a.bundle_root/name,num_steps=8,frameskip=5,keys_to_load=['pixels','action'],anchored_start=None)
        assert leaf.episode_count==160 and np.all(leaf._lengths==50) and np.all(leaf._clip_counts==11)
        group=str(delay) if delay<=4 else '5_to_10'
        location[(shard,delay)]=(group,len(physical[group]),leaf)
        physical[group].append(leaf);leaves.append(leaf)
    grouped={k:ScenarioBalancedDataset(v) for k,v in physical.items()};weights={k:1. for k in grouped}
    synthetic=LogicalGroupDataset(grouped,weights,epoch_size=_default_epoch_size(grouped,weights))
    with h5py.File(a.original_h5) as f:original_length=int(np.maximum(f['ep_len'][:].astype(np.int64)-40+1,0).sum())
    class Sized:
        def __init__(self,n):self.n=n
        def __len__(self):return self.n
    top_groups={'original':Sized(original_length),'synthetic':synthetic};top_weights={'original':.5,'synthetic':.5}
    mixture=LogicalGroupDataset(top_groups,top_weights,epoch_size=_default_epoch_size(top_groups,top_weights))
    assert mixture.schedule==[0,1] and synthetic.schedule==list(range(6))
    masks={}
    # This uses the same random_split implementation and freshly seeded generator as native training.
    for seed in [3072,3073,3074]:
        train,_=torch.utils.data.random_split(range(len(mixture)),[.9,.1],generator=torch.Generator().manual_seed(seed))
        mask=np.zeros(len(mixture),bool);mask[train.indices]=True;masks[seed]=mask
    train_occurrences={};eligible=[]
    for shard in range(32):
        for ep in range(160):
            per_delay=[];qualified=True
            for delay in range(11):
                group,member_index,leaf=location[(shard,delay)]
                local=ep*11*len(physical[group])+member_index
                group_position=synthetic.names.index(group)
                syn_indices=np.arange(local,len(synthetic)//6,len(grouped[group]),dtype=np.int64)*6+group_position
                global_indices=np.concatenate([syn_indices+offset for offset in range(0,len(mixture)//2,len(synthetic))])
                global_indices=global_indices[global_indices<len(mixture)//2]*2+1
                counts={str(seed):int(mask[global_indices].sum()) for seed,mask in masks.items()}
                qualified=qualified and all(n>0 for n in counts.values());per_delay.append(counts)
                # Check the inverse index against the actual two-level sampler.
                for index in global_indices[:1]:
                    name,si=mixture.locate(int(index));g,gi=synthetic.locate(si);member,ci=grouped[g].locate(gi)
                    assert name=='synthetic' and g==group and member==member_index and ci==ep*11
            if qualified:eligible.append((shard,ep));train_occurrences[(shard,ep)]=per_delay
    output={'schema':'contextworld.delay_train_development_panel.v1','normalization':component['development_evaluation']['action_normalization'],
      'protocol':{'history_frames':7,'frameskip':5,'native_window_start':0,'physical_prediction_step':5,'delays':list(range(11)),
        'sampling':'Equal room x query-direction strata; SHA256 source-identity order; no prediction/target-based selection.',
        'training_qualification':'Start-0 clip has at least one index in native optimizer train split for every training seed; exact consumed minibatch not logged.',
        'weights':'Eleven delays equal within each scene; one native query action; same geometry strata in both splits.',
        'native_runtime_starts':list(range(11)),'declared_strict_start':0,'window_limit':'Only start 0 is a same-current conditional query.'},
      'source':{'manifest_sha256':sha(a.bundle_root/'manifest.jsonl'),'task_registry_sha256':sha(a.bundle_root/'task_registry.json'),
        'member_list_sha256':expected,'recorded_member_names_match':True,'adapter_bytes_match':True,
        'whole_release_identity_matches':all(x['recorded_manifest_sha256']==sha(a.bundle_root/'manifest.jsonl') and x['recorded_task_registry_sha256']==sha(a.bundle_root/'task_registry.json') for x in identities),
        'release_limit':'Registered training members and adapter bytes match; mounted release metadata differs from the historical whole-release receipt. This does not certify historical equality of every payload byte.',
        'original_h5_windows':original_length,'synthetic_windows':len(synthetic),'mixture_windows':len(mixture),'eligible_training_groups':len(eligible),'checkpoint_identities':identities},'splits':{}}
    current_signatures={}
    for split,payload in payloads.items():
        shards={}
        for name in payload['members']:
            m=pattern.search(Path(name).name);shards.setdefault(int(m['shard']),{})[int(m['delay'])]=name
        candidates={}
        for shard,arms in sorted(shards.items()):
            ds=lance.dataset(str(a.bundle_root/arms[0]));n=ds.count_rows()//50
            table=ds.take(list(range(0,n*50,50)),columns=['proprio','action']).to_pylist()
            for ep,row in enumerate(table):
                if split=='training' and (shard,ep) not in train_occurrences:continue
                x,y=row['proprio'];u=np.asarray(row['action']);axis=int(np.flatnonzero(u)[0]);direction=('right' if u[0]>0 else 'left') if axis==0 else ('up' if u[1]>0 else 'down')
                room='left' if x<112 else 'right';key=f'{split}/p{shard:03d}/e{ep:03d}'
                candidates[key]={'shard':shard,'episode':ep,'room':room,'direction':direction,'start_position':[x,y]}
        strata={f'{r}/{d}':[] for r in ['left','right'] for d in ['up','down']}
        for key,entry in candidates.items():strata[entry['room']+'/'+entry['direction']].append(key)
        count=a.scenes_per_split//4;selected=[]
        for layer,keys in strata.items():
            assert len(keys)>=count,(split,layer,len(keys));selected.extend(sorted(keys,key=lambda k:hashlib.sha256(k.encode()).hexdigest())[:count])
        selected.sort();scene_rows=[];signatures=set()
        (a.output/split).mkdir(parents=True,exist_ok=True)
        for index,key in enumerate(selected):
            entry=candidates[key];shard=entry['shard'];ep=entry['episode'];frames=[];actions=[];positions=[];witnesses=[]
            for delay in range(11):
                name=shards[shard][delay];ds=lance.dataset(str(a.bundle_root/name));offset=ep*50
                # Same storage slice and reshape as _ProjectedLanceSequence.__getitem__.
                table=ds.take(list(range(offset,offset+40)),columns=['episode_idx','step_idx','pixels','action','proprio','variation_action_delay_steps']).to_pydict()
                assert set(table['episode_idx'])=={ep} and table['step_idx']==list(range(40))
                assert all(int(x[0])==delay for x in table['variation_action_delay_steps'])
                blobs=table['pixels'][::5]
                pix=np.stack([np.asarray(Image.open(io.BytesIO(b)).convert('RGB')) for b in blobs]);assert pix.shape==(8,224,224,3)
                cmd=np.asarray(table['action'],np.float32).reshape(8,5,2);pos=np.asarray(table['proprio'],np.float32)[::5]
                frames.append(pix);actions.append(cmd);positions.append(pos)
                witnesses.append({'delay':delay,'member':name,'episode':ep,'native_raw_start':0,'consumed_slice_sha256':hashlib.sha256(b''.join(blobs)+cmd.tobytes()).hexdigest()})
            pix=np.stack(frames);cmd=np.stack(actions);pos=np.stack(positions)
            assert np.array_equal(pix[:,-2],np.repeat(pix[:1,-2],11,axis=0))
            assert np.array_equal(cmd,np.repeat(cmd[:1],11,axis=0))
            assert len({hashlib.sha256(x[:7].tobytes()).hexdigest() for x in pix})==11
            assert np.array_equal(pos[:,6],np.repeat(pos[:1,6],11,axis=0))
            assert np.allclose(pos[:,7]-pos[:,6],cmd[:,6,0]*(7*np.maximum(5-np.arange(11),0))[:,None],atol=1e-4,rtol=0)
            signature=hashlib.sha256(pix[0,6].tobytes()+cmd[0,:7].tobytes()).hexdigest();assert signature not in signatures;signatures.add(signature)
            path=Path(split)/f'{index:03d}.npz';np.savez_compressed(a.output/path,history_pixels=pix[:,:7],future_pixels=pix[:,7:],action_blocks=cmd[:,:7])
            row={'scene_id':key,'path':str(path),'sha256':sha(a.output/path),**entry,'members':witnesses,'current_action_signature':signature}
            if split=='training':row['optimizer_train_occurrences']=train_occurrences[(shard,ep)]
            scene_rows.append(row)
        current_signatures[split]=signatures
        output['splits'][split]={'count':len(scene_rows),'strata_counts':{k:count for k in strata},'source_group_count':len(candidates),'scenes':scene_rows}
        print(split,len(scene_rows),'eligible source groups',len(candidates),flush=True)
    assert not (current_signatures['training'] & current_signatures['development'])
    write(a.output/'runtime_environment.json',{'torch_version':torch.__version__,'training_torch_version':recorded_torch,'optimizer_split_torch_matches':True})
    write(a.output/'manifest.json',output);print('manifest',a.output/'manifest.json',flush=True)

if __name__=='__main__':main()
