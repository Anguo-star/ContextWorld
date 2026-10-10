#!/usr/bin/env python3
"""No-update Strength H3 native visual-prediction MSE gradient diagnostic.

This measures only the visual prediction component, not the entire JEPA/SIGREG
objective. Cached native image latents are detached; only predictor parameters
receive autograd gradients. Selection uses metadata and a fixed seed only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import diagnose_delay_native_objective as generic

NAMES = ('final_response', 'final_common', 'earlier_two', 'final_native', 'native_all_three')


def evaluate(adapter, helper, history, future, actions, named_parameters, pair_id):
    import torch
    h = torch.as_tensor(history, dtype=torch.float32, device=adapter.device).detach()
    y = torch.as_tensor(future, dtype=torch.float32, device=adapter.device).detach()
    if h.shape[:2] != (2, 3) or y.shape[:2] != (2, 1) or actions.shape[:2] != (2, 3):
        raise ValueError(f'Invalid K2/H3 pair shapes: {h.shape}, {y.shape}, {actions.shape}')
    normalized = torch.as_tensor(adapter._normalize_actions(actions), device=adapter.device)
    prediction = adapter.model.predict(h, adapter.model.action_encoder(normalized))
    target = torch.cat((h[:, 1:], y), dim=1)
    if prediction.shape != target.shape:
        raise ValueError('Prediction shape does not match shifted native target')
    d = generic._loss_decomposition(prediction, target)
    losses = dict(zip(NAMES, (d['response_by_position'][-1]/3,
        d['common_by_position'][-1]/3, d['native_by_position'][:-1].sum()/3,
        d['native_by_position'][-1]/3, d['native'])))
    loss_residual = float((losses['native_all_three'] - sum(losses[n] for n in NAMES[:3])).detach().cpu())
    position_residual = float((d['native_by_position']-d['response_by_position']-d['common_by_position']).abs().max().detach().cpu())
    if max(abs(loss_residual), position_residual) > generic.MSE_CLOSURE_ATOL:
        raise RuntimeError('Native loss reconstruction failed')
    canonical = helper.predict_all(adapter, history, future, actions, 'lewm', modes=('free',))['free'][:, 0]
    delta = prediction[:, -1].detach().cpu().numpy() - canonical
    max_abs = float(np.abs(delta).max())
    rel = float(np.linalg.norm(delta)/max(np.linalg.norm(canonical), 1e-12))
    if max_abs >= generic.CANONICAL_MAX_ABS or rel >= generic.CANONICAL_RELATIVE_L2:
        raise RuntimeError(f'Adapter endpoint mismatch: {max_abs}, {rel}')
    params = tuple(p for _, p in named_parameters)
    gradients = {n: torch.autograd.grad(losses[n], params, retain_graph=i<len(NAMES)-1, allow_unused=True)
                 for i,n in enumerate(NAMES)}
    closure = generic._gradient_closure(gradients['native_all_three'], [gradients[n] for n in NAMES[:3]])
    if closure['relative_to_component_norm_sum'] > generic.GRADIENT_CLOSURE_RELATIVE_MAX or closure['max_abs'] > 1e-5:
        raise RuntimeError(f'Gradient reconstruction failed: {closure}')
    scalar = lambda x: float(x.detach().cpu())
    row = {'pair_id':pair_id, 'condition_count':2, 'position_count':3,
        'losses':{n:scalar(v) for n,v in losses.items()},
        'final_response_loss_share':scalar(losses['final_response']/losses['native_all_three']),
        'native_query_response_nre':(scalar(d['response_by_position'][-1]/d['target_response_energy_by_position'][-1]) if scalar(d['target_response_energy_by_position'][-1])>0 else None),
        'loss_reconstruction_residual':loss_residual, 'position_decomposition_max_abs':position_residual,
        'per_position':{n:d[n].detach().cpu().tolist() for n in d if n!='native'},
        'canonical_adapter_endpoint':{'max_abs':max_abs,'relative_l2':rel},
        'gradient':generic._pairwise_gradient_summary(gradients), 'gradient_reconstruction':closure}
    norms=row['gradient']['norms']
    row['gradient']['response_to_native_norm_ratio']=norms['final_response']/max(norms['native_all_three'],1e-30)
    return row, gradients, prediction[:, -1].detach().cpu().numpy()


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--id',required=True)
    p.add_argument('--cache',type=Path,required=True,help='Shared extraction root')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',required=True,choices=('cuda:3','cuda:4','cuda:5'))
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    specs=json.loads(a.models.read_text()); spec=next(s for s in specs if s['id']==a.id)
    if spec['task']!='action_strength' or spec['family']!='lewm' or spec['regime'] not in ('scratch','joint','frozen'):
        raise ValueError('Only existing Strength LeWM T1/T2/T3 are supported')
    manifest_path=a.cache/'manifest.json'; manifest=json.loads(manifest_path.read_text())
    training_meta=manifest['splits']['training']
    fit=training_meta['pair_ids']
    if len(fit)<512: raise ValueError('Expected at least512 Training fit pairs with complete source groups')
    raw_path=a.cache/'training.npz'
    raw_sha=generic._sha256(raw_path)
    if raw_sha!=training_meta['sha256']: raise ValueError('Shared raw cache hash mismatch')
    raw=np.load(raw_path,allow_pickle=False)
    chosen=[]; chosen_groups=[]; seen_groups=set()
    for pair_id in fit:
        indices=np.flatnonzero(raw['pair_ids']==pair_id)
        groups=set(raw['source_groups'][indices].astype(str).tolist())
        if len(indices)!=2 or len(groups)!=1: raise ValueError('Pair source group mismatch')
        group=next(iter(groups))
        if group not in seen_groups:
            chosen.append(pair_id); chosen_groups.append(group); seen_groups.add(group)
        if len(chosen)==16: break
    if len(chosen)!=16: raise ValueError('Need16distinct Training source groups')
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'result.json').exists(): raise FileExistsError('Result exists')
    selection={'seed':20261010,'algorithm':'first16 distinct source groups in metadata-selected seed20261010 Training manifest order; firstpair/group',
               'split':'training','fit_pair_count':len(fit),'selected_pairs':chosen,'source_groups':chosen_groups,'source_group_count':len(chosen_groups),'manifest_sha256':generic._sha256(manifest_path)}
    generic._write_json(a.output/'selection.json',selection)
    helper=generic._load_helper(); adapter=helper.load_adapter(spec,manifest['normalization'],a.output,a.device)
    model=adapter.model; model.eval(); before=adapter.frozen_state_hash()
    flags=[q.requires_grad for q in model.parameters()]
    for q in model.parameters(): q.requires_grad_(False)
    model.predictor.requires_grad_(True)
    named=[('predictor.'+n,q) for n,q in model.predictor.named_parameters() if q.requires_grad]
    totals={n:[torch.zeros_like(q,device='cpu') for _,q in named] for n in NAMES}
    sums={n:0.0 for n in NAMES}; rows=[]
    try:
        for pair_id in chosen:
            indices=np.flatnonzero(raw['pair_ids']==pair_id)
            if len(indices)!=2 or raw['labels'][indices].tolist()!=[0,1]:
                raise ValueError('Expected pair-major low/high condition order')
            pixels=raw['history_pixels'][indices]
            futures=raw['queryfuture_pixels'][indices,None]
            actions=raw['rawactions'][indices].reshape(2,3,5,2)
            entry={'pair_id':str(pair_id),'row_indices':indices.tolist(),'raw_sha256':raw_sha,
                   **{n:raw[n][indices].astype(str).tolist() for n in
                      ('modes','episode_ids','source_episode_ids','source_step_ids','source_row_ids','source_groups')}}
            if pixels.shape[:2]!=(2,3) or futures.shape[:2]!=(2,1): raise ValueError('Expected raw RGB0/5/10 and future15')
            history,future=helper.encode_unique(adapter,pixels,futures)
            row,grad,direct=evaluate(adapter,helper,history,future,actions,named,str(pair_id))
            if not rows:
                actual=adapter.rollout_latents(pixels,actions,batch_size=2)[:,0]
                delta=direct-actual
                max_abs=float(np.abs(delta).max())
                relative=float(np.linalg.norm(delta)/max(np.linalg.norm(actual),1e-12))
                if max_abs>=generic.CANONICAL_MAX_ABS or relative>=generic.CANONICAL_RELATIVE_L2:
                    raise RuntimeError(f'Actual public adapter endpoint mismatch: {max_abs}, {relative}')
                row['actual_public_adapter_endpoint']={'max_abs':max_abs,'relative_l2':relative,'input':'same shared RGB and raw action blocks as native-position calculation'}
            row['source']=entry; rows.append(row)
            for n,g in grad.items():
                generic._accumulate_gradient(totals[n],g); sums[n]+=generic._tensor_tuple_norm(g)
            print(a.id,len(rows),flush=True)
        after=adapter.frozen_state_hash()
        if before!=after: raise RuntimeError('Model state changed')
    finally:
        for q,flag in zip(model.parameters(),flags): q.requires_grad_(flag)
    result={'schema':'contextworld.strength_native_visual_objective.v1','model':spec,
        'selection':selection,'pair_count':len(rows),'condition_count':32,
        'cuda_precision':{'matmul_allow_tf32':False,'cudnn_allow_tf32':False,'float32_matmul_precision':'highest'},
        'scope':'Training only; H3/K2 raw RGB0,5,10; target15; native shifted targets5,10,15; visual prediction MSE component only, not entire JEPA/SIGREG objective',
        'optimizer_steps':0,'public_test_accessed':False,'state_hash_before':before,'state_hash_after':after,
        'predictor_parameter_count':sum(q.numel() for _,q in named),
        'script_sha256':generic._sha256(Path(__file__)), 'generic_helper_sha256':generic._sha256(Path(generic.__file__)),
        'trainer_alignment':{'native_visual_loss':'(prediction - embeddings[:,1:]).square().mean()', 'pinned_source':str(Path(spec['stable_repo'])/'scripts/train/lewm.py'),'pinned_source_sha256':generic._sha256(Path(spec['stable_repo'])/'scripts/train/lewm.py'),'source_expression':'tgt_emb=emb[:,n_preds:]; pred_emb=model.predict(ctx_emb,ctx_act); pred_loss=(pred_emb-tgt_emb).pow(2).mean()', 'n_preds':1,'ctx_len':3, 'other_objective_components_excluded':['JEPA/SIGREG regularizers']},
        'mean_losses':{n:float(np.mean([r['losses'][n] for r in rows])) for n in NAMES},
        'summary':{'mean_final_target_response_energy':float(np.mean([r['per_position']['target_response_energy_by_position'][-1] for r in rows])),
                   'mean_native_query_response_nre':(float(np.mean([r['native_query_response_nre'] for r in rows])) if all(r['native_query_response_nre'] is not None for r in rows) else None),
                   'zero_final_target_energy_pair_count':sum(r['per_position']['target_response_energy_by_position'][-1]==0 for r in rows),
                   'energy_pooled_native_query_response_nre':float(3*sum(r['losses']['final_response'] for r in rows)/sum(r['per_position']['target_response_energy_by_position'][-1] for r in rows)),
                   'mean_pair_final_response_loss_share':float(np.mean([r['final_response_loss_share'] for r in rows])),
                   'aggregate_final_response_loss_share':float(sum(r['losses']['final_response'] for r in rows)/sum(r['losses']['native_all_three'] for r in rows))},
        'mean_gradient':generic._aggregate_gradient_summary(totals,sums,len(rows)), 'per_pair':rows,
        'limitation':'Three posthoc checkpoint diagnostics cannot independently establish optimization causality'}
    generic._write_json(a.output/'result.json',result)

if __name__=='__main__': main()
