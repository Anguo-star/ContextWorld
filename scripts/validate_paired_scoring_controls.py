#!/usr/bin/env python3
"""Deterministic controls for Action Strength's paired latent scorer.

This is a scorer-function control test on cached native targets, not a
benchmark reevaluation: no model prediction, training, replay, or simulation.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, importlib, json, math, sys, traceback
from pathlib import Path
import numpy as np

CW_ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = None
RESULTS_ROOT = None
OUT_DIR = Path('/tmp/cw-paired-score-controls-20261008/candidate1')
CANDIDATE_INDEX = 1
VARIANTS = [('original', 's3073'), ('frozen', 's3072')]
SCENARIOS = ('oracle_correct_target', 'ignore_history_predict_A',
             'ignore_history_predict_B', 'shared_midpoint_strict_tie',
             'swapped_targets_wrong_history', 'tiny_correct_response_0.001')
ATOL = 2e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_official():
    sys.path.insert(0, str(CW_ROOT))
    paired_mod = importlib.import_module('contextworld.benchmarks.paired_latent_response')
    paired = paired_mod.paired_latent_response_metrics
    scorer_path = CW_ROOT / 'contextworld/benchmarks/action_strength_icl_score.py'
    scorer_hash = sha256(scorer_path)
    try:
        mod = importlib.import_module('contextworld.benchmarks.action_strength_icl_score')
        return mod._prediction_metrics, paired, 'imported_official_module', scorer_hash
    except Exception as exc:
        # Extract the exact two scorer functions from the source file; keep the
        # original source AST/code and pair it with the genuine shared helper.
        tree = ast.parse(scorer_path.read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in ('_mse', '_prediction_metrics')]
        ns = {'np': np, 'paired_latent_response_metrics': paired}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(scorer_path), 'exec'), ns)
        return ns['_prediction_metrics'], paired, 'exact_ast_functions_after_import_error: ' + repr(exc), scorer_hash


def source_manifest():
    raw = (PANEL_DIR / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    entries = {x['scene_id']: x for x in manifest['scenes'] if x.get('ok')}
    return manifest, entries, hashlib.sha256(raw).hexdigest()


def load_model(variant: str, seed: str, entries, manifest_hash, candidate_index: int):
    d = RESULTS_ROOT / variant / seed
    receipts = sorted(d.glob('pair_*.json'))
    if len(receipts) != 256:
        raise RuntimeError(f'{d}: expected 256 pair JSONs, found {len(receipts)}')
    scene_ids, a_rows, b_rows, receipt_rows = [], [], [], []
    input_checks = {'receipt_count': len(receipts), 'pair_receipt_errors': [],
                    'feature_sha_verified': 0, 'panel_source_sha_verified': 0,
                    'manifest_sha_verified': 0, 'native_layout_receipts': 0}
    checkpoint_hashes = set()
    latent_nonzero_by_candidate = {}
    candidate_available_by_index = {}
    for rp in receipts:
        receipt = json.loads(rp.read_text())
        scene_id = receipt['scene_id']
        idx = int(rp.stem.split('_')[-1])
        expected_id = f'phrm-validation-{idx:05d}'
        errors = []
        if scene_id != expected_id: errors.append(f'scene_id index mismatch {expected_id}')
        if receipt.get('model_id') != f'action_strength/lewm/{variant}/{seed}': errors.append('model_id mismatch')
        if receipt.get('task') != 'action_strength' or receipt.get('family') != 'lewm': errors.append('task/family mismatch')
        if receipt.get('manifest_sha256') != manifest_hash: errors.append('manifest hash mismatch')
        else: input_checks['manifest_sha_verified'] += 1
        if receipt.get('feature_layout', {}).get('kind') != 'native_latent' or receipt.get('feature_not_used_in_native_score') is not True:
            errors.append('not confirmed native latent target cache')
        else: input_checks['native_layout_receipts'] += 1
        if scene_id not in entries:
            errors.append('scene absent from panel manifest')
        else:
            source = entries[scene_id]
            if source.get('sha256') != receipt.get('source_sha256'):
                errors.append('panel source SHA mismatch')
            else: input_checks['panel_source_sha_verified'] += 1
            panel_file = PANEL_DIR / source['path']
            if sha256(panel_file) != source.get('sha256'):
                errors.append('panel source bytes do not match manifest SHA')
        feat_path = d / receipt['features_file']
        if sha256(feat_path) != receipt.get('features_file_sha256'):
            errors.append('feature file SHA mismatch')
        else: input_checks['feature_sha_verified'] += 1
        if errors:
            input_checks['pair_receipt_errors'].append({'receipt': rp.name, 'errors': errors})
        checkpoint_hashes.add(receipt.get('checkpoint_sha256'))
        with np.load(feat_path, allow_pickle=False) as z:
            if 'target' not in z.files: raise RuntimeError(f'{feat_path}: target missing')
            target = z['target']
            if target.ndim != 4 or target.shape[0] != 2 or target.shape[1] < 1 or target.shape[2] < 1:
                raise RuntimeError(f'{feat_path}: unexpected target shape {target.shape}')
            if not 0 <= candidate_index < target.shape[1]:
                raise RuntimeError(f'{feat_path}: candidate index {candidate_index} outside K={target.shape[1]}')
            coverage = np.any(target[0, :, 0] != target[1, :, 0], axis=-1)
            for ci, is_nonzero in enumerate(coverage):
                latent_nonzero_by_candidate[ci] = latent_nonzero_by_candidate.get(ci, 0) + int(is_nonzero)
                candidate_available_by_index[ci] = candidate_available_by_index.get(ci, 0) + 1
            if tuple(receipt.get('free_prediction_shape', ())) != tuple(z['pred'].shape):
                input_checks['pair_receipt_errors'].append({'receipt': rp.name, 'errors': ['prediction shape receipt mismatch']})
            if receipt.get('candidate_axis_semantics') and 'canonical unique K' not in receipt['candidate_axis_semantics']:
                input_checks['pair_receipt_errors'].append({'receipt': rp.name, 'errors': ['candidate axis not stated canonical unique K']})
            if target.shape[1] != int(receipt.get('candidates', -1)):
                input_checks['pair_receipt_errors'].append({'receipt': rp.name, 'errors': ['candidate count vs target K mismatch']})
            a = np.asarray(target[0, candidate_index, 0], dtype=np.float64)
            b = np.asarray(target[1, candidate_index, 0], dtype=np.float64)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise RuntimeError(f'{feat_path}: non-finite target')
            scene_ids.append(scene_id); a_rows.append(a); b_rows.append(b)
            receipt_rows.append({'scene_id': scene_id, 'index': idx,
                                 'source_sha256': receipt.get('source_sha256'),
                                 'features_file': receipt['features_file'],
                                 'features_file_sha256': receipt.get('features_file_sha256'),
                                 'checkpoint_sha256': receipt.get('checkpoint_sha256'),
                                 'target_dtype': str(target.dtype),
                                 'target_shape': list(target.shape), 'candidate_index': candidate_index,
                                 'target_pair_sha256': hashlib.sha256(np.stack([target[0,0,0], target[1,0,0]]).tobytes()).hexdigest()})
    if len(set(scene_ids)) != 256:
        raise RuntimeError(f'{variant}/{seed}: scene ID coverage not 256 unique')
    return {'variant': variant, 'seed': seed, 'scene_ids': scene_ids,
            'A': np.stack(a_rows), 'B': np.stack(b_rows), 'receipts': receipt_rows,
            'input_checks': input_checks, 'checkpoint_hashes': sorted(checkpoint_hashes),
            'latent_nonzero_by_candidate_step5': {str(i):{'nonzero':n,'available':candidate_available_by_index[i],
             'rate':n/candidate_available_by_index[i]} for i,n in sorted(latent_nonzero_by_candidate.items())}}


def predictions(a, b):
    mid = a / 2.0 + b / 2.0
    return {
        'oracle_correct_target': (a.copy(), b.copy()),
        'ignore_history_predict_A': (a.copy(), a.copy()),
        'ignore_history_predict_B': (b.copy(), b.copy()),
        'shared_midpoint_strict_tie': (mid.copy(), mid.copy()),
        'swapped_targets_wrong_history': (b.copy(), a.copy()),
        'tiny_correct_response_0.001': (mid + 0.001 * (a-mid), mid + 0.001 * (b-mid)),
    }


def hand_one(pa, pb, a, b):
    # Exact formulas from the official source, independently spelled out.
    ll = float(np.mean((pa-a)**2)); lh = float(np.mean((pa-b)**2))
    hl = float(np.mean((pb-a)**2)); hh = float(np.mean((pb-b)**2))
    ft = (ll < lh, hh < hl)
    hist = (ll < hl, hh < lh)
    switch = float(np.sum((pb-pa)*(b-a)) > 0)
    d = b-a
    energy = float(np.mean(d*d))
    if energy == 0.0:
        gain = alignment = nre = None
        calibrated = None
    else:
        response = pb-pa
        pred_energy = float(np.mean(response*response))
        cross = float(np.mean(response*d))
        gain = cross / energy
        alignment = cross / math.sqrt(pred_energy*energy) if pred_energy > 0 else 0.0
        nre = float(np.mean((response-d)**2) / energy)
        calibrated = bool(nre < 1.0)
    joint = None if calibrated is None else bool(all(ft) and all(hist) and calibrated)
    return {'ll':ll,'lh':lh,'hl':hl,'hh':hh,
            'correct_future':ft,'correct_history':hist,'switch':bool(switch),
            'correct_future_mse_mean':(ll+hh)/2.0,
            'other_future_mse_mean':(lh+hl)/2.0,
            'target_response_mse':energy,'response_gain':gain,
            'response_alignment':alignment,'normalized_response_error':nre,
            'calibrated_response_success':calibrated,'joint_icl_pair_success':joint,
            'exact_future_tie_count':int(ll == lh)+int(hh == hl),
            'near_future_tie_count_1e-12':int(abs(ll-lh)<=1e-12)+int(abs(hh-hl)<=1e-12)}


def main():
    global CW_ROOT, PANEL_DIR, RESULTS_ROOT, OUT_DIR, CANDIDATE_INDEX
    ap=argparse.ArgumentParser()
    ap.add_argument('--contextworld-root',type=Path,default=CW_ROOT)
    ap.add_argument('--panel-dir',type=Path,required=True)
    ap.add_argument('--results-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--candidate-index',type=int,default=1)
    args=ap.parse_args()
    CW_ROOT=args.contextworld_root.resolve(); PANEL_DIR=args.panel_dir.resolve(); RESULTS_ROOT=args.results_root.resolve(); OUT_DIR=args.output_dir.resolve(); CANDIDATE_INDEX=args.candidate_index
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    score_fn, paired_fn, import_mode, scorer_hash = load_official()
    manifest, entries, manifest_hash = source_manifest()
    if len(entries) != 256: raise RuntimeError(f'panel manifest has {len(entries)} successful scene records')
    models = [load_model(v,s,entries,manifest_hash,CANDIDATE_INDEX) for v,s in VARIANTS]
    csv_path = OUT_DIR / 'paired_score_scene_results.csv'
    summary = {
        'title': 'Action Strength paired scorer deterministic control test',
        'interpretation': 'Function control on cached native target latents; not an original benchmark reevaluation, no training/inference/replay/simulation.',
        'inputs': {'panel_dir':str(PANEL_DIR),'panel_manifest_sha256':manifest_hash,
                   'panel_manifest_scene_count':len(entries),
                   'models':[{'model_id':f"action_strength/lewm/{m['variant']}/{m['seed']}",
                              'cache_root':str(RESULTS_ROOT/m['variant']/m['seed']),
                              'scene_count':len(m['scene_ids']), 'checkpoint_sha256_set':m['checkpoint_hashes'],
                              'input_checks':m['input_checks']} for m in models]},
        'scorer': {'path':str(CW_ROOT/'contextworld/benchmarks/action_strength_icl_score.py'),
                   'sha256':scorer_hash,'load_mode':import_mode,
                   'paired_response_path':str(CW_ROOT/'contextworld/benchmarks/paired_latent_response.py'),
                   'paired_response_is_genuine_import':True},
        'selection': {'candidate_axis':'canonical unique K axis', 'candidate_index':CANDIDATE_INDEX,
                      'horizon_index':0,'horizon_native_step':5,
                      'target_source':'features_pair_*.npz[target], conditions 0 and 1, native latent 192d',
                      'all_candidate_step5_nonzero_coverage_by_model':{f"action_strength/lewm/{m['variant']}/{m['seed']}":m['latent_nonzero_by_candidate_step5'] for m in models}},
        'numerical_conventions': {'zero_separation':'Exact target equality only; no epsilon and no rows silently removed. The official paired scorer rejects the entire call if any zero-separation pair is present.',
                                  'comparison':'Official nearest-target and paired response decisions are strict inequalities.',
                                  'handcheck_atol':ATOL},
        'scenarios': {}, 'model_scenarios': {}, 'cross_task_formula_note': {},
        'validation': {'scorer_vs_independent_handcheck': True}
    }
    rows = []
    for model in models:
        a,b=model['A'],model['B']; ids=model['scene_ids']; pred_map=predictions(a,b)
        sep=np.mean((b-a)**2,axis=1)
        zero=sep==0.0
        all_hand={}
        for scenario,(pa,pb) in pred_map.items():
            h=[hand_one(pa[i],pb[i],a[i],b[i]) for i in range(len(ids))]
            all_hand[scenario]=h
            # Try the official scorer on the full requested cohort. If any target pair
            # is exactly collapsed, preserve the failure and do not report an official
            # full-cohort number. Also call the official scorer one pair at a time for
            # each mathematically defined pair to verify the control formulas.
            full_call_status='defined'
            full_exception=None
            try:
                official_summary, official_records = score_fn(pair_ids=tuple(ids), predicted_low=pa, predicted_high=pb, target_low=a, target_high=b)
            except Exception as exc:
                full_call_status='undefined_due_to_zero_target_separation' if zero.any() else 'error'
                full_exception=f'{type(exc).__name__}: {exc}'
                official_summary=official_records=None
            single_defined=0; single_mismatches=[]
            for i in range(len(ids)):
                if zero[i]: continue
                try:
                    osum, orec = score_fn(pair_ids=(ids[i],), predicted_low=pa[i:i+1], predicted_high=pb[i:i+1], target_low=a[i:i+1], target_high=b[i:i+1])
                    single_defined += 1
                    hand=h[i]
                    pairs=[('correct_future_rate',float(np.mean(hand['correct_future']))),
                           ('correct_history_rate',float(np.mean(hand['correct_history'])),),
                           ('joint_icl_pair_success_rate',float(hand['joint_icl_pair_success'])),
                           ('correct_future_mse_mean',hand['correct_future_mse_mean']),
                           ('other_future_mse_mean',hand['other_future_mse_mean'])]
                    pairs += [('latent_response.response_gain',hand['response_gain']),
                              ('latent_response.normalized_response_error',hand['normalized_response_error'])]
                    observed={'correct_future_rate':osum['correct_future_rate'],
                              'correct_history_rate':osum['correct_history_rate'],
                              'joint_icl_pair_success_rate':osum['joint_icl_pair_success_rate'],
                              'correct_future_mse_mean':osum['correct_future_mse_mean'],
                              'other_future_mse_mean':osum['other_future_mse_mean'],
                              'latent_response.response_gain':osum['latent_response']['response_gain'],
                              'latent_response.normalized_response_error':osum['latent_response']['normalized_response_error']}
                    for key,val in pairs:
                        if not math.isclose(float(observed[key]),float(val),rel_tol=ATOL,abs_tol=ATOL):
                            single_mismatches.append({'scene_id':ids[i],'key':key,'official':observed[key],'hand':val})
                except Exception as exc:
                    single_mismatches.append({'scene_id':ids[i],'error':f'{type(exc).__name__}: {exc}'})
            # full-cohort component rates are hand-calculated over all scenes; they
            # remain clearly separate from the official paired score if any zero exists.
            future=[v for row in h for v in row['correct_future']]
            hist=[v for row in h for v in row['correct_history']]
            full_component={
                'correct_future_rate_formula_all_256':float(np.mean(future)),
                'correct_history_rate_formula_all_256':float(np.mean(hist)),
                'rule_switch_rate_formula_all_256':float(np.mean([x['switch'] for x in h])),
                'joint_defined_pair_rate_all_256':float(np.mean([x['joint_icl_pair_success'] is True for x in h])),
                'correct_future_mse_mean_all_256':float(np.mean([x['correct_future_mse_mean'] for x in h])),
                'other_future_mse_mean_all_256':float(np.mean([x['other_future_mse_mean'] for x in h])),
            }
            defined=[x for i,x in enumerate(h) if not zero[i]]
            if defined:
                full_component['paired_response_means_conditional_on_nonzero_separation']={
                    'coverage':len(defined),'response_gain_mean':float(np.mean([x['response_gain'] for x in defined])),
                    'normalized_response_error_mean':float(np.mean([x['normalized_response_error'] for x in defined])),
                    'calibrated_success_rate':float(np.mean([x['calibrated_response_success'] for x in defined]))}
            model_key=f"action_strength/lewm/{model['variant']}/{model['seed']}"
            expectation=expectations()[scenario]
            summary['scenarios'].setdefault(scenario, expectation)
            summary['model_scenarios'][f'{model_key}::{scenario}']={
                'n_scenes':len(ids),'exact_zero_target_separation_count':int(zero.sum()),
                'exact_zero_target_separation_scene_ids':[ids[i] for i in np.flatnonzero(zero)],
                'official_full_cohort_status':full_call_status,
                'official_full_cohort_exception':full_exception,
                'official_full_cohort_summary':official_summary,
                'official_single_pair_calls_defined_count':single_defined,
                'official_single_pair_handcheck_mismatches':single_mismatches[:20],
                'official_single_pair_handcheck_mismatch_count':len(single_mismatches),
                'all_scene_component_formula_summary':full_component,
                'midpoint_tie_diagnostics':{
                    'strict_exact_future_tie_decisions':int(sum(x['exact_future_tie_count'] for x in h)),
                    'near_tie_decisions_abs_mse_diff_le_1e-12':int(sum(x['near_future_tie_count_1e-12'] for x in h)),
                    'total_future_comparisons':2*len(ids)} if scenario=='shared_midpoint_strict_tie' else None,
                'expected_control':expectation
            }
            for i,hand in enumerate(h):
                rows.append({
                    'model_id':model_key,'scene_id':ids[i],'scenario':scenario,
                    'target_separation_mse':sep[i],'target_separation_rms':float(np.sqrt(sep[i])),
                    'zero_target_separation':bool(zero[i]),
                    'official_full_cohort_score_defined':full_call_status=='defined',
                    'official_single_pair_score_defined':bool(not zero[i]),
                    'official_single_pair_vs_handcheck': 'passed' if not zero[i] and not single_mismatches else ('undefined_zero_separation' if zero[i] else 'mismatch'),
                    'correct_future_rate_pair':float(np.mean(hand['correct_future'])),
                    'correct_history_rate_pair':float(np.mean(hand['correct_history'])),
                    'rule_switch_correct':hand['switch'],'joint_pair_success_formula':hand['joint_icl_pair_success'],
                    'correct_future_mse_mean':hand['correct_future_mse_mean'],
                    'other_future_mse_mean':hand['other_future_mse_mean'],
                    'response_gain':hand['response_gain'],'response_alignment':hand['response_alignment'],
                    'normalized_response_error':hand['normalized_response_error'],
                    'calibrated_response_success':hand['calibrated_response_success'],
                    'exact_future_tie_count':hand['exact_future_tie_count'],
                    'near_future_tie_count_1e-12':hand['near_future_tie_count_1e-12']})
    # Expected invariants, including finite-precision midpoint result, are checked
    # against all scene-level formula rows rather than imposing a task gate.
    with csv_path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary['outputs']={'scene_csv':str(csv_path),'scene_csv_rows':len(rows),'summary_json':str(OUT_DIR/'paired_score_controls.json'), 'candidate_index':CANDIDATE_INDEX}
    summary['synthetic_sanity_controls'] = synthetic_sanity(score_fn)
    summary['cross_task_formula_note']={
        'scope':'Static source-location check only; no other benchmark data rerun.',
        'commonality':'The five paired binary scorers below import and call the shared paired_latent_response_metrics helper; its target-separation rejection and response gain/alignment/NRE definition are shared.',
        'files':[
            {'task':'contact_friction','file':'contextworld/benchmarks/contact_friction_icl_score.py','prediction_metrics_line':40,'paired_response_call_line':86,'difference':'task-specific friction/conditioning submetrics around the shared response metric'},
            {'task':'motion_damping','file':'contextworld/benchmarks/motion_damping_icl_score.py','prediction_metrics_line':41,'paired_response_call_line':89,'difference':'task-specific damping submetrics around the shared response metric'},
            {'task':'reacher_arm_mass','file':'contextworld/benchmarks/reacher_arm_mass_icl_score.py','prediction_metrics_line':57,'paired_response_call_line':110,'difference':'mass-specific future strata and paired-bootstrap gate summaries'},
            {'task':'door','file':'contextworld/benchmarks/door_icl_score.py','prediction_metrics_line':None,'paired_response_call_line':95,'difference':'inline nearest-condition decisions plus door-specific summary; same response helper'},
            {'task':'portal_exit','file':'contextworld/benchmarks/portal_exit_icl_score.py','prediction_metrics_line':31,'paired_response_call_line':107,'difference':'exit-specific strata and strict baseline-relative gate; same response helper'}]}
    out=OUT_DIR/'paired_score_controls.json'
    out.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'json':str(out),'csv':str(csv_path),'scenes':256,'csv_rows':len(rows),
                      'model_scenarios':len(summary['model_scenarios']),
                      'zero_counts':{k:v['exact_zero_target_separation_count'] for k,v in summary['model_scenarios'].items()},
                      'official_status':{k:v['official_full_cohort_status'] for k,v in summary['model_scenarios'].items()},
                      'mismatch_counts':{k:v['official_single_pair_handcheck_mismatch_count'] for k,v in summary['model_scenarios'].items()}},indent=2))



def synthetic_sanity(score_fn):
    """Known separated 1-D pair: validate the scorer independently of cached inputs."""
    a=np.array([[0.0]],dtype=np.float64); b=np.array([[1.0]],dtype=np.float64)
    mid=(a/2.0+b/2.0)
    scenarios={
      'oracle_correct_target':(a,b),
      'ignore_history_predict_A':(a,a),
      'ignore_history_predict_B':(b,b),
      'shared_midpoint_strict_tie':(mid,mid),
      'swapped_targets_wrong_history':(b,a),
      'tiny_correct_response_0.001':(mid+0.001*(a-mid),mid+0.001*(b-mid)),
    }
    results={}
    for name,(pa,pb) in scenarios.items():
        actual,_=score_fn(pair_ids=('synthetic-known-separated',),predicted_low=pa,predicted_high=pb,target_low=a,target_high=b)
        hand=hand_one(pa[0],pb[0],a[0],b[0])
        observed={'correct_future_rate':actual['correct_future_rate'],
                  'correct_history_rate':actual['correct_history_rate'],
                  'joint_icl_pair_success_rate':actual['joint_icl_pair_success_rate'],
                  'correct_future_mse_mean':actual['correct_future_mse_mean'],
                  'other_future_mse_mean':actual['other_future_mse_mean'],
                  'response_gain':actual['latent_response']['response_gain'],
                  'normalized_response_error':actual['latent_response']['normalized_response_error']}
        expected_hand={'correct_future_rate':float(np.mean(hand['correct_future'])),
                       'correct_history_rate':float(np.mean(hand['correct_history'])),
                       'joint_icl_pair_success_rate':float(hand['joint_icl_pair_success']),
                       'correct_future_mse_mean':hand['correct_future_mse_mean'],
                       'other_future_mse_mean':hand['other_future_mse_mean'],
                       'response_gain':hand['response_gain'],
                       'normalized_response_error':hand['normalized_response_error']}
        mismatch={k:{'official':observed[k],'hand':expected_hand[k]} for k in observed
                  if not math.isclose(float(observed[k]),float(expected_hand[k]),rel_tol=ATOL,abs_tol=ATOL)}
        results[name]={'target_A':[0.0],'target_B':[1.0],'official_summary':actual,
                       'independent_hand_formula':expected_hand,'mismatches':mismatch,
                       'handcheck_passed':not mismatch,
                       'strict_future_tie_decisions':hand['exact_future_tie_count']}
    results['interpretation']='Synthetic one-dimensional target pair is only a scorer sanity check; it is not part of the cached benchmark evaluation.'
    return results

def expectations():
    return {
      'oracle_correct_target': {'correct_future_rate':1.0,'correct_history_rate':1.0,'joint':1.0,'response_gain':1.0,'NRE':0.0,'matched_MSE':0.0},
      'ignore_history_predict_A': {'correct_future_rate':0.5,'correct_history_rate':0.0,'joint':0.0,'response_gain':0.0,'NRE':1.0},
      'ignore_history_predict_B': {'correct_future_rate':0.5,'correct_history_rate':0.0,'joint':0.0,'response_gain':0.0,'NRE':1.0},
      'shared_midpoint_strict_tie': {'strict_tie_decisions_expected_per_pair':2,'correct_history_rate':0.0,'joint':0.0,'response_gain':0.0,'NRE':1.0},
      'swapped_targets_wrong_history': {'correct_future_rate':0.0,'correct_history_rate':0.0,'joint':0.0,'response_gain':-1.0,'NRE':4.0,'matched_MSE_equals_target_response_MSE':True},
      'tiny_correct_response_0.001': {'correct_future_rate':1.0,'correct_history_rate':1.0,'joint':1.0,'response_gain':0.001,'NRE':0.998001},
    }

if __name__=='__main__': main()
