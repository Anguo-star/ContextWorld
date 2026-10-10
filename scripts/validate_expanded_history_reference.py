#!/usr/bin/env python3
"""CPU Training->Development RGB reference. Hidden mode is target only; no Test reads."""
import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import lance
import numpy as np
import sklearn
import importlib.metadata
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def sha(b):
    """Hash immutable bytes."""
    return hashlib.sha256(b).hexdigest()

def module(n):
    """Load existing exact RGB feature extractors."""
    s = importlib.util.spec_from_file_location(n, REPO / 'scripts' / f'{n}.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

def group(r, task):
    """Return the independent source group for selection and bootstrap."""
    if task == 'cube':
        return str(r['source_episode'])
    pair = str(r['pair_id'])
    if task == 'damping':
        match = re.fullmatch(r'(.*)-(\d+)-(forward|reverse|mirror)', pair)
        if match is None:
            raise ValueError(f'Unrecognized damping source identity: {pair}')
        index = int(match.group(2))
        if 'catalog_index' in r and index != int(np.asarray(r['catalog_index']).reshape(-1)[0]):
            raise ValueError('Damping catalog identity disagrees with pair identity')
        return f'{match.group(1)}-source-{index // 2:06d}'
    return re.sub(r'-a\d+$', '', pair)

def versions(p):
    """Capture Lance version file identities before and after reads."""
    return {str(f.relative_to(p)): [f.stat().st_size, f.stat().st_mtime_ns] for f in p.rglob('*') if f.is_file() and '_versions' in f.parts}

def manifests(p):
    """Hash required release identity files at the release root."""
    root = p.parents[2]
    return {name: sha((root / name).read_bytes()) for name in ['manifest.jsonl', 'manifest.sha256', 'task_registry.json']}

def read(root, split, task, budget, workers):
    """Read metadata first, then only selected permitted history/action rows."""
    path = root / split / 'data.lance'
    before = versions(path)
    d = lance.dataset(path)
    step = 'model_step_idx' if task == 'cube' else 'step_idx'
    action = 'action_block' if task == 'cube' else 'action'
    cols = ['episode_idx', step, 'pair_id', 'hidden_mode', 'split'] + (['source_episode', 'action_profile_id', 'scene_template_content_hash', 'pair_content_hash'] if task == 'cube' else ['catalog_index'])
    meta = d.to_table(filter=f'{step} = 0', columns=cols).to_pylist()
    groups = defaultdict(list)
    for r in meta:
        groups[group(r, task)].append(r)
    selected = []
    count = 0
    for k in sorted(groups, key=lambda k: sha(k.encode())):
        if split == 'training' and budget and (count >= budget):
            break
        selected.extend(groups[k])
        count += len({r['pair_id'] for r in groups[k]})
    ids = sorted({int(r['episode_idx']) for r in selected})
    rows = defaultdict(dict)
    for j in range(0, len(ids), 256):
        sel = ','.join(map(str, ids[j:j + 256]))
        pos = '0,1,2,3' if task == 'cube' else ','.join(map(str, range(16)))
        for r in d.to_table(filter=f'episode_idx IN ({sel}) AND {step} IN ({pos})', columns=['episode_idx', step, action]).to_pylist():
            rows[int(r['episode_idx'])][int(r[step])] = r
        pixelpos = '0,1,2,3' if task == 'cube' else '0,5,10,15'
        for r in d.to_table(filter=f'episode_idx IN ({sel}) AND {step} IN ({pixelpos})', columns=['episode_idx', step, 'pixels']).to_pylist():
            rows[int(r['episode_idx'])][int(r[step])]['pixels'] = r['pixels']
    m = module('audit_pusht_contact_friction_rgb_identifiability') if task == 'friction' else module('audit_pusht_motion_damping_history_identifiability') if task == 'damping' else None

    def extract(r):
        rr = rows[int(r['episode_idx'])]
        hs = [0, 1, 2] if task == 'cube' else [0, 5, 10]
        pixels = [bytes(rr[i]['pixels']) for i in hs]
        future = bytes(rr[3 if task == 'cube' else 15]['pixels'])
        acts = np.asarray([rr[i][action] for i in (hs if task == 'cube' else range(15))], dtype=np.float64).reshape(-1)
        if task == 'friction':
            (x0, x1, x2) = [m._visible_geometry(b) for b in pixels]
            feat = np.concatenate([x0, x1 - x0, x2 - x1, acts])
        elif task == 'damping':
            (x0, x1, x2) = [m._block_centroid_from_rgb(b)[0] for b in pixels]
            den = np.linalg.norm(x1 - x0)
            if den <= 0:
                raise ValueError('zero RGB displacement')
            feat = np.array([np.linalg.norm(x2 - x1) / den])
        else:
            ims = []
            for b in pixels:
                with Image.open(BytesIO(b)) as im:
                    ims.append(np.asarray(im.convert('RGB').resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float64))
            feat = (2 * ims[1] - ims[0] - ims[2]).reshape(-1)
        return dict(pair=str(r['pair_id']), mode=str(r['hidden_mode']), group=group(r, task), feature=feat, actions=acts, pixels=pixels, future=future)
    out = []
    fail = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(extract, r) for r in selected]
        for (r, j) in zip(selected, jobs):
            try:
                out.append(j.result())
            except Exception as e:
                fail.append({'episode_idx': r['episode_idx'], 'pair': r['pair_id'], 'error': str(e)})
    h = hashlib.sha256()
    for r in sorted(out, key=lambda r: (r['pair'], r['mode'])):
        h.update(r['pair'].encode())
        h.update(r['mode'].encode())
        for b in r['pixels']:
            h.update(b)
        h.update(r['actions'].tobytes())
        h.update(r['feature'].tobytes())
    after = versions(path)
    if before != after:
        raise RuntimeError('Lance versions changed')
    return (out, dict(path=str(path.relative_to(root.parents[2])), lance_version=d.version, metadata_rows=d.count_rows(), metadata_pairs=len({r['pair_id'] for r in meta}), selected_conditions=len(selected), versions_before=before, versions_after=after, metadata_conditions=len(meta), isolation_values={k: sorted({str(r[k]) for r in meta}) for k in ['source_episode', 'action_profile_id', 'scene_template_content_hash', 'pair_content_hash']} if task == 'cube' else {}, metadata_source_groups=sorted(groups), metadata_group_count=len(groups), selected_group_count=len({group(r, task) for r in selected}), selected_pairs=count, selection_ids_sha256=sha(json.dumps(ids).encode()), selected_history_feature_and_label_sha256=h.hexdigest(), mask_or_extraction_failure_count=len(fail), mask_or_extraction_failures=fail))

def rgb_equal(left, right):
    """Compare decoded RGB, avoiding JPEG encoding differences."""
    with Image.open(BytesIO(left)) as image:
        a = np.asarray(image.convert('RGB'))
    with Image.open(BytesIO(right)) as image:
        b = np.asarray(image.convert('RGB'))
    return np.array_equal(a, b)

def fit_threshold(values, labels):
    """Choose direction and minimum-error threshold from Training only."""
    values = np.asarray(values)
    labels = np.asarray(labels)
    unique = np.unique(values)
    thresholds = np.r_[np.nextafter(unique[0], -np.inf), (unique[:-1] + unique[1:]) / 2, np.nextafter(unique[-1], np.inf)]
    order = np.argsort(values)
    ones = np.r_[0, np.cumsum(labels[order])]
    indices = np.searchsorted(values[order], thresholds, side='right')
    errors = ones[indices] + (len(labels) - indices) - (ones[-1] - ones[indices])
    best = int(np.argmin(np.r_[errors, len(labels) - errors]))
    overlap = max(values[labels == 0].min(), values[labels == 1].min()) <= min(values[labels == 0].max(), values[labels == 1].max())
    return (float(thresholds[best % len(thresholds)]), int(best >= len(thresholds)), bool(overlap))

def recipe(task):
    """Describe the fixed readout; Development never selects a recipe."""
    return dict(history_frames=[0, 1, 2] if task == 'cube' else [0, 5, 10], future_integrity_frame=3 if task == 'cube' else 15, action_tokens=3, action_scalars=75 if task == 'cube' else 30, action_features_used=task == 'friction', estimator={'friction': 'Training-fit StandardScaler + ExtraTreesClassifier(256, random_state=20261009)', 'damping': 'Training-only minimum-error single threshold; both directions', 'cube': '16x16 bilinear decoded RGB flatten(2*x1-x0-x2); Training-fit StandardScaler + RidgeClassifier(alpha=1)'}[task])

def controls(data):
    """Measure paired decoded input equality and deterministic accuracy ceilings."""
    pairs = defaultdict(list)
    for r in data:
        pairs[r['pair']].append(r)
    sums = dict(x0_equal=0, current_only_equal=0, action_only_equal=0, future_different=0)
    for (p, rs) in pairs.items():
        if len(rs) != 2 or rs[0]['mode'] == rs[1]['mode']:
            raise ValueError(f'incomplete binary pair {p}')
        (a, b) = rs
        for (k, v) in [('x0_equal', rgb_equal(a['pixels'][0], b['pixels'][0])), ('current_only_equal', rgb_equal(a['pixels'][2], b['pixels'][2])), ('action_only_equal', np.array_equal(a['actions'], b['actions'])), ('future_different', not rgb_equal(a['future'], b['future']))]:
            sums[k] += int(v)
    sums['current_plus_action_equal'] = sum((int(rgb_equal(rs[0]['pixels'][2], rs[1]['pixels'][2]) and np.array_equal(rs[0]['actions'], rs[1]['actions'])) for rs in pairs.values()))
    n = len(pairs)
    result = {k: dict(count=v, pairs=n, fraction=v / n) for (k, v) in sums.items()}
    for k in ['current_only_equal', 'action_only_equal', 'current_plus_action_equal']:
        result[k]['deterministic_classifier_theoretical_accuracy_upper_bound_not_trained_score'] = 1 - 0.5 * sums[k] / n
    return result

def score(data, pred, modes):
    """Report overall and worst-condition readout accuracy."""
    y = np.array([modes.index(r['mode']) for r in data])
    ok = pred == y
    per = {m: float(ok[y == i].mean()) for (i, m) in enumerate(modes)}
    return dict(accuracy=float(ok.mean()), condition_accuracy=per, worst_condition_accuracy=min(per.values()))

def main():
    """Run one fixed CPU readout on Training and Development."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['friction', 'damping', 'cube'], required=True)
    p.add_argument('--bundle-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-pairs', type=int, default=4096)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.train_pairs < 1 or a.workers < 1:
        p.error('--train-pairs and --workers must be positive')
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    mb = manifests(a.bundle_root)
    (train, tr) = read(a.bundle_root, 'training', a.task, a.train_pairs, a.workers)
    (dev, dr) = read(a.bundle_root, 'development', a.task, None, a.workers)
    result = dict(version='expanded_history_reference_v1', task=a.task, release=a.bundle_root.parents[2].name, program_sha256=sha(Path(__file__).read_bytes()), versions={'numpy': np.__version__, 'sklearn': sklearn.__version__, 'lance': importlib.metadata.version('pylance')}, sampling={'train_pair_target': a.train_pairs, 'workers': a.workers, 'selection': 'SHA256 source-group order, retain full last group', 'bootstrap_resamples': 4000, 'seed': 20261009}, recipe=recipe(a.task), training=tr, development=dr, manifest_sha256_before=mb, input_contract={'allowed': 'history pixels/actions only', 'hidden_mode': 'supervised target only', 'future_pixels': 'equality check only', 'Test_read': False, 'failure_interpretation': 'decoder failure does not establish unidentifiability; fixed recipe is not retried based on Development', 'friction_action_scalars': 30, 'bootstrap_unit': 'source group; mirror pairs kept together'})
    if tr['mask_or_extraction_failures'] or dr['mask_or_extraction_failures']:
        result['status'] = 'failed_extraction_no_rows_silently_dropped'
    else:
        result['controls'] = {'training': controls(train), 'development': controls(dev)}
        modes = sorted({r['mode'] for r in train})
        assert len(modes) == 2 and set(modes) == {r['mode'] for r in dev}
        x = np.stack([r['feature'] for r in train])
        z = np.stack([r['feature'] for r in dev])
        y = np.array([modes.index(r['mode']) for r in train])
        if a.task == 'damping':
            (threshold, flip, overlap) = fit_threshold(x[:, 0], y)
            predict = lambda q: (q[:, 0] > threshold).astype(int) ^ flip
            result['estimator'] = dict(threshold=threshold, flip=flip, training_classes_overlap=overlap)
        else:
            scaler = StandardScaler().fit(x)
            model = ExtraTreesClassifier(n_estimators=256, random_state=20261009, n_jobs=a.workers) if a.task == 'friction' else RidgeClassifier(alpha=1)
            model.fit(scaler.transform(x), y)
            predict = lambda q: model.predict(scaler.transform(q))
            result['estimator'] = {'name': type(model).__name__, 'parameters': model.get_params(deep=False), 'scaler': scaler.get_params(deep=False)}
        tp = predict(x)
        dp = predict(z)
        result['training_scores'] = score(train, tp, modes)
        result['development_scores'] = score(dev, dp, modes)
        result['development_predictions'] = [dict(pair_id=r['pair'], source_group=r['group'], mode=r['mode'], prediction=modes[int(v)], correct=bool(r['mode'] == modes[int(v)])) for (r, v) in zip(dev, dp)]
        by = defaultdict(list)
        for r in result['development_predictions']:
            by[r['source_group']].append(r['correct'])
        keys = sorted(by)
        tot = np.array([len(by[k]) for k in keys])
        cor = np.array([sum(by[k]) for k in keys])
        rng = np.random.default_rng(20261009)
        draw = rng.integers(len(keys), size=(4000, len(keys)))
        boot = cor[draw].sum(1) / tot[draw].sum(1)
        result['development_group_bootstrap_95ci'] = dict(resamples=4000, seed=20261009, groups=len(keys), lower=float(np.quantile(boot, 0.025)), upper=float(np.quantile(boot, 0.975)))
        result['status'] = 'completed_empirical_reference_not_mathematical_identifiability'
    if a.task == 'cube':
        result['full_train_development_metadata_overlap'] = {}
        for key in tr['isolation_values']:
            overlap = sorted(set(tr['isolation_values'][key]) & set(dr['isolation_values'][key]))
            result['full_train_development_metadata_overlap'][key] = {'count': len(overlap), 'values': overlap}
    for split_result in [tr, dr]:
        source_groups = split_result.pop('metadata_source_groups')
        split_result['metadata_source_groups_sha256'] = sha(json.dumps(source_groups).encode())
        isolation = split_result.pop('isolation_values')
        split_result['isolation_metadata'] = {key: {'distinct_count': len(values), 'sha256': sha(json.dumps(values).encode())} for key, values in isolation.items()}
    result['manifest_sha256_after'] = manifests(a.bundle_root)
    if mb != result['manifest_sha256_after']:
        raise RuntimeError('manifest changed')
    with a.output.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
if __name__ == '__main__':
    main()
