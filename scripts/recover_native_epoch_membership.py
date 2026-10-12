#!/usr/bin/env python3
"""Recover fit membership of native PLDM query windows from the training recipe."""
import hashlib
import json
import argparse
import importlib.util
from pathlib import Path

import h5py
import lance
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Recover PLDM native panel loader fit membership.')
parser.add_argument('--models', type=Path, required=True)
parser.add_argument('--panels', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
PANEL = args.panels
models = json.loads(args.models.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def mirror(path, data_root):
    path = Path(path)
    if path.exists():
        return path
    marker = '/dataset/ag_data/'
    assert marker in str(path)
    candidate = data_root.parents[2] / str(path).split(marker, 1)[1]
    assert candidate.exists(), candidate
    return candidate

rows = {}
for task in ('contact_friction', 'motion_damping'):
    rows[task] = {}
    for regime in ('joint', 'frozen'):
        matches = [m for m in models if m['task'] == task and m['family'] == 'pldm'
                   and m['regime'] == regime and m['training_seed'] == 3072]
        assert len(matches) == 1
        rows[task][regime] = matches[0]

joint_first = rows['contact_friction']['joint']
first_config = json.loads((Path(joint_first['checkpoint']).parent/'config.json').read_text())
first_identity = json.loads((Path(joint_first['checkpoint']).parent/'contextworld_training_identity_v1.json').read_text())['identity']
first_contract = first_identity['target']['dataset']['contract']
first_root = mirror(first_contract['root'], Path(json.loads((PANEL/'contact_friction'/'manifest.json').read_text())['root']))
ORIGINAL = mirror(first_contract['original_dataset'], first_root)

with h5py.File(ORIGINAL) as f:
    original_lengths = np.asarray(f['ep_len'], dtype=np.int64)
original_clips = int(np.maximum(original_lengths - 20 + 1, 0).sum())

for task in ('contact_friction', 'motion_damping'):
    joint = rows[task]['joint']
    frozen = rows[task]['frozen']
    run = Path(joint['checkpoint']).parent
    cfg = json.loads((run / 'config.json').read_text())
    identity = json.loads((run / 'contextworld_training_identity_v1.json').read_text())['identity']
    contract = identity['target']['dataset']['contract']
    assert cfg['seed'] == identity['seed'] == 3072
    assert cfg['train_split'] == .9
    assert cfg['data']['dataset']['num_steps'] == 4 and cfg['data']['dataset']['frameskip'] == 5
    assert contract['conditional_joint'] is None and contract['weights'] == {'original': .5, 'synthetic': .5}
    frozen_run = Path(frozen['checkpoint']).parent
    frozen_cfg = json.loads((frozen_run/'config.json').read_text())
    frozen_identity = json.loads((frozen_run/'contextworld_training_identity_v1.json').read_text())['identity']
    frozen_contract = frozen_identity['target']['dataset']['contract']
    assert frozen_cfg['seed'] == cfg['seed'] and frozen_cfg['train_split'] == cfg['train_split']
    assert frozen_cfg['data']['dataset']['name'] == cfg['data']['dataset']['name']
    assert frozen_contract == contract
    assert sha(Path(joint['checkpoint'])) == joint['checkpoint_sha256']
    assert sha(Path(frozen['checkpoint'])) == frozen['checkpoint_sha256']
    root = mirror(contract['root'], Path(json.loads((PANEL/task/'manifest.json').read_text())['root']))
    registry = root / 'task_registry.json'
    registry_hash = hashlib.sha256(registry.read_bytes()).hexdigest()
    assert registry_hash == contract['task_registry_sha256']
    manifest = root/'manifest.jsonl'
    assert sha(manifest) == contract['manifest_sha256']
    assert (root/'manifest.sha256').read_text().split()[0] == sha(manifest)
    panel_meta = json.loads((PANEL / task / 'manifest.json').read_text())['splits']['training']
    assert sha(Path(panel_meta['path'])) == panel_meta['sha256']
    registry_data = json.loads(registry.read_text())
    entry = next(x for x in registry_data['components'] if x['dataset_id'] == contract['dataset_id'])
    payload = next(x for x in entry['payloads'] if x['split'] == 'training' and x['payload_id'] == contract['payload_id'])
    assert len(payload['members']) == 1
    lance_path = root / payload['members'][0]
    assert str(lance_path) == panel_meta['lance_path']
    ds = lance.dataset(lance_path)
    assert ds.version == panel_meta['lance_version']
    table = ds.to_table(columns=['episode_idx', 'step_idx'])
    ep = np.asarray(table.column('episode_idx').to_pylist(), dtype=np.int64)
    step = np.asarray(table.column('step_idx').to_pylist(), dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(ep[1:] != ep[:-1]) + 1]
    ends = np.r_[starts[1:], len(ep)]
    lengths = ends - starts
    assert len(np.unique(ep[starts])) == len(starts)
    assert np.array_equal(step[starts], np.zeros(len(starts), dtype=np.int64))
    assert np.array_equal(step[ends-1], lengths-1)
    clips = np.maximum(lengths - 20 + 1, 0)
    first = np.r_[0, np.cumsum(clips[:-1])]
    synthetic_clips = int(clips.sum())
    epoch_size = 2 * max(original_clips, synthetic_clips)
    # Cross-check the closed-form 50/50 mapping against the group loader.
    repo = Path(__file__).resolve().parents[1]
    groups_path = repo/'contextworld/training/groups.py'
    spec = importlib.util.spec_from_file_location('pinned_groups', groups_path)
    groups = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(groups)
    class LengthOnly:
        def __init__(self, n): self.n = n
        def __len__(self): return self.n
    mixture = groups.LogicalGroupDataset({'original': LengthOnly(original_clips),
                                          'synthetic': LengthOnly(synthetic_clips)},
                                         {'original': .5, 'synthetic': .5}, epoch_size=epoch_size)
    assert mixture.schedule == [0, 1]
    for index in (0, 1, 2, 3, epoch_size//2, epoch_size-2, epoch_size-1):
        expected = ('original', (index//2)%original_clips) if index%2 == 0 else ('synthetic', (index//2)%synthetic_clips)
        assert mixture.locate(index) == expected
    fit, validation = torch.utils.data.random_split(
        range(epoch_size), [.9, .1], generator=torch.Generator().manual_seed(3072)
    )
    fit_indices = np.asarray(fit.indices, dtype=np.int64)
    fit_lookup = np.zeros(epoch_size, dtype=bool)
    fit_lookup[fit_indices] = True
    z = np.load(PANEL / task / 'training.npz')
    row_ep = np.asarray(z['episode_ids'], dtype=np.int64)
    query_ids = np.asarray(z['query_ids'])
    assert len(row_ep) == 1024 and len(query_ids) == 1024
    episode_to_pos = {int(value): i for i, value in enumerate(ep[starts])}
    positions = np.asarray([episode_to_pos[int(value)] for value in row_ep])
    assert np.all(clips[positions] > 0)
    local_clip = first[positions]
    # Equal weights, order original then synthetic: schedule [0,1],
    # synthetic local clip = (global_index//2) % synthetic_clips.
    occurrence_count = (epoch_size//2 - 1 - local_clip)//synthetic_clips + 1
    row_indices = [
        (2 * (int(idx) + np.arange(int(count), dtype=np.int64) * synthetic_clips) + 1)
        for idx, count in zip(local_clip, occurrence_count)
    ]
    row_fit = np.asarray([bool(fit_lookup[x].any()) for x in row_indices])
    row_validation = np.asarray([bool((~fit_lookup[x]).any()) for x in row_indices])
    assert len(set(query_ids.tolist())) == 512
    assert np.array_equal(query_ids[0::2], query_ids[1::2])
    both_fit = row_fit[0::2] & row_fit[1::2]
    both_validation = row_validation[0::2] & row_validation[1::2]
    fit_only = both_fit & ~np.any(row_validation.reshape(-1, 2), axis=1)
    validation_only = both_validation & ~np.any(row_fit.reshape(-1, 2), axis=1)
    mixed = ~(both_fit | both_validation)
    overlap = both_fit & both_validation
    pair_rows = np.arange(len(row_ep), dtype=np.int64).reshape(-1, 2)
    np.savez_compressed(
        OUT / f'{task}.npz', query_ids=query_ids[0::2], pair_query_ids=query_ids[0::2],
        pair_rows=pair_rows, episode_ids=row_ep,
        row_fit=row_fit, row_validation=row_validation,
        both_fit=both_fit, mixed=mixed, both_validation=both_validation,
        fit_only=fit_only, validation_only=validation_only, overlap=overlap,
        synthetic_clip_index=local_clip,
        synthetic_global_index=np.asarray([x[0] for x in row_indices]),
    )
    output = {
        'task': task, 'checkpoint': str(joint['checkpoint']),
        'checkpoint_sha256': joint['checkpoint_sha256'],
        'frozen_checkpoint': str(frozen['checkpoint']),
        'frozen_checkpoint_sha256': frozen['checkpoint_sha256'],
        'identity_path': str(run/'contextworld_training_identity_v1.json'),
        'config_path': str(run/'config.json'),
        'training_dataset_manifest_sha256': contract['manifest_sha256'],
        'task_registry': str(registry), 'task_registry_sha256': registry_hash,
        'original_dataset': str(ORIGINAL), 'original_clips': original_clips,
        'synthetic_dataset': str(lance_path), 'synthetic_lance_version': ds.version,
        'synthetic_clips': synthetic_clips, 'num_steps': 4, 'frameskip': 5,
        'clip_span': 20, 'epoch_size': epoch_size, 'seed': 3072,
        'train_split': .9, 'fit_size': len(fit), 'validation_size': len(validation),
        'mixture_schedule': ['original', 'synthetic'],
        'row_count': len(row_ep), 'pair_count': len(query_ids)//2,
        'both_fit_count': int(both_fit.sum()), 'mixed_count': int(mixed.sum()),
        'both_validation_count': int(both_validation.sum()),
        'fit_only_count': int(fit_only.sum()),
        'validation_only_count': int(validation_only.sum()),
        'fit_validation_overlap_count': int(overlap.sum()),
        'membership_semantics': 'loader split; per-minibatch consumption is not logged',
        'panel_manifest': str(PANEL/task/'manifest.json'),
        'panel_npz': str(PANEL/task/'training.npz'),
        'panel_npz_sha256': panel_meta['sha256'],
        'manifest_path': str(manifest),
        'manifest_sha256': sha(manifest),
        'source_files': {str(groups_path): sha(groups_path),
                         str(repo/'contextworld/training/stablewm_bundle.py'): sha(repo/'contextworld/training/stablewm_bundle.py'),
                         str(Path(joint['stable_repo'])/'scripts/train/pldm.py'): sha(Path(joint['stable_repo'])/'scripts/train/pldm.py')},
        'masks_npz': str(OUT/f'{task}.npz'),
    }
    (OUT / f'{task}.json').write_text(json.dumps(output, indent=2)+'\n')
    print(task, output['both_fit_count'], output['mixed_count'], output['both_validation_count'], flush=True)
