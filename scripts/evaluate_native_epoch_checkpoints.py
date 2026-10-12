#!/usr/bin/env python3
"""Evaluate a registered PLDM checkpoint at a saved training epoch on native panels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_task_history_readout import helper, sha, write  # noqa: E402
from diagnose_native_policy_contrast import matched_pairs, pair_statistics, aggregate, source_bootstrap  # noqa: E402
from diagnose_matched_encoder_contrast import conditional_fraction  # noqa: E402


def named_state_hashes(model):
    """Hash native state_dict entries grouped by their encoder/projector names."""
    groups = {'encoder': [], 'projector': []}
    for name, tensor in model.state_dict().items():
        for group in groups:
            if name.startswith(group + '.'):
                groups[group].append((name, tensor))
    result = {}
    for group, entries in groups.items():
        assert entries, f'No {group} state entries found'
        digest = hashlib.sha256()
        for name, tensor in sorted(entries):
            raw = tensor.detach().cpu().contiguous().numpy()
            digest.update(name.encode()); digest.update(str(raw.shape).encode())
            digest.update(str(raw.dtype).encode()); digest.update(raw.tobytes())
        result[group] = {'sha256': digest.hexdigest(), 'keys': [name for name, _ in entries]}
    return result


def run(args):
    import torch
    from contextworld.evaluation.protocol import _prepare_optional_flash_attention
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2, user_api='blas')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _prepare_optional_flash_attention()
    candidates = [x for x in json.loads(args.models.read_text())
                  if x['task'] == args.task and x['family'] == 'pldm'
                  and x['regime'] == args.regime and x['training_seed'] == 3072]
    assert len(candidates) == 1, candidates
    registered = candidates[0]
    assert sha(registered['checkpoint']) == registered['checkpoint_sha256']
    epoch_checkpoint = Path(registered['checkpoint']).with_name(f'weights_epoch_{args.epoch}.pt')
    assert epoch_checkpoint.is_file(), epoch_checkpoint
    epoch_sha = sha(epoch_checkpoint)
    spec = dict(registered, checkpoint=str(epoch_checkpoint), checkpoint_sha256=epoch_sha)
    panel = args.panels / args.task
    manifest_path = panel / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    output = args.output / f'epoch{args.epoch}' / args.task / 'pldm' / args.regime
    output.mkdir(parents=True, exist_ok=True)
    adapter = helper().load_adapter(spec, manifest['normalization'], output, args.device)
    adapter.model.eval()
    before = adapter.frozen_state_hash()
    named_before = named_state_hashes(adapter.model)
    if args.refresh_state_hashes:
        result_path = output / 'result.json'
        result = json.loads(result_path.read_text())
        assert result['checkpoint_sha256'] == epoch_sha
        assert result['model_state_hash_before'] == before
        result['named_state_hashes'] = named_before
        write(result_path, result)
        return result
    splits = {}
    for split, expected in (('training', 512), ('development', 256)):
        info = manifest['splits'][split]
        path = panel / info['path']
        assert sha(path) == info['sha256']
        with np.load(path, allow_pickle=False) as raw:
            data = {key: raw[key] for key in raw.files}
        rows, swap = matched_pairs(data)
        assert len(rows) == expected and len(data['labels']) == expected * 2
        assert len(np.unique(data['source_groups'][rows[:, 0]])) == info['source_group_count']
        h, y = helper().encode_unique(adapter, data['history_pixels'], data['queryfuture_pixels'])
        actions = data['rawactions'] if 'action_blocks' not in data else data['action_blocks']
        with torch.inference_mode():
            p = helper().predict_all(adapter, h, y[:, None], actions, 'pldm', modes=('free',))['free'][:, 0]
            w = helper().predict_all(adapter, h[swap], y[:, None], actions, 'pldm', modes=('free',))['free'][:, 0]
        swap_difference = float(np.max(np.abs(w-p[swap])))
        assert swap_difference < 3e-4, swap_difference
        indices = np.array([0, len(h)-1])
        native = adapter.rollout_latents(data['history_pixels'][indices],
                                         data['action_blocks'][indices, :h.shape[1]], batch_size=2)[:, 0]
        rollout_difference = float(np.max(np.abs(native-p[indices])))
        assert rollout_difference < 3e-4, rollout_difference
        stats = pair_statistics(y, p, w, rows)
        ids = data['query_ids'][rows[:, 0]].astype(str)
        groups = data['source_groups'][rows[:, 0]].astype(str)
        stats_path = output / f'{split}_query_stats.npz'
        np.savez_compressed(stats_path, query_ids=ids, source_groups=groups,
                            labels=data['labels'][rows], **stats)
        splits[split] = dict(summary=aggregate(stats), bootstrap=source_bootstrap(stats, groups),
                             target_conditional_fraction=conditional_fraction(y, data['query_ids']),
                             query_stats_sha256=sha(stats_path),
                             max_abs_swap_prediction_difference=swap_difference,
                             max_abs_native_rollout_difference=rollout_difference)
    after = adapter.frozen_state_hash()
    named_after = named_state_hashes(adapter.model)
    assert before == after and named_before == named_after
    result = dict(schema='contextworld.native_epoch_checkpoint.v1', model_id=registered['id'],
                  task=args.task, family='pldm', regime=args.regime, epoch=args.epoch,
                  registered_checkpoint=registered['checkpoint'],
                  registered_checkpoint_sha256=registered['checkpoint_sha256'],
                  checkpoint=str(epoch_checkpoint), checkpoint_sha256=epoch_sha,
                  panel_manifest_sha256=sha(manifest_path),
                  split_sha256={s: manifest['splits'][s]['sha256'] for s in splits},
                  model_state_hash_before=before, model_state_hash_after=after,
                  named_state_hashes=named_before, no_model_update=True, no_test_read=True,
                  metrics=splits,
                  comparison_note='All latent metrics are within-checkpoint normalized; target-space drift is tracked by named state hashes.')
    write(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('contact_friction', 'motion_damping'), required=True)
    parser.add_argument('--regime', choices=('joint', 'frozen'), required=True)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--models', type=Path, default=Path('/tmp/cw-icl-validity-20261007/models.json'))
    parser.add_argument('--panels', type=Path, default=Path('/tmp/cw-cross-task-mechanism-20261010/native'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/cw-native-epoch-20261012'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--refresh-state-hashes', action='store_true',
                        help='Refresh existing receipt hashes without rerunning inference')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
