"""Explicit PreJEPA state-to-RGB/action subspace migration, not function equivalence."""
import re


def adapt_prejepa_state(model, source):
    import torch
    from stable_worldmodel.wm.prejepa.prejepa import PreJEPA
    target = model.state_dict()
    if not isinstance(model, PreJEPA):
        return source, None
    if set(source) == set(target) and all(source[k].shape == target[k].shape for k in target):
        return source, None
    # state_dict iteration preserves ModuleDict concatenation order, unlike sorted keys.
    names = list(dict.fromkeys(k.split('.')[1] for k in source if k.startswith('extra_encoders.')))
    if list(model.extra_encoders) != ['action'] or 'action' not in names:
        raise ValueError('PreJEPA migration requires an action-only target and source action encoder')
    removed = [n for n in names if n != 'action']
    if not removed or any(n not in ('proprio', 'observation') for n in removed):
        raise ValueError('Unsupported PreJEPA source modalities')
    widths = {}
    for n in names:
        w = source[f'extra_encoders.{n}.patch_embed.weight']
        b = source[f'extra_encoders.{n}.patch_embed.bias']
        if w.ndim != 3 or w.shape[-1] != 1 or b.shape != (w.shape[0],):
            raise ValueError('Unsupported modality encoder shape')
        widths[n] = w.shape[0]
    old_dim = source['predictor.pos_embedding'].shape[-1]
    visual_dim = old_dim - sum(widths.values())
    new_dim = target['predictor.pos_embedding'].shape[-1]
    if visual_dim <= 0 or new_dim != visual_dim + widths['action']:
        raise ValueError('PreJEPA source and target visual/action widths differ')
    offset = visual_dim
    keep = list(range(visual_dim))
    for n in names:
        if n == 'action': keep.extend(range(offset, offset + widths[n]))
        offset += widths[n]
    index = torch.tensor(keep, dtype=torch.long)
    dropped = [k for k in source if any(k.startswith(f'extra_encoders.{n}.') for n in removed)]
    if set(source) - set(dropped) != set(target):
        raise ValueError('Unexpected missing/extra keys in PreJEPA migration')
    migrated = {}; transforms = []
    for k, expected in target.items():
        value = source[k]
        if value.shape != expected.shape:
            axis = None
            if k == 'predictor.pos_embedding': axis = 2
            elif re.fullmatch(r'predictor.transformer.(norm|layers.\d+.0.norm|layers.\d+.1.net.0).(weight|bias)', k): axis = 0
            elif re.fullmatch(r'predictor.transformer.layers.\d+.(0.to_qkv|1.net.1).weight', k): axis = 1
            elif re.fullmatch(r'predictor.transformer.layers.\d+.(0.to_out.0|1.net.4).(weight|bias)', k): axis = 0
            if axis is None or value.shape[axis] != old_dim or expected.shape[axis] != new_dim:
                raise ValueError(f'Unsupported PreJEPA mismatch: {k}: {value.shape} -> {expected.shape}')
            value = value.index_select(axis, index.to(value.device))
            transforms.append({'key': k, 'axis': axis, 'source_shape': list(source[k].shape), 'target_shape': list(value.shape)})
        if value.shape != expected.shape:
            raise ValueError(f'PreJEPA migrated shape mismatch: {k}')
        migrated[k] = value
    return migrated, {
        'adapter': 'prejepa_visual_action_subspace_v1',
        'function_equivalent_to_source': False,
        'semantics': 'parameter_subspace_transfer_not_zero_context_equivalence',
        'source_modalities_in_order': names, 'removed_modalities': removed,
        'source_dim': old_dim, 'target_dim': new_dim, 'retained_channels': keep,
        'dropped_keys': dropped, 'transformed_tensors': transforms,
    }
