#!/usr/bin/env python3
"""Build and verify the local nine-task expanded-Training HF release candidate."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import prepare_contextworld_hf_release as v3

REPO = Path(__file__).resolve().parents[1]
TASKS = ('speed', 'door', 'action_delay', 'action_strength', 'contact_friction',
         'motion_damping', 'robot_arm_mass', 'cube_gripper_carry', 'portal_exit')
BASE = 'ContextWorld-v1'
TEST = 'ContextWorld-v1-full'
VERSION = 'contextworld-expanded-training-v1-hf-rc2'
SPEED_TEST_SOURCE = 'context_world_speed_multistep_extrap_v5'
SPEED_TEST_REL = 'artifacts/evaluation/history3/speed_multistep_extrap_v5'
SPEED_TEST_SOURCE_REL = 'context_world/evaluation/history3/speed_multistep_extrap_v5'


def speed_test_payload_rows(dataset_root, frozen_test):
    """Close the frozen Speed catalogs over their SHA-pinned NPZ payloads."""
    prefix = SPEED_TEST_REL + '/payloads/'
    source_root = dataset_root / SPEED_TEST_SOURCE_REL
    frozen_rows = {row['path']: row for row in frozen_test['rows']}
    seen = set()
    selected = []
    for track in ('seen_for_multi', 'unseen_interpolation', 'extrapolation_low',
                  'extrapolation_high'):
        catalog_rel = f'{SPEED_TEST_REL}/catalogs/{track}.json'
        catalog_row = frozen_rows.get(catalog_rel)
        if not catalog_row or catalog_row.get('role') != 'dataset_payload':
            raise ValueError(f'Missing frozen Speed catalog: {catalog_rel}')
        catalog_path = frozen_test['root'] / catalog_rel
        if v3.sha256_file(catalog_path) != (catalog_row['bytes'], catalog_row['sha256']):
            raise ValueError(f'Frozen Speed catalog hash differs: {catalog_rel}')
        catalog = json.loads(catalog_path.read_text())
        if catalog.get('track') != track or not catalog.get('bundles'):
            raise ValueError(f'Invalid Speed catalog track: {catalog_rel}')
        for entry in catalog['bundles']:
            rel = entry.get('payload')
            digest = entry.get('payload_sha256')
            if (not v3.is_safe_rel(rel) or not rel.startswith(prefix)
                    or not isinstance(digest, str) or len(digest) != 64
                    or rel in seen):
                raise ValueError(f'Unsafe, duplicate, or unpinned Speed payload: {rel}')
            seen.add(rel)
            source_rel = rel[len(SPEED_TEST_REL) + 1:]
            source = v3.regular_file(source_root, source_rel)
            size, observed = v3.sha256_file(source)
            if observed != digest:
                raise ValueError(f'Speed catalog payload hash differs: {rel}')
            selected.append({'path': rel, 'bytes': size, 'sha256': digest,
                             'role': 'dataset_payload', 'component': 'speed', 'split': 'test',
                             'source_bundle': SPEED_TEST_SOURCE, 'source_path': source_rel,
                             'source_sha256': digest})
    actual = {f'{SPEED_TEST_REL}/{p.relative_to(source_root).as_posix()}'
              for p in (source_root / 'payloads').rglob('*.npz') if p.is_file()}
    if actual != seen:
        raise ValueError(f'Speed source/catalog payload mismatch: {len(actual ^ seen)} paths')
    return selected


def inventory(root, name, pin=None):
    base = root / name
    data = (base / 'manifest.jsonl').read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if pin and digest != pin:
        raise ValueError(f'{name}: manifest pin mismatch')
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError(f'{name}: duplicate manifest paths')
    registry_bytes = (base / 'task_registry.json').read_bytes()
    return {'root': base, 'rows': rows, 'manifest_sha256': digest,
            'registry_sha256': hashlib.sha256(registry_bytes).hexdigest(),
            'registry': json.loads(registry_bytes)}


def plan(root):
    study = json.loads((REPO / 'docs/research/data/icl_training_study_v2.json').read_text())
    expanded = {}
    for row in study['rows']:
        if (row.get('training_data_version') and row['regime'] != 'original'
                and not row.get('comparison_variant') and row['training_data_version'] != BASE):
            pair = (row['training_data_version'], row['training_dataset_manifest_sha256'])
            if row['task'] in expanded and expanded[row['task']] != pair:
                raise ValueError(f'{row["task"]}: inconsistent study Training identity')
            expanded[row['task']] = pair
    bundles = {name: inventory(root, name, pin) for name, pin in
               [(BASE, v3.PINS['dev_manifest']), (TEST, v3.PINS['test_manifest'])]}
    for name, pin in expanded.values():
        bundles[name] = inventory(root, name, pin)
    if bundles[BASE]['registry_sha256'] != v3.PINS['dev_registry'] or bundles[TEST]['registry_sha256'] != v3.PINS['test_registry']:
        raise ValueError('Frozen registry pin mismatch')
    components = {c['component_id']: json.loads(json.dumps(c)) for c in bundles[BASE]['registry']['components']}
    tests = {c['component_id']: c for c in bundles[TEST]['registry']['components']}
    selected = []
    for task in TASKS:
        train_name = expanded.get(task, (BASE, None))[0]
        train_comp = next(c for c in bundles[train_name]['registry']['components'] if c['component_id'] == task)
        comp = components[task]
        if train_comp['release_config_sha256'] != comp['release_config_sha256']:
            raise ValueError(f'{task}: release config differs')
        comp['payloads'] = ([json.loads(json.dumps(p)) for p in train_comp['payloads'] if p['split'] == 'training']
                            + [p for p in comp['payloads'] if p['split'] == 'development']
                            + [p for p in tests[task]['payloads'] if p['split'] == 'test'])
        comp['public_test_evaluation'] = tests[task]['public_test_evaluation']
        for split, name in [('training', train_name), ('development', BASE), ('test', TEST)]:
            rows = [r for r in bundles[name]['rows'] if r['role'] == 'dataset_payload'
                    and r.get('component') == task and r.get('split') == split]
            payloads = [p for p in comp['payloads'] if p['split'] == split]
            if not rows or sum(p['file_count'] for p in payloads) != len(rows):
                raise ValueError(f'{task}/{split}: registry file coverage mismatch')
            for row in rows:
                hits = [p for p in payloads if row['path'].startswith(p['public_path'] + '/')]
                if len(hits) != 1:
                    raise ValueError(f'{task}/{split}: unregistered payload path {row["path"]}')
                selected.append({**row, 'source_bundle': name, 'source_path': row['path'],
                                 'source_sha256': row['sha256']})
    speed_extra = speed_test_payload_rows(root, bundles[TEST])
    selected.extend(speed_extra)
    speed_test_payload = next(p for p in components['speed']['payloads'] if p['split'] == 'test')
    speed_test_payload['file_count'] += len(speed_extra)
    speed_test_payload['total_bytes'] += sum(row['bytes'] for row in speed_extra)
    if len({r['path'] for r in selected}) != len(selected):
        raise ValueError('Destination paths collide')
    merged = json.loads(json.dumps(bundles[BASE]['registry']))
    merged['components'] = [components[c['component_id']] for c in merged['components']]
    merged['public_test'] = bundles[TEST]['registry']['public_test']
    merged['export_id'] = VERSION
    merged['release_status'] = 'staging_not_public_release'
    return bundles, selected, merged, expanded


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def copy_verified(src, dst, row):
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with src.open('rb') as fin, dst.open('xb') as fout:
        while chunk := fin.read(4 << 20):
            fout.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    if (total, digest.hexdigest()) != (row['bytes'], row['sha256']):
        raise ValueError(f'Payload hash/size mismatch: {src}')
    return row


def package_file_paths(output):
    """List package files, excluding only the two root manifest self-references."""
    excluded = {output / '.build_in_progress', output / 'manifest.jsonl',
                output / 'manifest.sha256'}
    return [path for path in output.rglob('*') if path.is_file() and path not in excluded]


def render_card(expanded):
    card = (REPO / 'docs/templates/ContextWorld_Expanded_Training_Card.md').read_text()
    card = card.replace(
        '**Unpublished preparation draft.** This card describes a proposed nine-task package. Local package validation and the final release revision are pending.',
        f'**Unpublished release candidate `{VERSION}`.** This package contains nine tasks and passed local manifest and payload loading checks.')
    card = card.replace(
        'Final download and loader commands will be added after the assembled package is validated and a release revision is assigned.',
        'Open each Lance table at the paths in `task_registry.json` with `lance.dataset(path)`. Collection payloads list individual table paths in `members`.')
    card += '\n## Package inventory\n\n' + '\n'.join(
        f'- `{task}`: Training `{expanded.get(task, (BASE, None))[0]}`; Development `{BASE}`; Test `{TEST}`.'
        for task in sorted(TASKS)) + ('\n\nThe Speed Test directory also includes every NPZ payload referenced by its four frozen catalogs. '
                                      'The inherited `registry_contract.json` files describe Development only.\n\n'
                                      'Local manifest and payload loading checks do not establish a self-contained Test runtime. '
                                      'The Motion Damping Test still depends on historical release source identity and runtime artifacts under separate review.\n')
    return card


def render_version(bundles, expanded):
    return {'schema_version': 1, 'dataset_version': VERSION,
            'representation': 'native_lance_snapshot', 'release_status': 'staging_not_public_release',
            'public_test_included': True, 'evaluation_source_policy': 'frozen_development_and_test',
            'source_bundles': {name: {'manifest_sha256': bundle['manifest_sha256'],
                                     'task_registry_sha256': bundle['registry_sha256']}
                               for name, bundle in sorted(bundles.items())},
            'supplemental_sources': {SPEED_TEST_SOURCE: {
                'source_relative_root': SPEED_TEST_SOURCE_REL,
                'identity': 'payload_sha256 in the four frozen Speed Test catalogs'}},
            'training_sources': {task: expanded.get(task, (BASE, None))[0] for task in TASKS},
            'baseline_id': v3.BASELINE_ID, 'baseline_sha256': v3.PINS['baseline']}


def render_inventory(selected):
    by_split = Counter((row['component'], row['split']) for row in selected)
    return {'schema_version': 'contextworld.inventory.v1', 'dataset_version': VERSION,
            'totals': {'payload_files': len(selected),
                       'payload_bytes': sum(row['bytes'] for row in selected)},
            'by_component_split_files': {f'{component}|{split}': count
                                         for (component, split), count in sorted(by_split.items())}}


def execute(root, output, workers):
    bundles, selected, registry, expanded = plan(root)
    if output.exists() or output.is_symlink():
        raise ValueError(f'Output exists: {output}')
    output.mkdir(parents=True)
    marker = output / '.build_in_progress'
    marker.write_text('candidate incomplete\n')
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda r: copy_verified((root / SPEED_TEST_SOURCE_REL / r['source_path'])
                                                 if r['source_bundle'] == SPEED_TEST_SOURCE else
                                                 (bundles[r['source_bundle']]['root'] / r['source_path']),
                                                 output / r['path'], r), selected))
        write_json(output / 'task_registry.json', registry)
        write_json(output / 'VERSION.json', render_version(bundles, expanded))
        write_json(output / 'inventory.json', render_inventory(selected))
        for name, b in bundles.items():
            for filename in ('manifest.jsonl', 'manifest.sha256', 'task_registry.json'):
                dst = output / 'provenance' / name / filename
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(b['root'] / filename, dst)
        base_rows = {r['path']: r for r in bundles[BASE]['rows']}
        refs = set(v3.NORMALIZER_RE.findall(json.dumps(registry)))
        refs.update(r['path'] for r in bundles[BASE]['rows'] if r['role'] == 'release_metadata'
                    and r['path'].endswith('/development/registry_contract.json'))
        for rel in sorted(refs):
            row = base_rows.get(rel)
            if not row or row['role'] != 'release_metadata':
                raise ValueError(f'Missing normalizer/contract metadata: {rel}')
            copy_verified(bundles[BASE]['root'] / rel, output / rel, row)
        ref = REPO / v3.BASELINE_REL
        dst = output / 'reference' / ref.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref, dst)
        if v3.sha256_file(dst)[1] != v3.PINS['baseline']:
            raise ValueError('Reference baseline pin mismatch')
        for name in v3.LEGAL_FILES:
            shutil.copyfile(REPO / name, output / name)
        (output / 'README.md').write_text(render_card(expanded))
        (output / '.gitattributes').write_text('*.lance filter=lfs diff=lfs merge=lfs -text\n*.txn filter=lfs diff=lfs merge=lfs -text\n*.manifest filter=lfs diff=lfs merge=lfs -text\n')
        manifest = []
        payload_by_path = {r['path']: r for r in selected}
        for path in package_file_paths(output):
            rel = path.relative_to(output).as_posix()
            size, digest = v3.sha256_file(path)
            row = payload_by_path.get(rel)
            manifest.append({'path': rel, 'bytes': size, 'sha256': digest,
                             'role': 'dataset_payload' if row else 'release_metadata',
                             **({k: row[k] for k in ('component','split','source_bundle','source_path','source_sha256')} if row else {})})
        data = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in sorted(manifest, key=lambda r: r['path'])).encode()
        (output / 'manifest.jsonl').write_bytes(data)
        (output / 'manifest.sha256').write_text(f'{hashlib.sha256(data).hexdigest()}  manifest.jsonl\n')
        marker.unlink()
        return verify(root, output, workers)
    except BaseException:
        print(f'Incomplete candidate retained at {output}', file=sys.stderr)
        raise


def verify(root, output, workers):
    bundles, expected, registry, _ = plan(root)
    if (output / '.build_in_progress').exists():
        raise ValueError('Build incomplete')
    rows = [json.loads(line) for line in (output / 'manifest.jsonl').read_text().splitlines()]
    digest = v3.sha256_file(output / 'manifest.jsonl')[1]
    if (output / 'manifest.sha256').read_text().split() != [digest, 'manifest.jsonl']:
        raise ValueError('Manifest sidecar mismatch')
    actual = {r['path']: r for r in rows if r['role'] == 'dataset_payload'}
    if set(actual) != {r['path'] for r in expected}:
        raise ValueError('Payload coverage differs from plan')
    for r in expected:
        got = actual[r['path']]
        for k in ('bytes','sha256','component','split','source_bundle','source_path','source_sha256'):
            if got[k] != r[k]:
                raise ValueError(f'{r["path"]}: {k} mismatch')
    if json.loads((output / 'task_registry.json').read_text()) != registry:
        raise ValueError('Registry merge mismatch')
    def check(row):
        p = v3.regular_file(output, row['path'])
        if v3.sha256_file(p) != (row['bytes'], row['sha256']):
            raise ValueError(f'Candidate hash mismatch: {row["path"]}')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(check, rows))
    listed = {r['path'] for r in rows} | {'manifest.jsonl', 'manifest.sha256'}
    found = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if found != listed:
        raise ValueError(f'Unlisted or missing files: {len(found ^ listed)}')
    return {'status': 'passed', 'output': str(output), 'payload_files': len(expected),
            'payload_bytes': sum(r['bytes'] for r in expected), 'manifest_sha256': digest,
            'component_split_count': len({(r['component'], r['split']) for r in expected})}


def finalize(root, output, workers):
    """Finish a complete copy interrupted while writing its metadata manifest."""
    if not (output / '.build_in_progress').is_file() or (output / 'manifest.jsonl').exists():
        raise ValueError('Finalize requires an incomplete candidate without a manifest')
    _, selected, _, expanded = plan(root)
    (output / 'README.md').write_text(render_card(expanded))
    payload_by_path = {r['path']: r for r in selected}
    paths = package_file_paths(output)
    def row_for(path):
        rel = path.relative_to(output).as_posix()
        size, digest = v3.sha256_file(path)
        source = payload_by_path.get(rel)
        if source and (size, digest) != (source['bytes'], source['sha256']):
            raise ValueError(f'Copied payload differs from source manifest: {rel}')
        return {'path': rel, 'bytes': size, 'sha256': digest,
                'role': 'dataset_payload' if source else 'release_metadata',
                **({k: source[k] for k in ('component','split','source_bundle','source_path','source_sha256')} if source else {})}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(row_for, paths))
    if {r['path'] for r in rows if r['role'] == 'dataset_payload'} != set(payload_by_path):
        raise ValueError('Incomplete payload copy')
    data = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in sorted(rows, key=lambda r: r['path'])).encode()
    (output / 'manifest.jsonl').write_bytes(data)
    (output / 'manifest.sha256').write_text(f'{hashlib.sha256(data).hexdigest()}  manifest.jsonl\n')
    (output / '.build_in_progress').unlink()
    return verify(root, output, workers)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--finalize', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    try:
        if args.finalize:
            result = finalize(args.dataset_root, args.output, args.workers)
        elif args.execute:
            result = execute(args.dataset_root, args.output, args.workers)
        elif args.verify:
            result = verify(args.dataset_root, args.output, args.workers)
        else:
            _, selected, _, _ = plan(args.dataset_root)
            result = {'status': 'plan', 'payload_files': len(selected),
                      'payload_bytes': sum(r['bytes'] for r in selected)}
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, OSError, KeyError) as exc:
        print(f'Expanded release failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
