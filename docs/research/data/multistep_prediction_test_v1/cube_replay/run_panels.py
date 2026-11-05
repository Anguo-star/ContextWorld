#!/usr/bin/env python3
"""Build 256 coherent Cube v4r1 Test panels from the registered public source.

Original-source exact replay is recorded as a diagnostic.  The output uses
one fixed current runtime for coherent history and extended-future rendering.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LP_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import cube_coherent as b

ROOT = None
SCENES = None

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def worker(shard):
 out=ROOT/f'{shard:02d}';out.mkdir(exist_ok=True);sim=CubeGraspRuleV4Simulator();rows=[]
 for idx in range(shard*8,(shard+1)*8):
  scene=SCENES[idx];rec={'index':idx,'pair_id':scene['pair_id'],'source_row':scene['source_row'],'source_episode':scene['source_episode'],'source_step':scene['source_step'],'catalog_index':scene['catalog_index'],'status':'started'}
  try:
   candidate=b.cube_candidate(scene);blocks=action_blocks(candidate.action_profile)
   rec['source_identity']={'qpos_float64_equal':bool(np.array_equal(np.asarray(candidate.qpos,np.float64),np.asarray(scene['qpos'],np.float64))),'control_float64_equal':bool(np.array_equal(np.asarray(candidate.control,np.float64),np.asarray(scene['control'],np.float64))),'action_profile_id':scene['expected_profile_id'],'action_profile_sha256':hashlib.sha256(np.ascontiguousarray(blocks).tobytes()).hexdigest(),'lance_action_equal':bool(all(np.array_equal(scene['episodes'][mode]['blocks'],blocks) for mode in scene['modes']))}
   if not all((rec['source_identity']['qpos_float64_equal'],rec['source_identity']['control_float64_equal'],rec['source_identity']['action_profile_id']==rec['source_identity']['action_profile_sha256'],rec['source_identity']['lance_action_equal'])):raise RuntimeError('source identity mismatch')
   arr=b.run_cube_scene(scene,sim,rec);native=list(arr['candidate_ids']).index('amp_+1.0_repeat')
   assert np.array_equal(arr['goal_pixels'],arr['future_pixels'][0,native,0]);assert np.array_equal(arr['goal_state'],arr['future_states'][0,native,0]);assert np.array_equal(arr['history_pixels'][0,-1],arr['history_pixels'][1,-1]);assert arr['future_pixels'].shape==(2,14,5,224,224,3)
   path=out/(scene['pair_id']+'.npz');np.savez_compressed(path,**arr)
   rec.update(status='complete',npz=path.name,npz_sha256=sha(path),npz_bytes=path.stat().st_size,source_exact_replay_passed=bool(scene['audits']['source_exact_replay_passed']),source_controlled_position_max=float(arr['source_controlled_position_max']),source_history_pixel_max=int(arr['source_history_pixel_max']),native_goal_equal=True,current_pixel_equal=True,current_full_simulator_state_max=float(scene['audits']['current_full_simulator_state_max']),paired_audit_passed=bool(scene['audits']['paired_audit_passed']),candidate_raw_prefix_equal=bool(scene['audits']['candidate_raw_prefix_equal']),candidate_prefix_equal=True,current_full_state_tolerance=1e-10,render_mode='coherent_test_source_replay',source_audits=scene['audits'])
  except Exception as e:rec.update(status='failed',error=repr(e),traceback=traceback.format_exc())
  rows.append(rec)
  receipt_path=out/(scene['pair_id']+'.receipt.json')
  temporary=receipt_path.with_suffix('.json.tmp')
  temporary.write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
  temporary.replace(receipt_path)
  print('shard',shard,'scene',idx,rec['status'],flush=True)
 sim.close();(out/'receipt.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n');return str(out/'receipt.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=b.DEFAULT_BUNDLE)
    parser.add_argument("--cube-h5", type=Path, default=b.DEFAULT_CUBE_H5)
    parser.add_argument("--contextworld-repo", type=Path, default=b.DEFAULT_CW_REPO)
    parser.add_argument("--stable-repo", type=Path, default=b.DEFAULT_STABLE_REPO)
    parser.add_argument("--stable-ref", default=b.DEFAULT_STABLE_REF)
    parser.add_argument("--source-measurement", type=Path,
                        help="existing source-versus-current replay diagnostic JSON")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.workers != 32:
        raise ValueError("this frozen 256-pair build uses 32 shards of eight pairs")
    if args.stable_ref != b.DEFAULT_STABLE_REF:
        raise ValueError("stable-worldmodel ref differs from the pinned Cube v4r1 source")
    for path in (args.contextworld_repo, args.stable_repo):
        sys.path.insert(0, str(path))
    import scipy
    import mujoco
    from contextworld.evaluation.cube_grasp_rule_h3_v4 import (
        CubeGraspRuleV4Simulator as Simulator,
        action_blocks as profile_action_blocks,
    )
    global ROOT, SCENES, CubeGraspRuleV4Simulator, action_blocks
    CubeGraspRuleV4Simulator = Simulator
    action_blocks = profile_action_blocks
    ROOT = args.output.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    SCENES = b.load_cube_scenes(args.bundle, 256, args.cube_h5)
    if len(SCENES) != 256 or len({scene["pair_id"] for scene in SCENES}) != 256:
        raise RuntimeError("registered Test source does not contain 256 distinct pairs")
    started = time.monotonic()
    with mp.get_context("fork").Pool(32) as pool:
        receipts = pool.map(worker, range(32))
    rows = [row for path in receipts for row in json.loads(Path(path).read_text())]
    report = {
        "schema": "contextworld.cube-coherent-test-panel.v1",
        "render_mode": "coherent_test_source_replay",
        "source_exact_replay_criterion_unchanged": True,
        "source_exact_replay_is_required_for_panel_validity": False,
        "source_measurement_archive": str(args.source_measurement) if args.source_measurement else None,
        "source_bundle": str(args.bundle),
        "source_lance": b.CUBE_DATA,
        "source_h5": str(args.cube_h5),
        "stable_repo": str(args.stable_repo),
        "stable_ref": args.stable_ref,
        "physics_module": "contextworld/evaluation/cube_grasp_rule_h3_v4.py",
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__, "mujoco": mujoco.__version__},
        "count": len(rows),
        "complete": sum(row["status"] == "complete" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "source_exact_replay_passed": sum(row.get("source_exact_replay_passed") is True for row in rows),
        "elapsed_seconds": time.monotonic() - started,
        "scenes": rows,
    }
    (ROOT / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("count", "complete", "failed", "source_exact_replay_passed")}, indent=2))
    return 0 if report["failed"] == 0 and report["complete"] == 256 else 1

if __name__ == "__main__":
    raise SystemExit(main())
