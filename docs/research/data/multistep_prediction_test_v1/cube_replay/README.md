# Cube v4r1 coherent Test panels

This archive provides the Cube panel builder and its source checks. It reads the registered 256-pair Public Test source from `ContextWorld-v1-full/artifacts/synthesis/cube_gripper_carry_rule_h3_v4r1_release_projection_v1/validation.lance` and the pinned Cube H5 source. It does not train or select checkpoints.

`cube_coherent.py` contains the Cube functions extracted from the active builder. `run_panels.py` keeps the source identity checks, original paired-protocol audit, identical A/B current raw pixels and full simulator states matching within the paired-protocol tolerance of `1e-10`, equal candidate prefixes, and native first-future checks. It renders histories and five future blocks continuously in one current runtime. The saved `source_exact_replay_passed` flag remains a diagnostic: some reconstructed frames differ from the original Lance source, and this output does **not** claim to reproduce the original generation environment exactly. The coherent panels are not a replacement for the registered Test data.

## Inputs and runtime

- ContextWorld repository with `contextworld.evaluation.cube_grasp_rule_h3_v4`.
- stable-worldmodel at commit `875e607fc08aa72eacb94d5d178127804134cc06`, including its OGBench Cube environment.
- The registered `ContextWorld-v1-full` Test bundle and the original `cube_single_expert.h5`.
- Python packages: NumPy, SciPy, Lance, h5py, Pillow, MuJoCo, Gymnasium, OGBench, and stable-worldmodel. Use one fixed environment for the whole build; the manifest records NumPy, SciPy, and MuJoCo versions. Rendering uses `MUJOCO_GL=osmesa`; `LP_NUM_THREADS=1` and the usual BLAS thread limits are set before imports.

The active coherent build uses `/tmp/cw-strength-ablation-20260922/venv/bin/python` (Python 3.10.12, NumPy 1.24.4 with OpenBLAS ILP64, SciPy 1.11.3, MuJoCo 3.6.0, OGBench 1.2.1, PyLance 11.0.0, h5py 3.14.0, Pillow 9.2.0, Gymnasium 1.3.0). These are the **coherent build** dependencies; the original August Test generator's package versions are not established by the surviving release metadata.

## Build

Set paths to your local copy of the registered sources and pinned repositories. The archived source measurements are in `measurement_results.json`. Run from this directory with an output directory outside the source repositories:

```bash
export CONTEXTWORLD_REPO=/path/to/ContextWorld
export STABLE_REPO=/path/to/stable-worldmodel-875e607fc08a
export DATA_ROOT=/path/to/world_model
export CUBE_H5=/path/to/cube_single_expert.h5
python run_panels.py \
  --output /path/to/cube-coherent-test-panels \
  --contextworld-repo "$CONTEXTWORLD_REPO" \
  --stable-repo "$STABLE_REPO" \
  --bundle "$DATA_ROOT/ContextWorld-v1-full" \
  --cube-h5 "$CUBE_H5" \
  --source-measurement ./measurement_results.json
```

The runner writes 32 eight-pair shard receipts, one per-scene receipt and `.npz` per pair, and `manifest.json`. `manifest.json` reports the number of source-exact replays separately from the 256 coherent panel completions; it never marks failed source replay as passed.
