---
license: cc-by-4.0
language:
- en
tags:
- world-models
- in-context-learning
- robotics
- simulation
viewer: false
---

# ContextWorld: predicting with hidden dynamics

ContextWorld tests whether a visual world model can use recent images and actions to predict what happens under a hidden physical rule, without changing its weights at evaluation time. For example, two PushT scenes can show the same current image and receive the same planned pushing actions, yet lead to different future images because the contact friction differs. The preceding interaction shows how the object responded and provides evidence about the rule. The dataset covers nine tasks in TwoRoom, PushT, Reacher and Cube.

## What a query contains

A query provides a continuous history of images and actions ending at the current image, plus the planned action sequence. Scoring compares the model's predicted numerical features of the future (latent representations) with features of the saved future images. Those images are scoring targets, not prediction inputs; the same model's frozen target encoder converts them into target features.

In six tasks, queries are paired across two hidden conditions: the current image and planned actions match, while the histories and true futures differ. This tests whether the model uses the history to choose the appropriate future. Other tasks use their own query and scoring rules, described in the [technical report](https://github.com/Anguo-star/ContextWorld/blob/main/docs/ContextWorld_ICL_Benchmark.md).

Models may read only images (`pixels`) and actions (`action`). Hidden parameters, simulator states, pair identifiers and catalog indexes are for data checks and scoring, not model inputs.

## Tasks and splits

| Task (environment) | Hidden change | Training | Development | Test |
|---|---|---:|---:|---:|
| Action Strength (PushT) | Pusher movement under the same actions | 32,768 pairs | 256 pairs | 256 pairs |
| Contact Friction (PushT) | Friction during contact | 32,768 pairs | 256 pairs | 256 pairs |
| Motion Damping (PushT) | How quickly the object slows | 32,768 pairs | 256 pairs | 256 pairs |
| Robot Arm Mass (Reacher) | Arm mass under the same torque | 32,768 pairs | 256 pairs | 256 pairs |
| Cube Gripper Carry (Cube) | Whether a closed gripper carries the cube | 10,000 pairs from 10,000 source episodes | 256 pairs | 256 pairs |
| Portal Exit (TwoRoom) | Exit reached from the same portal | 32,768 pairs | 256 pairs | 256 pairs |
| Speed (TwoRoom) | Movement speed | 32 speed settings | 300 queries per reference speed | 300 queries per reference speed |
| Action Delay (TwoRoom) | Delay before an action takes effect | Delays 0–10 | 300 queries × 11 delays | 300 queries × 11 delays |
| Door (TwoRoom) | Whether a door can be crossed | 96 door positions | 300 queries across 16 door positions | 300 queries across 42 door positions |

A **pair** contains trajectories under two hidden conditions; it is not a count of independent source episodes. An episode is one recorded trajectory, and multiple action windows can come from the same episode. Speed settings, delays and door positions count distinct parameter values or locations, not training examples. The [report](https://github.com/Anguo-star/ContextWorld/blob/main/docs/ContextWorld_ICL_Benchmark.md) gives the construction and split rules.

Use Training to fit a model, Development to choose a method and checkpoint, and the public Test split for final offline reporting. Keep source groups and split boundaries intact. There is no hosted submission service.

## Download and use

The data files total about 50.20 GiB; metadata adds to the download. The bundle’s Lance tables and NPZ arrays hold task data; JSON catalogs describe the files and evaluation rules. `components/` contains Training and Development data; `artifacts/` contains Test data. Model weights, original-environment H5 datasets and training logs are separate.

Start with `task_registry.json` to find each task, split and file path. `manifest.jsonl` and `manifest.sha256` let you check file sizes and hashes. Included normalizers provide the stored scaling parameters used by the loaders.

After downloading the bundle, install the [ContextWorld code](https://github.com/Anguo-star/ContextWorld) and point its loader to the local bundle:

```bash
git clone https://github.com/Anguo-star/ContextWorld.git
cd ContextWorld
pip install -e ".[eval]"
export CONTEXTWORLD_BENCHMARK_ROOT=/absolute/path/to/downloaded/ContextWorld
contextworld-benchmark info
contextworld-benchmark audit --full
```

The audit checks the downloaded files without the original source datasets. For a revision-pinned download and evaluation instructions, see the [download and loading guide](https://github.com/Anguo-star/ContextWorld/blob/main/docs/HF_Dataset_Export.md). This package has no `datasets.load_dataset()` script or dataset viewer.

## Reading results

The task score asks whether the model identifies the appropriate future under each task's scoring rule. A high score alone does not establish accurate full trajectories or better planning decisions. The [report](https://github.com/Anguo-star/ContextWorld/blob/main/docs/ContextWorld_ICL_Benchmark.md) reports task scores, response and prediction diagnostics, and planning measures separately for three model families. For the reported ContextWorld training experiments, models are trained separately for each task; a nine-task result row therefore combines multiple checkpoints. Prediction errors measured in a model's own learned representation cannot be treated as a shared physical error scale across different encoders.

## License and attribution

Generated benchmark data use CC BY 4.0; ContextWorld code has a separate MIT license. Preserve `DATA_LICENSE`, `LICENSE` and `NOTICE`. Original-environment datasets and pretrained weights have their own terms and are not redistributed or relicensed by this package.

Credit **ContextWorld (Guo An, 2026)**, link to the [project](https://github.com/Anguo-star/ContextWorld), and specify the dataset revision used in your results.
