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

# ContextWorld: adaptation to hidden dynamics from visual history

ContextWorld tests whether a visual world model can use recent image and action history to predict the effects of hidden dynamics, without updating its weights during evaluation. It contains nine tasks across TwoRoom, PushT, Reacher and Cube. Paired queries control the current observation and query actions so that prior interaction supplies evidence about the hidden rule.

## Contents and splits

| Task | Environment | Training coverage | Development / Test |
|---|---|---|---|
| Action Strength | PushT | 32,768 pairs | 256 pairs each |
| Contact Friction | PushT | 32,768 pairs | 256 pairs each |
| Motion Damping | PushT | 32,768 pairs | 256 pairs each |
| Robot Arm Mass | Reacher | 32,768 pairs | 256 pairs each |
| Cube Gripper Carry | Cube | 10,000 pairs from 10,000 source episodes | 256 pairs each |
| Portal Exit | TwoRoom | 32,768 pairs | 256 pairs each |
| Speed | TwoRoom | 32 speed settings | 300 queries per reference speed |
| Action Delay | TwoRoom | Delays 0–10 | 300 queries × 11 delays each |
| Door | TwoRoom | 96 door positions | 300 queries each; 16 / 42 door positions |

Pairs, source episodes and parameter settings are different units. A pair contains trajectories under two hidden conditions and need not represent two independent source episodes. The [technical report](https://github.com/Anguo-star/ContextWorld/blob/main/docs/ContextWorld_ICL_Benchmark.md) specifies the construction, split sizes and pairing constraints. The package contains 11,306 data files, totaling 53,902,593,727 bytes (50.20 GiB), plus metadata.

Use Training for optimization, Development for method and checkpoint selection, and Test for final offline reporting. Preserve source grouping and split boundaries. Models receive images and actions only; hidden parameters and simulator states are evaluator-side information. Test is public and has no hosted submission service.

## Format and use

The package preserves native Lance tables, JSON catalogs and NPZ arrays. `task_registry.json` lists each task's split, payload format, relative path and evaluation contract. `manifest.jsonl` and `manifest.sha256` record file sizes and SHA-256 hashes. `components/` holds Training and Development data; `artifacts/` holds Test data. Normalizers and source manifests are included. Model weights, original-environment H5 datasets and training logs are separate dependencies.

Install the [ContextWorld code](https://github.com/Anguo-star/ContextWorld), then point its loader at the downloaded bundle:

```bash
pip install -e ".[eval]"
export CONTEXTWORLD_BENCHMARK_ROOT=/path/to/downloaded/ContextWorld
contextworld-benchmark info
contextworld-benchmark audit --full
```

The integrity check uses only the downloaded bundle; it does not need the original source datasets. Individual Lance tables can also be opened with `lance.dataset(path)`, using table paths from the registry. See the [download and loading guide](https://github.com/Anguo-star/ContextWorld/blob/main/docs/HF_Dataset_Export.md) for revision-pinned downloads and model evaluation. This package does not provide a `datasets.load_dataset()` script or dataset viewer.

## Interpretation

The [report](https://github.com/Anguo-star/ContextWorld/blob/main/docs/ContextWorld_ICL_Benchmark.md) defines task scores and reports three model families under several training schemes. Each task uses a separately trained checkpoint. Future selection, conditional response, complete prediction error and planning are evaluated separately: a high selection score alone does not establish accurate full trajectories or better decisions. Native latent errors are not a common physical accuracy scale across different encoders.

## License and attribution

Generated benchmark data use CC BY 4.0; ContextWorld code has a separate MIT license. Preserve `DATA_LICENSE`, `LICENSE` and `NOTICE`. Original-environment datasets and pretrained weights are external dependencies with their own terms; this package does not redistribute or relicense them.

Credit **ContextWorld (Guo An, 2026)**, link to the [project](https://github.com/Anguo-star/ContextWorld), and specify the dataset revision used in your results.
