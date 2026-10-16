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

# ContextWorld: visual world-model adaptation to hidden dynamics

ContextWorld evaluates whether a world model can use image and action history to identify hidden dynamics and predict action outcomes without updating its weights. It contains nine tasks across TwoRoom, PushT, Reacher and Cube. Task scores, conditional-response diagnostics, complete prediction errors and planning results are reported separately.

## Data and splits

| Task | Training coverage |
|---|---|
| Action Strength, Contact Friction, Motion Damping, Robot Arm Mass, Portal Exit | 32,768 pairs per task |
| Cube Gripper Carry | 10,000 pairs from 10,000 distinct source episodes |
| Speed | 32 speed settings |
| Action Delay | Delays 0–10 |
| Door | 96 door positions |

Pairs, independent source episodes and parameter settings are different units. The [technical report](https://github.com/Anguo-star/ContextWorld/blob/main/docs/ContextWorld_ICL_Benchmark.md) defines the sampling units, split sizes and pairing constraints.

Use Training for optimization, Development for method and checkpoint selection, and public Test for final offline reporting. Preserve source grouping and split boundaries. Models receive images and actions only; hidden parameters and simulator states are available to the evaluator, not to the model.

## Format and loading

Files use native Lance, JSON and NPZ formats. `task_registry.json` identifies task payloads and evaluation contracts; `manifest.jsonl` records file sizes and SHA-256 hashes. Use the ContextWorld loader as described in the [loading guide](https://github.com/Anguo-star/ContextWorld/blob/main/docs/HF_Dataset_Export.md). The package does not provide a Hugging Face `datasets.load_dataset()` script or dataset viewer.

The assembled package has passed local integrity and loading checks. Data download availability, model checkpoints and reproduction instructions are documented in the [dataset guide](https://github.com/Anguo-star/ContextWorld/blob/main/docs/Expanded_Training_Release.md).

## Results and scope

The technical report provides task definitions, scoring formulas and model comparisons. Each task is trained separately. High future-selection accuracy does not by itself establish accurate full-trajectory prediction or improved planning. Native latent-space errors are not a shared physical accuracy scale across different encoders. Original-environment CEM measures planning retention, separately from hidden-dynamics adaptation.

## License and attribution

Synthetic data use CC BY 4.0; code uses its separate MIT license. Preserve `DATA_LICENSE`, `LICENSE` and `NOTICE`. Original-environment datasets and pretrained weights are external dependencies with their own terms and are not included or relicensed by this package.
