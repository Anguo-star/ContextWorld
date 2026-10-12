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

# ContextWorld — expanded training data (release candidate)

**Unpublished preparation draft.** This card describes a proposed nine-task package. Local package validation and the final release revision are pending. It does not replace the frozen ContextWorld v3 reference results.

## Dataset description

ContextWorld evaluates whether a world model can infer hidden dynamics from observation/action history and use them to predict conditional futures. It separates prediction accuracy, history-conditioned response and planning evaluation.

The candidate expands Training data for Action Strength, Contact Friction, Motion Damping, Robot Arm Mass, Cube Gripper Carry and Portal Exit. Speed, Action Delay and Door retain their previous Training data. All Development and Test data are inherited from the frozen sources. Training sampling units differ across tasks; pair counts are not interchangeable with independent source-episode counts.

## Format and use

Data remain in native Lance format. `task_registry.json` defines task payloads and evaluation contracts; `manifest.jsonl` records file sizes and SHA256 hashes. Use the ContextWorld loader rather than assuming Hugging Face `datasets.load_dataset` support. Final download and loader commands will be added after the assembled package is validated and a release revision is assigned.

Use Training for optimization, Development for model and recipe selection, and Test for final reporting. Preserve grouping and split boundaries. The original-environment CEM evaluation measures planning retention; it does not by itself establish history-conditioned planning improvements.

## Available evidence and limitations

The [ContextWorld technical report](https://github.com/Anguo-star/ContextWorld/blob/main/docs/ContextWorld_ICL_Benchmark.md#training-comparison) is the source for reported model results and their interpretation. The package includes a frozen v3 reference result for provenance; that result does not measure models trained on this expanded candidate. No Test model evaluation was performed as part of packaging.

Training coverage, representation and initialization differ across the reported studies. The package inventory establishes data identity and loadability, not a causal effect of more Training examples or a new benchmark score.

## License and attribution

Synthetic data use CC BY 4.0; code uses its separate MIT license. Preserve `DATA_LICENSE`, `LICENSE` and `NOTICE`. External original-environment datasets and pretrained weights retain their own terms and are not relicensed by this card.

See the ContextWorld repository's release preparation guide and training study for exact source identities, results and outstanding acceptance checks.
