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

**Unpublished preparation draft.** This card describes a proposed nine-task package. Package validation and the final release revision are pending. It does not replace the frozen ContextWorld v3 reference results.

## Dataset description

ContextWorld evaluates whether a world model can infer hidden dynamics from observation/action history and use them to predict conditional futures. It separates prediction accuracy, history-conditioned response and planning evaluation.

The candidate expands Training data for Action Strength, Contact Friction, Motion Damping, Robot Arm Mass, Cube Gripper Carry and Portal Exit. Speed, Action Delay and Door retain their previous Training data. All Development and Test data are inherited from the frozen sources. Training sampling units differ across tasks; pair counts are not interchangeable with independent source-episode counts.

## Format and use

Data remain in native Lance format. `task_registry.json` defines task payloads and evaluation contracts; `manifest.jsonl` records file sizes and SHA256 hashes. Use the ContextWorld loader rather than assuming Hugging Face `datasets.load_dataset` support. Final download and loader commands will be added after the assembled package is validated and a release revision is assigned.

Use Training for optimization, Development for model and recipe selection, and Test for final reporting. Preserve grouping and split boundaries. The original-environment CEM evaluation measures planning retention; it does not by itself establish history-conditioned planning improvements.

## Available evidence and limitations

The accompanying study includes six tasks × three models × two initialization conditions, one training seed per condition. DINO-WM initialization involves parameter projection rather than a function-equivalent full checkpoint transfer. The study reports Development results, not full benchmark pass rates. The remaining three tasks were not rerun under this study.

Synthetic task coverage, model objective and initialization all affect results. The observations do not establish a universal data-scaling law or identify sample count as the sole cause of failure.

## License and attribution

Synthetic data use CC BY 4.0; code uses its separate MIT license. Preserve `DATA_LICENSE`, `LICENSE` and `NOTICE`. External original-environment datasets and pretrained weights retain their own terms and are not relicensed by this card.

See the ContextWorld repository's release preparation guide and training study for exact source identities, results and outstanding acceptance checks.
