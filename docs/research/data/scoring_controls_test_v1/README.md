# Public Test constructed-prediction scorer control

Run from the ContextWorld checkout with the established Python runtime:

```sh
CUDA_VISIBLE_DEVICES=5 /tmp/cw-strength-ablation-20260922/venv/bin/python docs/research/data/scoring_controls_test_v1/runner.py action_strength/lewm/original/s3073
CUDA_VISIBLE_DEVICES=5 /tmp/cw-strength-ablation-20260922/venv/bin/python docs/research/data/scoring_controls_test_v1/runner.py action_strength/lewm/frozen/s3072
```

The runner loads 256 public Test action-strength pairs and encodes their true five-step future images in each checkpoint's native target space. It constructs predictions directly from the target pairs and calls the official scorer; it does not run the predictor or train. Each checkpoint runs in a separate process because its Stable WorldModel source checkout differs. The `results.json` file combines the two per-checkpoint results. Checkpoint hashes are verified against `configs/benchmark/current_reference_test_v1.json`.
