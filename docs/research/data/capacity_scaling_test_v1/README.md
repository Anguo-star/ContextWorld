# Capacity-scaling Test results

`final_comparison.json` combines four new 183M Test reports in `results/` with the matching 18M Test reports from `current_reference_test_v1`. Original-environment CEM values are carried over from `model_capacity_scaling_v1/final_comparison.json`; they are not Test or Development task scores.

Rebuild the comparison without inference:

```bash
python docs/research/data/capacity_scaling_test_v1/evaluate.py --summarize-only
```

Running `evaluate.py --device cuda:6` evaluates only missing 183M Test reports from existing epoch-10 checkpoints using the frozen Test plan.
