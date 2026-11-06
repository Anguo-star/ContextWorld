# Data-scale Test results

`summary.json` contains the 40 appendix Data Scale rows on the frozen Test split. The `runs` lists retain training seeds, checkpoint SHA-256 values, raw Test report paths, and report SHA-256 values. The 2k LeWM pusher results reuse `figure3_data_scaling_test_v1`; standard 32k and gripper 10k endpoints reuse `current_reference_test_v1`. Other smaller-data endpoints have raw reports under `results/`.

Rebuild the summary without inference:

```bash
python docs/research/data/data_scaling_test_v1/evaluate.py --summarize-only
```

Re-evaluate only missing reports from the existing checkpoint map, using GPUs 6 and 7:

```bash
python docs/research/data/data_scaling_test_v1/evaluate.py --worker 0 --workers 2 --device cuda:6
python docs/research/data/data_scaling_test_v1/evaluate.py --worker 1 --workers 2 --device cuda:7
```

DINO-WM reports use the original Test runtime with Transformers 4.57.1 and Torch 2.9.1. If that Python executable moves, pass `--dino-python /absolute/path/to/python` with those package versions. The script verifies checkpoint hashes before evaluation and checks Test split, frozen model state, runtime, and bundle identity when summarizing.
