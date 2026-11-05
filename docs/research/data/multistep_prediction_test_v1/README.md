# Multi-step Test prediction results

`summary.json` contains 121 frozen-checkpoint runs and 75 model/task/scheme combinations across eight tasks. `summary_queries.json.gz` contains 33,176 per-query energy records; `summary.csv` is the corresponding row summary. All scores use Test panels and the checkpoints registered for the main Test task comparison.

Cube gripper carry has 15 registered checkpoints (11 combinations), but no multi-step score: its Test panel did not pass the source-state replay tolerance. The `missing` entries and `coverage.unavailable_combinations` in `summary.json` record this gap. Cube's main task-query Test metrics are separate and remain available.

`models.json`, `protocol.json`, and `provenance.json` identify the checkpoints, panel manifests, scoring code, frozen-state receipts, and source checks. `runner_free.py` records the free-rollout evaluation command used for the final subset; earlier completed runs also include optional full-observation diagnostics. The paper's multi-step metrics use only free rollouts.

To recompute the aggregate scores from the saved per-scene predictions, restore the Test panel and prediction directories at the runtime paths recorded in `provenance.json` (under `/tmp/cw-test-multistep-20261028`), with the registered checkpoints and the Python environment available, then run from the ContextWorld repository:

```bash
python scripts/multistep_prediction_score.py \
  --models /tmp/cw-test-multistep-20261028/models.json \
  --evaluation-split test \
  --results-root /tmp/cw-test-multistep-20261028/results \
  --panels-root /tmp/cw-test-multistep-20261028/panels \
  --output /tmp/cw-test-multistep-20261028/summary.json \
  --split-query-records
```

The archived query energies and summary can be inspected without those runtime directories. Re-running the scorer from raw predictions requires them; it does not train models or run inference.
