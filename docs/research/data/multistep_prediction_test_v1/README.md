# Multi-step Test prediction results

`summary.json`, `summary.csv`, and `summary_queries.json.gz` contain 136 registered checkpoint runs and 86 model/task/scheme settings across nine Test tasks. The 13 unavailable model settings in `coverage.unavailable_combinations` reflect untrained or unregistered schemes; `missing` is empty.

`models.json`, `protocol.json`, `provenance.json`, `panel_checks/`, and `receipts/` record checkpoint, panel, root-check, and frozen inference identities.

For Cube gripper carry, `cube_replay/` contains the coherent Test source replay builder, 256 scene receipts, panel manifest, and full replay measurements. Its root check is in `panel_checks/cube_gripper_carry.json`. The registered Test source identities and paired constraints are retained. The original source pixels/states were not exactly replayed; the measurement files record that diagnostic. This archive does not claim exact reproduction of the original source generation environment. See `cube_replay/README.md` for dependencies and the portable build command.

`summary_queries.json.gz` contains 37,016 per-query energy records. To recompute aggregate scores from saved per-scene predictions, restore the Test panel and prediction directories at the paths recorded in `provenance.json`, then run from the ContextWorld repository:

```bash
python scripts/multistep_prediction_score.py \
  --models /tmp/cw-test-multistep-20261028/models.json \
  --evaluation-split test \
  --results-root /tmp/cw-test-multistep-20261028/results \
  --panels-root /tmp/cw-test-multistep-20261028/panels \
  --output /tmp/cw-test-multistep-20261028/summary.json \
  --split-query-records
```

The archived summaries can be inspected without those runtime directories. Re-running the scorer from raw predictions requires them and does not train or run inference.
