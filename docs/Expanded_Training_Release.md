# ContextWorld nine-task dataset release

The current nine-task Training, Development and public Test bundle is assembled and verified locally. It has not been uploaded to Hugging Face; no stable public data revision or downloadable reference checkpoints are available. This page describes its composition, local verification and reproduction. For task definitions, scores and interpretation, see the [ContextWorld report](ContextWorld_ICL_Benchmark.md#training-comparison).

## Data composition

| Task | Training coverage in the current bundle |
|---|---|
| Action Strength | 32,768 pairs |
| Contact Friction | 32,768 pairs |
| Motion Damping | 32,768 pairs |
| Robot Arm Mass | 32,768 pairs |
| Cube Gripper Carry | 10,000 pairs from 10,000 independent source episodes |
| Portal Exit | 32,768 pairs with expanded physical coverage |
| Speed | 32 speed settings |
| Action Delay | Delays 0–10 |
| Door | 96 door positions |

Pair counts, source episodes and condition counts are different units. The [report's data table](ContextWorld_ICL_Benchmark.md#22-数据规模与统计单位) defines the coverage and Development/Test counts. The [study record](research/data/icl_training_study_v2.json) identifies the source bundles for each split. Training-scale comparisons use fixed Development and Test queries. Original-environment H5 datasets and model weights are external dependencies, outside this package.

The package contains native Lance and JSON payloads under `components/` and `artifacts/`, plus `task_registry.json`, `inventory.json`, `manifest.jsonl`, `manifest.sha256`, licenses and attribution. `provenance/` retains source manifests and registries. The loader follows `task_registry.json`; see the [loading guide](HF_Dataset_Export.md). The package has no `datasets.load_dataset()` script or supported dataset viewer. Dataset metadata describe the payloads; the [model Test scores](ContextWorld_ICL_Benchmark.md#training-comparison) are reported separately.

## Build and verify locally

From a ContextWorld checkout, set `CONTEXTWORLD_DATASET_ROOT` to the parent of the required source bundles and choose any new absolute output directory. The build reads the pinned source files and writes a new bundle:

```bash
export CONTEXTWORLD_DATASET_ROOT=/path/to/source-data-root
export CONTEXTWORLD_BENCHMARK_ROOT=/path/to/ContextWorld-bundle
python scripts/package_expanded_training_release.py \
  --dataset-root "$CONTEXTWORLD_DATASET_ROOT" \
  --output "$CONTEXTWORLD_BENCHMARK_ROOT"
python scripts/package_expanded_training_release.py \
  --dataset-root "$CONTEXTWORLD_DATASET_ROOT" \
  --output "$CONTEXTWORLD_BENCHMARK_ROOT" --execute
python scripts/package_expanded_training_release.py \
  --dataset-root "$CONTEXTWORLD_DATASET_ROOT" \
  --output "$CONTEXTWORLD_BENCHMARK_ROOT" --verify
python scripts/validate_expanded_training_release_loads.py \
  --release-root "$CONTEXTWORLD_BENCHMARK_ROOT"
```

The first command prints a plan without copying data. `--execute` requires an output directory that does not already exist. `--verify` checks the package inventory and file hashes against its pinned sources; the loading validator checks the registered payloads and loader entry points without scoring a model. The [dataset-card template](templates/ContextWorld_Expanded_Training_Card.md) records intended use and attribution.

## Verified scope and result provenance

The [validation record](research/data/expanded_training_release_validation_v2.json) reports 11,306 payload files (53,902,593,727 bytes; 50.20 GiB) across all nine tasks and three splits. Its package manifest SHA-256 is `000558936acf89c12bfbfb407b70c5cd57ecfd7f06c0e9d90e7edb7c4f4d4eb4`. Checks covered file sizes and SHA-256 hashes, all 1,316 registered Lance tables, three directory-format Test payloads, ten Training views, all nine Development normalizers, public Test bindings, and one Training and Development model-input row per task. The Speed Test catalogs' 4,200 referenced NPZ files were included and hash checked. These checks establish package integrity and loading behavior; they do not rerun model inference or prove independent end-to-end reproduction.

The [Public Test results](research/data/current_reference_test_v1/overview.md) provide the aggregate table. The [checkpoint manifest](research/data/current_reference_test_v1/plan.json) and [per-run records](research/data/current_reference_test_v1/results/) identify the checkpoints, scores and evaluation receipts. From the checkout, the aggregate can be recomputed without weights:

```bash
python scripts/summarize_current_reference_test.py \
  --plan docs/research/data/current_reference_test_v1/plan.json \
  --results-root docs/research/data/current_reference_test_v1/results \
  --output-dir /tmp/contextworld-test-summary
```

Rerunning inference requires the checkpoints and source dependencies recorded in that plan. Motion Damping scoring also requires its exact-hash [runtime provenance receipt](research/data/current_reference_test_v1/damping_runtime_provenance.json) and matching Test files. That receipt records the runtime and data-file identities for the evaluation. Package validation alone does not constitute a model Test result.
