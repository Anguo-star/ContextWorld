# ContextWorld dataset: format and loading

The current ContextWorld dataset contains Training, Development and public Test data for nine tasks. The complete bundle is assembled and verified locally, but no stable public download revision or model checkpoints are available yet. This guide describes how to use a complete local bundle; see the [dataset package guide](Expanded_Training_Release.md) to build and verify one from the source bundles.

## Bundle and splits

The bundle keeps its native Lance and JSON files. `task_registry.json` is the entry point for task payloads and evaluation contracts; `manifest.jsonl` and `manifest.sha256` record file integrity. `components/` contains Training and Development data, and `artifacts/` contains Test data. `inventory.json` describes the included task and split files. Use Development to choose a method, recipe and checkpoint, then use Test for the final offline report. Test is not a hosted leaderboard.

The bundle does not provide a `datasets.load_dataset()` script or a Hugging Face dataset viewer. Original-environment H5 datasets, model weights and training logs are separate dependencies and are not included in this package. Dataset metadata describe the payloads; model scores are reported separately in the [report's Public Test table](ContextWorld_ICL_Benchmark.md#training-comparison) and its linked result files.

## Load a local bundle

Set an absolute path to the complete bundle, regardless of its directory name:

```bash
export CONTEXTWORLD_BENCHMARK_ROOT=/path/to/ContextWorld-bundle
contextworld-benchmark info
```

The training and evaluation entry points accept `--benchmark-root` or `CONTEXTWORLD_BENCHMARK_ROOT`. An explicit path takes priority over the default directory inferred from `CONTEXTWORLD_DATASET_ROOT`; an incomplete bundle is an error. For an external model implementing the [adapter contract](External_Model_Adapter_Contract.md), run a Development evaluation first:

```bash
python -m contextworld.benchmarks.external_model_cli \
  --benchmark-root "$CONTEXTWORLD_BENCHMARK_ROOT" \
  --evaluation-split development \
  --task contact_friction \
  --adapter your_package.module:YourAdapter \
  --checkpoint /path/to/checkpoint \
  --model-name your-model \
  --output /path/to/dev-result.json
```

After fixing all method and checkpoint choices, run the same command with `--evaluation-split test` and a new output path for the final Test result. See the [Stable-WorldModel training guide](StableWM_Training.md) for the built-in LeWM, PLDM and PreJEPA launchers.

Original-environment H5 files used by original-data training, mixed training and optional CEM evaluation must be obtained separately. Point `CONTEXTWORLD_DATASET_ROOT` at their parent data root as described in the [training guide](StableWM_Training.md); it is distinct from `CONTEXTWORLD_BENCHMARK_ROOT`, which selects the nine-task bundle.

## Verify a bundle

From the ContextWorld repository, with the source bundles available under a separate data root, run:

```bash
export CONTEXTWORLD_DATASET_ROOT=/path/to/source-data-root
python scripts/package_expanded_training_release.py \
  --dataset-root "$CONTEXTWORLD_DATASET_ROOT" \
  --output "$CONTEXTWORLD_BENCHMARK_ROOT" --verify
python scripts/validate_expanded_training_release_loads.py \
  --release-root "$CONTEXTWORLD_BENCHMARK_ROOT"
```

The package check compares the bundle with pinned source manifests; the loading check opens the registered payloads and loader entry points. The [release guide](Expanded_Training_Release.md) gives the build command, exact validation scope and publication status.
