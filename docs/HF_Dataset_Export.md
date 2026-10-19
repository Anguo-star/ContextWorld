# ContextWorld dataset: format and loading

ContextWorld packages Training, Development and Test data for nine tasks. The complete bundle is prepared locally; a public Hugging Face repository and download revision have not yet been published. The commands below explain how to download a published revision and use the same package locally. Model weights are distributed separately from the dataset. See the [package guide](Expanded_Training_Release.md) for composition and verification results.

## Bundle and splits

The bundle keeps its native Lance and JSON files. `task_registry.json` is the entry point for task payloads and evaluation contracts; `manifest.jsonl` and `manifest.sha256` record file integrity. `components/` contains Training and Development data, and `artifacts/` contains Test data. `inventory.json` describes the included task and split files. Use Development to choose a method, recipe and checkpoint, then use Test for the final offline report. Test is not a hosted leaderboard.

The bundle does not provide a `datasets.load_dataset()` script or a Hugging Face dataset viewer. Original-environment H5 datasets, model weights and training logs are separate dependencies and are not included in this package. Dataset metadata describe the payloads; model scores are reported separately in the [report's Public Test table](ContextWorld_ICL_Benchmark.md#training-comparison) and its linked result files.

## Download a fixed revision

Once the dataset is published, use its announced repository ID and commit hash. These placeholders are not an existing download address:

```bash
pip install huggingface_hub
export CW_HF_DATASET_ID=your-account/ContextWorld
export CW_HF_REVISION=published-commit-hash
export CONTEXTWORLD_BENCHMARK_ROOT=/absolute/path/to/ContextWorld
hf download "$CW_HF_DATASET_ID" --repo-type dataset \
  --revision "$CW_HF_REVISION" --local-dir "$CONTEXTWORLD_BENCHMARK_ROOT"
```

Download the complete repository, including manifests, registries and normalizers. Native Lance tables contain several interdependent files. Reserve at least 55 GiB for the dataset; model weights and runtime dependencies require additional space.

## Load a local bundle

From a ContextWorld code checkout, install the readers:

```bash
pip install -e ".[eval]"
```

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

Run these commands from the ContextWorld checkout:

```bash
contextworld-benchmark audit --full \
  --benchmark-root "$CONTEXTWORLD_BENCHMARK_ROOT"
python scripts/validate_expanded_training_release_loads.py \
  --release-root "$CONTEXTWORLD_BENCHMARK_ROOT"
```

Both commands use only the downloaded bundle. The first checks the manifest, file sizes and SHA-256 hashes; omitting `--full` gives a faster size and metadata check. The second opens registered tables, validates split bindings and normalizers, and reads sample Training and Development inputs. These are data checks, not model scores. The source-build verifier is for maintainers rebuilding from the original source bundles; downloaders do not need those sources.

For final Test scoring, the public entry point verifies the selected task's Test files and scoring configuration against the bundle before running the task scorer. Images and actions are read from the bundle, while the installed ContextWorld checkout provides the scoring code. Record both the dataset commit and code commit with each result. A complete rerun of the report additionally requires its model checkpoints and model-runtime dependencies.
