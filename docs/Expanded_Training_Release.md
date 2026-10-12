# Expanded-training release candidate

**The nine-task dataset is assembled and its files and loading interfaces are verified locally. It has not been uploaded to Hugging Face.** The candidate version is `contextworld-expanded-training-v1-hf-rc2`; it preserves the frozen Development and Test files and packages the expanded Training data used in the current study. It does not replace the frozen v3 reference results.

Task definitions, training regimes and findings are in the [ContextWorld report](ContextWorld_ICL_Benchmark.md#training-comparison). This page describes the dataset package, its validation and how to reproduce it. Model Test results are a separate deliverable: loading Test files does not evaluate a model.

## Data coverage

| Task | Training data in this candidate |
|---|---|
| Action Strength | 32,768 pairs |
| Contact Friction | 32,768 pairs |
| Motion Damping | 32,768 pairs |
| Robot Arm Mass | 32,768 pairs |
| Cube Gripper Carry | 10,000 pairs from 10,000 independent source episodes |
| Portal Exit | 32,768 pairs with expanded physical coverage |
| Speed | Reused frozen-source Training data |
| Action Delay | Reused frozen-source Training data |
| Door | Reused frozen-source Training data |

Pair counts are not interchangeable with independent episode counts. Sampling and coverage definitions are in the [report](ContextWorld_ICL_Benchmark.md#2-数据与划分). Development comes from `ContextWorld-v1`; Test comes from `ContextWorld-v1-full`. Neither is regenerated when Training expands. Original-environment datasets and pretrained weights are external dependencies, not included or relicensed by this package.

## Validation results

The [validation record](research/data/expanded_training_release_validation_v2.json) identifies the candidate and exact scope of the checks.

| Check | Result |
|---|---|
| Dataset payload | 11,306 files; 53,902,593,727 bytes (50.20 GiB) |
| Split coverage | All 9 tasks × Training / Development / Test |
| Integrity | Full file sizes, SHA256 hashes and exact package inventory verified |
| Registry and evaluation contracts | Expanded Training sources merged; frozen Development and Test contracts preserved |
| Lance loading | All 1,316 registered tables opened; three directory-format Test payloads checked structurally |
| ContextWorld loading | 10 Training views resolved; all nine Development normalizers and public Test bindings resolved |
| Model-input row loading | One Training and one Development row per task loaded; no Test examples scored |
| Speed Test closure | All 4,200 catalog-referenced NPZ files included and SHA256-verified; one bundle per distribution loaded without external artifact lookup |

The package manifest SHA256 is `000558936acf89c12bfbfb407b70c5cd57ecfd7f06c0e9d90e7edb7c4f4d4eb4`. These checks establish file integrity and the listed loading behavior. They do not establish an independent end-to-end Test runtime. Motion Damping’s historical scorer also checks source-release provenance files; these are separate from the dataset candidate. The current evaluation uses an exact-hash provenance snapshot and verifies that its Test table is byte-identical to the frozen bundle before scoring. It does not execute the historical code in that snapshot.

## Package format

```text
README.md                    # Hugging Face dataset card with YAML metadata
DATA_LICENSE, LICENSE, NOTICE
VERSION.json                 # Candidate version and source identities
inventory.json               # Task/split file counts
manifest.jsonl, manifest.sha256
task_registry.json           # Tasks, payloads and evaluation contracts
components/                  # Native Lance payloads and required metadata
artifacts/                   # Frozen Test payloads, including Speed NPZ files
provenance/                  # Source manifests and registries
reference/                   # Frozen v3 results, explicitly historical
```

Use the [ContextWorld loader](HF_Dataset_Export.md), following the entrypoints in `task_registry.json`. Data remain in native Lance format; this package does not provide a `datasets.load_dataset` script. The dataset card disables the unsupported viewer. Source manifests are retained so the merged layout can be traced to the original task bundles.

## Reproduce the package

From a ContextWorld checkout, with the source bundles under `CONTEXTWORLD_DATASET_ROOT`:

```bash
python scripts/package_expanded_training_release.py \
  --dataset-root "$CONTEXTWORLD_DATASET_ROOT" \
  --output /path/to/ContextWorld-expanded-training-v1-hf-rc2 --execute

python scripts/package_expanded_training_release.py \
  --dataset-root "$CONTEXTWORLD_DATASET_ROOT" \
  --output /path/to/ContextWorld-expanded-training-v1-hf-rc2 --verify

python scripts/validate_expanded_training_release_loads.py \
  --release-root /path/to/ContextWorld-expanded-training-v1-hf-rc2
```

The build requires a new output directory and checks copied bytes against the pinned source manifests. It rejects truncated or altered files. The validation command checks the released loader and payload entrypoints without running a model. The [dataset-card template](templates/ContextWorld_Expanded_Training_Card.md) describes intended use, attribution and evaluation limits.

## Reproduce the reported Test table

The [Test archive](research/data/current_reference_test_v1/overview.md) contains the frozen checkpoint plan, individual result files, receipts and aggregates. To recompute the aggregate table from those files, without loading model weights:

```bash
python scripts/summarize_current_reference_test.py \
  --plan docs/research/data/current_reference_test_v1/plan.json \
  --results-root docs/research/data/current_reference_test_v1/results \
  --output-dir /tmp/contextworld-test-summary
```

The evaluator checks native input protocols and unchanged model state. The summarizer verifies result hashes and checkpoint identities, then averages all selected training runs equally. It reports task scores in percent; low scores remain in the report.

Rerunning model inference additionally requires the weights and source dependencies recorded in the plan. `scripts/run_current_reference_test.py` is the frozen runner. `scripts/retry_current_reference_test.py` provides a compatibility entry for the existing cohort: it preserves completed results, records failed attempts, handles an unavailable optional FlashAttention extension, and accepts explicitly bound external artifacts. Motion Damping’s provenance snapshot is constructed by `scripts/prepare_damping_test_audit_overlay.py`; both its release audit and its Test-file equality check must pass. The [runtime provenance receipt](research/data/current_reference_test_v1/damping_runtime_provenance.json) records the exact historical source hashes and matching Test files. Runtime compatibility is recorded separately from the unchanged scoring protocol.

## Remaining release requirements

A public release still needs a pinned download revision, downloadable reference weights and a clean-environment end-to-end reproduction. The corresponding nine-task model Test report must retain its own checkpoint and scoring identities; Development research results cannot be relabeled as Test. Dataset upload is a separate publication step and has not been performed.
