# Expanded-training release preparation

This is the preparation plan for the next ContextWorld dataset release. **It is not an uploaded release or an accepted replacement for the frozen v3 benchmark.** The proposed dataset version is `expanded-training-v1`; the final tag will be assigned after package validation.

Task definitions, all reported results, training regimes and scientific interpretation are maintained in the [ContextWorld technical report](ContextWorld_ICL_Benchmark.md#training-comparison). This page only describes dataset packaging and release checks. The current study includes native ICL scratch results for all nine tasks and encoder-freeze comparisons for six tasks; reported and unmeasured regimes are distinguished in the report.

## What is included

| Task | Proposed Training data |
|---|---|
| Action Strength | 32,768 pairs |
| Contact Friction | 32,768 pairs |
| Motion Damping | 32,768 pairs |
| Robot Arm Mass | 32,768 pairs |
| Cube Gripper Carry | 10,000 pairs from 10,000 independent source episodes |
| Portal Exit | 32,768 pairs with expanded physical coverage |
| Speed | Inherited v3 data |
| Action Delay | Inherited v3 data |
| Door | Inherited v3 data |

Counts measure different sampling units and should not be treated as equal numbers of independent episodes. Coverage definitions are in the technical report. Development and Test are inherited from frozen sources, not regenerated alongside expanded Training data. Existing v3 results remain attached to v3 and must not be relabeled as results from expanded Training data.

## Proposed Hugging Face layout

```text
README.md                    # Dataset card with YAML metadata
DATA_LICENSE, LICENSE, NOTICE
VERSION.json                 # Dataset version and source identities
manifest.jsonl               # Per-file sizes, hashes and split provenance
manifest.sha256
task_registry.json           # Task, payload and evaluation definitions
components/                  # Native Lance payloads and required normalizers
reference/                   # Results with explicit dataset and evaluation identity
```

Retain the native Lance representation and document the ContextWorld loader. Do not claim compatibility with `datasets.load_dataset` or enable an unsupported dataset viewer. Hugging Face renders the repository README as a [dataset card](https://huggingface.co/docs/hub/datasets-cards); its metadata supports [disabling the viewer](https://huggingface.co/docs/hub/datasets-viewer-configure). The [candidate card](templates/ContextWorld_Expanded_Training_Card.md) is a draft until package checks finish.

## Reproduce the preparation inventory

From the ContextWorld checkout:

```bash
python scripts/prepare_training_study_release.py \
  --dataset-root "$CONTEXTWORLD_DATASET_ROOT" \
  --output /tmp/contextworld-release-plan
```

This reads manifests, checks expanded Training identities against the published study, selects frozen evaluation files and checks destination path uniqueness. It writes a proposed file manifest and a [release inventory](research/data/expanded_training_release_plan.json). It does **not** copy payloads, open Test examples, merge registries, verify every payload byte or upload anything. The current inventory covers all nine tasks and their three splits; metadata coverage alone does not establish that a package is loadable.

## Before release

1. Assemble files under a new versioned root. Validate every selected file's size and SHA256, including Lance metadata and data fragments; reject truncated files.
2. Merge task registries, preserve evaluation contracts and normalizers, and verify all references against the assembled package.
3. Load each task and split with the released loader. Use Development for smoke evaluation; keep Test for final reporting under the chosen acceptance protocol.
4. Declare the reference scope: dataset version, tasks, models, training regimes, seeds and evaluation split. Publish only observed results under that scope; unmeasured experimental regimes need not be filled merely to make a rectangular table. Development research results must not be relabeled as final Test scores.
5. Finalize the dataset card, attribution, version, download instructions and result identities. Retain v3 as a separate reproducible release.

No dataset has been uploaded by this preparation procedure. A public release requires a validated package and a pinned download revision.
