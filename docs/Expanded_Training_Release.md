# Expanded-training release preparation

This is the preparation plan for the next ContextWorld dataset release. **It is not an uploaded release or an accepted replacement for the frozen v3 benchmark.** The proposed dataset version is `expanded-training-v1`; the final tag will be assigned after package validation.

The [training study](research/Training_Data_and_Initialization.md) covers six tasks, three models and two initialization conditions: 36 runs, with one training seed per condition. DINO-WM uses projected initialization rather than function-equivalent warmstart. These Development results establish useful comparisons; they do not establish a nine-task, multi-seed Test baseline.

## What is included

| Task | Proposed Training data | New three-model initialization study |
|---|---|---|
| Action Strength | 32k pairs | Complete |
| Contact Friction | 32k pairs | Complete |
| Motion Damping | 32k pairs | Complete |
| Robot Arm Mass | 32k pairs | Complete |
| Cube Gripper Carry | 10k independent source episodes | Complete |
| Portal Exit | 32k coverage expansion | Complete |
| Speed | Inherited v3 data | Not rerun |
| Action Delay | Inherited v3 data | Not rerun |
| Door | Inherited v3 data | Not rerun |

Counts measure different sampling units and should not be treated as equal numbers of independent episodes. See the study's data coverage table. Development and Test are inherited from the frozen sources, not regenerated alongside expanded Training data. Existing v3 reference results remain attached to v3; they must not be relabeled as results from the new Training data.

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
4. Decide the reference scope explicitly. A complete new nine-task baseline requires the missing task/model/initialization coverage and the declared seed and Test requirements. Alternatively, publish the data with the six-task exploratory study clearly labeled as such.
5. Finalize the dataset card, attribution, version, download instructions and result identities. Retain v3 as a separate reproducible release.

No HF upload is authorized by this preparation step. Publication is a separate action after the assembled package and its claims are ready.
