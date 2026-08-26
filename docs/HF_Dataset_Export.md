# ContextWorld-v1 dataset guide

This document has two audiences. Dataset users should read **Bundle layout**
and **Loading the data**. Repository maintainers who prepare a Hugging Face
revision should also read **Building a distribution bundle**.

`ContextWorld-v1` is the only ContextWorld-specific directory intended for
distribution. It contains the Training and Development splits for all nine
components, together with task metadata and file-integrity manifests. Public
Test examples are not included.

## How the payloads were produced

This guide describes the **distribution bundle**: its layout, loading contract,
and clean export. It is not the data-generation specification. The component
payloads are first produced from continuous environment-simulator trajectories,
then checked for causal continuity, pair construction, and split isolation.
They are not generated or edited by an image-generation model.

See [Data generation methodology](Data_Generation.md) for the common causal
contract and the task-specific builders and configurations. The clean exporter
described below copies approved Training and Development artifacts into the
public layout; it does not regenerate trajectories or change benchmark
semantics.

## Bundle layout

```text
ContextWorld-v1/
├── README.md
├── LICENSE
├── DATA_LICENSE
├── NOTICE
├── VERSION.json
├── task_registry.json
├── manifest.jsonl
├── manifest.sha256
├── normalizers/
│   └── tworoom_original_train_s3072.json
└── components/
    ├── tworoom-speed/v1/{training,development}/
    ├── tworoom-door/v1/{training,development}/
    ├── tworoom-action-delay/v1/{training,development}/
    ├── tworoom-portal-exit/v1/{training,development}/
    ├── pusht-action-strength/v1/{training,development}/
    ├── pusht-contact-friction/v1/{training,development}/
    ├── pusht-motion-damping/v1/{training,development}/
    ├── reacher-arm-mass/v1/{training,development}/
    └── cube-gripper-carry/v1/{training,development}/
```

`task_registry.json` is the entry point for software. For each component it
records the environment, capability class, history length, action dimension,
payload layout, table members, sequence schema, Development selection rule,
normalization information, and required runtime view.

`manifest.jsonl` records the relative path, byte size, SHA-256 digest,
component, and split for every distributed file. `manifest.sha256` protects
the manifest itself.

## Loading the data

Set the benchmark root to the extracted directory:

```bash
export CONTEXTWORLD_BENCHMARK_ROOT=/absolute/path/ContextWorld-v1
contextworld-benchmark info
```

For built-in Stable-WorldModel training, select a task and family through the
ContextWorld launcher:

```text
CW_TASK=action_strength
CW_FAMILY=lewm
CW_TRAINING_TRACK=joint_scratch_v1
CONTEXTWORLD_BENCHMARK_ROOT=/absolute/path/ContextWorld-v1
CW_CHECKPOINT_ROOT=/absolute/path/checkpoints/lewm-contextworld
```

Do not set `CW_DATASET` when using the registered bundle. The launcher reads
`task_registry.json`, selects the component's Training payload, and constructs
a manifest-bound dataset request. This prevents an accidental path from
silently selecting the wrong component or split.

Another model family may read the registered Training payloads with its own
loader. Development scoring is model-independent once the model implements
`LatentWorldModelAdapter`; see the
[external model adapter contract](External_Model_Adapter_Contract.md).

### Why some payloads use runtime views

The nine components do not all share one physical table layout:

- PushT action strength, contact friction, motion damping, Reacher arm mass,
  and TwoRoom portal exit use sequence tables that also contain per-step
  metadata. The registered view exposes the numeric model inputs without
  changing the distributed files.
- TwoRoom speed, door, and action delay contain multiple Lance tables. The
  registry lists the members and deterministic balancing rule instead of
  treating the split directory as one table.
- Cube gripper carry stores blocked transitions with five raw actions per
  model step. Its registered view presents the action block in the sequence
  geometry expected by the model while preserving the original payload.

These views adapt storage layout, not benchmark semantics. They do not invent
frames, alter actions, or expose hidden simulator state.

## What the bundle does not contain

The distribution intentionally excludes:

- Public Test examples and evaluation payloads;
- model checkpoints, training logs, and experiment-tracker metadata;
- third-party source checkouts;
- original LeWM training datasets;
- repository-maintenance records and failed-run artifacts.

Original environment training and CEM evaluation use separately distributed
environment datasets. They are not part of `ContextWorld-v1`.

## Building a distribution bundle

This section is for repository maintainers. End users should download an
already built bundle rather than run the exporter.

The source root supplied here is the reviewed synthesis-artifact tree described
in [Data generation methodology](Data_Generation.md). Running this exporter is
therefore a packaging operation, not a substitute for synthesis or causal-data
validation.

First validate all registered source mappings without copying payloads:

```bash
python scripts/export_contextworld_hf_clean.py \
  --suite-export-root /absolute/path/to/source-artifacts \
  --output /absolute/path/to/ContextWorld-v1
```

After reviewing the plan, add `--execute`. The destination must not already
exist. The exporter:

1. copies only registered Training and Development files;
2. rejects symbolic links and credential-like text;
3. verifies every copied file against its source;
4. writes the registry, component cards, and manifests;
5. publishes the completed directory atomically.

On a managed mount that permits file creation but not the final directory
rename, add `--direct-write`. Direct mode still refuses an existing output
directory and removes its own incomplete output after a failed copy.

If payloads are unchanged and only generated documentation or registry
metadata has changed, refresh them without copying the data again:

```bash
python scripts/export_contextworld_hf_clean.py \
  --output /absolute/path/to/ContextWorld-v1 \
  --refresh-metadata
```

Refresh mode verifies the existing manifest and confirms that every payload
still has the same path, size, component, split, and provenance mapping before
updating generated metadata.

The current plan selects 4,352 files and 18,121,449,288 bytes (about 17 GiB).
Use `--full-plan` only when the complete Lance member list is needed.

## Publication status

The distribution bundle has been assembled and validated locally. Publishing
it as Public v1 still requires a stable Hugging Face revision, final citation
and license metadata, a clean-environment loading check, and synchronization
with the public reference-results package. Building the directory alone does
not publish the dataset or unseal Public Test.
