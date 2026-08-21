# Hugging Face dataset export

The directory under `data/world_model/context_world` is a research workspace,
not a distributable dataset. It contains data-generation history, checkpoints,
logs, evaluation records, and an upstream source checkout. Do not upload that
directory to Hugging Face.

ContextWorld instead builds a new, self-contained staging directory. The clean
export has one stable path per benchmark component and contains only Training
and Development data. Public Test examples remain withheld.

Create it beside the research workspace, not inside it. A typical layout is:

```text
data/world_model/
├── quentinll/          original model-training data
├── context_world/      internal research workspace
└── ContextWorld-v1/    clean distribution staging
```

This command produces a staging export, not a public release. Creating the
directory does not by itself authorize public distribution or benchmark
submission.

## Build the staging directory

First validate the registered mappings for all nine components without
copying data:

```bash
python scripts/export_contextworld_hf_clean.py \
  --suite-export-root /absolute/path/to/contextworld-artifact-root \
  --output /absolute/path/to/ContextWorld-v1
```

When the printed plan is correct, add `--execute`. The output path must not
already exist. The exporter copies regular files into a temporary sibling
directory, rejects symlinks and credential-like text, verifies every copied
file, and only then renames the staging directory into place.

Atomic directory publication is the default. On a managed dataset mount that
allows file creation but rejects the final directory rename, add
`--direct-write`. Direct mode still requires a nonexistent output directory,
never overwrites files, and removes its own incomplete output if copying
fails.

If only the checked-in registry contract or generated component cards change,
refresh an existing staging tree without copying the payloads again:

```bash
python scripts/export_contextworld_hf_clean.py \
  --output /absolute/path/to/ContextWorld-v1 \
  --refresh-metadata
```

Refresh mode first verifies the existing manifest digest and confirms that
every registered payload still has the same public path, size, component,
split, and provenance mapping. It is restricted to staging exports. This
operation updates only generated documentation and indexes; it does not modify
dataset payloads or change any payload's direct/adapter status.

Plan-only mode recursively checks every registered payload and reports its
layout, file count and byte count without copying. Against the current frozen
suite it selects 4,352 files (18,121,449,288 bytes, about 17 GiB). Add
`--full-plan` only when the complete list of Lance members is needed.

The resulting layout is:

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

`task_registry.json` records each component's environment, capability class,
history length, action dimension, frozen release-config digest, and provenance
mapping. Each payload also records its public path, layout, table count,
members, sequence schema, required adapter, and whether the path is a direct
Stable-WorldModel training input. `single_dataset_entrypoint` describes only
physical layout; it is not by itself a training-compatibility claim.
`manifest.jsonl` records the relative path, size, SHA-256 digest, component,
and split of every distributed file.

## Training paths

The clean export is a distribution tree, not a replacement for
`CONTEXTWORLD_ARTIFACT_ROOT`. Existing frozen LeWM/PLDM reference launchers
continue to use the research artifact tree and their registered paths.

Five components expose one physically addressable sequence table per payload:
PushT action strength, contact friction and motion damping; Reacher arm mass;
and TwoRoom portal exit. Their temporal and model-input columns are valid, but
the tables also contain string-valued metadata on every step. The pinned public
Stable-WorldModel reader rejects that legacy layout before applying
`keys_to_load`. These paths therefore remain non-direct until
`stablewm_step_metadata_to_episode_table_v1` moves constant metadata into the
episode side table. An older internal loader may accept them, but that is not a
public compatibility guarantee.

TwoRoom speed, door and action delay are collections of Lance tables. Their
split roots are not direct `CW_DATASET` entry points: speed contains nested
regime/shard tables, door contains multiple scenario tables, and action delay
contains separate coarse/full collections. Use the component's registered
training recipe or select and compose the stable member list from
`task_registry.json`. Passing a collection root to Stable-WorldModel either
cannot be detected or is ambiguous.

Cube gripper carry is a different case. Its distributed table is a
blocked-transition projection with `model_step_idx`, a 25-value
`action_block`, and only four model steps per episode. It does not contain the
literal `step_idx`, per-raw-step `action`, and `proprio` columns required by
the standard sequence trainer. The table remains useful for the frozen
benchmark protocol, but it is not a direct `CW_DATASET` input. A future
`cube_block_projection_to_sequence_v1` adapter must reconstruct and validate
the raw sequence; renaming columns would be scientifically incorrect.

The five tables were checked with the frozen internal Lance reader using
`num_steps=4` and `frameskip=5`; their sampled tensor geometry is correct. The
pinned public revision was separately checked at source level and rejects the
per-step string metadata described above. Neither check is a claim that every
model family or training recipe has completed optimization.

This keeps source data, staged benchmark data, and generated checkpoints
as three separate roots. The launcher prints all three before starting.

## Deliberate exclusions

The clean exporter does not copy:

- Public Test examples or evaluation payloads;
- model checkpoints, training logs, or experiment-tracker metadata;
- Stable-WorldModel or other third-party source checkouts;
- the original LeWM datasets;
- historical staging, recovery, diagnostics, or failed-run directories.

The generated `VERSION.json` remains `staging_not_public_release`. A final
public release additionally requires a citation file, stable Hugging Face
revision, clean-environment loading test, and synchronized reference-results
package.
