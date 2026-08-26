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

`task_registry.json` records each component's environment, capability class,
history length, action dimension, frozen release-config digest, and provenance
mapping. Each payload also records its public path, layout, table count,
members, sequence schema, required adapter, and whether the path is a direct
Stable-WorldModel training input. The registry additionally binds the public
Development payload, deterministic selection rule, action normalization, and
whether the result is a matched comparison or a diagnostic. The portable
TwoRoom normalizer is itself covered by the manifest. `single_dataset_entrypoint`
describes only physical layout; it is not by itself a training-compatibility
claim.
`manifest.jsonl` records the relative path, size, SHA-256 digest, component,
and split of every distributed file.

## Training paths

The clean export replaces the internal artifact tree for every current public-
facing ContextWorld Training and Development input. Set
`CONTEXTWORLD_BENCHMARK_ROOT` to this directory; do not point
`CONTEXTWORLD_ARTIFACT_ROOT` at it. The latter variable is retained only for an
explicit, authorized reproduction of a historical frozen LeWM/PLDM release.

Benchmark training for all three built-in StableWM families can consume the
clean tree directly through the ContextWorld launcher:

```text
CW_TASK=action_strength
CW_FAMILY=lewm
CW_TRAINING_TRACK=joint_scratch_v1
CONTEXTWORLD_DATASET_ROOT=/absolute/path/data/world_model
CONTEXTWORLD_BENCHMARK_ROOT=/absolute/path/data/world_model/ContextWorld-v1
CW_CHECKPOINT_ROOT=/absolute/path/checkpoints/lewm-contextworld
```

When the bundle is located at
`<CONTEXTWORLD_DATASET_ROOT>/ContextWorld-v1`, the benchmark-root variable is
optional. Do not set `CW_DATASET` for this path. The launcher creates a compact
`contextworld://` identifier that binds the manifest, registry, selected
payload and mixture recipe. StableWM opens the registered member tables
lazily, so the export is neither rewritten nor duplicated.

Five components expose one physically addressable sequence table per payload:
PushT action strength, contact friction and motion damping; Reacher arm mass;
and TwoRoom portal exit. Their temporal and model-input columns are valid, but
the tables also contain string-valued metadata on every step. The pinned public
Stable-WorldModel reader rejects that legacy layout before applying
`keys_to_load`. ContextWorld's
`stablewm_step_metadata_to_episode_table_v1` runtime view reads only the numeric
model columns while leaving the distributed metadata unchanged. The physical
table remains non-direct; the registered launcher view is the public training
entry.

TwoRoom speed, door and action delay are collections of Lance tables. Their
split roots are not direct `CW_DATASET` entry points: speed contains nested
regime/shard tables, door contains multiple scenario tables, and action delay
contains separate coarse/full collections. The runtime reader uses the exact
member list from `task_registry.json` and balances scenarios deterministically.
For Action Delay `full`, delays 0 through 4 and the combined 5–10 group receive
equal top-level weight.

Cube gripper carry is a different case. Its distributed table is a
blocked-transition projection with `model_step_idx`, a 25-value
`action_block`, and only four model steps per episode. It does not contain the
literal `step_idx`, per-raw-step `action`, and `proprio` columns required by
the standard sequence reader, so it is not a direct `CW_DATASET` input. The
`cube_block_projection_to_sequence_v1` runtime view preserves the four model
steps, treats each block as five five-dimensional raw actions for
normalization, and then exposes the expected 25 action features. It does not
invent missing raw frames or rename the projection into a different dataset.

The runtime views are wired to LeWM, PLDM and PreJEPA as reference StableWM
integrations. This list does not restrict the dataset: another model can read
the manifest-bound Training payloads with its own loader and implement
`LatentWorldModelAdapter` for Development scoring. Runtime compatibility is
not a claim that every model family or training recipe has completed
optimization.

This keeps original environment data, the staged benchmark bundle, and
generated checkpoints as three separate roles. The launcher prints each role
that the selected run actually uses; the private research archive is not a
normal input to current Training or Development evaluation.

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
