# Cloud training entry point

The canonical Stable-WorldModel options, family mappings, credential handling
and resume/evaluation behavior are documented in
[Stable-WorldModel training](StableWM_Training.md). This page focuses on the
cloud router and the distinction between original and benchmark data.

The cloud platform enters through `algorithm/lewm_ag/startup_cce.sh`. Set its
working directory to the absolute ContextWorld checkout and give it the
repository-relative training script:

| platform field | value |
|---|---|
| `work_dir` | absolute path to the ContextWorld checkout |
| `run_shell_script` | `scripts/cloud_train.sh` |

`startup_cce.sh` is supplied by the cloud runtime and is not part of this
repository. It changes into `work_dir` before invoking the relative
`run_shell_script`.

## Original DINO-WM training: one shared data root

Stable-WorldModel calls its DINO-WM training entry `prejepa`; that is why the
cloud family value below is `prejepa`.

The four standard original-data jobs do not require four hand-written dataset
paths. Set the dataset root, checkpoint root, and Stable-WorldModel checkout
once in the common job configuration:

```text
CW_TASK=original
CW_FAMILY=prejepa
CW_ALL_SEEDS=1
CW_MAX_EPOCHS=10
CW_BATCH_SIZE=128
CONTEXTWORLD_DATASET_ROOT=/absolute/path/data/world_model
CW_CHECKPOINT_ROOT=/absolute/path/checkpoints/dino-wm
CW_OUTPUT=/absolute/path/training-logs/dino-wm
CONTEXTWORLD_STABLE_WORLDMODEL_REPO=/absolute/path/stable-worldmodel
HF_HUB_CACHE=/absolute/path/huggingface-cache
```

Each job then changes only `CW_ENV`:

```text
CW_ENV=tworoom
CW_ENV=pusht
CW_ENV=reacher
CW_ENV=cube
```

With `CW_ALL_SEEDS=1`, each job runs seeds 3072, 3073 and 3074 in sequence.
The complete copy-ready job environment is recorded in
[`dinowm_original_cloud_v1.env.example`](../configs/training/dinowm_original_cloud_v1.env.example).
The checked-in template sets the comparison recipe recorded in the profile:
10 epochs and batch 128 for comparison with the existing LeWM/PLDM baselines.
The remaining model and optimizer defaults come from the pinned public
PreJEPA YAML.

The public family-profile launcher translates the environment into the
configuration form expected by the selected model family. LeWM and PLDM choose
a Hydra `data` group; PreJEPA uses a flat `dataset_name`. Operators therefore
select an environment, not a YAML file or a dataset filename.

For DINO-WM/PreJEPA, the built-in mapping resolves these files below
`CONTEXTWORLD_DATASET_ROOT`; the middle column shows the corresponding
LeWM/PLDM data group:

| `CW_ENV` | LeWM/PLDM data group | PreJEPA relative dataset path |
|---|---|---|
| `tworoom` | `data=tworoom` | `quentinll/tworoom.h5` |
| `pusht` | `data=pusht` | `quentinll/pusht_expert_train.h5` |
| `reacher` | `data=dmc` | `quentinll/reacher.h5` |
| `cube` | `data=ogb` | `quentinll/ogbench/cube_single_expert.h5` |

`CW_DATASET=/absolute/path/to/file` remains available as a higher-priority
override for a one-off file or a non-standard layout. It is not needed for
the four jobs above. Both the root-based and exact-file forms are validated
before training, and a print-only run shows the resolved upstream command.

`CW_CHECKPOINT_ROOT` is exported as Stable-WorldModel's `STABLEWM_HOME`.
Every run is isolated below
`$CW_CHECKPOINT_ROOT/checkpoints/<environment>_<family>_original_s<seed>/`,
so all four environments and all three seeds can safely share one root.
`CW_OUTPUT` is a separate optional setting for Hydra run files; it is not the
checkpoint root.

For public reproduction or any mount outside the project's established
internal layout, set `CONTEXTWORLD_STABLE_WORLDMODEL_REPO` explicitly. Source
auto-discovery is only a local convenience and should not be treated as part
of a portable job configuration. The backbone cache may also be explicit:

| variable | meaning |
|---|---|
| `CONTEXTWORLD_STABLE_WORLDMODEL_REPO` | Stable-WorldModel source checkout containing `scripts/train/` |
| `HF_HUB_CACHE` | Hugging Face cache containing the DINOv2 backbone |

`CW_DATA_ROOT` remains an optional compatibility shortcut for local
auto-detection. It is not required when concrete paths are supplied and never
overrides them.

### Two data trees, deliberately separate

```
<data root>/data/world_model/
├── quentinll/                 original LeWM open data   <- CONTEXTWORLD_DATASET_ROOT
│   ├── tworoom.h5                 (resolved as quentinll/tworoom.h5)
│   ├── pusht_expert_train.h5
│   ├── reacher.h5
│   └── ogbench/cube_single_expert.h5
└── context_world/             ContextWorld's own data   <- CONTEXTWORLD_ARTIFACT_ROOT
    ├── synthesis/                 synthesized benchmark data
    ├── training/                  checkpoints and run logs
    └── upstream/                  the Stable-WorldModel source checkout
```

`CW_TASK=original` reads the first tree. The nine benchmark capabilities read
the second through `CONTEXTWORLD_ARTIFACT_ROOT`. Original-data training does
not need the ContextWorld artifact tree at all.

`CONTEXTWORLD_ARTIFACT_ROOT` matters especially in the cloud: without it,
`contextworld.paths.artifact_root` infers the location from the checkout
(`repo.parents[1]/data/world_model/context_world`), which is only correct
when `work_dir` happens to sit two levels below the data root.

The cloud commonly mounts the data root as `/opt/huawei/dataset/ag_data`; the
development box has an extra `explorer-env` segment. Those locations are only
fallback candidates, not the primary cloud interface.

Then per run:

| variable | default | meaning |
|---|---|---|
| `CW_TASK` | *(required)* | one of the nine benchmark tasks, or `original` |
| `CW_ENV` | — | with `CW_TASK=original`: `tworoom`, `pusht`, `reacher`, `cube` |
| `CW_FAMILY` | `lewm` | `lewm`, `pldm` or `prejepa` |
| `CW_SEED` | `3072` | training seed |
| `CW_ALL_SEEDS` | unset | run all three baseline seeds in sequence |
| `CW_MODE` | `preflight` | mode for the shell-backed tasks |
| `CW_STAGE` | `paired` | `action_delay` only: `paired` or `curriculum` |
| `CW_VARIANT` | recipe of record | override the launcher's variant |
| `CW_DATASET` | — | optional exact-file override for original training; required for benchmark PreJEPA |
| `CW_OUTPUT` | launcher default | optional per-run/Hydra output directory; not the Stable-WorldModel checkpoint root |
| `CW_CHECKPOINT_ROOT` | — | Stable-WorldModel cache/checkpoint root (`STABLEWM_HOME`) |
| `CW_BATCH_SIZE` | 128 for cloud PreJEPA; family YAML otherwise | see below |
| `CW_MAX_EPOCHS` | family YAML | training epochs |
| `CW_NUM_WORKERS` | family YAML | data loader workers |
| `CW_DEVICES` | family YAML | Lightning devices (`auto`, integer, or Hydra value) |
| `CW_LOGGER` | `none` | optional `wandb`; compatible LeWM/PLDM checkouts also support `swanlab` |
| `CW_RESUME` | `never` | `never`, `auto`, or `required` |
| `CW_POST_TRAIN_EVAL` | unset | optional original-environment MPC evaluation after successful training |
| `CW_PRINT_ONLY` | unset | resolve and print without running |

Anything after `--` is forwarded to the underlying launcher untouched. For a
family-profile run, prefer the typed `CW_*` variables; an uncommon raw Hydra
setting must be forwarded as `-- --override KEY=VALUE`.

For benchmark data exported with the clean Hugging Face layout, pass the
absolute training payload recorded in `task_registry.json` through
`CW_DATASET` only when `direct_stable_worldmodel_load` is true, or pass the
output of the adapter named by that registry entry. Do not assume every
component has the same shape: for example, PushT action strength uses
`training/data.lance`, while Action Delay has separate `training/coarse` and
`training/full` payloads. Do not point `CONTEXTWORLD_ARTIFACT_ROOT` at the
clean package; that variable continues to identify the internal
research/artifact tree used by frozen reference launchers. See
[Hugging Face dataset export](HF_Dataset_Export.md).

## Examples

```bash
CW_TASK=speed CW_FAMILY=lewm CW_MODE=formal bash scripts/cloud_train.sh
CW_TASK=door CW_FAMILY=pldm CW_MODE=formal bash scripts/cloud_train.sh
CW_TASK=action_delay CW_FAMILY=lewm CW_STAGE=paired bash scripts/cloud_train.sh
CW_TASK=action_delay CW_FAMILY=lewm CW_STAGE=curriculum bash scripts/cloud_train.sh
CW_TASK=contact_friction CW_FAMILY=pldm bash scripts/cloud_train.sh

# a benchmark capability after applying its registered data adapter
CW_TASK=action_strength CW_FAMILY=prejepa \
    CW_DATASET=/absolute/path/to/adapter-output.lance \
    bash scripts/cloud_train.sh
```

## Original task training

The four unmodified environments the nine capabilities are built on. This is
the baseline regime — same families, different data.

```bash
CW_TASK=original CW_ENV=tworoom CW_FAMILY=prejepa \
    CONTEXTWORLD_DATASET_ROOT=/abs/data/world_model \
    CW_CHECKPOINT_ROOT=/abs/checkpoints/dino-wm \
    bash scripts/cloud_train.sh

# all three baseline seeds in sequence
CW_TASK=original CW_ENV=pusht CW_FAMILY=prejepa CW_ALL_SEEDS=1 \
    CONTEXTWORLD_DATASET_ROOT=/abs/data/world_model \
    CW_CHECKPOINT_ROOT=/abs/checkpoints/dino-wm \
    bash scripts/cloud_train.sh
```

`CW_ENV` takes the environment, not a capability — `tworoom`, not `speed`.

### What the environments differ in

Verified by loading each dataset and composing each config, not read off
documentation:

| env | lewm/pldm group | prejepa `dataset_name` | action | aux column |
|---|---|---|---|---|
| tworoom | `data=tworoom` | `quentinll/tworoom.h5` | 2 | `proprio` (2) |
| pusht | `data=pusht` | `quentinll/pusht_expert_train.h5` | 2 | `proprio` (4) |
| reacher | `data=dmc` | `quentinll/reacher.h5` | 2 | `observation` (6) |
| cube | `data=ogb` | `quentinll/ogbench/cube_single_expert.h5` | 5 | `observation` (28) |

Three traps the launcher absorbs:

* **The data group names do not match the environment names.** Reacher is
  `dmc` and cube is `ogb`.
* **`prejepa.yaml` hardcodes `wm.encoding.proprio`**, and `prejepa.py` raises
  if an encoding key is missing from the dataset. Reacher and Cube carry
  `observation` instead, so the encoding is remapped for those two.
* **A relative dataset name resolves under `$STABLEWM_HOME/datasets/`.** An
  empty directory left by an interrupted download shadows the real file and
  silently re-downloads several GB. The original-task launcher avoids this by
  resolving PreJEPA's built-in dataset name against the absolute
  `CONTEXTWORLD_DATASET_ROOT`. `CW_DATASET` is only the exact-file override.

Action width is **not** passed — `lewm.py:285` and `prejepa.py:209` both
derive it from the loaded dataset. Before Hydra starts, ContextWorld opens the
H5 file read-only and checks the required top-level model columns and raw
action/auxiliary-observation widths. This is why Cube remains five-dimensional
while the other three original datasets remain two-dimensional.

### GPUs

`trainer.devices` is `auto` in all three family configs, and the launcher
only overrides it when `--devices` is passed. Leaving it unset uses every
visible GPU, which is normally what you want.

Set it only to reach a specific effective batch. The comparison recipe is
`batch × devices × accumulate = 1024`: batch 128 with accumulation 1 on eight
GPUs, or accumulation 2 on four GPUs. The four-GPU cloud variables are:

```text
CW_DEVICES=4
CW_BATCH_SIZE=128
CW_ACCUMULATE=2
```

For two GPUs, use `CW_ACCUMULATE=4`. Reducing per-device batch to 64 requires
doubling those accumulation values to preserve the same effective batch.

### The pretrained backbone

`prejepa` loads `facebook/dinov2-small` (22M params, patch 14, hidden 384)
through `transformers.AutoModel`. `dinov2_small` in the config is an alias
resolved by `BACKBONE_ALIASES` in `wm/prejepa/module.py`.

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download facebook/dinov2-small
```

Anything in `BACKBONE_ALIASES` works as a drop-in — `dinov2_base`,
`dinov2_large`, `dinov2_giant` — via `-- backbone.name=dinov2_base
backbone.type=dinov2_base`. Changing it is a second variable, so the
baseline-comparable runs should stay on `small`.

Note that a torch-hub `.pth` (e.g. `dinov2_vitb14_pretrain.pth`) will **not**
load: `from_pretrained` needs the HF layout of `config.json` plus
`model.safetensors`.

### The baseline already exists for lewm and pldm

`artifacts/evaluation/original_baseline_matrix_v1/` holds 4 environments ×
2 families × 3 seeds (3072/3073/3074), 300 evaluations per cell, frozen
2026-08-17. This matrix records completed evaluation of frozen checkpoints;
it neither requires nor authorizes retraining those reference checkpoints. A
new family needs **4 environments × 3 seeds = 12 runs** to report a comparable
mean ± std.

Check what a combination resolves to before spending a GPU on it:

```bash
CW_PRINT_ONLY=1 CW_TASK=door CW_FAMILY=pldm bash scripts/cloud_train.sh
```

## Why this is a router and not a wrapper

The nine tasks were built over months and do not share an interface. The
router exists to absorb that, and each divergence below is a command a
hand-written wrapper gets silently wrong:

| task | how family and seed arrive |
|---|---|
| `speed` | lewm reads `MODEL_VARIANT`/`TRAINING_SEED` from the environment; **pldm is a different program** (`run_pldm_reference_completion.py`) |
| `door` | positional variant spelled `fixed-mixed` / `pldm-mixed` — **not** `lewm` / `pldm` |
| `action_delay` | positional `$1` family, positional `$2` seed, **two ordered stages** |
| the five hidden-property tasks | `--model` / `--seed` / `--output` flags |
| `action_strength` | lewm needs `--variants mixed_dynamics_response_sigreg_0p02`; pldm is the completion program again |
| `prejepa`, any task | uniform, through the public `run_stablewm_train.py` family profile |

The door variant names and the action_strength recipe string are asserted
against the frozen release configs in `tests/test_cloud_train.py`, so a change
there fails a test rather than quietly training a different mixture.

## Batch size

`lewm.yaml` and `pldm.yaml` both ship `batch_size: 128` with no gradient
accumulation, so **the baselines are aligned by upstream default and need no
override**. `prejepa.yaml` ships `32`, so the router passes `128` for prejepa
runs, including original-environment runs, unless `CW_BATCH_SIZE` says
otherwise.

On eight GPUs that is an effective batch of 1024, which is what the frozen
configs record as `effective_global_batch`. If you have fewer GPUs than the
recipe assumed, `CW_ACCUMULATE=N` reaches the same effective batch.

## What it does not do

* **It does not validate modes or variants.** Those vocabularies belong to the
  launchers and change over time; guessing here would make valid runs
  unreachable.
* **It does not edit a launcher.** `run_h3_hidden_passage_train.sh` and
  `train_tworoom_step1.py` are byte-pinned by frozen release configs — routing
  above them is why this file exists rather than a new mode inside them.
* **It does not train.** Every command it emits is one you could type by hand.

Some launchers refuse to run because their reference matrix is already frozen
(`RuntimeError: The frozen reference matrix permits only its registered
reproduction run`). That is the launcher's own governance, not a routing
failure — the baseline in question has already been produced and published.
