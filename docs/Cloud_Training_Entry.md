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

`cloud_train.sh` performs path normalization and then always enters
`run_stablewm_train.py`, independent of model family. For modern training, the
entry composes the family profile with an explicit dataset or registered
runtime bundle view. Benchmark components default to the current
`joint_scratch_v1` comparison: LeWM, VIS-WM, PLDM and PreJEPA all train on the same
registered 50/50 original/synthetic view without loading an original-task
checkpoint. The old LeWM/PLDM launchers remain available only when
`CW_TRAINING_TRACK=historical_release` is set explicitly.

| request | execution selected by `run_stablewm_train.py` |
|---|---|
| original environment, any family | current family profile |
| benchmark component with an explicit `CW_DATASET` | current family profile |
| benchmark component, any built-in family, no `CW_DATASET` | `joint_scratch_v1` runtime view over `ContextWorld-v1` |
| benchmark component, LeWM/PLDM, `CW_TRAINING_TRACK=historical_release` | frozen historical recipe |

Huawei's `startup_cce.sh` may both export GUI custom parameters and repeat
them as command-line pairs such as `--CW_FAMILY prejepa`. `cloud_train.sh`
consumes those duplicate platform arguments before entering Python. Their
environment values remain authoritative, while ordinary lowercase options
such as `--max-epochs 10` are still accepted for local use. Secret parameter
values are never included in the filtering log.

## Base family and training method are orthogonal

Two independent public variables describe a run:

* `CW_FAMILY` selects the **base model family** — the backbone, forward pass,
  objective and optimizer supplied by the Stable-WorldModel checkout:
  `lewm`, `viswm`, `pldm` or `prejepa`.
* `CW_METHOD` selects the **training method overlay** applied on top of that
  family: `native` (default) or `coja_v1`.

A method is not a family. `coja_v1` adds no model parameter, encoder, adapter,
head or inference change; it enables the checkout's own one-step
conditional-joint loss keys and keeps publicly related Contact pairs together
inside the family's native flat batch, over the same registered 50/50
original/ContextWorld mixture. There is therefore no `lewm_coja` family value,
and each family keeps its own run directory and immutable training identity —
a non-native method appends its name to the default run name rather than
sharing the native run.

Current support matrix (`CW_TRAINING_TRACK=joint_scratch_v1`):

| `CW_METHOD` | `lewm` | `viswm` | `pldm` | `prejepa` | components |
|---|---|---|---|---|---|
| `native` | yes | yes | yes | yes | all nine benchmark components, and original tasks |
| `coja_v1` | yes | yes | yes | yes | `contact_friction` only |

Everything outside that matrix fails closed before training: another
component, `CW_TASK=original`, `CW_TRAINING_TRACK=historical_release`, an
operator-supplied `CW_DATASET`, or a payload/mixture override. `coja_v1` also
fails closed on a checkout whose family config does not expose
`loss.conditional_joint`; ContextWorld does not add a loss family of its own.
The support envelope lives in
[`stablewm_family_profiles_v1.yaml`](../configs/training/stablewm_family_profiles_v1.yaml),
not in launcher code.

```bash
# same base family choice, one extra orthogonal variable
CW_TASK=contact_friction CW_FAMILY=pldm CW_METHOD=coja_v1 \
    CONTEXTWORLD_DATASET_ROOT=/abs/data/world_model \
    CONTEXTWORLD_BENCHMARK_ROOT=/abs/data/world_model/ContextWorld-v1 \
    CW_CHECKPOINT_ROOT=/abs/checkpoints/pldm-contextworld-v1 \
    bash scripts/cloud_train.sh
```

Swap `CW_FAMILY=pldm` for `lewm`, `viswm` or `prejepa` to run the same overlay on
another base family; drop `CW_METHOD` to get the family's native objective.

`CW_FAMILY=viswm` selects the independent VIS-WM entry and its published
VISReg defaults. LeWM remains prediction MSE + SIGReg and has no VISReg
selector. Optional VIS-WM ablations use `CW_VISWM_WEIGHT`,
`CW_VISWM_NUM_PROJECTIONS`, and `CW_VISWM_LAMBDA_{SCALE,SHAPE,CENTER}`; none is
needed for the method-of-record recipe.

## Original DINO-WM training: one shared data root

Stable-WorldModel calls its DINO-WM training entry `prejepa`; that is why the
cloud family value below is `prejepa`.

The four standard original-data jobs do not require four hand-written dataset
paths. Set the dataset root, checkpoint root, and Stable-WorldModel checkout
once in the common job configuration:

```text
CW_TASK=original
CW_FAMILY=prejepa
CW_SEEDS=3072
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

`CW_SEEDS` accepts one seed or a comma-separated list. Submit 3072, 3073 and
3074 as separate scheduler jobs so each task keeps its recovery state separate.
Non-SLURM launchers may use `CW_SEEDS=3072,3073,3074` for a serial sweep;
omitting the variable runs only seed 3072. For such a sweep, all training runs
finish or resume before post-training evaluation begins. Each seed keeps its
own immutable recipe identity and full-state `last.ckpt`.
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

### Data roles (the historical archive is optional)

```
<data root>/data/world_model/
├── quentinll/                 original LeWM open data   <- CONTEXTWORLD_DATASET_ROOT
│   ├── tworoom.h5                 (resolved as quentinll/tworoom.h5)
│   ├── pusht_expert_train.h5
│   ├── reacher.h5
│   └── ogbench/cube_single_expert.h5
├── ContextWorld-v1/           clean benchmark bundle    <- CONTEXTWORLD_BENCHMARK_ROOT
│   ├── task_registry.json
│   └── components/
└── context_world/             optional historical archive <- CONTEXTWORLD_ARTIFACT_ROOT
    ├── synthesis/                 synthesized benchmark data
    ├── training/                  checkpoints and run logs
    └── upstream/                  the Stable-WorldModel source checkout
```

`CW_TASK=original` reads the first tree. Current benchmark runs for all three
built-in families compose a runtime view from the first two trees. Post-training ICL evaluation
also reads the clean bundle's Development payloads, while CEM keeps using the
original data. The internal artifact tree is only for frozen LeWM/PLDM release
reproduction. Original-data training without post-evaluation needs neither
ContextWorld tree.

`CONTEXTWORLD_ARTIFACT_ROOT` is therefore only needed for a frozen historical
release recipe. When it is needed, keep it explicit: inferring it from the
checkout (`repo.parents[1]/data/world_model/context_world`) is unreliable in a
cloud mount.

The cloud commonly mounts the data root as `/opt/huawei/dataset/ag_data`; the
development box has an extra `explorer-env` segment. Those locations are only
fallback candidates, not the primary cloud interface.

Then per run:

| variable | default | meaning |
|---|---|---|
| `CW_TASK` | *(required)* | one of the nine benchmark tasks, or `original` |
| `CW_ENV` | — | with `CW_TASK=original`: `tworoom`, `pusht`, `reacher`, `cube` |
| `CW_FAMILY` | `lewm` | base method family: `lewm`, `viswm`, `pldm` or `prejepa` |
| `CW_METHOD` | `native` | training method overlay applied to that family: `native` or `coja_v1` (currently `contact_friction` on `joint_scratch_v1` only) |
| `CW_TRAINING_TRACK` | `joint_scratch_v1` | current component comparison; use `historical_release` only to reproduce an old frozen LeWM/PLDM release |
| `CW_SEEDS` | `3072` | one seed, or a comma-separated sequence such as `3072,3073,3074` |
| `CW_MODE` | `preflight` | mode for the shell-backed tasks |
| `CW_STAGE` | `paired` | `action_delay` only: `paired` or `curriculum` |
| `CW_VARIANT` | recipe of record | override the launcher's variant |
| `CW_DATASET` | — | optional exact-file override; omit it for the standard registered component view |
| `CONTEXTWORLD_BENCHMARK_ROOT` | `<CONTEXTWORLD_DATASET_ROOT>/ContextWorld-v1` | clean export root used by current component training and every Development ICL suite |
| `CW_COMPONENT_PAYLOAD` | task profile | optional registered payload override; Action Delay supports `coarse` or `full` |
| `CW_MIX_ORIGINAL_WEIGHT`, `CW_MIX_SYNTHETIC_WEIGHT` | task profile | optional benchmark mixture override |
| `CW_COMPONENT_EPOCH_SIZE` | balanced full coverage | optional virtual samples per training epoch |
| `CW_OUTPUT` | launcher default | optional per-run/Hydra output directory; not the Stable-WorldModel checkpoint root |
| `CW_CHECKPOINT_ROOT` | — | Stable-WorldModel cache/checkpoint root (`STABLEWM_HOME`) |
| `CW_BATCH_SIZE` | 128 for cloud PreJEPA; family YAML otherwise | see below |
| `CW_MAX_EPOCHS` | family YAML | training epochs |
| `CW_NUM_WORKERS` | `2` per DDP process for any ContextWorld-v1 view; family YAML otherwise | data loader workers |
| `CW_DEVICES` | family YAML | Lightning devices (`auto`, integer, or Hydra value) |
| `CW_LOGGER` | `none` | `wandb` or `swanlab` when the selected family trainer uses the common logger factory |
| `CW_RESUME` | `auto` | `never`, `auto`, `required`, or `reset`; `reset` preserves the run name, archives its exact local state, and starts from epoch zero |
| `CW_POST_TRAIN_EVAL` | unset | for current family-profile runs, run the complete applicable original CEM and benchmark ICL suite; the standard CEM default is 50 episodes × seeds 42–47 = 300 episodes per checkpoint; component runs fail early if the matching original dataset cannot be resolved; frozen historical reproductions retain their component evaluator |
| `CW_EVAL_NUM` | `50` | CEM episodes per evaluation seed; changing it produces a non-standard diagnostic or repair run |
| `CW_EVAL_SEEDS` | `42,43,44,45,46,47` | six standard CEM evaluation seeds; changing them produces a non-standard diagnostic or repair run |
| `CW_EVAL_ONLY` | unset | skip training/resume and evaluate an existing family-profile checkpoint selected by `CW_ENV`, `CW_FAMILY`, `CW_SEEDS` and `CW_EVAL_EPOCH`/`CW_MAX_EPOCHS` |
| `CW_EVAL_RESULT_SUBDIR` | unset | new immutable name below `eval_results/` for a later evaluation attempt |
| `CW_PRINT_ONLY` | unset | resolve and print without running |

Arguments given to `cloud_train.sh` are parsed by the same public
`run_stablewm_train.py` entry used outside the cloud. Prefer the typed `CW_*`
variables; an uncommon upstream Hydra setting can be passed with a repeated
`--override KEY=VALUE` option.

For a `ContextWorld-v1` training view, the launcher uses the multiprocessing
`spawn` method and defaults to two workers in each DDP process. Worker counts
are per process, not per job: on eight GPUs, `CW_NUM_WORKERS=16` would create
up to 128 data-loading workers. Override the default only after measuring the
host's CPU, memory and file-handle capacity.

Post-training evaluation needs `CONTEXTWORLD_BENCHMARK_ROOT` because every ICL
step reads the clean export's Development payload. It does not read Public Test
or `CONTEXTWORLD_ARTIFACT_ROOT`; CEM continues to read the matching original
dataset. Results are written beside each checkpoint under `eval_results/`; see
[Stable-WorldModel training](StableWM_Training.md#optional-post-training-evaluation)
for the exact layout and execution matrix.

The `50 × 6` budget applies to CEM only. ICL consumes the deterministic
Development selection registered for each component; it is not resampled to a
common seed/episode geometry.

`CW_CHECKPOINT_ROOT` is a persistent run root: the cloud entry sets both
`STABLEWM_HOME` and `SPT_CACHE_DIR` to it. Evaluation weights live in
`checkpoints/`; StablePretraining's native recovery and scheduler-requeue state
live in `runs/`. StablePretraining handles a same-job scheduler requeue
directly. ContextWorld also marks each SPT UUID run with its run name and
immutable recipe identity. A newly submitted job can therefore locate the
newest matching `last.ckpt` and pass it back to the upstream `spt.Manager` with
full-state semantics. No scheduler ID is copied or forged, and replacement
containers work as long as the checkpoint root remains mounted. When a
requested epoch weight already exists,
post-training evaluation skips training only under `CW_RESUME=auto` or
`required` and only after the saved `config.yaml` and
`contextworld_training_identity_v1.json` prove an exact recipe match. Older
checkpoints without that launcher identity are never accepted automatically;
use `CW_EVAL_ONLY=1` after reviewing them. Set `CW_EVAL_RESULT_SUBDIR` to
preserve a failed, interrupted, or changed evaluation attempt and write new
evidence separately. An exact repeat of a completed suite is accepted
idempotently only when its request digest matches and every output file still
has its recorded size and SHA-256.

Use `CW_RESUME=reset` when a changed recipe must replace a failed attempt
without changing its run name. Before training, the launcher moves that exact
run's `checkpoints/<run>`, marker-bound StablePretraining UUID directories,
and current `CW_OUTPUT/<run>` into timestamped, recoverable
`.contextworld_reset_archive/` directories. Other runs and seeds are not
changed. `CW_PRINT_ONLY=1` previews the reset without moving anything, and
`reset` cannot be combined with `CW_EVAL_ONLY=1`.

For PreJEPA, the CEM smoke uses the upstream planner with the checkpoint's
declared history stream. Its Development ICL eligibility is checked separately;
a state-conditioned original checkpoint keeps a Development `not_compatible`
row and receives a separate normalized-zero diagnostic score. Benchmark-
component PreJEPA runs remove the state encoder and must complete the matching
RGB/action Development ICL row. The two result tracks are never merged. See
the training guide for the exact boundary.

For current benchmark training with LeWM, PLDM or PreJEPA, point
`CONTEXTWORLD_BENCHMARK_ROOT` at the clean export and omit `CW_DATASET`. The
launcher verifies the manifest and registry, selects the component payload,
and registers the same lazy reader in every DDP rank. Do not point
`CONTEXTWORLD_ARTIFACT_ROOT` at the clean package; that variable identifies
the internal research tree used only by explicit historical reproduction. See
[Hugging Face dataset export](HF_Dataset_Export.md).

## Examples

```bash
# current joint-from-scratch component comparison; choose any built-in family
CW_TASK=action_strength CW_FAMILY=lewm \
    CONTEXTWORLD_DATASET_ROOT=/abs/data/world_model \
    CONTEXTWORLD_BENCHMARK_ROOT=/abs/data/world_model/ContextWorld-v1 \
    CW_CHECKPOINT_ROOT=/abs/checkpoints/lewm-contextworld-v1 \
    bash scripts/cloud_train.sh

# old protocol evidence only; this does not enter the current comparison
CW_TASK=door CW_FAMILY=pldm CW_TRAINING_TRACK=historical_release \
    CONTEXTWORLD_ARTIFACT_ROOT=/abs/data/world_model/context_world \
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

# optional non-SLURM serial sweep; use one seed per requeueable scheduler job
CW_TASK=original CW_ENV=pusht CW_FAMILY=prejepa CW_SEEDS=3072,3073,3074 \
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

`lewm.yaml`, `viswm.yaml` and `pldm.yaml` all ship `batch_size: 128` with no gradient
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
