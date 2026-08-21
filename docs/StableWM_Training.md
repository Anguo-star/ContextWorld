# Stable-WorldModel training

ContextWorld provides one public training command for LeWM, PLDM and PreJEPA
(DINO-WM):

```bash
python scripts/run_stablewm_train.py --help
```

The command calls the selected Stable-WorldModel checkout's own trainer. It
does not copy or reimplement the model, forward pass, loss, optimizer or
training loop. Its job is narrower: select the correct family profile,
validate paths, translate common options to the family's Hydra schema, keep
runs isolated, and optionally start an original-environment MPC evaluation
after training.

The machine-readable profile is
[`configs/training/stablewm_family_profiles_v1.yaml`](../configs/training/stablewm_family_profiles_v1.yaml).
It pins the public upstream revision used to validate the interface and makes
the dataset and parameter mappings reviewable without reading the launcher.

## Public upstream checkout

Core reproduction does not require the ContextWorld team's private
Stable-WorldModel fork. Use the public checkout and revision recorded by the
profile:

```bash
git clone https://github.com/galilai-group/stable-worldmodel
cd stable-worldmodel
git checkout addbab40377da680dadbfbc90250fe749f6f57e3
uv sync --extra all
```

Run the ContextWorld entry from an environment containing the upstream
training dependencies, and point `CONTEXTWORLD_STABLE_WORLDMODEL_REPO` at
that checkout. Optional ContextWorld research extensions such as VISReg are a
separate capability: the launcher detects their YAML fields and rejects the
request when the selected public checkout does not implement them.

## Cloud entry

The cloud platform enters through `algorithm/lewm_ag/startup_cce.sh`. Set:

| platform field | value |
|---|---|
| `work_dir` | absolute path to the ContextWorld checkout |
| `run_shell_script` | `scripts/cloud_train.sh` |

For original-data training, the minimal job variables are:

```text
CW_TASK=original
CW_ENV=tworoom
CW_FAMILY=prejepa
CONTEXTWORLD_DATASET_ROOT=/absolute/path/data/world_model
CW_CHECKPOINT_ROOT=/absolute/path/stablewm-runs
CONTEXTWORLD_STABLE_WORLDMODEL_REPO=/absolute/path/stable-worldmodel
```

Change only `CW_ENV` for the other original tasks: `pusht`, `reacher`, or
`cube`. Set `CW_SEEDS=3072,3073,3074` to run the three frozen baseline seeds
sequentially; the default `CW_SEEDS=3072` runs once. `CW_DATASET` is a higher-priority
one-off override when the standard root layout is not used.

For the DINO-WM comparison run, use the checked-in cloud template:

```text
configs/training/dinowm_original_cloud_v1.env.example
```

It makes the scientific settings explicit: 10 epochs, history 3, one-step
prediction, frameskip 5, per-device batch 128, and an effective global batch
of 1024 on eight GPUs. On four GPUs set `CW_ACCUMULATE=2`; on two GPUs set it
to 4. These changes retain the effective batch while adapting only to the
available hardware.

Before allocating a GPU, render and validate the final upstream command:

```bash
CW_PRINT_ONLY=1 bash scripts/cloud_train.sh
```

Print-only mode does not create directories, authenticate a tracker, start
training, or run evaluation.

## Paths have separate meanings

| variable | meaning |
|---|---|
| `CW_DATASET` | exact source H5/HDF5 file or `.lance` table |
| `CONTEXTWORLD_DATASET_ROOT` | root used only to resolve the four built-in original datasets |
| `CW_DATASET_CACHE_ROOT` | Stable-WorldModel download/data cache (`LOCAL_DATASET_DIR`) |
| `CW_CHECKPOINT_ROOT` | Stable-WorldModel storage root; models are written below `<root>/checkpoints/` |
| `CW_OUTPUT` | Hydra logs and run files, written below `<output>/<run-name>/` |
| `HF_HUB_CACHE` | pretrained backbone cache, used by PreJEPA |

For `.lance`, “exact table” means that the path is physically addressable;
it does not guarantee training compatibility. The launcher preflight and the
export registry's `direct_stable_worldmodel_load` field are authoritative.

`CW_OUTPUT` is not a checkpoint directory. Changing it does not move or hide
model weights. Stable-WorldModel currently uses `STABLEWM_HOME` as a common
storage base, so `CW_CHECKPOINT_ROOT` is translated to that upstream variable;
ContextWorld passes an exact dataset path to prevent the checkpoint root from
also becoming the data source by accident.

For LeWM and PLDM, the selected task determines the upstream data YAML; users
do not need to name it separately for registered targets. PreJEPA uses its
flat dataset settings instead of a Hydra data group.

| Stable-WorldModel data group | registered targets |
|---|---|
| `tworoom` | original TwoRoom; speed, door, action delay, portal exit |
| `pusht` | original PushT; action strength, contact friction, motion damping |
| `dmc` | original Reacher; arm mass |
| `ogb` | original Cube; gripper carry |

`CW_DATASET` replaces only the concrete file/table path. It does not erase the
task's YAML selection, history length, action width, or encoding convention.
`CW_DATA_GROUP` remains available as an explicit override for a compatible
custom checkout.

The launcher accepts a single H5/HDF5 file or a single `.lance` table. A
directory containing several tables is a dataset collection, not one training
table, and is rejected with the discovered members listed. This matters for
the clean Hugging Face package: some components contain several split- or
condition-specific tables and require the training recipe to select them
explicitly.

The launcher opens both H5 and Lance schemas before allocating a trainer. For
H5 it checks the selected model columns plus raw action and auxiliary-input
widths. For Lance it also checks episode/step indices. A `.lance` suffix
therefore does not by itself imply trainer compatibility. If the selected
checkout uses the pinned public reader that forbids per-step string metadata,
the preflight detects that implementation and rejects affected tables with the
required projection named. The distributed Cube gripper-carry projection is
rejected for a separate reason: it stores blocked actions and model steps
rather than a native raw-step sequence, and requires the audited
`cube_block_projection_to_sequence_v1` adapter.

## Common training options

The following cloud variables are translated to each family's actual YAML
keys. Equivalent command-line flags are shown by `--help`.

| variable | purpose |
|---|---|
| `CW_SEEDS` | one seed, or a comma-separated sequence of seeds |
| `CW_RUN_NAME` | model name and checkpoint subdirectory |
| `CW_MAX_EPOCHS` | training epochs |
| `CW_BATCH_SIZE`, `CW_NUM_WORKERS` | input throughput |
| `CW_TRAIN_SPLIT`, `CW_FRAMESKIP` | data sampling geometry |
| `CW_HISTORY_SIZE`, `CW_NUM_PREDS` | context and prediction horizons |
| `CW_DEVICES`, `CW_ACCELERATOR`, `CW_STRATEGY` | Lightning execution |
| `CW_PRECISION`, `CW_ACCUMULATE` | numerical precision and gradient accumulation |
| `CW_GRADIENT_CLIP_VAL` | gradient clipping |
| `CW_LEARNING_RATE`, `CW_WEIGHT_DECAY` | optimizer overrides |
| `CW_FAST_DEV_RUN` | Lightning fast-development run |
| `CW_LIMIT_TRAIN_BATCHES`, `CW_LIMIT_VAL_BATCHES` | bounded smoke runs |
| `CW_RESUME` | `never` (default), `auto`, or `required` |

LeWM and PLDM additionally expose loader controls through
`CW_PERSISTENT_WORKERS`, `CW_PREFETCH_FACTOR` and `CW_PIN_MEMORY`. PreJEPA's
current trainer hard-codes those DataLoader choices, so the profile rejects
them instead of accepting an option that has no effect.
All three supported trainers require `CW_NUM_WORKERS` to be positive: their
active persistent-worker or prefetch settings are incompatible with zero
workers in PyTorch.

`CW_EMBED_DIM` maps to `embed_dim` for LeWM and `wm.embed_dim` for PLDM. It is
rejected for PreJEPA, whose representation width is determined by the selected
backbone. Action width is never passed: all three trainers derive it from the
loaded dataset. The preflight verifies that width against the selected task
instead of padding, truncating, or replacing it. The original Cube H5 dataset
therefore remains five-dimensional rather than inheriting a two-dimensional
assumption; this does not make the blocked benchmark projection a native
training sequence.

Advanced users may repeat `--override KEY=VALUE`. These raw Hydra overrides
are appended last and therefore win over the typed interface. They are useful
for upstream options that are not part of the stable public contract, but the
result should be recorded with the experiment because ContextWorld cannot
validate their scientific meaning. Dataset paths, run identity, Hydra output
paths and secret-like keys are excluded from this escape hatch; use their
typed options or standard secret environment variables.

## Family-specific options

The generic entry deliberately keeps objective options outside the common
namespace. LeWM exposes `CW_LEWM_REGULARIZER`, SIGReg weight and the basic
VISReg weight/projection/lambda controls. The launcher first checks that the
selected checkout actually declares those fields. The pinned public upstream
supports native SIGReg but does not contain the full ContextWorld VISReg and
conditional-regularizer extension; unsupported requests fail before GPU work.

Frozen ContextWorld reference recipes remain authoritative for released LeWM
and PLDM component checkpoints. The generic entry is suitable for original
baselines, new model families and reproducible public experiments; it does not
replace a component's registered objective and data-mixture launcher.

In the clean staging package, five components each have a single table with
native temporal columns, but they are not direct inputs to the pinned public
reader: PushT action strength, contact friction and motion damping; Reacher arm
mass; and TwoRoom portal exit. They require
`stablewm_step_metadata_to_episode_table_v1` for the pinned public reader,
because their string metadata is stored per step. Speed, door and action delay
are multi-table collections whose split roots require an explicit composition
recipe. Cube is a single table physically, but requires the raw-sequence
adapter described above. These distribution constraints do not affect the
four original H5 datasets used by the cloud baseline command.

## Experiment tracking and credentials

Tracking is optional and disabled by default:

```text
CW_LOGGER=none
```

| family | `none` | `wandb` | `swanlab` |
|---|---:|---:|---:|
| LeWM | yes | compatible checkout required | compatible checkout required |
| PLDM | yes | compatible checkout required | compatible checkout required |
| PreJEPA | yes | yes | compatible checkout required |

The public Stable-WorldModel revision pinned by this repository exposes only
a WandB hook for PreJEPA; it does not provide PreJEPA SwanLab logging. SwanLab
is available only with a separately identified compatible checkout whose
PreJEPA trainer calls `build_training_logger` and declares the corresponding
logger schema. ContextWorld detects that interface at launch and then supplies
the SwanLab or WandB configuration. For every family, an unsupported backend
fails before training instead of silently producing an untracked run.

For SwanLab, inject the standard secret `SWANLAB_API_KEY` through the cloud
platform. Do not place the key in a command-line option or YAML file. The
launcher reads it only from the child environment and authenticates through
the Python SDK, so the value is absent from process arguments, rendered
commands, Hydra configuration and launcher logs. Non-secret settings use
`CW_SWANLAB_PROJECT`, `CW_SWANLAB_WORKSPACE`, `CW_TRACKER_NAME`,
`CW_TRACKER_ID`, `CW_SWANLAB_LOGDIR` and `CW_SWANLAB_MODE`.

WandB uses the standard `WANDB_API_KEY` environment variable, plus
`CW_WANDB_PROJECT`, `CW_WANDB_ENTITY`, `CW_TRACKER_NAME` and `CW_TRACKER_ID`.

## Resume behavior

The new entry defaults to `CW_RESUME=never`. A non-empty checkpoint directory
then fails before training, preventing an old run from being resumed or
overwritten without notice.

- `never`: require a fresh run directory.
- `auto`: allow the selected trainer to resume if its full-state checkpoint
  exists.
- `required`: require
  `<CW_CHECKPOINT_ROOT>/checkpoints/<run>/<run>_weights.ckpt` before launch.

Exported `weights_epoch_*.pt` files contain model weights for evaluation; they
are not a guarantee that optimizer and scheduler state can be resumed. LeWM's
compatible extension can additionally set
`LEWM_SAVE_FULL_RESUME_EACH_EPOCH=1`. PLDM and PreJEPA do not currently make
the same per-epoch full-state guarantee.

## Optional post-training evaluation

`CW_POST_TRAIN_EVAL=1` starts the separate
`scripts/run_stablewm_eval.py` command only after training succeeds. This
stage runs Stable-WorldModel's original-environment MPC/CEM evaluator and
writes seed-specific logs, metrics and receipts below the checkpoint's
`eval_results/` directory. Existing outputs are never overwritten.

The automatic hand-off is enabled for the checkpoint families already
validated with this evaluator (LeWM and PLDM). PreJEPA can be trained now, but
automatic post-eval remains disabled until a real saved checkpoint completes
the family smoke test. This restriction does not affect training.

Useful variables are `CW_EVAL_EPOCH`, `CW_EVAL_NUM`, `CW_EVAL_SEEDS` and
`CW_EVAL_KEEP_VIDEOS`. If `CW_EVAL_EPOCH` is omitted, the launcher uses
`CW_MAX_EPOCHS` or the selected upstream YAML default.

The evaluation profile carries the training `CW_FRAMESKIP` into the planner's
action block. This keeps the candidate-action width equal to the model's
action-encoder width when a run intentionally differs from the default of 5.

This optional evaluation is not the published ContextWorld benchmark score.
ICL and component CEM results must still be produced by the component's frozen
evaluation protocol.

## Independent sweeps are not distributed training

Multiple values in `CW_SEEDS`, for example `3072,3073,3074`, run independent
seeds sequentially in one job. A platform may instead submit one job per
seed. The legacy Stable-WorldModel
`run_trainer_batch.sh` assigns independent comma-separated runs to hosts; it
does not configure `torchrun`, ranks or a shared multi-node DDP process.
Distributed strategy remains a trainer/Lightning setting selected with
`CW_STRATEGY` and the cloud platform's own process topology.
