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
runs isolated, and optionally start the applicable evaluation suite after
training.

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

Portable full-state recovery requires `stable-pretraining>=0.1.8`. For an
offline cloud installation, keep one compatible `stable_pretraining` wheel in
the package directory; multiple versions allow pip to select a runtime that
does not provide the required Manager interface. Version 0.1.8 also requires
Kornia. On Python 3.10, the verified offline set is
`stable-pretraining==0.1.8`, `kornia==0.8.2`, and `kornia_rs==0.1.14`.

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
`cube`. The default `CW_SEEDS=3072` runs one seed. Submit seeds 3072, 3073 and
3074 as separate scheduler jobs so each task keeps its recovery state separate.
A comma-separated list remains available for non-SLURM serial sweeps.
`CW_DATASET` is a higher-priority one-off override when the standard root
layout is not used.

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
| `SPT_CACHE_DIR` | StablePretraining full-state/requeue storage; the launcher sets it equal to `CW_CHECKPOINT_ROOT` |
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
| `CW_RESUME` | `auto` (default), `never`, or `required` |

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

`CW_CHECKPOINT_ROOT` is one persistent run root. ContextWorld sets both
`STABLEWM_HOME` and `SPT_CACHE_DIR` to that same absolute path and rejects a
different `SPT_CACHE_DIR`. This co-location does not merge formats:
Stable-WorldModel writes evaluation weights under `checkpoints/<run>/`, while
StablePretraining keeps its native full-state `last.ckpt` and
scheduler-requeue state under `runs/`.

The default is `CW_RESUME=auto`.

- `never`: require a fresh run directory.
- `auto`: resume full state either through StablePretraining's same-job
  scheduler requeue or from the newest `last.ckpt` carrying the same run name
  and immutable recipe identity. A completed target-epoch weight instead
  skips directly to post-training evaluation.
- `required`: require one of those two full-state paths and fail if neither is
  available.

`SPT_CACHE_DIR` persists StablePretraining's native recovery state. Automatic
same-job recovery still uses StablePretraining's SLURM index. For a newly
submitted job, ContextWorld locates the previous SPT UUID run by an immutable
`contextworld_run_identity_v1.json` marker and passes its `last.ckpt` to the
upstream `spt.Manager` with `weights_only=False`. This works across replacement
containers as long as `CW_CHECKPOINT_ROOT` is mounted persistently.

Exported `weights_epoch_*.pt` files contain model weights for evaluation; they
are not optimizer checkpoints. StablePretraining's `last.ckpt` contains the
model, optimizer, scheduler and progress state. The ContextWorld family entry
supplies this checkpoint to the unchanged upstream LeWM, PLDM or PreJEPA
trainer. If `auto` sees an incomplete run without an identity-matched
`last.ckpt`, it fails rather than silently starting again from epoch zero.
The one safe pre-check exception is a directory containing only the exact
immutable identity written by this launcher: no trainer state exists yet, so
the unchanged request may retry from the beginning.

## Optional post-training evaluation

`CW_POST_TRAIN_EVAL=1` calls `scripts/run_stablewm_eval.py --suite` only after
all requested training seeds succeed. In a comma-separated seed sweep, the
launcher finishes or resumes every training run before it starts the first
evaluation, so an evaluation failure cannot prevent a later seed from reaching
its requested checkpoint. The same evaluation script can be invoked later
against an existing checkpoint; training does not contain a second copy of the
evaluation logic.

This automatic hand-off applies to current family-profile runs: every
original-environment run and any component run supplied with an explicit
dataset. A historical LeWM/PLDM component reproduction selected by omitting
`CW_DATASET` keeps its frozen evaluator and does not accept
`CW_POST_TRAIN_EVAL`; that evaluator is part of the historical release
protocol.

With `CW_RESUME=auto` or `required`, an existing target-epoch weight can skip
training only when two records are present: StableWM's resolved `config.yaml`
and ContextWorld's `contextworld_training_identity_v1.json`. The latter binds
the complete Hydra override vector, dataset metadata, family profile, and the
relevant StableWM source/configuration tree to one digest. An exact match is
required; a same-named checkpoint from another recipe is rejected.
`CW_RESUME=never` never takes this shortcut. Runs created before this identity
record was introduced require the explicit `CW_EVAL_ONLY=1` path after manual
review. Evaluation evidence is immutable. Repeating the exact request reuses a
completed manifest only after its request digest and every output size/SHA-256
still match; no evaluator is run again. A failed or interrupted suite, a
changed request, or damaged output requires a new namespace with
`CW_EVAL_RESULT_SUBDIR=<name>`.

For an original-environment run, the suite runs that environment's MPC/CEM
evaluation and its registered benchmark components:

| original environment | benchmark ICL evaluations |
|---|---|
| TwoRoom | speed, door rule, action delay, portal exit |
| PushT | action strength, contact friction, motion damping |
| Reacher | arm mass |
| Cube | gripper-carry rule |

For a benchmark-component run, the suite runs that component's ICL evaluator
and the matching original-environment CEM retention check. Component training
therefore also needs either `CONTEXTWORLD_DATASET_ROOT` or an exact
`CW_EVAL_ORIGINAL_DATASET`. If neither is available, only the CEM step is
recorded as skipped; the ICL step still runs.

Contact Friction and Motion Damping can currently be evaluated only on their
Development splits; Public Test remains closed. Other ICL
steps use the component's registered external-evaluation path and are marked
as unofficial results; this command never adds or changes a scoreboard row.
Run it only after the training recipe and checkpoint rule are fixed, not as a
Public-Test model-selection loop.

For PreJEPA, the original-environment CEM path has been execution-validated on
TwoRoom, PushT, Reacher, and Cube, but remains checkpoint-level evidence rather
than a published benchmark score. ContextWorld invokes the upstream planner
through a small history-key bridge:
it supplies pixels plus the state stream declared by the checkpoint
(`proprio` or `observation`) and selects the upstream split pixel/state goal
objective. Before launch, it also requires the checkpoint's trained history
length and action block to match the planner request. This makes the planner
inputs and objective agree with the trained checkpoint; it does not turn the
smoke check into a published benchmark score.

The frozen v1 external ICL adapter has a narrower contract: pixel history and
actions only. Current original-dataset PreJEPA checkpoints additionally
require `proprio` or `observation`, so the suite records their ICL rows as
`not_compatible` and emits no score. This is an input-protocol mismatch, not a
model failure. Supplying zero state or widening the frozen ICL interface with
simulator state would make the comparison invalid. The suite also records
`not_compatible` when a checkpoint's trained history length differs from a
component's frozen protocol.

Useful variables are `CW_EVAL_EPOCH`, `CW_EVAL_NUM`, `CW_EVAL_SEEDS`,
`CW_EVAL_DEVICE`, `CW_EVAL_BATCH_SIZE`, `CW_EVAL_ORIGINAL_DATASET`,
`CW_EVAL_RESULT_SUBDIR` and `CW_EVAL_KEEP_VIDEOS`. `CW_EVAL_RESULT_SUBDIR` is a
new named directory below `eval_results/`; it never overwrites an earlier
attempt. If `CW_EVAL_EPOCH` is omitted, the launcher uses `CW_MAX_EPOCHS` or
the selected upstream YAML default.

The same suite can be run later without retraining by naming the exact saved
checkpoint:

```bash
python scripts/run_stablewm_eval.py --suite \
  --family prejepa \
  --original-env tworoom \
  --dataset /absolute/path/to/tworoom.h5 \
  --checkpoint /absolute/path/to/checkpoints/run/weights_epoch_10.pt \
  --stablewm-repo /absolute/path/to/stable-worldmodel \
  --training-seed 3072
```

Use `--component <id>` instead of `--original-env` for a component-trained
checkpoint. `--result-subdir <name>` (or `CW_EVAL_RESULT_SUBDIR=<name>` through
the training entry) creates a new immutable namespace below `eval_results/`
for a later attempt.

The evaluation profile carries the training `CW_FRAMESKIP` into the planner's
action block. This keeps the candidate-action width equal to the model's
action-encoder width when a run intentionally differs from the default of 5.

With the default namespace, all outputs are colocated with the evaluated
checkpoint:

```text
<checkpoint run>/eval_results/
├── original_cem/                 # original-environment CEM
├── benchmark_cem/<component>/    # component-training retention
├── benchmark_icl/<component>/result.json
└── manifest.json
```

The manifest records the checkpoint path and SHA-256, Stable-WorldModel
revision, commands, completed/skipped/failed status, and output identities.
Each original-CEM JSON receipt also records the success rate, successful
episode count, and evaluation time as typed values; the upstream text report
is retained alongside it.
Existing results are never overwritten. A completed suite is evaluation
evidence, not by itself a formal release or scoreboard registration. If one
evaluator returns a failure status, the suite still attempts the remaining
applicable steps and returns a nonzero status after recording all outcomes.

The evaluator reads the Stable-WorldModel revision directly from the
checkout's `.git/HEAD` and refs. This avoids Git `safe.directory` failures on
read-only cloud mounts while preserving the exact source identity required by
the model adapters. Keep `.git` metadata with the checkout used for
evaluation. `CW_STABLEWM_REF` can record a caller-supplied full SHA when a
separately identified source snapshot has no Git metadata; that value is an
assertion by the caller, not an independent source-tree verification.

## Independent sweeps are not distributed training

StablePretraining indexes recovery state by `JOB_ID[_ARRAY_TASK_ID]`, so
serializing multiple `CW_SEEDS` within one requeueable SLURM task is unsafe and
the launcher rejects it. Submit one seed per job or array task. Non-SLURM
serial sweeps remain supported. Each seed has a separate run identity; a newly
submitted non-SLURM job can resume each incomplete seed from its own
identity-matched `last.ckpt`. This is ContextWorld's portable new-job hand-off,
not StablePretraining's scheduler-index requeue. The legacy Stable-WorldModel
`run_trainer_batch.sh` assigns independent comma-separated runs to hosts; it
does not configure `torchrun`, ranks or a shared multi-node DDP process.
Distributed strategy remains a trainer/Lightning setting selected with
`CW_STRATEGY` and the cloud platform's own process topology.
