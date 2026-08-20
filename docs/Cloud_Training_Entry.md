# Cloud training entry point

The cloud job template ends in:

```bash
bash ${run_shell_script} "$@"
```

so the platform holds exactly **one** script path. Point `run_shell_script` at
`scripts/cloud_train.sh` once, and switch tasks with environment variables
instead of creating a job configuration per task.

## Job configuration

| variable | value |
|---|---|
| `work_dir` | the ContextWorld checkout |
| `run_shell_script` | `scripts/cloud_train.sh` |

Then per run:

| variable | default | meaning |
|---|---|---|
| `CW_TASK` | *(required)* | one of the nine benchmark tasks |
| `CW_FAMILY` | `lewm` | `lewm`, `pldm` or `prejepa` |
| `CW_SEED` | `3072` | training seed |
| `CW_MODE` | `preflight` | mode for the shell-backed tasks |
| `CW_STAGE` | `paired` | `action_delay` only: `paired` or `curriculum` |
| `CW_VARIANT` | recipe of record | override the launcher's variant |
| `CW_DATASET` | — | required for `prejepa` |
| `CW_OUTPUT` | under the artifact root | output directory |
| `CW_BATCH_SIZE` | family default | see below |
| `CW_PRINT_ONLY` | unset | resolve and print without running |

Anything after `--` is forwarded to the underlying launcher untouched.

## Examples

```bash
CW_TASK=speed CW_FAMILY=lewm CW_MODE=formal bash scripts/cloud_train.sh
CW_TASK=door CW_FAMILY=pldm CW_MODE=formal bash scripts/cloud_train.sh
CW_TASK=action_delay CW_FAMILY=lewm CW_STAGE=paired bash scripts/cloud_train.sh
CW_TASK=action_delay CW_FAMILY=lewm CW_STAGE=curriculum bash scripts/cloud_train.sh
CW_TASK=contact_friction CW_FAMILY=pldm bash scripts/cloud_train.sh

# a new family, same interface
CW_TASK=speed CW_FAMILY=prejepa CW_DATASET=/path/to/data \
    bash scripts/cloud_train.sh
```

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
| `prejepa`, any task | uniform, through `run_prejepa_train.py` |

The door variant names and the action_strength recipe string are asserted
against the frozen release configs in `tests/test_cloud_train.py`, so a change
there fails a test rather than quietly training a different mixture.

## Batch size

`lewm.yaml` and `pldm.yaml` both ship `batch_size: 128` with no gradient
accumulation, so **the baselines are aligned by upstream default and need no
override**. `prejepa.yaml` ships `32`, so the router passes `128` for prejepa
runs unless `CW_BATCH_SIZE` says otherwise.

On eight GPUs that is an effective batch of 1024, which is what the frozen
configs record as `effective_global_batch`. If you have fewer GPUs than the
recipe assumed, `run_prejepa_train.py --accumulate N` reaches the same
effective batch.

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
