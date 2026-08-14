# Cube Gripper Carry History=3 v4r1 Reference-Training Protocol

Status: `preregistered_before_reference_training`. This protocol authorizes a
fixed LeWM/PLDM Development matrix on the already frozen v4r1 data. It does not
authorize Public Test generation, access, hashing, scoring, suite registration,
or a release claim.

## Starting evidence

The v4r1 Training/Development build and its frozen RGB-history probe have
passed their preregistered data-readiness gates. The retained build contains
2,048 Training pairs and 256 split-disjoint Development pairs over the four
balanced action-template families `endpoint4`, `front_hold`, `plateau`, and
`ramp4`. Every action profile satisfies `sum(p)=0` and `p[-1]=0`; paired modes
share query pixels, physical query state, and actions, while their histories
and real futures differ. No Public split was generated.

The data-readiness decision explicitly ran zero optimizer steps and requires a
separate model preregistration. This document and its machine-readable YAML are
that separate preregistration. They do not reinterpret the RGB probe as a
reference-model result.

## Fixed model matrix

The complete matrix is frozen before any v4r1 model result exists:

- model families: LeWM and PLDM;
- training seeds: 17321, 17322, and 17323;
- six jobs total, with no adaptive stopping or result-dependent recipe change;
- LeWM recipe: `mixed_frozen_image_paired_future_fit_1p00`;
- PLDM recipe: `mixed_pldm_joint`;
- 4,096 AdamW optimizer steps and fixed final checkpoint step 4,096;
- monitor-only snapshots at 512, 1,024, 2,048, and 4,096 steps;
- batch size 128: 64 original Cube rows and 64 capability rows arranged as 32
  complete pairs;
- four training data-loader workers and monitor batch size 64;
- learning rate `5e-5`, weight decay `1e-3`, gradient clip norm `1.0`, and
  bfloat16 on CUDA;
- no Development-based checkpoint selection and no scientific CLI overrides.

The seed block is a fresh Cube-specific block selected before v4r1 model
training. The recipes, optimizer, batch composition, fixed step, and thresholds
reuse the existing Cube reference contract rather than being tuned against the
new Development split.

## Five-axis action contract

Cube exposes five raw action axes and each model action block contains five raw
steps. The flattened action-encoder input is therefore 25, not 10 and not 2.
The wrapper must bind all shared layers before materialization or model
instantiation:

- shared trainer `ACTION_INPUT_DIM=25`;
- pilot `ACTION_DIM=5` and `ACTION_INPUT_DIM=25`;
- mixed engine `ACTION_INPUT_DIM=25`;
- H5 normalizer source shape `(rows, 5)`;
- Lance capability blocks ending in `(5, 5)`.

Any two-axis normalizer, ten-value action tensor, or non-5x5 capability block
fails before the first optimizer step.

## Freeze and input identity

The training freezer must run before any job. It verifies and records:

- the exact preregistration, protocol, loader, trainer, matrix runner,
  Development scorer/CLI, adapters, shared trainer/engine, tests, and package
  identities;
- the v4r1 manifest, request, build report, success marker, recovery/prior
  receipts, RGB probe, and data-readiness decision;
- the complete generated-data tree hash and the independent Training and
  Development Lance table-tree hashes;
- absence of `validation.lance` and the closed Public declaration;
- the original 100 GB Cube H5 content hash, row/episode counts, and five-axis
  action shape;
- all four original Cube Lance file hashes, sizes, schema, row count, and
  five-axis action column;
- both initialization checkpoint hashes and sizes;
- a clean isolated Stable-WorldModel worktree at commit
  `875e607fc08aa72eacb94d5d178127804134cc06`, all required runtime file
  hashes, and strict CPU loading of both checkpoints with a 25-value action
  encoder;
- absence of the new training and Development-score output roots.

The freezer writes one x-exclusive receipt. The formal trainer refuses to run
without that receipt or after preregistration, code, generated-data, upstream
input, checkpoint, or runtime drift. The shared released PushT trainer remains
byte-for-byte unchanged; Cube's old-runtime compatibility is process-local to
the Cube wrapper. The receipt authorizes only the six fixed jobs and
Development scoring.

## Development scoring

Each fixed step-4096 checkpoint is independently scored on
`loader_validation.lance` with frozen inference batch size 64. The scorer never
resolves or opens a validation or Public table. Every score is bound to the
exact matrix cell, training-report hash chain, checkpoint file hash, and model
state hash; relabeling another checkpoint as an authorized family/seed fails.
Per checkpoint it requires:

- correct-future rate at least 0.75;
- correct-history rate at least 0.75;
- context-switch rate at least 0.90;
- worst-rule correct-future rate at least 0.70;
- separated paired target latents;
- response gain at least 0.50;
- normalized response error strictly below 1.00;
- paired-bootstrap lower bounds of 0.70 for future/history and 0.85 for
  context switching.

Raw latent MSE is never compared across LeWM and PLDM. A model family passes
only when all three distinct checkpoints pass every gate. Family decisions are
independent; at least one complete family is required to proceed toward a
release. A failed family remains a reported negative reference result and may
not be rescued by its mean score.

## Stop rule and later stages

If the freeze fails, any job exits nonzero, provenance drifts, or neither model
family passes 3/3, stop with `failed_development`. Do not alter the recipe,
threshold, seed, checkpoint step, or data under this preregistration.
CUDA visibility and frozen inputs are preflighted before the one-use training
root is created; all six completed training cells are preflighted before the
one-use score root is created.

If at least one family passes, write an immutable Development decision first.
Original Cube CEM retention still requires a separate post-Development
authorization and paired baseline/query protocol. Public Test remains closed
through CEM. Only a later release freeze may authorize one Public evaluation
for the families that passed both Development and retention.

## Explicitly prohibited

- Public/Test generation, opening, reading, decoding, hashing, or scoring;
- using Development to select a checkpoint step;
- overriding the data root, upstream inputs, initialization checkpoint,
  variant, optimizer steps, or contrast scales;
- training with a two-axis or ten-value action path;
- continuing after identity or provenance drift;
- claiming release readiness from data readiness or a single training seed;
- rerunning failed jobs under changed settings without a new preregistration.
