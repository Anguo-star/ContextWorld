# Cube Gripper Carry History=3 v4r1 Public Release v1 Protocol

Status: `preregistered_before_public_generation_or_access`. This protocol
authorizes one deterministic Public data generation and, only after successful
publication and integrity validation, one fixed LeWM three-checkpoint scoring
campaign. It does not authorize training, checkpoint selection, threshold
changes, retries after Public access, or Suite registration.

## Starting evidence

The v4r1 non-Public data build passed with 2,048 Training pairs and 256
split-disjoint Development pairs. Both splits are balanced over `endpoint4`,
`front_hold`, `plateau`, and `ramp4`; every action perturbation satisfies
`sum(p)=0` and `p[-1]=0`. No Public split was generated or opened.

The frozen reference Development decision is `passed_development`: LeWM passed
3/3 checkpoints and PLDM passed 0/3. The subsequent original-Cube CEM decision
is `passed_retention`: the three LeWM checkpoints passed 3/3 against the frozen
baseline/query protocol. PLDM is therefore excluded from this Public campaign.

## Pre-access freeze

Before any Public generation, opening, hashing, or scoring, the freezer must
verify and bind:

- the exact v4r1 data-readiness, reference Development, and CEM-retention
  decisions and their preregistration/freeze chains;
- the restored byte-exact Reference Training v3 protocol;
- all Public builder, authorization, scoring, runner, finalizer, shared physics,
  shared data/scoring, adapter, path, protocol, and test implementations;
- the clean isolated Stable-WorldModel worktree at commit
  `875e607fc08aa72eacb94d5d178127804134cc06` and its required LeWM files;
- the exact three LeWM step-4096 checkpoint files and model-state identities;
- absence of the one-use Public data, score, and final-decision paths;
- the still-closed Public state in every prerequisite.

The freeze receipt is written x-exclusively. It records its logical path rather
than attempting to embed its own impossible self-hash. Each later entrypoint
rehashes the current receipt and binds that observed identity into its output.

## Fixed Public data

The only authorized split is `validation`, representing Public Test. It
contains 256 pairs (512 model conditions) and is generated from a fixed pool of
512 candidates with:

- catalog offset `3000000`;
- candidate-assignment seed `2026081400`;
- catalog seed `2026081401`;
- action-profile seed `2026081402`;
- JPEG quality 95 and 16 workers;
- the four templates balanced at 64 accepted pairs each;
- paired `cannot_hold` and `can_hold` trajectories sharing the query pixels,
  query physical state, and complete five-axis query action;
- three history frames and one real simulator future frame;
- action tensor shape `(4, 5, 5)` and flattened model action input 25;
- zero overlap with every prior source episode, exact action profile, scene
  template, pair content, or query-pixel identity.

The exclusion union covers the historical receipt plus the retained v4r1
Training and Development splits. It is frozen before Public candidate
assignment. The complete 100 GB source H5 is rehashed immediately before that
assignment. Generation uses local `/tmp` staging, validates the Lance tree, and
publishes `_SUCCESS.json` last. Integrity validation may open generated data;
model scoring may not begin unless publication succeeds.

## Fixed Public evaluation

Public data is loaded once and reused read-only for the complete fixed matrix:

- family: LeWM only;
- training seeds: 17321, 17322, 17323;
- recipe: `mixed_frozen_image_paired_future_fit_1p00`;
- checkpoint: the exact final step-4096 file for each seed;
- devices: `cuda:0`, `cuda:1`, and `cuda:2`, assigned in seed order;
- inference batch size: 64;
- original Cube five-axis finite-action population normalization;
- zero online environment calls;
- no adaptation, checkpoint selection, or recipe change.

This artifact is a reproducible local Public Test, matching the existing
ContextWorld release template; it is not a sealed leaderboard payload. The
future target frame and evaluator audit/provenance columns therefore remain in
the Public Lance tree so independent users can reproduce the score. They are
never model inputs: the adapter receives only the three RGB history frames and
the frozen five-axis query-action blocks. In particular, hidden mode, hidden
value, physical state, pair/provenance IDs, action-profile IDs, and content
hashes are evaluator-only fields. A result cannot claim compliance if any of
those fields is passed to the model.

All checkpoint paths, SHA-256 digests, sizes, and model-state SHA-256 values are
preflighted. The score namespace and an irreversible access marker are created
before the Public Lance table is read. Any failure after that marker consumes
the authorization; retry requires a distinct preregistration and namespace.

## Frozen scoring gates

Each checkpoint is scored with its own frozen target encoder. Cross-checkpoint
absolute latent MSE comparison is prohibited. Each checkpoint must satisfy:

- correct-future rate at least 0.75;
- correct-history rate at least 0.75;
- context-switch rate at least 0.90;
- worst-rule correct-future rate at least 0.70;
- separated paired target latents;
- response gain at least 0.50;
- normalized response error strictly below 1.00;
- paired-bootstrap 95% lower bounds of 0.70 for future/history and 0.85 for
  context switching, using 10,000 resamples and seed `2026080314`.

The reference Public method passes only if all three checkpoints pass every
gate. A mean score cannot rescue a failed seed.

## Outcome branches

- Pre-access freeze failure: Public remains untouched and unauthorized.
- Generation failure before publication: archive the failure; do not reuse the
  namespace.
- Any failure after the access marker: Public may have been read; archive the
  failure and create a new recovery preregistration before any retry.
- Completed Public scoring with 3/3 passing checkpoints: authorize packaging a
  local Public release candidate with a positive LeWM reference claim.
- Completed Public scoring with 0–2/3 passing checkpoints: authorize only a
  local data-and-scoring candidate carrying the explicit negative reference
  result.

Neither completed branch directly authorizes ContextWorld Suite registration.
That requires a later packaging and suite-identity audit.

## Explicitly prohibited

- Public generation, directory traversal, table opening, hashing, or scoring
  before a valid freeze receipt exists;
- PLDM Public scoring under this authorization;
- training, fine-tuning, checkpoint selection, or Public-derived normalization;
- changing pair count, seeds, devices, thresholds, bootstrap recipe, or output
  paths;
- rerunning within either one-use namespace after it has been created;
- claiming Suite release status from the pre-access freeze alone.
