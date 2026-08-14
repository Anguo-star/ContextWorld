# Cube Gripper-Carry History=3 v4r1 Original-Task Retention v1

## Purpose and predecessor gate

This is a post-Development authorization for the Cube gripper-carry benchmark.
It does not reopen training and it does not authorize Public Test.  Its only
question is whether the three LeWM checkpoints that passed the frozen v3
Development gate retain the original Cube planning ability.

The prerequisite decision is
`cube_gripper_carry_h3_v4r1_reference_development_v3`.  It records LeWM as
3/3 passing and PLDM as 0/3 passing.  Consequently this retention stage
authorizes only:

- the original LeWM initialization checkpoint as the paired baseline;
- LeWM training seeds 17321, 17322, and 17323 at the fixed 4096-step endpoint;
- one standard original-Cube CEM evaluation for each of those four files.

PLDM, new checkpoints, new training, recipe changes, and checkpoint selection
are outside this protocol.

## Released-capability template

The gate follows the completed Reacher arm-mass release template:

- evaluation seeds 42, 43, and 44;
- 100 deterministic queries for each evaluation seed;
- 300 episodes per checkpoint;
- baseline and candidates evaluated on the identical query catalog;
- a candidate is noninferior when its total success count is no more than
  15/300 below the corresponding original checkpoint;
- every checkpoint in the passing three-seed family must be noninferior.

This is a success-count noninferiority contract, not a comparison of latent MSE
between different checkpoints.

## Runtime and standard CEM

The evaluator is bound to the clean Stable-WorldModel checkout at commit
`875e607fc08aa72eacb94d5d178127804134cc06`, the same pinned runtime used by
the v3 reference training.  The older Cube release draft is not an execution
authority: its wrapper resolves a mutable sibling Stable-WorldModel checkout
and its data/results belong to the superseded failed Development attempt.

The frozen evaluator preserves the upstream standard Cube planning contract:

- original `cube_single_expert.h5` dataset;
- goal offset 25 raw steps;
- evaluation budget 50 raw steps;
- history length 3;
- horizon 5, receding horizon 5, action block 5;
- CEM 300 samples, 30 iterations, and top-k 30;
- no rollout videos.

The original Cube action has five coordinates.  One action block therefore
enters LeWM as 25 values.  The freezer requires strict CPU loading of the
baseline and all three candidates in the pinned runtime before any CEM run is
authorized.

## Frozen paired query catalog

Before evaluation, the freezer deterministically materializes a single query
catalog from the original Cube H5.  For each seed, it reproduces the upstream
`numpy.default_rng(seed).choice` selection over valid start rows, including the
historical final-index exclusion, and then sorts the selected rows.  The
catalog stores row index, episode index, and start step for all 300 queries.

Each of the four independent jobs must consume the byte-identical catalog and
must verify its episode and start-step identities against the H5.  The result
aggregator rejects any catalog, protocol, dataset, checkpoint, configuration,
episode count, or per-query outcome drift.

## Gate

Let `B` be the baseline success count over the 300 frozen queries and `C_i` be
the success count for LeWM training seed `i`.  Candidate `i` passes iff

```text
C_i >= B - 15
```

The LeWM family passes retention iff seeds 17321, 17322, and 17323 all pass.
The receipt additionally records per-evaluation-seed successes, regressions,
and improvements, but those diagnostics do not replace the preregistered total
success gate.

## One-use outputs and stop rule

The preregistration fixes distinct one-use paths for:

- the freeze receipt;
- the shared query catalog;
- the four-job retention result root;
- the immutable retention decision.

Existing output paths are never overwritten.  A missing freeze, identity
drift, nonzero job, changed query catalog, or incomplete 300-episode result is
an infrastructure failure and may not be repaired in the same namespace.  If
any of the three candidates fails noninferiority, the decision is
`failed_retention` and Public evaluation remains unauthorized.

If all three pass, the decision is `passed_retention`.  That permits only a
limited positive Development-plus-retention reference claim.  A later,
separate one-use release freeze is still required before Public Test can be
generated, opened, read, hashed, or scored.

## Public boundary

Throughout this stage Public Test is:

- not generated;
- not opened;
- not read or decoded;
- not hashed;
- not scored;
- unavailable for recipe, checkpoint, threshold, or family selection.

This protocol does not authorize suite registration or a release-candidate
claim.
