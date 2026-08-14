# Cube Gripper Carry History=3 v4r1 Public Recovery v1 Protocol

Status date: 2026-08-14. Status:
`authorized_before_recovery_freeze_not_generated_not_read_not_scored`.
The user explicitly authorized this distinct recovery campaign after the first
Public v1 namespace was consumed by a publication-receipt implementation
failure. This authorization permits one recovery generation attempt and, only
after successful publication, one fixed three-checkpoint LeWM scoring attempt.

## Recovery lineage

The original scientific protocol remains
`cube_gripper_carry_rule_history3_v4r1_public_release_v1`. The distinct
recovery authorization is
`cube_gripper_carry_rule_history3_v4r1_public_recovery_v1`; it does not rename
or change the scientific task.

The recovery freeze must rehash and semantically validate this complete failed
campaign chain:

- original preregistration: SHA256
  `633589015d23279a859b20d3ce02d6804fb25ced95858e4a419f733d8794903c`,
  17,240 bytes;
- original freeze receipt: SHA256
  `9215a8fbb74677717d088df536f26f62922667f501a1b168481df1c8007066c7`,
  2,183,220 bytes;
- `_GENERATION_STARTED.json`: SHA256
  `a8c985f2f13fff93a0ac3629ffb5feee19803848ec15b6b2ac128ca7fb0e1965`,
  908 bytes;
- `_GENERATION_FAILURE.json`: SHA256
  `fc5e6e21b43af548102c105ec21e75bdd7542808f3ede818d65c683063907fcc`,
  979 bytes.

The failed root must contain exactly the two marker files. The failure must
remain `KeyError("preregistration")`, with no published Lance table, success
marker, model read, score root, or final decision. The original namespace is
immutable and is never retried.

## Frozen scientific invariants

Recovery changes only authorization and artifact identities. It preserves the
original Public v1 science byte-for-byte in meaning:

- 256 pair-balanced Public Test pairs from a fixed 512-candidate pool;
- catalog offset `3000000` and seeds `2026081400`, `2026081401`,
  `2026081402`;
- 64 accepted pairs for each of `endpoint4`, `front_hold`, `plateau`, and
  `ramp4`;
- split-disjoint source episodes, action profiles, scene templates, pair
  content, and query pixels;
- five-axis action blocks satisfying `sum(p)=0` and `p[-1]=0`;
- JPEG quality 95, 16 workers, and `/tmp` staging;
- the same complete source H5 identity and mandatory rehash before candidate
  selection;
- LeWM only, seeds 17321/17322/17323, their exact step-4096 checkpoints,
  devices `cuda:0`/`cuda:1`/`cuda:2`, and batch size 64;
- the same latent gates, 10,000 paired bootstrap resamples, and bootstrap seed;
- no training, adaptation, model/checkpoint selection, threshold change, or
  online environment call.

PLDM remains excluded because it failed frozen Development 0/3. Public fields
are never used for training or selection. The adapter receives only history
pixels and query action blocks.

## Distinct one-use paths

- preregistration:
  `configs/benchmark/cube_gripper_carry_h3_v4r1_public_recovery_prereg_v1.yaml`;
- freeze receipt:
  `artifacts/evaluation/history3/cube_gripper_carry_h3_public_recovery_v1/public_recovery_freeze_receipt_v1.json`;
- Public data:
  `artifacts/synthesis/cube_gripper_carry_rule_h3_public_v4r1_recovery_v1`;
- Public score:
  `artifacts/evaluation/history3/cube_gripper_carry_h3_public_recovery_v1/public_score_v1`;
- decision:
  `artifacts/evaluation/history3/cube_gripper_carry_h3_public_recovery_v1/public_recovery_decision_v1.json`.

Every path must be absent at freeze. Each namespace is created exclusively and
cannot be deleted or reused after an attempt begins.

## Execution guardrails

Before generation, the recovery loader, source/checkpoint hashes, complete
lineage, Stable-WorldModel runtime, free space, and all three real CUDA adapter
preflights must pass. The generation budget is not consumed if this preflight
fails.

Generation reserves its new root before rehashing the source, builds in local
staging, validates every frozen data gate, copies the staged tree, and writes
`_SUCCESS.json` last. Scoring is prohibited unless publication validation
passes. The scoring runner performs checkpoint and adapter preflight before it
creates the score namespace or reads the Public table, writes an irreversible
access marker before table access, evaluates all three fixed checkpoints, and
does not retry after any consumed attempt.

The finalizer independently revalidates the published tree and recomputes every
checkpoint and method-level gate. Its decision binds the recovery prereg,
freeze, original failed lineage, Public success marker, score success marker,
and matrix score.

## Outcome boundary

If all three LeWM checkpoints pass, the result may be packaged as a local Cube
Public release candidate with a positive reference claim. If fewer than three
pass, only a data/scoring candidate carrying the explicit negative reference
result may be packaged. Neither branch automatically adds Cube to the current
eight-component Suite; Suite registration requires a separate post-result
packaging and identity audit.
