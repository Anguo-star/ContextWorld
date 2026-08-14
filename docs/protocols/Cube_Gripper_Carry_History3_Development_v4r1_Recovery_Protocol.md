# Cube Gripper Carry History=3 v4r1 Recovery Protocol

Status: `preregistered_before_v4r1_recovery_build`. This document authorizes
infrastructure recovery only. The scientific protocol remains
`cube_gripper_carry_rule_history3_development_v4`; the separate recovery
authorization is `cube_gripper_carry_h3_development_v4r1`.

## Why recovery is necessary

The one authorized v4 formal build generated all 2,048 Training pairs, then
failed when Lance attempted its NFS atomic commit rename. The process exited 1
with `EPERM`; `_versions` and `_transactions` remained empty, Development did
not start, and no manifest or build report was written. This is an
infrastructure failure, not a scientific gate result.

The failed tree and its complete 16,384-row fragment are immutable evidence.
The original attempt budget is consumed. It is forbidden to retry, overwrite,
repair, promote, or silently discard that attempt under the original v4
preregistration. The failure decision, failed-attempt receipt, and deterministic
raw-query reconstruction receipt must all be frozen before recovery.

## Scientific protocol remains v4

v4r1 changes no scientific variable. The following remain exactly as frozen
for v4:

- History=3: x0, x1, x2 are model-visible history and x3 is the future;
- hidden modes `cannot_hold` and `can_hold`, with paired actions bitwise equal;
- the only v3-to-v4 scientific change, 0.40 N `can_hold` vertical-force
  coupling;
- continuous `[p, -p, p]` action-block sequence without installing state
  after x0;
- four action templates: `endpoint4`, `plateau`, `ramp4`, `front_hold`;
- `sum(p)=0`, `p[-1]=0`, and
  `dot([4,3,2,1,0], p)=1`, at absolute tolerance `1e-6`;
- the all-zero terminal action block, camera, JPEG95, source filter, causal
  gates, action-support gate, split sizes, seeds, and RGB probe recipe and
  thresholds.

The scientific `protocol_id` therefore remains v4. v4r1 is an authorization
and population/publication label, not a scientific protocol version.

## New population and mandatory exclusions

The failed v4 Training batch is scientifically inspectable and cannot be
reused. Its five exclusion classes each contain 2,048 identities: source
episode, action profile, scene template, pair content, and raw query pixel.
All five have zero overlap with the old final prior.

The v4r1 finalizer must union the old prior with the failed batch before the
recovery build. The required union counts are:

| Identity | Required count |
|---|---:|
| source episodes | 4,369 |
| action profiles | 4,370 |
| scene templates | 4,378 |
| pair contents | 4,378 |
| raw query pixels | 4,378 |

The formal catalog offset is `2,000,000`. It is positive and four-aligned, so
anchor assignment is unchanged while the concrete action-profile namespace is
disjoint from preformal indices 0/1 and failed-v4 indices
1,000,000–1,002,047. The only authorizing action-support input is canonical
audit v2, `cube_gripper_carry_h3_v4r1_action_support_v2`. It exhaustively
checks the full candidate pools: 4,096 Training profiles and 512 Development
profiles, for 4,608 unique profiles total, with exact four-template balance,
zero Training/Development profile overlap, and zero failed-batch profile
overlap.

Action-support audit v1 checked only the 2,304 finally selected pair profiles,
not the complete 4,608-profile candidate pools. It is explicitly superseded
and non-authorizing. A recovery freeze presented with v1, even if v1 reports
PASS, must fail closed; only the SHA256-bound v2 artifact may satisfy this
prerequisite.

The frozen old request establishes 9,998 eligible source episodes before the
old prior. Removing the 2,321 old-prior episodes left 7,677; the 2,048 failed
Training episodes are disjoint, so 5,629 remain after the recovery union. The
candidate-pool requirement is 4,608 (`4096 + 512`), leaving a deterministic
margin of 1,021. This is a non-scientific capacity check and must be rechecked
by the final prior and recovery catalog, not used to select scientific values.

## Recovery build scope

Exactly one recovery build is authorized:

| Split | pairs | episodes | rows | pairs per template |
|---|---:|---:|---:|---:|
| `train` | 2,048 | 4,096 | 16,384 | 512 |
| `loader_validation` | 256 | 512 | 2,048 | 64 |

The build worker count is frozen at exactly 16 and is part of the recorded
request. Any other value fails before source enumeration or output creation.
No builder/Lance smoke is authorized. Public generation is explicitly false;
no Public split may be generated, opened, read, decoded, scanned, sampled,
hashed, or scored. Reference-model
training and scoring are unauthorized both in the structured phase declaration
and the top-level authorization flag; optimizer steps are zero and no
checkpoint may be created.

## Local commit and x-exclusive publication

The recovery builder must create a unique staging directory on the local
`/tmp` filesystem. Both Lance tables must commit and reopen there, and all
table, causal, replay, cross-split, manifest, and build-report checks must pass
before publication.

The final NFS root is
`artifacts/synthesis/cube_gripper_carry_rule_h3_development_v4r1`. It must not
already exist. Publication uses an x-exclusive `shutil.copytree` whose files
are copied with `shutil.copy2` and `dirs_exist_ok=False`; it must not rename a
non-empty directory. Source and destination path/size/SHA receipts, tree hash,
and reopened Lance identities must match.

`_SUCCESS.json` is never present in local staging and is created exclusively
at the destination only after every copied file has been verified. An
interrupted or failed copy remains incomplete and must never receive a success
marker. No existing destination may be removed or overwritten.

## Frozen RGB-history probe and stop rule

Only a completed publication containing the one valid `_SUCCESS.json` may be
probed. The probe CLI must receive explicit `--prereg`, `--freeze-receipt`,
and `--prior-exclusion-receipt` paths in addition to the artifact root and
output. Before opening either Lance table it must validate the exact
preregistration → recovery freeze → final recovery-prior chain, then parse and
cross-bind `request.json`, `build_report.json`, and `manifest.json` against
that chain and the success marker. In particular, the request must bind the
exact freeze and prior receipts, the build-report request must equal
`request.json`, and the manifest must bind the exact prior and published file
set. It must reject symlinks, Public-shaped paths, identity/status drift, split
drift, or a non-closed Public declaration and must reverify all trusted input
bytes after the Lance reads.

Exactly one RGB-history probe is authorized, using the unchanged v4
16×16 `flatten(2*x1-x0-x2)` recipe, Training-only scaler fit,
`RidgeClassifier(alpha=1)`, thresholds, bootstrap, permutation controls, and
x0/query/action-only controls.

- If recovery data construction, local commit/reopen, publication, prior
  exclusion, causal, replay, action-support, or content isolation fails, stop
  with `failed_development`; do not probe.
- If the one frozen probe fails any gate, stop with `failed_development`; do
  not rerun or alter the recipe.
- If all data and probe gates pass, the result is data readiness only. Model
  training still requires a separate preregistration.

Public Test remains closed in every outcome.

## Freeze and identity chain

Before recovery, the freeze receipt must bind:

- this preregistration and protocol;
- the original v4 preregistration snapshot and freeze receipt;
- the old final prior, infrastructure-failure decision, immutable failed
  attempt, and raw-query reconstruction receipts;
- the canonical v4r1 action-support audit v2; audit v1 is superseded and
  non-authorizing;
- the source H5 content identity without recording its local path;
- current v4r1 builder, v4 physics wrapper and inherited v2/v3 dependencies,
  action auditor, probe, prior finalizer, recovery freezer, their tests, the
  shared causal contract, and this protocol.

Every input is supplied explicitly. Public-shaped paths, symlinks, missing or
extra authorization bindings, hash/size drift, unresolved identity
placeholders, altered scientific/recovery contracts, contaminated Public or
model scope, and an existing output must fail before the receipt is written.
The receipt is created with exclusive file mode and all input bytes are
reverified immediately before output.

## Canonical identity state

The infrastructure-failure decision, authorizing action-support audit v2,
recovery implementation, tests, and protocol are canonical and SHA256-bound in
the preregistration. No unresolved identity placeholder remains. The recovery
freezer continues to reject placeholders and any byte drift. Canonical identity
completion makes the preregistration eligible for its one freeze; it does not
itself consume or execute the freeze, build, or probe attempt.
