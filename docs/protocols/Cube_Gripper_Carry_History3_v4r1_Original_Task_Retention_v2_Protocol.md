# Cube Gripper-Carry History=3 v4r1 Original-Task Retention v2 Recovery

## Recovery basis

The v1 retention freeze was valid, but all four jobs terminated while creating
their first Cube environment.  The common exception was `Cannot initialize a
headless EGL display`; no call to `World.evaluate` occurred, no CEM episode
completed, and no baseline or candidate aggregate was created.  The immutable
v1 failure receipt therefore classifies the attempt as an infrastructure
failure, not a retention result.

This v2 preregistration creates a new freeze, query-catalog, result, and
decision namespace.  The v1 directory remains unchanged and is not reused as
an output directory.

## Only authorized change

The only runtime change is:

```text
evaluation.mujoco_gl: egl -> osmesa
```

OSMesa is the backend recorded by the completed Reacher retention template and
successfully constructs the pinned original-Cube environment in this runtime.
The backend is shared by the original LeWM baseline and all three candidates.
It does not change the frozen query identities, restored physical states, goal
offset, execution budget, model checkpoints, training seeds, CEM settings, or
noninferiority threshold.

Before v2 can be frozen, an isolated one-environment OSMesa preflight must
complete without calling `World.evaluate` or consuming a CEM episode.  The
baseline and three candidates must also strict-load with 18,034,628 parameters
each in the clean Stable-WorldModel checkout at commit
`875e607fc08aa72eacb94d5d178127804134cc06`.

## Exact carry-over contract

The v2 effective preregistration inherits the v1 scientific contract:

- LeWM baseline plus training seeds 17321, 17322, and 17323;
- evaluation seeds 42, 43, and 44;
- 100 queries per evaluation seed and 300 episodes per checkpoint;
- byte-identical shared query catalog with SHA-256
  `0ebf7fa68078f76e1172cb47ef594129acc5563ed148ef4dafbd96b4c855d725`;
- goal offset 25 and evaluation budget 50;
- history 3, horizon 5, receding horizon 5, action block 5;
- CEM 300 samples, 30 iterations, and top-k 30;
- candidate success count no more than 15/300 below the paired baseline;
- all three candidates must pass.

The v2 freezer deterministically rematerializes the catalog and rejects the
freeze unless its hash is exactly the v1 frozen catalog hash.

## Output and stop rules

The v2 paths are one-use and distinct from v1.  Any identity drift, failed
OSMesa preflight, nonzero matrix job, incomplete 300-episode result, or query
catalog change stops the attempt.  There is no retry in the same namespace.

Public Test remains not generated, not opened, not read, not hashed, and not
scored.  Even a passed retention decision does not authorize Public access,
suite registration, or release status; those require a later separate one-use
release freeze.
