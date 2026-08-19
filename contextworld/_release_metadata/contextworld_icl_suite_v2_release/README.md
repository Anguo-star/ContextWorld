# Packaged Suite v2 release metadata

This directory contains byte-identical distribution copies of the immutable
historical Suite v2 public-scoreboard specification and aggregate scoreboard.
They are not new results, benchmark samples, training data, checkpoints, or
artifact-tree replacements.

At runtime, ContextWorld first reads the canonical external receipt.  Only
when that receipt is unavailable does the read-only Suite v2 `info` path use
one of these copies, after verifying the SHA-256 declared by the frozen suite
configuration.

Canonical logical paths:

- `artifacts/evaluation/contextworld_icl_suite_v2_release/public_scoreboard_spec.json`
- `artifacts/evaluation/contextworld_icl_suite_v2_release/public_scoreboard.json`
