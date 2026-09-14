# Agent Memory: corrected WSRL AntMaze freeze

- Date: 2026-09-14
- Sequence: 005
- Agent: Codex
- Branch: `integration/upstream-sync-20260914`
- Commit before: `cae6937`
- Commit after: `352451a`
- Status: completed

## Goal

Adopt only the validated upstream AntMaze fixes and freeze base parity for generation 2.

## What changed

- Added the four substantive 3c9c46 CQL corrections and made three already-equal CQL values explicit.
- Froze `bootstrap_at_done=truncated`, retained `target_entropy=0.0`, and removed legacy `obs_mode`.
- Unified TensorBoard/final-checkpoint infrastructure across WSRL, Raw, Centered, and Lancet.
- Added reproducible resolved-config and three-way audits.

## Validation

All four online configs parsed under upstream CLI; 119 resolved base inputs are identical after excluding only Lancet-specific parameters (`base_config_parity.json`: PASS).

## Decisions

Upstream `target_entropy=auto` came from unrelated commit 823f58b, so it is not folded into the WSRL stability fix. Exact 20-episode evaluation remains frozen.

## Problems / caveats

Corrected baseline performance has not yet been run; config correctness is not performance evidence.

## Next dependency

Finish dataset/semantic/checkpoint tests and real AntMaze smoke before main promotion.

## Resume hint

Read `wsrl_config_three_way_diff.md` and `base_config_parity.json`.
