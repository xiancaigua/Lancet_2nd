# Agent Memory: Current Lancet implementation

- Date: 2026-09-09
- Sequence: 004
- Agent: Codex
- Branch: `main`
- Commit before: `51d014d058605f92ecc823ea3b2b42ef25ab4dd3`
- Commit after: working tree
- Status: completed

## Goal

Move the confirmed legacy executable to Lancet V1 and implement the frozen current Lancet contract.

## What changed

- Added critic-aligned `ResidualEnsemble`, zero heads, K=8 local actions, centering, detached U weights, post-base-step target reuse, corrected ensemble actor Q, constant 50k lifecycle, and full checkpoint state.
- Moved legacy code/registration/config/tests to `LancetV1` / `lancet_v1`.
- Added deterministic evaluation seed state plus initial/final actual-step evaluation and exact episode-count plumbing.

## Key files

- `rl_garden/algorithms/lancet.py`
- `rl_garden/training/off2on/lancet.py`
- `tests/test_lancet.py`

## Validation

`211 passed` for the combined Lancet, SAC/CQL/WSRL, off2on, evaluation, registry, and checkpoint selection; a later focused evaluation regression passed 93 tests. Current/legacy/registry focus passed 47 tests, dry-run construction passed, and focused Ruff checks passed.

## Decisions

The latest registry remains `lancet`; the variants are `raw|centered|lancet`. WSRL Bellman/CQL/Cal-QL/REDQ target code was not changed.

## Problems / caveats

Real AntMaze smoke and seed-0 stability pilots have not run yet.

## Next dependency

Review/commit a clean implementation revision, then create experiment archives before runtime tests.

## Resume hint

Run the focused/regression suites and inspect `git diff`, then use the shared WSRL smoke initializer.
