# Agent Memory: Generation-2 offline queue safeguard

- Date: 2026-09-15
- Sequence: 005
- Branch: main
- Commit before: 1bfa118ccd33af3876be705addba373667a59c55
- Commit after: pending
- Status: completed

## Goal

Submit remaining corrected WSRL offline initializers safely without bypassing
the seed-0 corrected-baseline gate for formal online branches.

## What changed

- Added an explicit lifecycle opt-in for post-initializer online preparation.
- The default queue policy records a deferred online launch after validation.
- Added a regression test for the default deferred behavior.

## Validation

`./dev d4rl ruff check scripts/experiments/run_training.py tests/test_lancet_training_lifecycle.py` and the lifecycle test module passed (7 tests).

## Decisions

GPU 4 remains excluded. Seed 0 is untouched; newly queued initializer jobs use
the dynamic resource gate and the existing email lifecycle.

## Resume hint

Use the dedicated generation-2 lifecycle state under `data/runs/formal/.lifecycle/`; do not reuse the legacy formal pipeline state.
