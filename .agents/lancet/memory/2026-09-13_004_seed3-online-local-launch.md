# Agent Memory: seed-3 online local launch

- Date: 2026-09-13
- Sequence: 004
- Agent: Codex
- Branch: `main`
- Commit before: working tree
- Commit after: working tree; not committed
- Status: completed

## Goal

Restore the locally blocked seed-3 online WSRL/Lancet pair when capacity became available.

## What changed

- Returned the two seed-3 online jobs to the dynamic local queue.
- Existing capacity gates assigned WSRL to GPU 0 and Lancet to GPU 2.

## Validation

Both jobs were unstarted and `blocked`; lifecycle now reports `running` with
training PIDs 2980269 and 2980274 respectively.

## Decisions

GPU 4 remains excluded. No scientific configuration or algorithm behavior changed.

## Resume hint

Use `run_training.py status` or the lifecycle state for noninvasive progress checks.
