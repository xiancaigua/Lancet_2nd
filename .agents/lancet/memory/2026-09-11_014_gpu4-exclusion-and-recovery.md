# Agent Memory: GPU 4 exclusion and formal recovery

- Date: 2026-09-11
- Sequence: 014
- Agent: Codex
- Branch: `main`
- Commit before: `14f37e22927470a253e5477f0dc07e99ae5e1ae9`
- Commit after: this infrastructure commit
- Status: completed

## Goal

Exclude another user's GPU 4 and preserve formal initializer integrity.

## What changed

- Stopped only Lancet seed 4 PID 4081054 on GPU 4; return code 143 was archived.
- Added a persistent, audited `excluded_gpu_ids` scheduler policy and CLI.
- Restarted only the lifecycle controller; seeds 0/2/3 retained their PIDs.

## Key files

- `scripts/experiments/run_training.py`
- `tests/test_lancet_training_lifecycle.py`
- `/data/lancet/runs/formal/.lifecycle/formal_pipeline.json`

## Validation

`6 passed`; Ruff passed. GPU 4 exclusion is `[4]`; GPU 4 has no Lancet process.

## Decisions

Seed 1 stopped externally at update 487335 (last checkpoint 450k). Direct resume
would execute another configured 1M updates, so recovery uses a new full-run
archive rather than changing the frozen scientific configuration.

## Problems / caveats

GPU 4 remains unavailable until the user explicitly releases it.

## Next dependency

Prepare fresh seed 1 and seed 4 archives and enqueue them on non-4 GPUs.

## Resume hint

Inspect `formal_pipeline.json` and run the lifecycle status command first.
