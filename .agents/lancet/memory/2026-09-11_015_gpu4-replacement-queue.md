# Agent Memory: GPU 4 replacement queue

- Date: 2026-09-11
- Sequence: 015
- Agent: Codex
- Branch: `main`
- Commit before: `e712a88ed28a1296a164eca51e5276893f93084d`
- Commit after: this infrastructure commit
- Status: completed

## Goal

Requeue the formal initializers interrupted on reserved GPU 4 without changing
the frozen training configuration.

## What changed

- Preserved and superseded seed 1 (update 487335) and seed 4 (update 5117).
- Prepared fresh full 1M archives for both seeds and added them to the queue.
- Batch completion now ignores explicitly superseded runs.

## Key files

- `scripts/experiments/run_training.py`
- `/data/lancet/runs/formal/.lifecycle/formal_pipeline.json`

## Validation

Both replacements are queued with no GPU allocation. Stable audit rejected GPU
4 by policy and found no currently safe unused GPU. Lifecycle tests: 6 passed;
Ruff passed.

## Decisions

Do not resume seed 1 from 450k: the current runner would execute another full
configured 1M updates. A clean full rerun preserves the frozen semantics.

## Problems / caveats

The current protocol file only adds infrastructure/readiness notes relative to
the formal snapshot; replacement metadata records both hashes and commit lineage.

## Next dependency

Wait for a non-4 GPU to pass the stable resource gate; the single controller
will then start seed 1 first and seed 4 second.

## Resume hint

Run `python3 scripts/experiments/run_training.py status` and inspect the last
GPU audit before changing any queue state.
