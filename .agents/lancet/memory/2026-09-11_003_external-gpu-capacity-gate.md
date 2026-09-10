# Agent Memory: External GPU capacity gate

- Date: 2026-09-11
- Sequence: 003
- Agent: Codex
- Branch: main
- Commit before: b41dfed3e6ecf0597bbe13cc7243253bc58499f0
- Commit after: this commit (GPU capacity gate)
- Status: completed

## Goal

Queue formal work without colliding with unrelated processes already using every GPU.

## What changed

- Added an optional free-memory threshold checked under the per-GPU archive lock.
- Added capacity state, last free-memory reading, and check time to metadata.
- Documented and unit-tested the NVIDIA SMI parsing path.

## Key files

- `scripts/experiments/archive_run.py`
- `scripts/experiments/README.md`
- `tests/test_lancet_experiment_archive.py`

## Validation

Focused archive tests and Ruff must pass before formal launch.

## Decisions

Formal initializers will wait for 45,000 MiB free and will not stop existing jobs.

## Problems / caveats

All six GPUs were occupied by unrelated work at the launch audit.

## Next dependency

Freeze this commit and pre-register the five initializer runs in detached tmux sessions.

## Resume hint

Use metadata capacity/lock states to distinguish waiting archives from active training.
