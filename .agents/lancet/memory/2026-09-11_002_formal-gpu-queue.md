# Agent Memory: Formal GPU queue

- Date: 2026-09-11
- Sequence: 002
- Agent: Codex
- Branch: main
- Commit before: b1999275b9ce032f25d0ed1a1049ec106baa884c
- Commit after: this commit (formal queue support)
- Status: completed

## Goal

Allow all five formal initializers to be registered safely on three idle GPUs.

## What changed

- Added per-GPU advisory locks and explicit waiting/acquired metadata to the archive launcher.
- Serialized completion-time Memory, current-state, and Handoff writes.
- Documented and tested the queue primitive.

## Key files

- `scripts/experiments/archive_run.py`
- `scripts/experiments/README.md`
- `tests/test_lancet_experiment_archive.py`

## Validation

Focused archive tests and Ruff must pass before commit and launch.

## Decisions

Use only idle GPUs 0, 2, and 4; queue seeds 3 and 4 behind seeds 0 and 1.

## Problems / caveats

GPUs 1, 3, and 5 host unrelated workloads and are excluded.

## Next dependency

Commit and push the queue support, then freeze and launch all initializer archives.

## Resume hint

Inspect formal metadata `gpu_lock_state`, tmux sessions, train logs, and `nvidia-smi`.
