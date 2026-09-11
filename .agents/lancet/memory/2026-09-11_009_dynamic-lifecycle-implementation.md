# Agent Memory: Dynamic formal lifecycle implementation

- Date: 2026-09-11
- Sequence: 009
- Agent: Codex
- Branch: `main`
- Commit before: `c1b784de8a2b5e84a12e5ad0a1f522552c4480ce`
- Commit after: working tree, not committed
- Status: completed

## Goal

Replace fixed 45 GiB waiters with one low-frequency, any-GPU formal queue.

## What changed

- Added `run_training.py`: attach/managed lifecycle, atomic state, dynamic GPU
  gate, progress files, initializer validation, seed-wise online preparation,
  and idempotent notifications.
- Added a recursive finite checkpoint/optimizer validator and reusable SMTP API.
- Unified WSRL/Lancet online logging to TensorBoard and final checkpoint saving.

## Validation

Focused archive/email/lifecycle suite: 16 passed. Ruff passed. The checkpoint
validator passed on the real 20k debug initializer (144 finite tensors).
Resolved WSRL/Lancet base configs had zero differences.

## Decisions

Gate = observed 5,564 MiB + 8 GiB margin, stable util <=20%, foreign memory
<=2 GiB, free lock, and at most one Lancet formal job/GPU. Normal poll is 5h.

## Problems / caveats

Current GPUs all have active workloads; the first queue pass may launch no job.

## Next dependency

Commit/push infrastructure, enable lifecycle email, initialize state, and run
the controller detached.

## Resume hint

Run `python3 scripts/experiments/run_training.py status` first.
