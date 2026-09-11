# Agent Memory: Dynamic formal queue activated

- Date: 2026-09-11
- Sequence: 010
- Agent: Codex
- Branch: `main`
- Commit before: `111839e0fa2123f2f05b506b5529c3edb22924b2`
- Commit after: working tree, not committed
- Status: completed

## Goal

Attach seed 0/1 and safely requeue seed 2/3/4 under the new lifecycle.

## What changed

- Enabled ignored local lifecycle email after a successful integration test.
- Created `/data/lancet/runs/formal/.lifecycle/formal_pipeline.json` and started
  tmux `lancet_formal_lifecycle` at infrastructure commit `111839e0`.
- Attached untouched PIDs 790322/790367; queued seeds 2–4 with no GPU/PID.

## Validation

First stable gate pass launched no worker: all six GPUs had foreign load or a
held Lancet lock. QUEUED notifications for seeds 2–4 each report `sent`.

## Decisions

Normal polling is 5h; 3×30s samples are taken only at scheduling time. The
formal algorithm identity remains `6281763`, separate from infrastructure.

## Problems / caveats

No additional GPU is currently safe, so seeds 2–4 remain queued.

## Next dependency

Allow the controller to schedule when capacity changes and validate each
initializer before its seed-wise online pair.

## Resume hint

Run `python3 scripts/experiments/run_training.py status`; do not restart seed 0/1.
