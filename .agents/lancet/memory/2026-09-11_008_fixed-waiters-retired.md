# Agent Memory: Fixed-GPU waiters retired

- Date: 2026-09-11
- Sequence: 008
- Agent: Codex
- Branch: `main`
- Commit before: `c1b784de8a2b5e84a12e5ad0a1f522552c4480ce`
- Commit after: working tree, not committed
- Status: completed

## Goal

Remove only the obsolete fixed-GPU waiters for formal initializer seeds 2–4.

## What changed

- Verified PIDs 838124/838159/838193 had no training child, CUDA context,
  `train.log`, checkpoint, start time, or update.
- Sent `Ctrl-C` only to tmux sessions `lancet_formal_init_s2/s3/s4`; preserved
  their prepared archives and all data.

## Validation

Seeds 0/1 remained alive as PIDs 790322/790367 on GPUs 2/4. All foreign GPU
processes and unrelated tmux session `0` remained untouched.

## Decisions

The preserved seed 2–4 archives will be reused by one dynamic any-GPU queue.

## Problems / caveats

No currently safe extra GPU was assumed from a single snapshot.

## Next dependency

Validate and activate the low-frequency dynamic lifecycle controller.

## Resume hint

Inspect `/data/lancet/runs/formal/.lifecycle/formal_pipeline.json` after queue initialization.
