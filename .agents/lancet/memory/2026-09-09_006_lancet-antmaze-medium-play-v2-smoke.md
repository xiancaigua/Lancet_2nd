# Agent Memory: lancet smoke

- Date: 2026-09-09
- Sequence: 006
- Agent: archived experiment launcher
- Branch: `main`
- Commit before: `991991d1e6e77fbe42b95d2cc8a82faa376601b8`
- Commit after: working tree, not committed
- Status: completed

## Goal

Validate current Lancet on the real 1M-transition AntMaze dataset from a shared WSRL checkpoint.

## What changed

Created and finalized one `smoke` experiment archive.

## Key files

- `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/lancet/seed_0/20260909_215436`

## Validation

Process result: passed; return code 0; checkpoints found: 1. Structural finite-state analysis and agent-construction reload passed. The checkpoint records 7 residual updates, nonzero residual parameters, and an initialized finite U EMA.

## Decisions

The experiment directory is authoritative; this entry is only an index summary.

## Problems / caveats

See `analysis.md` and `train.log` in the archive.

## Next dependency

Review algorithm-specific success criteria before the next experiment.

## Resume hint

Open `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/lancet/seed_0/20260909_215436/analysis.md` first.
