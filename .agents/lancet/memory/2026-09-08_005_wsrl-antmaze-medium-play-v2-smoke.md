# Agent Memory: wsrl smoke

- Date: 2026-09-08
- Sequence: 005
- Agent: archived experiment launcher
- Branch: `main`
- Commit before: `477abdac2f1e297c6ede82aafde20ce90b65de4b`
- Commit after: working tree, not committed
- Status: completed

## Goal

Validate the real AntMaze WSRL offline-to-online pipeline with a tiny budget.

## What changed

Created and finalized one `smoke` experiment archive.

## Key files

- `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/wsrl/seed_0/20260908_123022`

## Validation

Passed: 1M finite dataset; offline 2/2; online step/update 2→6; actor, critic, and target tensors changed; 8 checkpoints; final CLI reload rc=0.

## Decisions

The experiment directory is authoritative; this entry is only an index summary.

## Problems / caveats

See `analysis.md` and `train.log` in the archive.

## Next dependency

Review algorithm-specific success criteria before the next experiment.

## Resume hint

Open `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/wsrl/seed_0/20260908_123022/analysis.md` first.
