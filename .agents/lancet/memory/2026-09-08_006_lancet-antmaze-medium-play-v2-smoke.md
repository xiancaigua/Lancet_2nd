# Agent Memory: lancet smoke

- Date: 2026-09-08
- Sequence: 006
- Agent: archived experiment launcher
- Branch: `main`
- Commit before: `477abdac2f1e297c6ede82aafde20ce90b65de4b`
- Commit after: working tree, not committed
- Status: completed

## Goal

Validate the real AntMaze Lancet residual-correction pipeline with a tiny budget.

## What changed

Created and finalized one `smoke` experiment archive.

## Key files

- `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/lancet/seed_0/20260908_123802`

## Validation

Passed: 1M finite data; finite residual losses; positive residual delta in offline and online phases; all required TensorBoard tags; residual weights/optimizer checkpointed; final CLI reload rc=0. Episodic return and success were not collected because four steps did not finish an episode.

## Decisions

The experiment directory is authoritative; this entry is only an index summary.

## Problems / caveats

See `analysis.md` and `train.log` in the archive.

## Next dependency

Review algorithm-specific success criteria before the next experiment.

## Resume hint

Open `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/lancet/seed_0/20260908_123802/analysis.md` first.
