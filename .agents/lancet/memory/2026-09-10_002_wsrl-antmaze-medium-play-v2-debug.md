# Agent Memory: wsrl debug

- Date: 2026-09-10
- Sequence: 002
- Agent: archived experiment launcher
- Branch: `main`
- Commit before: `991991d1e6e77fbe42b95d2cc8a82faa376601b8`
- Commit after: working tree, not committed
- Status: completed

## Goal

Run the seed-0 20k WSRL online stability branch from the shared initializer.

## What changed

Created and finalized one `debug` experiment archive.

## Key files

- `/home/zhaozihan/Lancet/data/runs/debug/antmaze-medium-play-v2/wsrl/seed_0/20260910_095226`

## Validation

Process result: passed; return code 0; checkpoints found: 5. Final checkpoint step/update is 40032/80160; all policy, optimizer, and merged TensorBoard values are finite; agent-construction reload passed. Eleven debug evaluations all returned normalized score 0, which is stability evidence only.

## Decisions

The experiment directory is authoritative; this entry is only an index summary.

## Problems / caveats

See `analysis.md` and `train.log` in the archive.

## Next dependency

Review algorithm-specific success criteria before the next experiment.

## Resume hint

Open `/home/zhaozihan/Lancet/data/runs/debug/antmaze-medium-play-v2/wsrl/seed_0/20260910_095226/analysis.md` first.
