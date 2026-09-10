# Agent Memory: wsrl-lancet-initializer debug

- Date: 2026-09-10
- Sequence: 001
- Agent: archived experiment launcher
- Branch: `main`
- Commit before: `991991d1e6e77fbe42b95d2cc8a82faa376601b8`
- Commit after: working tree, not committed
- Status: completed

## Goal

Create the seed-0 shared 20k WSRL offline initializer for the stability gate.

## What changed

Created and finalized one `debug` experiment archive.

## Key files

- `/home/zhaozihan/Lancet/data/runs/debug/antmaze-medium-play-v2/wsrl-lancet-initializer/seed_0/20260910_091819`

## Validation

Process result: passed; return code 0; checkpoints found: 6. The offline_final checkpoint records step/update 20000/20000, all checkpoint and TensorBoard tensors are finite, and an agent-construction reload passed.

## Decisions

The experiment directory is authoritative; this entry is only an index summary.

## Problems / caveats

See `analysis.md` and `train.log` in the archive.

## Next dependency

Review algorithm-specific success criteria before the next experiment.

## Resume hint

Open `/home/zhaozihan/Lancet/data/runs/debug/antmaze-medium-play-v2/wsrl-lancet-initializer/seed_0/20260910_091819/analysis.md` first.
