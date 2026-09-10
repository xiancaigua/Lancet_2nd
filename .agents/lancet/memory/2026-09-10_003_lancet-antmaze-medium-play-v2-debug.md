# Agent Memory: lancet debug

- Date: 2026-09-10
- Sequence: 003
- Agent: archived experiment launcher
- Branch: `main`
- Commit before: `991991d1e6e77fbe42b95d2cc8a82faa376601b8`
- Commit after: working tree, not committed
- Status: completed

## Goal

Run the seed-0 20k current Lancet online stability branch from the shared initializer.

## What changed

Created and finalized one `debug` experiment archive.

## Key files

- `/home/zhaozihan/Lancet/data/runs/debug/antmaze-medium-play-v2/lancet/seed_0/20260910_102445`

## Validation

Process result: passed; return code 0; checkpoints found: 5. Final step/update
was 40032/80160; 60160 residual updates occurred after adaptation start 25056.
All checkpoint/scalar state was finite and agent-construction reload passed.
Maximum logged `|Delta|/|Q_base|` was about `7.13e-4`.

## Decisions

The experiment directory is authoritative; this entry is only an index summary.

## Problems / caveats

All 11 WSRL and Lancet debug evaluations were 0. The observed 14976-step
adaptation AUC is therefore 0 for both and is not formal performance evidence.

## Next dependency

Human-review and push the final evidence commit before formal preparation.

## Resume hint

Open `/home/zhaozihan/Lancet/data/runs/debug/antmaze-medium-play-v2/lancet/seed_0/20260910_102445/analysis.md` first.
