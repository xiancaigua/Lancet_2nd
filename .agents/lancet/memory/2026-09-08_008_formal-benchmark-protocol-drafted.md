# Agent Memory: Formal benchmark protocol drafted

- Date: 2026-09-08
- Sequence: 008
- Agent: Codex
- Branch: `main`
- Commit before: `477abdac2f1e297c6ede82aafde20ce90b65de4b`
- Commit after: working tree, not committed
- Status: completed

## Goal

Freeze a first-stage WSRL versus Lancet-TD protocol without launching training.

## What changed

- Added the AntMaze v1 fairness, budget, metric, pilot, statistics, rerun,
  hardware, and readiness contract.
- Classified current implementation as `NEEDS SMALL PATCH`.

## Key files

- `experiments/protocols/antmaze_wsrl_lancet_v1.md`

## Commands / validation

Both full-scale `--print-config` outputs matched on base fields. A no-training
temporary probe verified strict WSRL-to-Lancet checkpoint loading, base tensor
equality, reproducible residual initialization, and fresh residual optimizer
state. Formal run/checkpoint roots each contained zero files.

## Decisions

V1 is Lancet-TD (`lambda_u_variation=0`), uses seed-paired shared WSRL offline
checkpoints, a 20k/20k seed-0 debug gate, and 1M-offline/500k-online formal
budgets.

## Problems / caveats

Small patches are needed for eval seeding/final eval, stage configs,
paired RNG alignment, diagnostics, GPU selection, lineage, and periodic
checkpoint cadence.

## Next dependency

Human review, then bounded patches and archived debug pilots; no formal launch.

## Resume hint

Read the protocol readiness table and checklist before editing code.
