# Agent Memory: semantic and dataset validation

- Date: 2026-09-14
- Sequence: 006
- Branch: `integration/upstream-sync-20260914`
- Commit before: `c53a217`
- Commit after: `a6cb5d0`
- Status: completed

## Goal

Verify that the upstream API migration preserves Lancet/WSRL mathematics,
checkpoint continuation, and AntMaze dataset semantics.

## What changed

- Added exact inactive/active WSRL parity and deterministic save-load-continuation tests.
- Adapted the archived Lancet V1 executable to Dict state observations without changing its formula.
- Added old/new dataset fingerprint comparison and the semantic parity report.

## Key files

- `tests/test_lancet.py`
- `experiments/migrations/upstream_sync_20260914/lancet_semantic_parity_report.md`
- `experiments/migrations/upstream_sync_20260914/dataset_semantic_comparison.json`

## Validation

Focused Lancet: 71 passed. WSRL/SAC/CQL/checkpoint: 174 passed.
Observation/replay/off2on: 195 passed. Dataset semantic comparison: PASS.

## Decisions

The upstream Dict wrapper is representational only for this AntMaze dataset;
all 4096-prefix semantic field hashes and the raw dataset hash are unchanged.

## Problems / caveats

Repository-wide `ruff check .` exposes inherited upstream lint debt; changed
migration-owned files pass focused Ruff.

## Next dependency

Run the upstream-compatible real AntMaze smoke and email gate.

## Resume hint

Read the two migration reports above, then inspect smoke configs before running.
