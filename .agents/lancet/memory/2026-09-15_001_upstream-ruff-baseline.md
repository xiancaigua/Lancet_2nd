# Agent Memory: Upstream Ruff baseline

- Date: 2026-09-15
- Sequence: 001
- Branch: `integration/upstream-sync-20260914`
- Commit before: `824a5de`
- Commit after: pending continuity commit
- Status: completed

## Goal

Replace the raw zero-finding gate with a reproducible no-new-regressions gate.

## What changed

- Ran Ruff 0.16.6 on clean upstream `252d1e0` and integration.
- Corrected the sole integration-only `UP045` annotation in WSRL.
- Recorded the comparison in the migration gate report and JSON summary.

## Key files

- `rl_garden/algorithms/wsrl.py`
- `experiments/migrations/upstream_sync_20260914/ruff_baseline_comparison.json`
- `experiments/migrations/upstream_sync_20260914/upstream_migration_gate_report.md`

## Validation

Both trees have the identical normalized 2,610-finding multiset. All 24
integration-added Python files and core Lancet-owned files have zero findings;
the key Lancet/registry rerun passed 50 tests.

## Decisions

Do not mass-fix upstream lint debt. Promotion requires no new normalized
findings plus clean integration-added/Lancet-owned files.

## Resume hint

Commit the gate evidence, recheck origin/main, then promote without force push.
