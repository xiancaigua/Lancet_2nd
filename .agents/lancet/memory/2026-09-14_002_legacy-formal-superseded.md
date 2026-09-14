# Agent Memory: legacy formal generation superseded

- Date: 2026-09-14
- Sequence: 002
- Agent: Codex
- Branch: `integration/upstream-sync-20260914`
- Commit before: `db57e1e9e1ee6e81ffafd99b1547a96e26389df1`
- Commit after: pending migration-audit commit
- Status: completed

## Goal

Freeze generation-1 evidence and retire its idle scheduler before upstream resync.

## What changed

- Added a 17-job manifest and summary under `experiments/migrations/upstream_sync_20260914/`.
- Added the old AntMaze raw and transformed semantic fingerprint.
- Stopped only tmux `lancet_formal_lifecycle`; no training worker was running.

## Validation

The manifest records 13 completed, 2 failed, and 2 blocked operational jobs; all are overlaid as `SUPERSEDED_PRE_UPSTREAM_SYNC`. Five initializers and eight online runs completed. Raw/Centered did not run.

## Decisions

Generation 1 is historical evidence, not failed work, and must be excluded from generation-2 statistics. Old checkpoints are archival only.

## Problems / caveats

The failed seed-1/seed-4 initializer attempts remain preserved alongside their successful replacements.

## Next dependency

Commit the migration audit, fetch upstream, and perform a complete no-ff merge.

## Resume hint

Read `legacy_formal_manifest.json`, `legacy_formal_summary.md`, and `old_dataset_fingerprint.json` before touching upstream.
