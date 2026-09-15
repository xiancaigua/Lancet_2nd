# Agent Memory: lancet-upstream-sync smoke

- Date: 2026-09-14
- Sequence: 008
- Agent: archived experiment launcher
- Branch: `integration/upstream-sync-20260914`
- Commit before: `756fef4ee83230046f04c0d41dac37201f78fc03`
- Commit after: working tree, not committed
- Status: partial

## Goal

Validate current Lancet end-to-end after the full upstream observation and policy API merge.

## What changed

Created and finalized one `smoke` experiment archive.

## Key files

- `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/lancet-upstream-sync/seed_0/20260914_203538`

## Validation

Process return code 0; finite checkpoint and seven residual updates passed.
The evaluation gate failed because the resolved cap was 50 steps, so no full
1000-step AntMaze episode completed. A corrected smoke is required.

## Decisions

The experiment directory is authoritative; this entry is only an index summary.

## Problems / caveats

This is retained as a configuration-regression attempt, not final smoke evidence.

## Next dependency

Review algorithm-specific success criteria before the next experiment.

## Resume hint

Open `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/lancet-upstream-sync/seed_0/20260914_203538/analysis.md` first.
