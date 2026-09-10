# Agent Memory: Formal identity and coordinates

- Date: 2026-09-11
- Sequence: 001
- Agent: Codex
- Branch: `main`
- Commit before: `c83876c4e60365a1c80aba2d373c6f93d94effec`
- Commit after: working tree, not committed
- Status: completed

## Goal

Remove step-coordinate/readiness ambiguity and freeze formal archive identity.

## What changed

- Analysis now distinguishes global, online, and adaptation coordinates.
- Formal archives require and hash protocol/dataset plus source/resolved config;
  online forks can also hash their shared initializer.
- Readiness docs now reflect the user's formal-launch authorization.

## Key files

- `scripts/experiments/archive_run.py`
- `scripts/experiments/analyze_run.py`
- `scripts/experiments/compare_runs.py`
- `experiments/protocols/antmaze_wsrl_lancet.md`

## Validation

Archive/Lancet/evaluation selection: 28 passed. Pilot re-analysis yielded
global/online/adaptation endpoints 40032/20032/14976.

## Decisions

No algorithm or training-math change was made. `expnote` is unavailable, so
the existing experiment archive remains the authoritative record.

## Problems / caveats

Only three GPUs are currently fully idle; remaining initializer seeds require conservative queuing.

## Next dependency

Commit/push the frozen revision, write the external identity manifest, then launch formal initializers.

## Resume hint

Inspect formal `FORMAL_IDENTITY.json`, tmux sessions, and per-run metadata first.
