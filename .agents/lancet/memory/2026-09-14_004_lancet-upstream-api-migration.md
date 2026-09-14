# Agent Memory: Lancet upstream API migration

- Date: 2026-09-14
- Sequence: 004
- Agent: Codex
- Branch: `integration/upstream-sync-20260914`
- Commit before: `0605c42`
- Commit after: `ce6575f`
- Status: completed

## Goal

Move Lancet to upstream's canonical Dict observation and policy contracts without changing its mathematics.

## What changed

- Residual state extraction now uses `ObservationSchema` canonical state keys.
- B×K local-action evaluation repeats every Dict observation tensor correctly.
- Base Q still receives the full upstream observation; actor sampling uses the shared SACCore hook.
- Lancet builder now mirrors upstream WSRL encoder, policy-std, SARSA, and registry plumbing.

## Validation

`pytest -q tests/test_lancet.py tests/test_lancet_audit.py tests/test_training_registry.py`: 42 passed.

## Decisions

Lancet remains state-only for now; image observations fail explicitly rather than being silently discarded by the residual.

## Problems / caveats

Full regression, semantic parity, checkpoint continuation, and real AntMaze gates remain pending.

## Next dependency

Freeze corrected generation-2 WSRL/Lancet configurations and complete the scientific audits.

## Resume hint

Review commit `ce6575f` and `experiments/migrations/upstream_sync_20260914/base_config_parity.json`.
