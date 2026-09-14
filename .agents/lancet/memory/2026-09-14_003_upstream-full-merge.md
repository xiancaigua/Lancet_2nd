# Agent Memory: complete upstream merge

- Date: 2026-09-14
- Sequence: 003
- Agent: Codex
- Branch: `integration/upstream-sync-20260914`
- Commit before: `37c7e51f364178d399c22f2719a767e8d5f800e9`
- Commit after: `c1125de`
- Status: completed

## Goal

Merge the complete latest upstream codebase without replacing it with cherry-picks.

## What changed

- Merged upstream `252d1e0948618a0cd3675b9a05e0bcf4c29b5afb` with `--no-ff`.
- Preserved upstream framework changes and retained Lancet exports, arguments, and registry entries.

## Validation

Merge base was `c45d89d4ed22dfe2db5e31679308cb4796834d4e`; three conflicts were resolved and no unmerged paths remain.

## Decisions

The observation/policy/replay architecture comes from upstream. Lancet will migrate forward rather than reverting framework code.

## Problems / caveats

Lancet has not yet passed compatibility or semantic-parity tests against the merged API; `main` remains unchanged.

## Next dependency

Audit the merged API and migrate Lancet without changing its frozen mathematics.

## Resume hint

Start from `rl_garden/algorithms/lancet.py`, WSRL/SACCore hooks, observation helpers, and focused test collection.
