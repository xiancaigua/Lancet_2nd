# Agent Memory: Lancet v3 design frozen

- Date: 2026-09-09
- Sequence: 001
- Agent: Codex
- Branch: `main`
- Commit before: `ede0935942e94cc1a2b6f9889bb738b8f1e5b5ea`
- Commit after: working tree, not committed
- Status: completed

## Goal

Turn the supplied v3 theory and current WSRL/Lancet code into a code-ready specification.

## What changed

- Added the v3 implementation design and Theory–Code Alignment table.
- Reclassified current `Lancet` as Legacy Lancet-TD, not v3.

## Key files

- `docs/design/lancet-v3-implementation.md`
- `docs/design/lancet-v3-theory-code-alignment.md`
- `handoff/05_lancet_implementation.md`

## Validation

Read the supplied 26-page PDF and current SAC/CQL/CalQL/WSRL/Lancet update,
policy, phase, checkpoint, registry, config, and test paths. No training ran.

## Decisions

Use a shared backbone with N zero-output heads, per-critic centered TD fitting,
detached REDQ U weighting, and a 50k post-warmup linear actor handoff.

## Problems / caveats

The design is not implemented. Practical projection has only one observed-action TD target.

## Next dependency

Revise the AntMaze benchmark protocol around Full Lancet v3 and early-online metrics.

## Resume hint

Read both v3 design documents before editing `lancet.py` or adding `lancet_v3.py`.
