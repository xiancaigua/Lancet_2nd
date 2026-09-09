# Agent Memory: Lancet residual scaffold

- Date: 2026-09-07
- Sequence: 004
- Agent: recovered retrospectively
- Branch: `feat/lancet`
- Commit before: unknown; recovery found `c45d89d`
- Commit after: `c45d89d` (working-tree changes remain uncommitted)
- Status: completed

## Goal

Add a minimal Offline-to-Online critic-correction extension without changing WSRL's base TD path.

## What changed

- Added `algorithms/lancet.py`, off2on registration, configs, smoke script, and focused tests.
- Exported `Lancet` and `ResidualQNetwork` from `rl_garden.algorithms`.

## Decisions

Lancet subclasses WSRL. A shared `R_phi(s,a)` is added to every ensemble Q; base critic targets/updates stay unchanged. Actor uses corrected Q. U-variation is intentionally disabled until its equation is specified.

## Validation

`tests/test_lancet.py` passed 6 tests, including finite update, online switch, checkpoint round trip, and registry discovery.

## Resume hint

Read `rl_garden/algorithms/lancet.py` before changing loss semantics.
