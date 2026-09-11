# Agent Memory: Shared-fork runtime gate

- Date: 2026-09-11
- Sequence: 011
- Agent: Codex
- Branch: `main`
- Commit before: `f35488265403f2c7c9389df0fc4e80c83f2c37b9`
- Commit after: working tree, not committed
- Status: completed

## Goal

Make per-seed shared-fork equality a direct runtime gate, not an inference from
two successful checkpoint loads.

## What changed

Added `validate_shared_fork.py` and required it before an initializer may
prepare online branches.

## Validation

The real 20k debug initializer passed exact equality for policy/base critics,
target critics, alpha, four base optimizer states, counters, zero residual, and
`Q_use == Q_base`. Ruff and Python compilation passed.

## Decisions

Validation constructs CPU agents from checkpoint metadata; it does not create
a D4RL env, allocate GPU state, or touch a running process.

## Problems / caveats

None observed.

## Next dependency

Apply this gate automatically when each formal `offline_final.pt` completes.

## Resume hint

Read each archive's `validation/initializer_validation.json` before online work.
