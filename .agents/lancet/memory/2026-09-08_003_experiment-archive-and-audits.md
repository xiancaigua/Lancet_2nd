# Agent Memory: Experiment archive and audits

- Date: 2026-09-08
- Sequence: 003
- Agent: Codex
- Branch: `main`
- Commit before: `477abdac2f1e297c6ede82aafde20ce90b65de4b`
- Commit after: working tree, not committed
- Status: completed

## Goal

Enforce durable experiment records and independently auditable runtime checks.

## What changed

- Added a Host-side archive launcher that freezes intent, command, config,
  metadata, log, analysis, Memory, and Handoff index for each completed run.
- Enforced smoke/debug/formal path separation and formal clean-tree guard.
- Added runtime, mount, D4RL, and Lancet audit tools plus focused tests.
- Moved nine old non-archived logs to `runs/legacy_unclassified/`.

## Key files

- `scripts/experiments/archive_run.py`
- `scripts/audit/run_all.sh`
- `tests/test_lancet_audit.py`

## Validation

Archive guard tests: 3 passed. `scripts/audit/run_all.sh`: all audits passed;
AntMaze has 1,000,000 finite transitions and Lancet audit tests: 7 passed.

## Decisions

The launcher runs on Host with stdlib and delegates training to Docker. This
keeps Host free of training dependencies and repository records user-owned.

## Problems / caveats

Real-data WSRL and Lancet off2on smokes are still pending.

## Next dependency

Create full human Handoff, then run the archived WSRL smoke.

## Resume hint

Read `scripts/experiments/README.md` and the Handoff experiment workflow.
