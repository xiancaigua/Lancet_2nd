# Agent Memory: source-server migration audit

- Date: 2026-09-12
- Sequence: 002
- Agent: Codex
- Branch: `main`
- Commit before: `da95b63db48a9551bf31d5979741ed69b75e3fa2`
- Commit after: working tree; not committed
- Status: completed

## Goal

Produce a read-only source-server migration inventory without disturbing formal training.

## What changed

- Added `/home/zhaozihan/Lancet/MIGRATION_SOURCE_MANIFEST.md` with frozen identities, runtime, checkpoint hashes, transfer scope, and rsync templates.

## Validation

Audited lifecycle state, archive metadata, checkpoint hashes, process liveness,
Docker image/mounts, package versions, and D4RL dataset hash.

## Decisions

Use Git to restore `rl-garden`; transfer dataset and formal data separately.
Partial checkpoints are not strict off-policy resumes because no replay snapshots exist.

## Problems / caveats

Seeds 1–3 were active and seed 4 queued at the audit snapshot; active roots
require an initial rsync followed by a final reconciliation.

## Next dependency

Target server must recreate the Docker runtime, verify hashes, and decide when
the old running initializers may finish or be explicitly invalidated.

## Resume hint

Read `/home/zhaozihan/Lancet/MIGRATION_SOURCE_MANIFEST.md` first.
