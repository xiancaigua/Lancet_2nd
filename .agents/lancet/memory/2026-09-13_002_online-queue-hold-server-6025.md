# Agent Memory: online queue hold for server 6025

- Date: 2026-09-13
- Sequence: 002
- Agent: Codex
- Branch: `main`
- Commit before: working tree
- Commit after: working tree; not committed
- Status: completed

## Goal

Prevent this server from launching the unstarted seed-1 and seed-3 online pairs.

## What changed

- Marked four local lifecycle jobs `blocked` with a server-6025 migration hold.
- Preserved all archives, configurations, and checkpoints.

## Validation

All four targets were `queued` with no worker or training PID before mutation.
They are now `blocked` and cannot be selected by the local queue.

## Resume hint

Use their existing formal archives on server 6025; do not re-enable them here
unless the user explicitly requests it.
