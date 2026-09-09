# Agent Memory: Engineering consolidation

- Date: 2026-09-08
- Sequence: 002
- Agent: Codex
- Branch: `main`
- Commit before: `477abdac2f1e297c6ede82aafde20ce90b65de4b`
- Commit after: working tree, not committed
- Status: completed

## Goal

Unify Lancet continuity with rl-garden's native `.agents/` hierarchy.

## What changed

- Migrated all `.agent/` state, memories, and templates without loss into
  `.agents/lancet/`; removed the obsolete duplicate directory.
- Added the namespace README, updated `AGENTS.md`, and created verified local
  runtime bindings plus split run/checkpoint roots.

## Key files

- `.agents/lancet/README.md`
- `.agents/lancet/CURRENT_STATE.md`
- `AGENTS.md`

## Validation

All eight prior memories and both templates exist in the new namespace; Git
remotes and source/runtime bind mounts were unchanged.

## Decisions

Native rules remain authoritative. Experiment directories are the result
source of truth; Memory and Handoff are summaries.

## Problems / caveats

Handoff split, archival launcher, audits, and real-data smokes remain next.

## Next dependency

Build the archive workflow before any new smoke.

## Resume hint

Read `AGENTS.md`, current state, this memory, then the experiment workflow.
