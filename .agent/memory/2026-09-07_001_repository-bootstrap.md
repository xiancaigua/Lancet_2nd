# Agent Memory: repository bootstrap

- Date: 2026-09-07
- Sequence: 001
- Agent: recovered retrospectively
- Branch: `feat/lancet`
- Commit before: unknown; recovery found `c45d89d`
- Commit after: `c45d89d` (working-tree changes remain uncommitted)
- Status: completed

## Goal

Establish a single Host checkout and persistent data layout for Lancet work.

## What changed

- Added `/home/zhaozihan/Lancet/data/{datasets/d4rl,checkpoints,runs,videos,cache/torch_extensions}`.
- Cloned the upstream HTTPS repository and created the `feat/lancet` branch.

## Validation

`pwd` confirmed `/home/zhaozihan/Lancet/rl-garden`; current base commit is `c45d89d`.

## Decisions

The Host checkout is the only source of truth; datasets and run artifacts are outside Git.

## Resume hint

Read `AGENTS.md` and `.agent/CURRENT_STATE.md`; do not clone a second checkout in Docker.
