# Agent Memory: runtime access and dev wrapper

- Date: 2026-09-07
- Sequence: 003
- Agent: recovered retrospectively
- Branch: `feat/lancet`
- Commit before: unknown; recovery found `c45d89d`
- Commit after: `c45d89d` (working-tree changes remain uncommitted)
- Status: completed

## Goal

Make the D4RL container persistent, safely addressable, and share the Host source tree.

## What changed

- Added `dev` generic Docker passthrough and lifecycle wrapper.
- Started `lancet-d4rl` with host networking, GPU access, Host UID/GID, and two bind mounts.
- Added proxy, cache, D4RL dataset, and user PATH environment handling in `run.sh`/`dev`.

## Validation

Host-to-container and container-to-Host mount writes were verified; proxy requests to GitHub succeeded in both contexts.

## Decisions

Normal `./dev d4rl <command>` forwards argv unchanged. Only explicit `--shell` invokes `bash -lc`.

## Resume hint

Use `./dev d4rl ...`; do not use `docker rm` or affect non-Lancet containers.
