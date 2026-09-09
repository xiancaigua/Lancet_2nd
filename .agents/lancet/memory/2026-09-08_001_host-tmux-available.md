# Agent Memory: Host tmux available

- Date: 2026-09-08
- Sequence: 001
- Agent: administrator installation, verified by Codex
- Branch: `feat/lancet`
- Commit before: `c45d89d`
- Commit after: `c45d89d` (working tree remains uncommitted)
- Status: completed

## Goal

Make Host-side detached session management available for long-running control and training workflows.

## What changed

- The Host administrator installed `tmux`; no Docker image or container was changed.

## Commands / validation

```bash
command -v tmux && tmux -V
```

Result: `/usr/bin/tmux`, version `3.4` (`3.4-1ubuntu0.1`).

## Decisions

Use Host tmux for long-lived Codex/training-control sessions. The Docker container remains the runtime; do not bind long `docker exec` training directly to a normal SSH shell.

## Resume hint

For a formal run, create `tmux new -s lancet`, start `./dev d4rl ...` inside it, then detach with `Ctrl-b d`.
