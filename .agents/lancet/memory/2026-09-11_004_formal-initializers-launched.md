# Agent Memory: Formal initializers launched

- Date: 2026-09-11
- Sequence: 004
- Agent: Codex
- Branch: main
- Commit before: 62817637beffcbe8b1315d3c0daf03f1fc5fd9a0
- Commit after: documentation-only launch-state commit
- Status: partial

## Goal

Start all five frozen 1M WSRL offline initializer tasks without GPU conflicts.

## What changed

- Froze commit, dataset, source config, protocol, runtime, and image identity.
- Created formal archives for seeds 0–4 in detached tmux sessions.
- Started seeds 0/1 on GPUs 2/4 and queued seeds 2/3/4 under the same GPU locks.

## Key files

- `/home/zhaozihan/Lancet/data/runs/formal/FORMAL_IDENTITY.json`
- `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/`

## Validation

All archives record commit `6281763` and matching dataset/config/protocol hashes.
Seeds 0/1 reached update 9,000 with finite logged values and no runtime error signal.

## Decisions

Use only GPUs 2/4 and serialize all work; unrelated processes on other GPUs remain untouched.

## Problems / caveats

Initializers need about 19 hours each at the observed 14.3 updates/s; completion and reload remain pending.

## Next dependency

Validate every `offline_final.pt`, then run seed-wise shared-fork equality.

## Resume hint

Read archive metadata, short launcher logs, tmux sessions, and fixed-size tails of train logs.
