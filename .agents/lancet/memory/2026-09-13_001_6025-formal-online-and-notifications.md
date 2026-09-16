# Agent Memory: 6025 formal online launch and notification repair

- Date: 2026-09-13
- Sequence: 001
- Agent: Codex
- Branch: main
- Commit before: ccf1a28c9ddfd4200162335d5010fb2a6c56500e
- Commit after: this infrastructure commit
- Status: completed

## Goal

Continue the formal seed-1 WSRL/Lancet online pair on server 6025, exclude
seed 3 because it is running on the old server, and restore automatic lifecycle
email without changing algorithms, protocol, evaluation, or hyperparameters.

## What changed

- Started WSRL seed 1 on physical GPU 6 and Lancet seed 1 on dedicated GPU 1,
  each in an independent tmux session from its existing prepared archive.
- Kept the validated seed-1 shared initializer and frozen 500k online configs.
- Stopped the two seed-3 waiters before training and marked their 6025 lifecycle
  jobs cancelled; their prepared archives remain unchanged and have no log or
  checkpoint output.
- Added optional SMTP subject prefixing and configured the ignored local prefix
  as `[lancet6025]`.
- Made failed notification records retryable while preserving idempotence.
- Added the missing ignored host-direct personal runtime binding.
- Restored a detached monitor for progress, completion validation, and email.

## Validation

- Focused notification/lifecycle tests: 11 passed.
- Focused Ruff check: passed.
- Two seed-1 STARTED messages with `[lancet6025]` were accepted by SMTP.
- Both jobs entered online training with evaluation output and no OOM or
  traceback at the recorded launch audit.

## Decisions

- The 6025 machine identity is a mail subject prefix, not a requirement to
  rename the configured SMTP account.
- SMTP failure remains non-fatal to training.
- Seed-3 archives are preserved as scientific records; only their new-server
  queue entries are cancelled.

## Problems / caveats

- The root-owned STCOcc process on dedicated GPU 1 could not be terminated by
  the zhaozihan account; Lancet was started using the remaining memory under
  the user-authorized aggressive policy.
- Current workers imported the pre-repair notification module, so the detached
  monitor owns completion validation/email for these two runs.
- Training-window `return=nan` can mean no episode completed in that window;
  model loss and evaluation metrics must be checked before declaring divergence.

## Resume hint

Inspect `lancet_wsrl_s1`, `lancet_lancet_s1`, and
`lancet_formal_monitor`. Do not launch seed 3 on server 6025.
