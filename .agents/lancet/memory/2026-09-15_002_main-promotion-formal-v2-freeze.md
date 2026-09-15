# Agent Memory: Main promotion and formal V2 freeze

- Date: 2026-09-15
- Sequence: 002
- Branch: `main`
- Commit before: `5ab3053`
- Commit after: continuity-only commit pending
- Status: completed

## Goal

Promote the validated upstream integration and create a new formal generation.

## What changed

- Fast-forwarded main from `db57e1e` to `5ab3053` and pushed without force.
- Created and pushed tag `lancet-upstream-sync-20260914`.
- Froze `/data/lancet/runs/formal_v2/FORMAL_IDENTITY_V2.json` and isolated
  formal_v2 run/checkpoint roots; generation 1 remains untouched.
- Sent the required merge-complete and new-main emails.

## Validation

Local main and origin/main match `5ab3053`; identity JSON parses and hashes to
`f6435a187980b4c02e9e64d9d6950d6f52b86accdd0f87bb0cb8b6247f1039ed`.

## Decisions

`5ab3053` is the generation-2 scientific code identity. Later documentation or
launcher-only commits must be recorded separately as infrastructure identity.

## Resume hint

Check formal_v2 launcher isolation, then start only corrected WSRL seed 0.
