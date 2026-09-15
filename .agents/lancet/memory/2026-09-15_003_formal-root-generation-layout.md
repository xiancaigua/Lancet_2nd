# Agent Memory: Formal root generation layout

- Date: 2026-09-15
- Sequence: 003
- Branch: `main`
- Commit before: `1a9d99a`
- Commit after: infrastructure commit pending
- Status: completed

## Goal

Make `formal` mean the current generation and archive the old generation clearly.

## What changed

- Renamed stopped generation-1 run/checkpoint roots to `formal_v1`.
- Renamed the empty generation-2 roots to canonical `formal`.
- Added `/data/lancet/runs/FORMAL_GENERATIONS.json`; old metadata remains immutable.
- Updated protocol, handoff, and analysis defaults to the new canonical paths.

## Validation

No training process was active. All 286 MiB run artifacts and 9.3 GiB
checkpoints were preserved; V2 identity and generation-map JSON parse cleanly.

## Decisions

`formal` is always the current scientific generation. Historical generations
use explicit suffixes and are excluded by default analysis paths.

## Resume hint

Commit/push the infrastructure path update, then archive Stage A under formal.
