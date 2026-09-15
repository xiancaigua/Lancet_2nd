# Agent Memory: Corrected WSRL initializer started

- Date: 2026-09-15
- Sequence: 004
- Branch: `main`
- Commit before: `d7e9ad0`
- Commit after: continuity-only commit pending
- Status: partial

## Goal

Run generation-2 Stage A corrected WSRL seed-0 offline validation.

## What changed

- Started one archived 1M-update WSRL initializer on physical GPU 0.
- Used canonical current `formal` roots; generation-1 remains in `formal_v1`.
- Copied the frozen V2 identity into the run archive.

## Key files

- Run: `/data/lancet/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_0/20260915_105013`
- Checkpoints: `/data/lancet/checkpoints/formal/antmaze-medium-play-v2/wsrl-initializer/seed_0/20260915_105013`

## Validation

Archive metadata is `running`, hashes match V2 config/protocol/dataset identity,
tmux and training PID are alive, GPU 0 reached about 5.6 GiB and 98% compute,
and the initial rate is about 14.3 updates/s (roughly 19.4 hours total).

## Problems / caveats

The platform rejected a STARTED email containing internal path/PID details;
this notification failure does not affect or relabel training.

## Resume hint

Inspect only the log tail/metadata/GPU at low frequency. After `offline_final.pt`
exists and the process finishes, run finite/reload and the 100-episode diagnostic.
