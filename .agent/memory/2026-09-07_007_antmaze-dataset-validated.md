# Agent Memory: AntMaze dataset validated

- Date: 2026-09-07
- Sequence: 007
- Agent: Codex
- Branch: `feat/lancet`
- Commit before: `c45d89d`
- Commit after: `c45d89d` (working tree remains uncommitted)
- Status: completed

## Goal

Finish the interrupted AntMaze HDF5 transfer and verify it is usable by D4RL.

## What changed

- Completed the exact 231,110,764-byte `antmaze-medium-play-v2` HDF5 in `/data/lancet/datasets/d4rl` using a container-side resumable transfer.

## Commands / validation

```bash
./dev d4rl --shell 'python ... gym.make("antmaze-medium-play-v2").get_dataset() ...'
```

Result: D4RL parsed `observations (1000000, 29)` and `actions (1000000, 8)` successfully.

## Decisions

The D4RL backend is now data-ready; the next validation must be the deliberately tiny WSRL/Lancet off2on pipeline, not a formal run.

## Resume hint

Run the smoke configs in `configs/off2on/*_antmaze_medium_play_smoke.yaml`, save bounded logs under `/data/lancet/runs`, then record a new memory immediately.
