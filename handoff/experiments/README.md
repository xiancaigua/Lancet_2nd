# Experiment Index

The experiment directory under `/home/zhaozihan/Lancet/data/runs/` is the
authoritative record. This page is a human-readable index only.

## Planned benchmark protocols

| Protocol | Scope | Status | Entry point |
|---|---|---|---|
| AntMaze WSRL vs Lancet | `antmaze-medium-play-v2`; 5 paired seeds; Raw/Centered ablations | planned; not started; smoke and seed-0 pilot are current gates | [`antmaze_wsrl_lancet.md`](../../experiments/protocols/antmaze_wsrl_lancet.md) |
| Legacy WSRL vs Lancet V1 | `antmaze-medium-play-v2`, 5 paired seeds | superseded; never started; historical only | [`antmaze_wsrl_lancet_v1.md`](../../experiments/protocols/antmaze_wsrl_lancet_v1.md) |

## Smoke

| Experiment | Status | Main result | Output |
|---|---|---|---|
| 20260908_123022 | finished | WSRL 1M data, offline→online, updates, save/reload passed | `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/wsrl/seed_0/20260908_123022` |
| 20260908_123802 | finished | Lancet V1 shared-scalar residual updated/logged/checkpointed/reloaded; finite losses | `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/lancet/seed_0/20260908_123802` |
<!-- smoke-rows -->

## Debug

| Experiment | Status | Main result | Output |
|---|---|---|---|
<!-- debug-rows -->

## Formal

| Env | Algo | Seed | Status | Main metric | Output |
|---|---|---|---|---|---|
<!-- formal-rows -->

No formal experiment has been started. `runs/formal/` and
`checkpoints/formal/` were verified empty on 2026-09-09 after the current protocol revision.
