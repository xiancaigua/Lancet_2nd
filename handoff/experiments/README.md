# Experiment Index

The experiment directory under `/home/zhaozihan/Lancet/data/runs/` is the
authoritative record. This page is a human-readable index only.

## Planned benchmark protocols

| Protocol | Scope | Status | Entry point |
|---|---|---|---|
| AntMaze WSRL vs Lancet | `antmaze-medium-play-v2`; 5 paired seeds; Raw/Centered ablations | shared offline initializer stage running; online not started | [`antmaze_wsrl_lancet.md`](../../experiments/protocols/antmaze_wsrl_lancet.md) |
| Legacy WSRL vs Lancet V1 | `antmaze-medium-play-v2`, 5 paired seeds | superseded; never started; historical only | [`antmaze_wsrl_lancet_v1.md`](../../experiments/protocols/antmaze_wsrl_lancet_v1.md) |

## Smoke

| Experiment | Status | Main result | Output |
|---|---|---|---|
| 20260908_123022 | finished | WSRL 1M data, offline→online, updates, save/reload passed | `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/wsrl/seed_0/20260908_123022` |
| 20260908_123802 | finished | Lancet V1 shared-scalar residual updated/logged/checkpointed/reloaded; finite losses | `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/lancet/seed_0/20260908_123802` |
| 20260909_215057 | finished | real 1M data; shared checkpoint finite/reload passed | `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/wsrl-lancet-initializer/seed_0/20260909_215057` |
| 20260909_215436 | finished | current Lancet residual updated; finite/reload passed | `/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/lancet/seed_0/20260909_215436` |
<!-- smoke-rows -->

## Debug

| Experiment | Status | Main result | Output |
|---|---|---|---|
| 20260910_091819 | finished | 20k shared initializer finite; reload passed | `/home/zhaozihan/Lancet/data/runs/debug/antmaze-medium-play-v2/wsrl-lancet-initializer/seed_0/20260910_091819` |
| 20260910_095226 | finished | WSRL actual step 40032; finite/reload; 11 evals=0 | `/home/zhaozihan/Lancet/data/runs/debug/antmaze-medium-play-v2/wsrl/seed_0/20260910_095226` |
| 20260910_102445 | finished | Lancet actual step 40032; 60160 residual updates; finite/reload; paired debug AUC=0 | `/home/zhaozihan/Lancet/data/runs/debug/antmaze-medium-play-v2/lancet/seed_0/20260910_102445` |
<!-- debug-rows -->

## Formal

| Env | Algo | Seed | Status | Main metric | Output |
|---|---|---|---|---|---|
| antmaze-medium-play-v2 | wsrl-initializer | 0 | running on GPU 2 | initializer validation pending | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_0/20260911_013042` |
| antmaze-medium-play-v2 | wsrl-initializer | 1 | failed at 487335; GPU 4 external interruption; replacement pending | not collected | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_1/20260911_013042` |
| antmaze-medium-play-v2 | wsrl-initializer | 2 | running on GPU 3 | initializer validation pending | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_2/20260911_013834` |
| antmaze-medium-play-v2 | wsrl-initializer | 3 | running on GPU 5 | initializer validation pending | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_3/20260911_013834` |
| antmaze-medium-play-v2 | wsrl-initializer | 4 | failed before first checkpoint; stopped to release GPU 4; replacement pending | not collected | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_4/20260911_013834` |
| antmaze-medium-play-v2 | wsrl-initializer | 1 | replacement queued; GPU 4 excluded | not collected | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_1/20260911_175532` |
| antmaze-medium-play-v2 | wsrl-initializer | 4 | replacement queued; GPU 4 excluded | not collected | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_4/20260911_175542` |
| antmaze-medium-play-v2 | wsrl-initializer | 0 | finished | not collected | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_0/20260911_013042` |
<!-- formal-rows -->

Formal initializer training started on 2026-09-11 from frozen commit `6281763`.
The online main comparison and component ablations have not started.
