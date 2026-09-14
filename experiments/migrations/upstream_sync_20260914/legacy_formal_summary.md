# Legacy formal generation-1 summary

Generated: `2026-09-14T14:57:40+08:00`  
Scientific status: **SUPERSEDED_PRE_UPSTREAM_SYNC**

Generation 1 is superseded because upstream corrected the WSRL AntMaze
CQL/Cal-QL configuration after launch and the full upstream migration changes
observation, policy, and replay infrastructure. The rerun is not selected due
to unfavorable returns. Old/new results must never be mixed statistically.

No run, scalar, metric, or checkpoint was deleted or rewritten. The JSON
manifest is the authoritative supersession overlay.

## Canonical lifecycle counts

| Method | Operational status | Count |
|---|---|---:|
| lancet | blocked | 1 |
| lancet | completed | 4 |
| wsrl | blocked | 1 |
| wsrl | completed | 4 |
| wsrl-initializer | completed | 5 |

## Canonical jobs

| Method | Seed | Operational status | Final/global step | Finite | Archive |
|---|---:|---|---:|---|---|
| wsrl-initializer | 0 | completed | — | None | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_0/20260911_013042` |
| wsrl-initializer | 2 | completed | — | None | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_2/20260911_013834` |
| wsrl-initializer | 3 | completed | — | None | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_3/20260911_013834` |
| wsrl-initializer | 1 | completed | — | None | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_1/20260911_175532` |
| wsrl-initializer | 4 | completed | — | None | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl-initializer/seed_4/20260911_175542` |
| wsrl | 0 | completed | 1500032 | True | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl/seed_0/20260912_000600` |
| lancet | 0 | completed | 1500032 | True | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/lancet/seed_0/20260912_000610` |
| wsrl | 2 | completed | 1500032 | True | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl/seed_2/20260912_115947` |
| lancet | 2 | completed | 1500032 | True | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/lancet/seed_2/20260912_115955` |
| wsrl | 3 | completed | 1500032 | True | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl/seed_3/20260912_121449` |
| lancet | 3 | completed | 1500032 | True | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/lancet/seed_3/20260912_121500` |
| wsrl | 1 | blocked | — | None | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl/seed_1/20260912_224051` |
| lancet | 1 | blocked | — | None | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/lancet/seed_1/20260912_224100` |
| wsrl | 4 | completed | 1500032 | True | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/wsrl/seed_4/20260913_115245` |
| lancet | 4 | completed | 1500032 | True | `/home/zhaozihan/Lancet/data/runs/formal/antmaze-medium-play-v2/lancet/seed_4/20260913_115252` |

Raw and Centered generation-1 formal runs: **none**.

## Exclusion contract

- Preserve generation-1 archives for debugging, regression, and migration audit.
- Exclude `SUPERSEDED_PRE_UPSTREAM_SYNC` from default formal analysis.
- Do not use generation-1 checkpoints as generation-2 scientific initializers.
