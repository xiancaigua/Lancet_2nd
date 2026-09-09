# Testing and Validation

Only commands actually run are listed. Current Lancet unit/regression evidence
is kept separate from historical Lancet V1 runtime evidence.

## Current implementation checks (2026-09-09)

| Command | Result |
|---|---|
| focused current/legacy/registry selection | 47 passed, 1 warning |
| base evaluation and off-policy endpoint tests | 17 passed |
| Lancet + SAC/CQL/WSRL/off2on/checkpoint regression selection | 211 passed, 8 warnings |
| WSRL, Lancet, and Lancet V1 `--dry-run` construction | passed |
| shared initializer/online/debug config preflights | passed |

The current suite verifies exact-zero fork equality, per-critic shapes,
action-centering, pairwise/efficient U equivalence (including N=2), first-batch
EMA initialization, single-target reuse after the base critic step, actor and
optimizer gradient isolation, unchanged WSRL target semantics, offline/warmup/
50k lifecycle gates, four residual updates per UTD=4 unit, WSRL checkpoint
forking, Lancet round-trip, isolated evaluation seeds, and final evaluation at
the first actual rollout step at or above the nominal budget.

## Environment/audit evidence

Earlier audit runs verified container/Python/Torch/CUDA/GPU visibility,
bidirectional bind mounts, `rl_garden`, D4RL AntMaze reset/action space, all
required dataset keys, exactly 1,000,000 transitions, finite dataset arrays,
and checkpoint infrastructure. Legacy optional D4RL backend warnings (Flow,
CARLA, GymBullet, headless GLFW) do not affect the verified AntMaze path.

## Historical runtime evidence

- WSRL archive `20260908_123022` loaded the 1M dataset and completed its tiny
  offline-to-online path with checkpoint reload.
- Archive `20260908_123802` exercised the old shared-scalar implementation and
  is therefore **Lancet V1**, not evidence for current Lancet.

## Pending runtime gates

- Current Lancet archived real-data tiny smoke: not yet run.
- Shared seed-0 20k WSRL/Lancet stability pilot: not yet run.
- Formal benchmark: not started and not authorized.
- Kitchen dataset: not fetched; not a blocker for the first AntMaze study.
