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
| final affected regression selection (2026-09-10) | 202 passed, 7 warnings |

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

## Current Lancet real-data smoke (2026-09-09)

- Shared initializer `20260909_215057` loaded the real 1M-transition dataset,
  performed two offline updates, saved `offline_final.pt`, and passed
  structural and agent-construction reload validation.
- Lancet `20260909_215436` loaded that exact initializer, completed online
  handoff, recorded seven residual updates with changed residual parameters
  and initialized U EMA, and passed finite-state/checkpoint reload validation.

## Seed-0 stability gate (2026-09-10)

| Role | Archive | Actual final step | Result |
|---|---|---:|---|
| shared WSRL 20k offline initializer | `20260910_091819` | 20,000 | finite; reload passed |
| WSRL 20k online branch | `20260910_095226` | 40,032 | finite; reload passed |
| Lancet 20k online branch | `20260910_102445` | 40,032 | finite; reload passed |

Both online branches used the same `offline_final.pt`. Lancet adaptation began
at global step 25,056, ran 60,160 residual updates, and retained finite state.
Its maximum logged ratio-of-means `|Delta|/|Q_base|` was about `7.13e-4`; the
final ratio was about `6.25e-7`. U weights stayed at or below 3.0. All 11
normalized-score evaluations were 0 for both methods. The observed adaptation
horizon was only 14,976 steps, so its mean AUC of 0 is debug evidence and does
**not** complete or estimate the formal 0–50k primary comparison.

Evidence lives in each archive's `metrics/summary.json`,
`metrics/scalars.json`, and `analysis.md`; paired evidence is in the Lancet
archive's `metrics/paired_comparison.{json,md}`.

## Remaining validation notes

- Peak per-process CUDA memory was not collected as a scalar; no OOM occurred.
- Kitchen is not fetched and is not a blocker for the first AntMaze study.
- Formal benchmark has not started.
