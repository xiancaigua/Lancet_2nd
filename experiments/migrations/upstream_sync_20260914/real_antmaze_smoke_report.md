# Real AntMaze upstream-migration smoke

Date: 2026-09-14

## Final result

**PASS**. No formal or baseline-validation training was started.

The final evidence archive is:

`/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/lancet-upstream-sync/seed_0/20260914_204852`

Its seed-matched shared WSRL initializer is:

`/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/wsrl-upstream-sync-initializer/seed_0/20260914_202534`

## Evidence

- The upstream loader read the real 1,000,000-transition D4RL dataset.
- The initializer completed two offline critic update units and saved
  `offline_final.pt`; SHA256 is
  `e9b558a952238e44a7a5fc351fdc83aedd6f1924efb668dbf1e689b940731e12`.
- Finite model/optimizer validation and WSRL-to-Lancet shared-fork loading passed.
- At the fork, residual output was exactly zero and `Q_use == Q_base` with
  maximum absolute difference `0.0`.
- Lancet completed eight online environment steps after loading that checkpoint.
  Warmup ended at global step 4; the final checkpoint records seven residual
  updates, adaptation step 6, initialized finite U EMA, and online step 8.
- Three complete 1,000-step AntMaze evaluations ran. Returns were zero, which is
  expected to be scientifically meaningless at this tiny budget but is finite.
- TensorBoard contains finite base-Q, corrected-Q, Delta, TD-residual, U/weight,
  actor/critic/alpha, phase-coordinate, and evaluation scalars.
- Final checkpoint SHA256 is
  `746247a46e07d4327a7b2aa782145344a7dee3276552f4225bd25b9a3c96efc8`;
  model, optimizer, residual, and scalar finite gates passed, and agent reload
  was verified.

## Retained preliminary attempts

Archive `20260914_203538` completed training but resolved only 50 evaluation
steps, so no AntMaze episode completed. Its analysis records this as an
incomplete evaluation attempt. Archive `20260914_204623` fixed the full-episode
evaluation and passed finite/reload checks but had logging disabled. Neither is
used as the final migration smoke evidence; both remain preserved.

Optional-backend import warnings (Flow, CARLA, GymBullet) and headless GLFW
warnings did not affect AntMaze execution.
