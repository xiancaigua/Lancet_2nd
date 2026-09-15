# Known issues and next steps

## P0 — corrected baseline validation

The integration branch has passed merge, Lancet/WSRL semantic parity,
checkpoint, dataset, resolved-config, real AntMaze smoke, email, and the
revised no-new-regressions Ruff gate. Ruff 0.16.6 reports the same normalized
2,610 findings on clean upstream `252d1e0` and integration; integration-added
and Lancet-owned Python files are clean. Main promotion and generation-2
identity freeze are complete.

## P1 — seed-0 online sanity

1. Run the corrected 1M WSRL seed-0 initializer and 100-episode diagnostic.
2. Run WSRL seed-0 for the frozen 50k–100k online sanity horizon.
3. Only if the corrected baseline is stable, run Lancet seed 0 from the same
   initializer, then begin the five-seed generation-2 pipeline.

## P2 — non-blocking notes

- D4RL optional Flow/CARLA/GymBullet and headless GLFW warnings remain benign
  for the audited AntMaze path.
- Kitchen is not a first-stage AntMaze blocker.
- GPU 4 remains reserved for another user unless explicitly released.
- Generation-1 results and checkpoints are historical/debug evidence only.

## Current decision

`UPSTREAM MERGE COMPLETE - NEW MAIN ACTIVE - BASELINE VALIDATION PENDING`.
