# Known issues and next steps

## P0 — validation gates before formal work

1. Create a clean implementation commit after final diff/review.
2. Run one archived shared-checkpoint AntMaze tiny smoke for current Lancet.
3. Run the archived seed-0 20k WSRL/Lancet stability pilot.
4. Inspect finite losses, Q/Delta/U/weights, correction ratios, evaluation,
   GPU memory, and checkpoint reload before issuing formal readiness.

Formal training remains unauthorized until these gates pass.

## P1 — benchmark operations

1. Freeze five seed-specific WSRL 1M offline initializer configs and hashes.
2. Verify the common actual endpoint, adaptation-coordinate AUC analysis, and
   deterministic evaluation stream on pilot archives.
3. Freeze the shared commit/config/dataset lineage and empty formal roots.
4. Schedule the 10-run WSRL-vs-Lancet main table; schedule Raw/Centered
   component ablations afterward without outcome-based selection.

## P2 — empirical research questions

- Whether replay-action supervision plus action centering approximates the
  oracle local-action correction sufficiently.
- Whether REDQ U weighting improves Centered Residual.
- Whether K=8, sigma=.1, and the 50k active window transfer beyond
  `antmaze-medium-play-v2`.
- Optional action-wise rank/variation diagnostics for mechanism analysis.

These are experiment questions, not invitations to change the frozen objective
after seeing results.

## Non-blocking environment notes

- D4RL import emits legacy Gym/optional-backend warnings; AntMaze data/env
  audits passed.
- Kitchen dataset validation is not required for the first AntMaze benchmark.
- Long runs must use Host tmux/scheduler around the archive launcher.

## Current decision

The current Lancet implementation and focused tests exist in the working tree.
Formal readiness is still pending real-data tiny smoke and the seed-0 20k
stability pilot. No formal run has started.
