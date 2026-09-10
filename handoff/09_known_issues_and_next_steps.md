# Known issues and next steps

## P0 — initializer completion gate

The implementation, current-method tiny smoke, paired 20k pilot, checkpoint
reload, and finite diagnostics passed. Formal protocol and identity are frozen.
Five 1M shared initializer archives are active or queued; before online work:

1. Let seeds 0–4 finish without score-based stopping or reruns.
2. Validate update count, finite model/optimizer state, checkpoint hash, and reload.
3. Verify seed-matched shared-fork base-state equality and Lancet zero correction.

Formal online training remains unstarted until these gates pass.

## P1 — benchmark operations

1. Use the now-verified common actual endpoint and adaptation-coordinate AUC
   tooling; the debug pilot reached only adaptation step 14,976, not 50k.
2. Schedule the 10-run WSRL-vs-Lancet main table only after initializer/fork validation.
3. Schedule Raw/Centered
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
- The pilot did not log peak per-process CUDA memory, although no OOM occurred.
- WSRL and Lancet both scored 0 throughout this short pilot; this is not a
  failure of the stability gate and provides no performance claim.

## Current decision

The implementation/review gates passed, commit `6281763` is pushed, and five
formal initializer archives have been launched. Monitor and validate them; do
not start untracked or premature online runs.
