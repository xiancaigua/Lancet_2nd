# Known issues and next steps

## P0 — remaining launch gates

The implementation, current-method tiny smoke, seed-0 shared initializer,
paired 20k online pilot, checkpoint reload, and finite diagnostics passed.
Before formal work:

1. Treat the user's formal-launch authorization as acceptance of the already
   reviewed frozen implementation; do not reopen algorithm design.
2. Push the chosen clean formal commit to `origin`; formal metadata must reference
   that immutable remote commit.
3. Reconfirm empty formal roots and prepare the five seed-specific shared
   offline initializer archives, configs, and hashes.

Formal training remains unstarted and unauthorized in the current session.

## P1 — benchmark operations

1. Freeze five seed-specific WSRL 1M offline initializer configs and hashes.
2. Use the now-verified common actual endpoint and adaptation-coordinate AUC
   tooling; the debug pilot reached only adaptation step 14,976, not 50k.
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
- The pilot did not log peak per-process CUDA memory, although no OOM occurred.
- WSRL and Lancet both scored 0 throughout this short pilot; this is not a
  failure of the stability gate and provides no performance claim.

## Current decision

The implementation/review gates passed and the user authorized formal launch.
Freeze and push the formal-infrastructure commit, then start the five archived
1M shared WSRL offline initializers. Do not start untracked online runs.
