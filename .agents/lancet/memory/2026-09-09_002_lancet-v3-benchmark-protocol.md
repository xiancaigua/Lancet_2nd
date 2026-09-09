# Agent Memory: Lancet v3 benchmark protocol drafted

- Date: 2026-09-09
- Sequence: 002
- Branch: `main`
- Commit before: `ede0935942e94cc1a2b6f9889bb738b8f1e5b5ea`
- Commit after: uncommitted working tree
- Status: completed

## Goal

Replace the never-run Legacy Lancet-TD plan with a v3-aligned, pre-registered
benchmark contract without launching training.

## What changed

- Added `experiments/protocols/antmaze_wsrl_lancet_v3.md`.
- Marked v1 as a superseded historical protocol and updated Human Handoff.
- Froze shared per-seed offline initialization, WSRL-vs-Full main comparison,
  matched Raw/Centered ablations, early-online AUC, pilot gates, statistics,
  checkpoint cadence, and rerun policy.

## Validation

Protocol values were checked against current WSRL/Lancet configs and update,
evaluation, checkpoint, D4RL metric, and archive code. A pure Torch check matched pairwise
and efficient U to 6.7e-16 and verified centered mean below 1e-16. No experiment
ran.

## Decisions

Primary metric is normalized-score AUC 0–50k. The mandatory main table is
WSRL vs Full v3 at five paired seeds; Legacy Lancet-TD is excluded.

## Problems / caveats

V3 code, focused tests, stage configs, evaluation guarantees, smoke, and pilot
remain prerequisites. Formal launch is blocked.

## Next dependency

Human review, then implement the smallest `lancet_v3` code/test slice.

## Resume hint

Read the v3 implementation spec and protocol before touching algorithm code.
