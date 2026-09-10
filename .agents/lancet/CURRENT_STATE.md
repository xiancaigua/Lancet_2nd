# Lancet Current Agent State

Last updated: 2026-09-10 11:05 CST
Branch: `main`
Commit: `991991d1e6e77fbe42b95d2cc8a82faa376601b8` plus experiment archive evidence

## Current objective

Close the validated Lancet implementation at a clean evidence commit and hand
it to a human for review/push. Formal benchmarks remain unstarted.

## Repository and runtime

- Host: `/home/zhaozihan/Lancet/rl-garden`
- Container: `/workspace/rl-garden`
- Data: `/home/zhaozihan/Lancet/data` -> `/data/lancet`
- Image/container: `lancet-d4rl:cu128` / `lancet-d4rl`
- Python/Torch/CUDA: 3.10.21 / 2.7.0+cu128 / 12.8
- GPU: 6 × RTX 4090; choose and record one physical GPU per run
- Standard execution: `./dev d4rl <command>`

## Lancet architecture

```text
one unchanged WSRL Bellman target y
  -> normal base critic optimizer step
  -> recompute Q_i_updated
  -> shared residual backbone + N exact-zero heads
  -> fit per-critic stopgrad(y - Q_i_updated)
  -> K=8 Raw/Centered/Lancet correction; U is fit weight only
  -> actor min_i(Q_i + Delta_i)
```

The actor numerator retains action gradient and subtracts a detached local
residual mean. Lancet is inactive offline and through the 5k frozen warmup,
active at lambda=1 for 50k adaptation steps, then update/correction are off.

## Implementation status

### Implemented and unit/regression verified

- Latest identity: `Lancet` / registry `lancet`; one confirmed historical
  shared-scalar implementation is `LancetV1` / `lancet_v1`; no V2 executable
  was invented.
- Residual ensemble, local actions, centering, REDQ U weighting, post-base-step
  target reuse, actor aggregation/gradient isolation, lifecycle, RNG state,
  logging, checkpoint migration and round-trip.
- Unified `raw|centered|lancet` variants and frozen shared-fork configs.
- Isolated evaluation seed stream, exact-episode WSRL plumbing, initial anchor,
  and final evaluation at the first actual step at/above nominal budget.
- Aggregate affected suite: 211 passed; later evaluation regression: 93 passed;
  focused lint and archive/Lancet tests pass.

### Runtime evidence

- Shared-checkpoint WSRL initializer and current Lancet real-data tiny smoke passed finite-state and agent-construction reload validation.
- Seed-0 shared 20k WSRL initializer and paired 20k online WSRL/Lancet
  branches passed finite-state, evaluation, and agent-construction reload.
- Both online branches ended at actual step 40,032. Lancet started adaptation
  at 25,056 and completed 60,160 residual updates; logged Delta/Q ratio stayed
  below `7.13e-4`. Both short debug curves were identically zero, so no
  performance claim is supported.
- Formal benchmark (not started).
- Readiness: **NOT READY FOR FORMAL** only because the final evidence commit
  still needs human review and push; implementation/runtime gates passed.

## Environment status

- D4RL AntMaze env and 1,000,000-transition dataset previously audited.
- Docker GPU, mounts, proxy/runtime audits previously passed.
- Host tmux 3.4 is available for detached pilots.
- Legacy D4RL optional-backend/headless warnings are non-blocking for AntMaze.

## Active decisions

- WSRL Bellman/CQL/Cal-QL/REDQ target behavior is unchanged.
- All online branches fork one seed-matched WSRL offline checkpoint.
- Primary metric is Adaptation AUC 0–50k from the first trainable post-warmup
  step; endpoint is the first common actual step >= nominal 500k.
- Raw/Centered/Lancet share capacity, initialization, local actions, optimizer,
  lifecycle, and checkpoint machinery.
- Formal requires a clean pushed frozen commit and all runtime gates.

## Immediate next steps

1. Review and commit validation tooling, memories, and handoff updates.
2. Human-review and push the frozen commits to `origin`.
3. Recheck empty formal roots, then prepare the five archived 1M shared WSRL
   initializers only after explicit formal-launch authorization.
4. Do not launch unarchived or outcome-selected runs.

## Read next

1. `AGENTS.md` and task-relevant `.agents/rules/`
2. `.agents/lancet/PLAN.md` and `CHECKLIST.md`
3. `docs/design/lancet-implementation.md`
4. `docs/design/lancet-theory-code-alignment.md`
5. `experiments/protocols/antmaze_wsrl_lancet.md`
6. `.agents/lancet/memory/INDEX.md`
