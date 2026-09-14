# Lancet Current Agent State

Last updated: 2026-09-14 15:03 CST
Branch: `integration/upstream-sync-20260914`
Commit: pre-merge base `db57e1e9e1ee6e81ffafd99b1547a96e26389df1`; generation-1 formal identity remains `62817637beffcbe8b1315d3c0daf03f1fc5fd9a0`

## Current objective

Rebuild the code and experimental baseline by fully merging current
`upstream/main` on an isolated integration branch. The exact pre-resync code is
anchored and pushed at `backup/pre-upstream-sync-20260914` and
`pre-upstream-sync-20260914`; `main` is not yet changed.
The old-server migration audit is complete; see
`/home/zhaozihan/Lancet/MIGRATION_SOURCE_MANIFEST.md`. Do not interrupt active
initializers merely to migrate: their partial checkpoints do not include replay
snapshots and therefore are not strict off-policy continuation points.
Generation-1 evidence is frozen in
`experiments/migrations/upstream_sync_20260914/`. Its 17 lifecycle jobs are
scientifically overlaid as `SUPERSEDED_PRE_UPSTREAM_SYNC`; old data remains
untouched. The idle generation-1 lifecycle controller has been stopped and no
Lancet training worker is running on this host.

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
- Formal initializer archives were created from clean, pushed commit `6281763`.
  Seed 0 continues on GPU 2; seeds 2/3 continue on GPUs 3/5. Seed 1 was
  externally stopped on reserved GPU 4 at update 487335. A mistakenly assigned
  seed 4 run was stopped at 17:31 before its first checkpoint. Both failed
  archives are preserved. Fresh full-run replacements are queued at
  `seed_1/20260911_175532` and `seed_4/20260911_175542`; neither owns a GPU.
- At the 2026-09-11 12:03 CST health check, seeds 0/1 were at approximately
  479k/348k with current logs, finite reported losses/Q values, periodic
  checkpoints, and no NaN/Inf/OOM/traceback signal in the checked log tail.
- All five shared WSRL initializers have now completed and passed their recorded
  validation. Online seed 0 and seed 2 WSRL/Lancet pairs completed; seed 3 and
  seed 4 pairs are running locally; seed 1 remains blocked locally for server
  6025. Raw/Centered ablations have not launched.
- Preliminary snapshot: `/data/lancet/runs/formal/analysis/preliminary_2026-09-13/`.
  The two complete pairs have zero normalized score at every observed
  post-adaptation evaluation, while finite/reload checks pass. This is a
  performance concern, not an observed numerical crash; do not alter running
  formal jobs in response.

## Environment status

- D4RL AntMaze env and 1,000,000-transition dataset previously audited.
- Docker GPU, mounts, proxy/runtime audits previously passed.
- Host tmux 3.4 is available for detached pilots.
- Legacy D4RL optional-backend/headless warnings are non-blocking for AntMaze.

## Active decisions

- WSRL Bellman/CQL/Cal-QL/REDQ target behavior is unchanged.
- All online branches fork one seed-matched WSRL offline checkpoint.
- `global_step`, `online_step`, and `adaptation_step` are distinct. Primary
  Adaptation AUC 0–50k starts at the first trainable post-warmup step; the
  endpoint is the first common actual `online_step >= 500k`.
- Raw/Centered/Lancet share capacity, initialization, local actions, optimizer,
  lifecycle, and checkpoint machinery.
- Formal requires a clean pushed frozen commit and all runtime gates.
- Frozen hashes: dataset `c9fec1c1...7e5b`, source initializer config
  `5af6bf32...4233`, protocol `d17fb7e4...051d`.
- Resolved WSRL/Lancet base training values match. Both online configs now use
  TensorBoard and save final checkpoints; this is infrastructure-only parity.
- Paired analysis now computes Primary Adaptation AUC on the exact fixed 0–50k
  window and reports longer observed AUC only as a diagnostic.
- Local email credentials live in ignored `configs/local/lancet_email.env`.
  SMTP was tested; lifecycle notifications are state-transition-only and their
  failures cannot fail or relabel a training run.
- `scripts/experiments/run_training.py` remains the generation-1 queue/watch
  entry point, but its tmux controller was stopped gracefully after the
  supersession manifest was written. It must not be restarted for generation 1.
  It uses an approximately 5h normal cycle, atomic state/progress files,
  stable capacity sampling, one formal job per GPU, and attach mode for seed
  seed 0. It does not import the algorithm or add training/evaluation work.
- Physical GPU 4 belongs to another user and is hard-excluded from the dynamic
  scheduler until that user explicitly releases it.
- Initializer completion directly compares the WSRL and Lancet forked policy,
  base/target critics, alpha, base optimizer states, counters, exact-zero
  residual, and `Q_use==Q_base` on CPU before online archive preparation.
- Seeds 2/3 are running on GPUs 3/5. Replacement seeds 1/4 remain queued with
  no worker, training PID, or CUDA context because the stable resource pass
  found no safe unused non-4 GPU.
- Do not modify training code/config during these runs. Preserve and invalidate
  archives rather than overwriting if a real bug is found.
- The preliminary report treats the nonzero boundary-interpolated Lancet AUC as
  a measurement artifact: it is induced by a pre-adaptation evaluation point,
  not a post-adaptation success observation.

## Immediate next steps

1. Commit the pre-merge migration audit on the integration branch.
2. Fetch and fully merge current `upstream/main`; migrate Lancet to upstream APIs without changing frozen mathematics.
3. Pass config, dataset, semantic-parity, checkpoint, test, and real-AntMaze gates before promoting `main`.

## Read next

1. `AGENTS.md` and task-relevant `.agents/rules/`
2. `.agents/lancet/PLAN.md` and `CHECKLIST.md`
3. `docs/design/lancet-implementation.md`
4. `docs/design/lancet-theory-code-alignment.md`
5. `experiments/protocols/antmaze_wsrl_lancet.md`
6. `.agents/lancet/memory/INDEX.md`
