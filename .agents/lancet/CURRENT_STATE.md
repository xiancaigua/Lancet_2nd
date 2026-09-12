# Lancet Current Agent State

Last updated: 2026-09-12 00:05 CST  
Branch: `main`
Commit: current infrastructure HEAD; formal training identity remains `62817637beffcbe8b1315d3c0daf03f1fc5fd9a0`

## Current objective

Operate the five seed-specific shared WSRL offline initializers through one
dynamic lifecycle queue. Lancet mathematics and the formal protocol are frozen.

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
- Formal online branches have not started. Each seed may enter its WSRL/Lancet
  online queue immediately after that initializer passes finite/hash/reload
  validation; it no longer waits for all five seeds.

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
- `scripts/experiments/run_training.py` is the single queue/watch entry point;
  tmux `lancet_formal_lifecycle` is active with PID 66688. The exact pushed
  infrastructure commit is recorded in `formal_pipeline.json`.
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

## Immediate next steps

1. Keep the detached lifecycle controller alive; inspect it with
   `python3 scripts/experiments/run_training.py status`.
2. Let replacement seed 1/4 use any non-4 GPU that passes the dynamic gate.
3. Validate each `offline_final.pt`; a failed validation must block online.
4. Permit seed-wise paired online launch only after shared lineage/reload checks.

## Read next

1. `AGENTS.md` and task-relevant `.agents/rules/`
2. `.agents/lancet/PLAN.md` and `CHECKLIST.md`
3. `docs/design/lancet-implementation.md`
4. `docs/design/lancet-theory-code-alignment.md`
5. `experiments/protocols/antmaze_wsrl_lancet.md`
6. `.agents/lancet/memory/INDEX.md`
