# Lancet Current Agent State

Last updated: 2026-09-15 CST
Branch: `integration/upstream-sync-20260914`
Commit: `824a5de` plus pending gate/continuity commit

## Current objective

Promote the fully validated upstream integration to `main`, freeze generation-2
formal identity, then run only the corrected WSRL seed-0 baseline-validation
initializer. Generation-1 formal evidence is superseded and no training worker
from it is active.

## Repository and runtime

- Host: `/home/zhaozihan/Lancet/rl-garden`
- Container: `/workspace/rl-garden`
- Data: `/home/zhaozihan/Lancet/data` -> `/data/lancet`
- Image/container: `lancet-d4rl:cu128` / `lancet-d4rl`
- Python/Torch/CUDA: 3.10.21 / 2.7.0+cu128 / 12.8
- GPU: 6 × RTX 4090; GPU 4 is reserved for another user
- Standard execution: `./dev d4rl <command>`

## Upstream migration identity

- Old main: `db57e1e9e1ee6e81ffafd99b1547a96e26389df1`
- Upstream: `252d1e0948618a0cd3675b9a05e0bcf4c29b5afb`
- Full merge: `c1125de`
- Backup: `backup/pre-upstream-sync-20260914` and annotated tag of the same name
- Integration promotion gate: PASS; main promotion is pending

## Lancet architecture

```text
one unchanged WSRL Bellman target y
  -> normal base critic optimizer step
  -> recompute Q_i_updated
  -> fit exact-zero, critic-aligned residual to stopgrad(y - Q_i_updated)
  -> K=8 Raw/Centered/Lancet correction; U is a detached fit weight only
  -> actor min_i(Q_i + Delta_i) during the 50k active handoff window
```

Lancet is inactive offline and through the 5k frozen warmup. The base critic,
target critic, CQL/Cal-QL, and REDQ target semantics remain upstream WSRL.

## Verified status

- Lancet migrated to canonical Dict observations; B×K repeat covers all keys.
- Corrected AntMaze WSRL CQL settings and `bootstrap_at_done=truncated` frozen.
- Resolved base parity: 118 fields identical for WSRL/Raw/Centered/Lancet.
- Dataset: raw SHA and every audited semantic prefix field match old/new loaders.
- Tests: Lancet 71 passed; core 174; observation/replay/off2on 195; latest key rerun 50 passed.
- Exact checkpoint continuation and scientific inactive/active parity pass.
- Real AntMaze smoke passed with finite updates, residual activity, evaluation,
  checkpoint save/reload, and shared-fork equality.
- SMTP upstream-merge test passed without exposing credentials.
- Ruff 0.16.6 no-new-regressions gate passes: clean upstream and integration
  have identical normalized 2,610-finding multisets; all 24 added Python files
  and Lancet-owned files are clean.

## Formal generations

- Generation 1: `SUPERSEDED_PRE_UPSTREAM_SYNC`, retained unchanged for history.
- Generation 2: not yet created; no initializer or online run has started.
- Old checkpoints must never initialize generation-2 scientific runs.

## Known issues

1. Upstream carries 2,610 repository-wide Ruff findings; they are baseline debt,
   not integration regressions.
2. Legacy optional D4RL/headless warnings remain non-blocking for AntMaze.
3. Corrected WSRL seed-0 must pass Stage A/B before any multi-seed formal launch.

## Active decisions

- No force push and no mixing formal generations.
- `target_entropy=0.0` is retained; the upstream `auto` change was unrelated to
  the validated AntMaze CQL correction.
- Primary metric remains Adaptation AUC 0–50k.
- Promotion uses a no-new-lint-regressions gate against clean upstream rather
  than mass-editing unrelated lint debt.

## Immediate next steps

1. Commit and push the final integration gate evidence.
2. Recheck origin/main, promote integration normally, and tag the sync.
3. Freeze `FORMAL_IDENTITY_V2.json` and formal_v2 roots.
4. Archive and launch corrected WSRL seed-0 1M offline initializer only.

## Read next

1. `AGENTS.md` and task-relevant `.agents/rules/`
2. `experiments/migrations/upstream_sync_20260914/upstream_migration_gate_report.md`
3. `experiments/protocols/antmaze_wsrl_lancet.md`
4. `docs/design/lancet-implementation.md`
5. `.agents/lancet/memory/INDEX.md`
