# AntMaze WSRL vs Lancet Benchmark Protocol

Status: **pre-registered; implementation/runtime gates passed; formal runs not started**
Protocol date: 2026-09-09
Environment: `antmaze-medium-play-v2`
Implementation contract: `docs/design/lancet-implementation.md`

## 1. Research question

> With the seed-specific offline initializer, WSRL base algorithm, dataset,
> actor/base-critic architecture, reward transform, online budget, and
> evaluation protocol held fixed, does Lancet improve policy-facing critic
> repair and sample efficiency immediately after offline-to-online handoff?

The confirmatory claim is early adaptation. The latest method is **Lancet**,
not “Lancet v3”. Git history contains one confirmed legacy executable, now
called **Lancet V1**; it is excluded from this protocol.

## 2. Methods and contrasts

| Method | Residual | Centering | U fit weight | Role |
|---|:---:|:---:|:---:|---|
| WSRL | no | no | no | confirmatory baseline |
| Raw Residual | yes | no | no | component ablation |
| Centered Residual | yes | yes | no | component ablation |
| Lancet | yes | yes | yes | confirmatory method |

The three residual variants share the same shared-backbone/N-head network,
zero initialization, optimizer/LR, K=8 local actions, perturbation RNG,
regularizer, update frequency, checkpoint format, and 50k active lifecycle.
Thus WSRL→Raw tests correction, Raw→Centered tests projection, and
Centered→Lancet tests U weighting.

## 3. Frozen WSRL base recipe

The paper recipe is retained in `configs/off2on/wsrl_antmaze_medium_play_v2_paper.yaml`. Frozen fork-ready configs are:

- shared offline: `configs/off2on/wsrl_antmaze_medium_play_v2_offline_initializer.yaml`;
- WSRL online: `configs/off2on/wsrl_antmaze_medium_play_v2_online.yaml`;
- Lancet online: `configs/off2on/lancet_antmaze_medium_play_v2.yaml`;
- ablations: `configs/off2on/lancet_raw_antmaze_medium_play_v2.yaml` and `lancet_centered_antmaze_medium_play_v2.yaml`.

They must resolve identically for every base parameter across online branches.

| Parameter | Frozen value |
|---|---:|
| dataset | D4RL `antmaze-medium-play-v2`, 1,000,000 transitions |
| offline updates | 1,000,000 once per seed |
| nominal online budget | 500,000 env steps per branch |
| UTD / batch | 4 / 1024; four 256-example critic minibatches |
| train frequency | 64 env steps |
| replay | capacity 1,000,000; online `empty`; offline ratio 0 |
| frozen warmup | first 5,000 online env steps |
| reward scale / bias | 10 / -5 |
| gamma / tau | 0.99 / 0.005 |
| actor | 2×256, LayerNorm, uniform std |
| base critic | 10 × 4×256, LayerNorm |
| REDQ target subsample | 2 critics |
| actor / critic / alpha LR | 1e-4 / 3e-4 / 1e-4 |
| target entropy / entropy backup | 0 / false |
| offline objective | CQL + Cal-QL lower bound; alpha 5; target gap 0.8 |
| online CQL | disabled |
| evaluation cadence | nominally every 2,000 online steps; 20 episodes |
| formal checkpoint cadence | every 25,000 actual online steps plus final |

`use_calql=true` is the existing rl-garden WSRL recipe rather than the
official JAX plain-CQL initializer. It is shared by all branches and must be
named accurately in reporting.

## 4. Shared offline checkpoint fork

For each seed in `{0,1,2,3,4}`:

1. Train WSRL for 1,000,000 offline updates and save one `offline_final.pt`.
2. Archive its resolved config, Git commit, dataset/checkpoint hashes, and
   offline evaluation.
3. Load that exact checkpoint into WSRL, Raw, Centered, and Lancet online
   branches without replay state.
4. Verify actor, base critic, target critic, alpha, all base optimizer states,
   global counters, and phase state are identical at the fork.
5. Construct each residual branch only at the fork with an isolated seed and
   exact-zero final heads; verify `Q_use==Q_base` before online evidence.
6. Clear/collect replay and execute the same frozen 5k warmup.

Independent residual-method offline pretraining is forbidden.

## 5. Frozen Lancet parameters and update semantics

| Parameter | Value |
|---|---:|
| residual | shared 2×128 backbone + N=10 heads |
| output initialization | exact zero weight and bias |
| optimizer / LR | Adam / 1e-3 |
| local action set | K=8: replay, deterministic actor, six perturbations |
| perturbation | Gaussian sigma 0.1 × action half-range, clipped |
| fit / minimal-surgery coefficients | 1 / 1e-4 |
| U beta / EMA decay / cap / eps | 1 / 0.99 / 3 / 1e-8 |
| active handoff window | adaptation steps `[0,50000)` |
| active lambda | constant 1; then exactly 0 |
| residual updates | one after every base critic minibatch while active |

For each critic minibatch, compute Bellman target `y` once, perform the
unchanged base critic step, recompute updated `Q_i`, then fit
`stopgrad(y-Q_i_updated)` without calling stochastic `_target_q()` again.
The actor uses `min_i(Q_i+Delta_i)`. Its numerator `R(s,a_actor)` keeps action
gradient while the local residual mean is detached. The target critic never
uses residual correction.

For Lancet only:

```text
w = clip(1 + beta * U / (EMA(U) + eps), 1, w_max)
```

U is detached and only weights residual fitting. EMA is initialized to the
first active batch mean, not zero. There is no U loss, U target, or
`lambda_u_variation`.

## 6. Lifecycle and coordinates

- offline: residual update/correction inactive;
- online warmup 0–5k: inactive;
- first trainable step after warmup: `adaptation_step=0`;
- adaptation steps `[0,50000)`: lambda=1, residual updates active;
- adaptation step `>=50000`: lambda=0, residual updates stop, actor uses base Q.

Logs retain three distinct coordinates: `global_step` includes offline and
online work, `online_step` starts at the online switch, and `adaptation_step`
starts at the first trainable step after the frozen warmup. A 50k adaptation
window therefore corresponds approximately to online steps 5k–55k.

The nominal online budget is 500k. With `train_freq=64`, do not modify rollout
behavior to land on exactly 500000. The final endpoint is the first actual
`online_step >= 500k` common to all compared methods. Reports must include its
global and online coordinates and must never label `global_step` as online.

## 7. Validation sequence and seed-0 debug pilot

```text
implementation -> unit/regression tests -> archived tiny real-data smoke
-> shared seed-0 stability pilot -> formal readiness review -> formal runs
```

The debug pilot is not paper evidence:

- shared WSRL initializer: 20,000 offline updates, seed 0;
- WSRL branch: 20,000 nominal online steps;
- Lancet branch: 20,000 nominal online steps;
- UTD=4, batch=1024, warmup=5k;
- evaluation every 2k, 20 episodes;
- checkpoints every 5k plus final;
- outputs only under `/data/lancet/{runs,checkpoints}/debug/`.

Pass requires finite losses/Q/Delta/U/weights/parameters, successful evaluation
and checkpoint reload, no OOM, residual inactivity before adaptation, real
residual updates after adaptation, and no uncontrolled residual domination.
Record `mean(|Delta|)`, `mean(|Q_base|)`, their ratio of means, and the mean
pointwise ratio. A sustained ratio >1 or >10× early growth is a diagnostic
warning for review, not a theoretical threshold.

## 8. Formal matrices

Main comparison (10 online runs):

| Environment | Method | Seeds | Budget | Status |
|---|---|---|---:|---|
| antmaze-medium-play-v2 | WSRL | 0,1,2,3,4 | nominal 500k | planned |
| antmaze-medium-play-v2 | Lancet | 0,1,2,3,4 | nominal 500k | planned |

Component ablations (10 further online runs):

| Environment | Method | Seeds | Budget | Status |
|---|---|---|---:|---|
| antmaze-medium-play-v2 | Raw Residual | 0,1,2,3,4 | nominal 500k | planned |
| antmaze-medium-play-v2 | Centered Residual | 0,1,2,3,4 | nominal 500k | planned |

Raw and Centered launch registry `lancet` with archive `--method-label raw-residual` or `centered-residual`; the method label changes paths/metadata only. All 20 online cells reuse the same five offline initializers. Ablations may be
scheduled after the main pipeline is confirmed, but cannot be selectively
discarded after unfavorable results.

## 9. Evaluation metrics

Evaluation uses a separate deterministic seed stream shared within each paired
comparison, 20 complete AntMaze episodes, a common pre-adaptation anchor, and
actual logged coordinates. In this backend,
`normalized_score = 100 * success_at_end`; these are the same scientific
endpoint in different units.

### Primary

**Adaptation AUC 0–50k**, measured from the first post-warmup trainable step:

```text
AUC_adapt_0_50k = (1/50000) * integral_0^50000 normalized_score(t) dt
```

Use trapezoidal integration and interpolate only at the fixed adaptation
boundary when needed.

### Secondary

- Adaptation AUC 0–100k;
- full online AUC through the common actual online endpoint;
- final-window mean over the last 10 evaluations;
- score at the common actual final online endpoint;
- complete paired-seed learning curves.

### Diagnostic

- steps-to-threshold 20/40/60 when reached for two consecutive evaluations;
- best score (never headline evidence or a rerun criterion);
- wall-clock/GPU use and mechanism diagnostics.

## 10. Mechanism diagnostics

Log, without changing training RNG:

- base critic/actor losses, alpha, entropy/policy std;
- per-critic post-update TD residual mean/std/RMSE;
- base-Q and corrected-Q mean/std;
- raw R and Delta mean/std/absolute mean;
- residual fit, minimal-surgery, and total loss;
- U mean/std/EMA and weight mean/max;
- lambda, online/adaptation step, residual update count;
- residual/base-Q ratio-of-means and mean pointwise ratio;
- finite flags and peak GPU memory.

Action-wise rank/variation analysis is optional phase-2 evidence, not a gate.

## 11. Statistics

Report every seed value, mean, sample SD (`ddof=1`), and 95% t confidence
interval. The confirmatory estimate is paired `Lancet-WSRL` on primary
Adaptation AUC 0–50k. If a test is reported, use a predeclared two-sided paired
t-test at alpha .05. Report effect and CI regardless of p-value.

Ablation contrasts are Centered−Raw and Lancet−Centered. If reporting their
p-values, apply Holm correction as one family. Do not change the primary metric
after seeing results.

## 12. Checkpoints, archives, hardware, and reruns

Save the shared offline checkpoint, periodic online checkpoints, and final
checkpoint. Lancet reload includes inherited actor/base/target/alpha and base
optimizers plus residual network/optimizer, EMA, adaptation counters, variant,
and local RNG.

Use the existing archive roots and method labels:

```text
/data/lancet/runs/{debug,formal}/antmaze-medium-play-v2/{wsrl,raw-residual,centered-residual,lancet}/seed_<s>/<timestamp>/
/data/lancet/checkpoints/{debug,formal}/...
```

Each process uses one explicit GPU; parallel runs require distinct verified
idle GPUs. Long jobs use Host tmux or scheduler around the archive launcher.

Formal runs require one clean, pushed Git commit for all methods. Reruns are
allowed only for OOM, host/container/environment failure, corrupted checkpoint,
launcher failure, or a verified code/config bug. Preserve and invalidate old
archives explicitly. Poor return or an unfavorable seed never permits rerun.

## 13. Readiness checklist

- [x] Lancet implementation review gate accepted before formal authorization.
- [x] WSRL target/base-loss parity and baseline regression tests passed.
- [x] Tensor, gradient, lifecycle, UTD, and checkpoint tests passed.
- [x] Raw/Centered/Lancet matched-machinery behavior verified.
- [x] Shared checkpoint base-state and zero-correction equality verified.
- [x] Evaluation seed, adaptation coordinates, and common endpoint verified.
- [x] Tiny real-data smoke passed.
- [x] WSRL seed-0 debug pilot passed.
- [x] Lancet seed-0 debug pilot passed.
- [x] Diagnostics and reload are finite (except peak-memory scalar, which was
  not collected; no OOM occurred).
- [ ] Formal configs, dataset/checkpoint hashes, commit, and matrix frozen.
- [ ] Final clean formal-infrastructure commit pushed.
- [ ] Formal roots rechecked immediately before launch.

Formal launch remains unauthorized until every applicable item passes.

## 14. Open scientific questions

The implementation objective is frozen. Remaining questions are empirical:

1. whether observed-action TD supervision plus centering approximates the
   oracle action-relative correction sufficiently;
2. whether REDQ action disagreement usefully targets faster residual fitting;
3. whether K=8, sigma=.1, the 50k active window, and U weights transfer beyond
   `antmaze-medium-play-v2`.

These are answered by the declared experiments, not post-hoc objective changes.
