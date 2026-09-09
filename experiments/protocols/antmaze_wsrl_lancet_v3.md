# AntMaze WSRL vs Lancet v3 Benchmark Protocol

Status: **pre-registered draft; implementation and pilots not started**
Protocol date: 2026-09-09
Environment scope: `antmaze-medium-play-v2` only
Implementation specification: `docs/design/lancet-v3-implementation.md`

## 1. Research question

Primary question:

> With one seed-specific offline initialization, the WSRL base algorithm,
> dataset, actor/base-critic architecture, reward transform, online interaction
> budget, and evaluation protocol held fixed, does Lancet v3 repair the
> policy-facing critic faster immediately after offline-to-online handoff and
> improve early-online sample efficiency?

The main claim is early adaptation, not guaranteed asymptotic improvement.

Component questions are:

1. Does a fast residual branch help beyond WSRL?
2. Does action centering improve over an unrestricted but capacity-matched
   residual?
3. Does detached policy-sensitive disagreement weighting improve over the
   centered residual without changing correction direction?

## 2. Method identities

| Scientific label | Residual | Centering | U weighting | Role |
|---|:---:|:---:|:---:|---|
| WSRL | no | no | no | confirmatory baseline |
| Raw Residual v3 | yes | no | no | component ablation |
| Centered Residual v3 | yes | yes | no | component ablation |
| Full Lancet v3 | yes | yes | yes | confirmatory method |

All three v3 residual variants use the same shared-backbone/N-head network,
exact-zero output initialization, optimizer, local-action machinery,
minimal-surgery coefficient, update frequency, and handoff schedule.

The executable class currently registered as `lancet` is **Legacy Lancet-TD**:
one shared scalar, no action centering, mean-critic target, random nonzero
output, and no U weighting. It is excluded from this protocol. The historical
`antmaze_wsrl_lancet_v1.md` remains a legacy protocol record and must not be
used as v3 evidence.

## 3. Environment and base configuration

The first protocol covers only state-observation
`antmaze-medium-play-v2`. Later AntMaze environments require a new revision;
Kitchen is not a blocker.

The base recipe is the checked-in rl-garden WSRL AntMaze paper configuration:

| Parameter | Frozen value |
|---|---:|
| offline dataset | D4RL `antmaze-medium-play-v2`, 1,000,000 transitions |
| offline updates | 1,000,000 per seed, shared initializer |
| online steps | 500,000 per branch |
| UTD | 4.0 |
| batch size | 1024; high-UTD critic minibatch 256 |
| training frequency | 64 env steps |
| replay capacity | 1,000,000 |
| online replay | `empty`, offline ratio 0.0 |
| frozen warmup | 5,000 online env steps |
| reward transform | scale 10.0, bias -5.0 |
| gamma / tau | 0.99 / 0.005 |
| actor | 2 x 256, LayerNorm, uniform std |
| base critic | 10 critics, 4 x 256, LayerNorm |
| REDQ target subsample | 2 critics |
| base kernel init | orthogonal, near-zero output |
| actor / critic / alpha LR | 1e-4 / 3e-4 / 1e-4 |
| target entropy / entropy backup | 0.0 / false |
| offline objective | CQL + Cal-QL lower bound, alpha 5, autotune gap 0.8 |
| online CQL | disabled, alpha 0 |

`use_calql=true` is a deliberate rl-garden baseline choice and differs from
the official WSRL AntMaze script's plain-CQL initializer. It is not a Lancet
advantage because the identical checkpoint is shared by every branch, but
papers/results must name the baseline recipe accurately.

## 4. Shared offline checkpoint fork

For every seed `s` in `{0,1,2,3,4}`:

1. Run one WSRL-only initializer for exactly 1,000,000 offline updates and
   zero online steps.
2. Save `offline_final.pt`, frozen/resolved config, dataset hash, checkpoint
   hash, offline evaluation, and Git commit.
3. Construct each online branch from the same code commit and seed.
4. Strict-load the same checkpoint without replay state into WSRL, Raw,
   Centered, and Full branches.
5. Verify equality/hash of actor, base critic, target critic, alpha,
   actor/critic/alpha/CQL-alpha optimizers, global step/update, and phase state.
6. Verify every residual variant has exact `R=Delta=0`, a fresh optimizer,
   and therefore the same policy-facing Q as WSRL before online evidence.
7. Clear replay identically and collect the same 5,000-step frozen warmup.

Residual state is not added to the shared offline checkpoint. It is created at
the online branch with deterministic isolated initialization. This is fair
because every residual variant uses the same residual architecture/seed and
zero output, while WSRL receives no extra correction.

Independent Lancet offline pretraining is forbidden.

## 5. Lancet v3 frozen method parameters

These are design defaults for the implementation/smoke/pilot. They become
formal-frozen after the stability pilot confirms numerical viability; they
must not be tuned on five-seed formal results.

| Parameter | Value |
|---|---:|
| residual architecture | shared 2 x 128 backbone + N=10 scalar heads |
| output initialization | exact zero weight and bias |
| residual optimizer / LR | Adam / 1e-3 |
| local actions K | 8 |
| local set | replay action, current policy action, 6 perturbations |
| perturbation | clipped Gaussian, sigma 0.1 of action half-range |
| fitting coefficient | 1.0 |
| minimal-surgery coefficient mu | 1e-4 |
| U beta | 1.0 |
| U EMA decay | 0.99 |
| U weight clip | [1, 3] |
| U epsilon | 1e-8 |
| actor correction schedule | linear 1 to 0 |
| handoff window | 50,000 post-warmup env steps |
| residual update frequency | once per base critic minibatch |

U is detached REDQ action-dependent disagreement and enters only `w(s)` in
the residual fitting term. There is no `lambda_u_variation`, U target, or U
loss in v3.

## 6. Execution order and validation gates

```text
implementation design
  -> implement v3
  -> focused unit tests
  -> archived tiny real-data smoke for raw/centered/full construction
  -> seed-0 20k stability pilot (WSRL vs Full)
  -> review/freeze operational v3 config
  -> formal main comparison
  -> pre-registered component ablations
```

No later stage is authorized before the preceding gate passes. The pilot may
change a clearly unstable engineering hyperparameter, but the change requires
a protocol revision and new smoke evidence. Primary metrics cannot be changed
after pilot or formal results are inspected.

## 7. Seed-0 stability pilot

The pilot is `run_type=debug` and never enters a paper results table:

- one shared WSRL initializer: 20,000 offline updates, seed 0;
- WSRL online branch: 20,000 env steps;
- Full Lancet v3 online branch: 20,000 env steps;
- UTD 4, batch 1024, warmup 5,000;
- evaluation every 2,000 steps, 20 episodes;
- checkpoint every 5,000 steps plus final;
- outputs/checkpoints only under `/data/lancet/{runs,checkpoints}/debug/`.

Raw and Centered must each pass unit tests and a tiny real-data smoke, but do
not require separate 20k pilots unless their smoke exposes a variant-specific
problem.

### Pilot pass gate

- no NaN/Inf in losses, Q values, U/weights, parameters, or checkpoints;
- no unbounded critic/residual loss growth;
- actor/base critic/residual parameters remain finite;
- residual is exactly inactive offline and during warmup, then updates after
  warmup;
- exact-zero fork equality is recorded;
- evaluation completes and final endpoint evaluation is present;
- checkpoint save/load restores residual optimizer, EMA, schedule/counters,
  and local-action RNG;
- no unexpected OOM or environment/container failure;
- base and corrected Q, Delta, U, and weight trajectories are reviewable;
- residual domination diagnostics show no uncontrolled growth.

Record `mean(|Delta|)`, `mean(|Q_base|)`, ratio of means, and mean pointwise
ratio. A sustained ratio above 1 or >10x growth from the first active window is
an engineering warning requiring review, not a theoretical pass/fail threshold.

## 8. Formal matrices and compute staging

### Confirmatory main table: mandatory 10 online runs

| Environment | Method | Seeds | Online budget | Status |
|---|---|---|---:|---|
| antmaze-medium-play-v2 | WSRL | 0,1,2,3,4 | 500k each | planned |
| antmaze-medium-play-v2 | Full Lancet v3 | 0,1,2,3,4 | 500k each | planned |

This table directly tests the main early-repair claim at the lowest adequate
formal cost: 2 methods x 5 paired seeds = 10 online runs.

### Component ablation table: pre-registered 10 additional online runs

| Environment | Method | Seeds | Online budget | Status |
|---|---|---|---:|---|
| antmaze-medium-play-v2 | Raw Residual v3 | 0,1,2,3,4 | 500k each | planned ablation |
| antmaze-medium-play-v2 | Centered Residual v3 | 0,1,2,3,4 | 500k each | planned ablation |

These runs are required before claiming that centering or U weighting is the
source of improvement. They may be scheduled after the main table to control
cost, but they may not be selectively omitted because their early results are
unfavorable. If resources support only three-seed exploratory ablations, those
must be labeled exploratory and cannot support final component claims.

All 20 online cells reuse the same five offline initializers. Legacy Lancet-TD
is not a formal cell.

## 9. Evaluation metric contract

Evaluation uses one deterministic, separately seeded evaluation environment,
20 complete 1,000-step AntMaze episodes per evaluation, and nominal milestones
every 2,000 online steps, including online step 0 and forced 50k, 100k, and
500k endpoints.

The D4RL adapter reports `success_at_end`. rl-garden derives
`normalized_score = 100 * success_at_end`; these are two units for the same
scientific endpoint, not independent metrics.

### Primary

**Normalized early-online AUC from 0 to 50,000 steps:**

```text
AUC_0_50k = (1 / 50,000) * integral_0^50k normalized_score(x) dx.
```

Use trapezoidal integration over actual logged online-step coordinates with
linear interpolation at 50k if rollout boundaries overshoot it. The shared
offline checkpoint evaluation is the common x=0 anchor. This metric directly
tests the 50k handoff mechanism and remains in normalized-score units.

### Secondary

- normalized AUC 0–100k;
- full normalized AUC 0–500k;
- final-window mean over the last 10 evaluations;
- forced 500k endpoint score;
- complete learning curve with all seed traces.

### Diagnostic only

- first step reaching normalized-score thresholds 20, 40, and 60 for two
  consecutive evaluations; unreached values are right-censored;
- best score (never a headline or rerun criterion);
- wall-clock/GPU efficiency;
- critic/residual mechanism diagnostics below.

Return, success fraction, and normalized score may all be logged, but results
tables must not pretend success and normalized score are statistically
independent outcomes.

## 10. Mechanism diagnostics

At the same logging cadence for all methods where defined:

- base critic loss, actor loss, alpha, entropy, policy std;
- per-critic observed TD-residual mean/std/RMSE;
- base-Q and corrected-Q mean/std;
- raw R and Delta mean/std/absolute mean;
- residual fit, minimal-surgery, and total loss;
- U mean/std/EMA and weight mean/max;
- lambda, adaptation environment step, residual update count;
- `mean(|Delta|)/(mean(|Q_base|)+eps)` and pointwise ratio mean;
- parameter/gradient finite flags and peak GPU memory.

For Centered/Full, verify mean action-centered Delta is approximately zero per
critic/state. For Full, correlate U with observed TD-residual energy as a
diagnostic only; U is not claimed to be true error.

Optional phase-2 offline analysis may sample multiple actions for the same
state and report action-wise Q/Delta variation, rank consistency/change,
top-1 action match, MC regret, or oracle projection error where ground truth
is available. These do not block this AntMaze protocol.

## 11. Statistical reporting

For each method and metric report all five seed values, mean, sample standard
deviation (`ddof=1`), and 95% t confidence interval.

The confirmatory comparison is the paired per-seed difference
`Full Lancet v3 - WSRL` on primary `AUC_0_50k`. If a test is reported, use a
predeclared two-sided paired t-test at alpha 0.05. Effect size, paired values,
and confidence interval take precedence over binary significance with n=5.

Component contrasts are:

- Centered minus Raw: centering effect;
- Full minus Centered: U-weighting effect.

If inferential p-values are reported for these two contrasts, use Holm
correction as one predeclared family. Secondary metrics are estimation and
diagnostic evidence; do not choose a new headline after observing results.

## 12. Checkpoint and evaluation cadence

- one shared `offline_final.pt` per seed;
- formal online checkpoints every 25,000 env steps plus `final.pt`;
- debug checkpoints every 5,000 env steps plus final;
- evaluation every 2,000 online steps, with explicit step-0 and final
  endpoint evaluations;
- checkpoint hash and parent offline-checkpoint hash in metadata;
- reload validation on the shared offline checkpoint, one periodic online
  checkpoint, and final checkpoint.

Lancet v3 reload must restore residual network/optimizer, U EMA,
adaptation-start step, residual update count, lambda schedule state, local RNG,
and inherited WSRL state.

## 13. Archives, identity, and hardware

Use the existing archive workflow and type-separated roots:

```text
/data/lancet/runs/{debug,formal}/antmaze-medium-play-v2/<method>/seed_<s>/<timestamp>/
/data/lancet/checkpoints/{debug,formal}/antmaze-medium-play-v2/<method>/seed_<s>/<timestamp>/
```

Method path labels are `wsrl`, `lancet-v3-raw`, `lancet-v3-centered`, and
`lancet-v3-full`; metadata also records registry algorithm `lancet_v3`,
variant, protocol id, Git commit, parent checkpoint/hash, dataset hash, and
selected GPU id.

Each process uses one explicit GPU. Parallel seeds may use different verified
idle GPUs only after the pilot measures peak memory. Long runs use Host tmux or
a scheduler around the archive launcher; no Codex foreground formal run.

## 14. Seed and RNG protocol

Use seeds 0–4 for Python, NumPy, PyTorch CPU/CUDA, environment, replay/dataset
sampling, and network initialization. Lancet-exclusive residual initialization
and local perturbations use isolated deterministic RNG streams so they do not
advance the base WSRL training RNG. Evaluation uses a separate deterministic
seed stream identical within each paired comparison.

Bitwise CUDA determinism is not promised. Record numerical-determinism
limitations and never trade away the paper-scale algorithm merely to claim
bitwise identity.

## 15. Failure and rerun policy

Reruns are allowed only for OOM, host/container/environment failure, corrupted
checkpoint, launcher failure, or a verified code/config bug. Preserve the old
archive with failed/stopped/invalid status and an explicit reason.

Low return, high variance, a bad seed, or an unfavorable method result never
justifies rerun. A code fix requires a new Git commit and explicit invalidation
and replacement links. Never overwrite an archive.

## 16. Git and readiness gates

Formal runs require a clean, pushed Git commit shared by every compared
method, frozen configs, frozen dataset/checkpoint hashes, and protocol id.

### Benchmark protocol implementation readiness

Overall classification: **BLOCKER**. The protocol and base runtime are ready
for implementation, but no v3 algorithm exists yet. `NEEDS SMALL PATCH` items
are bounded infrastructure/configuration changes; none justifies starting a
legacy run as a substitute.

| Class | Requirement | Current implementation | Evidence / file | Action required |
|---|---|---|---|---|
| READY | Runtime and AntMaze dataset | Container/GPU and 1M-transition AntMaze data passed prior audits | `scripts/audit/`; archived audit/smoke evidence | preserve |
| READY | WSRL base target/loss and high UTD | UTD=4 performs four critic/hook steps and one actor step | `rl_garden/algorithms/sac_core.py`, `cql.py`, `wsrl.py` | add v3 parity/counting tests only |
| READY | Type-separated archives and clean-formal guard | launcher separates smoke/debug/formal and checks dirty Git state | `scripts/experiments/archive_run.py` | protocol forbids the emergency dirty override |
| READY | Generic extensible checkpoint mechanism | inherited policy/target/optimizer/training-state hooks exist | `rl_garden/algorithms/base_algorithm.py`, `common/checkpoint.py` | use hooks; do not replace format |
| NEEDS SMALL PATCH | Shared WSRL checkpoint fork | current full YAMLs each include offline and online phases; v3 loader does not exist | `configs/off2on/wsrl_antmaze_medium_play_v2_paper.yaml`, current Lancet config | add initializer/fork configs and strict base-equality test |
| NEEDS SMALL PATCH | Output/checkpoint cadence | generic paths work, but current paper config has no periodic checkpoint cadence | same configs and archive launcher | freeze 25k formal / 5k debug plus final |
| NEEDS SMALL PATCH | Seed/evaluation contract | training env is seeded; evaluation reset is not independently seeded and final endpoint is not forced | `rl_garden/algorithms/off_policy.py`, `base_algorithm.py` | add isolated eval seed and explicit step-0/final eval |
| NEEDS SMALL PATCH | Metrics and diagnostics | success/normalized score exists; required early AUC and most v3 mechanism tags do not | D4RL backend, logger, current legacy `lancet.py` | add analysis metric derivation and pure diagnostic logging |
| NEEDS SMALL PATCH | GPU/variant/lineage metadata | archives record generic hardware/Git data but not every v3 comparison field | `scripts/experiments/archive_run.py` | add selected GPU, variant, protocol, parent/hash fields |
| BLOCKER | V3 algorithm, registry, and configs | only shared-scalar Legacy Lancet-TD is executable | `rl_garden/algorithms/lancet.py`, `training/off2on/lancet.py` | implement the reviewed `lancet_v3` specification |
| BLOCKER | V3 unit/smoke/pilot evidence | no v3 test or experiment has run | tests and archives | pass focused tests, tiny smoke, then seed-0 debug pilot |

Formal launch is not authorized.

## 17. Readiness checklist

- [ ] Lancet v3 implementation manually reviewed.
- [ ] WSRL base target/loss parity test passed.
- [ ] V3 tensor/gradient/lifecycle/UTD tests passed.
- [ ] Raw/Centered/Full use matched architecture and local actions.
- [ ] Exact-zero fork `Q_use==Q_base` verified.
- [ ] Shared offline checkpoint strict-load/base-state equality verified.
- [ ] Eval seed and exact endpoint behavior verified.
- [ ] Variant/GPU/lineage metadata verified.
- [ ] Tiny v3 real-data smoke passed.
- [ ] WSRL seed-0 20k debug pilot passed.
- [ ] Full Lancet v3 seed-0 20k debug pilot passed.
- [ ] Diagnostics and reload are finite/complete.
- [ ] Working tree clean; frozen commit pushed.
- [ ] Five shared offline initializer configs/hashes frozen.
- [ ] Main 10-run matrix and metrics frozen.
- [ ] Formal output roots checked before launch.

## 18. Open scientific questions

The v3 equations are specified; U is no longer an undefined loss. Remaining
scientific uncertainties are empirical:

1. whether single observed-action TD supervision plus centering approximates
   the oracle action-relative projection well enough;
2. whether REDQ action-dependent disagreement identifies states where faster
   residual fitting improves policy repair;
3. whether the recommended K/noise/window/U-weight settings transfer beyond
   `antmaze-medium-play-v2`.

These questions are answered by predeclared experiments, not by changing the
objective after results are seen.
