# Lancet v3 Implementation Specification

Status: design frozen for implementation review; **not implemented**
Date: 2026-09-09
Theory source: `docs/design/Lancet_v3_技术实现思路与理论证明.pdf`
Code baseline: rl-garden commit `ede0935942e94cc1a2b6f9889bb738b8f1e5b5ea`

## 1. Scope and invariants

Lancet v3 is an online-handoff extension of WSRL. It is not a replacement
critic and it does not participate in offline pretraining.

The following WSRL behavior is invariant:

- the Bellman target and target-critic REDQ subsampling;
- the base critic TD/CQL/Cal-QL loss and optimizer step;
- the target-critic Polyak update;
- the replay transition and reward/observation preprocessing;
- the offline checkpoint contents and the 5,000-step frozen warmup;
- the fact that the target critic never contains a Lancet correction.

The existing `rl_garden/algorithms/lancet.py` does not satisfy this
specification. It remains the **Legacy Lancet-TD / shared raw-residual
scaffold** for historical checkpoint and smoke-test reproducibility.

## 2. Current update chain and v3 insertion point

The real WSRL chain is:

```text
WSRL
  -> _CalQLRolloutTrainingShell
     -> CQLCore / SACCore
        -> _target_q(data)                    # target [B, 1]
        -> _critic_loss(data)                 # base Q [N, B, 1]
        -> q_optimizer.step()
        -> _post_critic_update(data, info)    # Lancet v3 residual step
        -> _actor_loss_from_batch(data)       # Lancet v3 actor Q
        -> actor_optimizer.step()
        -> _target_update()                   # base critic only
```

For integer `utd=4`, `SACCore.train_high_utd()` samples one batch of 1024,
splits it into four minibatches of 256, performs four critic updates and four
`_post_critic_update` calls, then performs one actor update on the full batch.
Lancet v3 follows that schedule: one residual update per base-critic
minibatch, then one actor update.

## 3. Residual architecture decision

| Candidate | Theory alignment | Capacity/fairness | Decision |
|---|---|---|---|
| One shared scalar residual | Cannot fit critic-specific TD residuals; `min(Q)+R` commutes and does not repair ensemble members separately | smallest, but wrong v3 object | reject for v3; legacy only |
| N independent residual MLPs | Correct per-critic outputs | roughly N-fold trunk parameters; changes capacity and instability substantially | reject as default; optional future capacity ablation |
| Shared backbone + N heads | Per-critic correction with shared state-action representation | modest parameter growth; head identity aligns with REDQ critic identity | **selected** |

`ResidualEnsemble` supports state-observation inputs only in the first implementation
(it still consumes both state and action):

```text
[state, action]
  -> Linear(128) -> LayerNorm -> ReLU
  -> Linear(128) -> LayerNorm -> ReLU
  -> Linear(N)                         # N independent output rows/heads
```

The final `Linear(N)` weight and bias are initialized to exact zero. The
backbone may use the normal deterministic initialization. Consequently the
first residual backward updates only the zero head; backbone gradients become
nonzero after the head leaves zero. This one-step behavior is expected and
preserves exact fork equality. A forward pass
returns `[N, B, 1]`; a local-action forward returns `[N, B, K, 1]`.

The shared backbone deliberately couples representation learning between
heads, but the base REDQ ensemble remains untouched and retains its epistemic
structure. An `H -> N` linear layer is exactly N independent scalar heads over
one shared feature vector.

Recommended v3 defaults are two 128-wide hidden layers and residual LR
`1e-3` (3.33 times the base critic LR `3e-4`). These values must be frozen
before the stability pilot, not selected after formal results.

## 4. Tensor contracts

Let N be critic count, B minibatch size, K local-action count, and A action
dimension. The first AntMaze configuration uses `N=10`, `K=8`, full batch
1024, and high-UTD minibatch `B=256`.

| Symbol / code object | Shape | Gradient owner |
|---|---|---|
| `target_y` | `[B, 1]` | none; detached target |
| `Q_all` | `[N, B, 1]` | base critic only in base loss |
| `local_actions` | `[B, K, A]` | none; action-set generation detached |
| `Q_local` | `[N, B, K, 1]` | none in U computation |
| `R_local` | `[N, B, K, 1]` | residual |
| `Delta_local` | `[N, B, K, 1]` | residual |
| `Delta_observed` | `[N, B, 1]` | residual |
| `TD_residual` | `[N, B, 1]` | none; `stopgrad(target_y - Q_all)` |
| `U` | `[B, 1]` | none |
| `w` | `[B, 1]` | none; broadcasts over N |
| `Q_use_actor` | `[N, B, 1]` | actor through action only |

`target_y.unsqueeze(0)` broadcasts to `[N,B,1]`. No mean over critics is
taken when constructing the TD-residual target.

## 5. Local action set

Use `K=8` and finite Box action bounds. For residual fitting at replay state
`s_b`, construct:

```text
index 0: replay action a_b
index 1: deterministic current actor action pi_mean(s_b)
index 2..7: clip(pi_mean(s_b) + sigma * action_scale * epsilon)
```

where `epsilon ~ Normal(0,I)`, `sigma=0.1`, and
`action_scale=(high-low)/2`. Every action is clipped elementwise to the actual
Box bounds. Deterministic actor action is preferred for residual fitting
because it introduces no extra policy-sampling noise into the TD target. The
current concrete API is `self.policy.predict(data.obs, deterministic=True)`
under `torch.no_grad()`; the actor path continues to use
`self.policy.actor_action_log_prob(...)` for its existing reparameterized
sample.

For the actor update, index 1 is a detached copy of the stochastic action
already sampled by SAC for its actor loss, and perturbations are centered on
that detached copy. The center is evaluated from this detached local set,
while `R(s,a_actor)` is evaluated once more with the differentiable actor
action. Therefore the action distribution `q(.|s)` is stop-gradient, but
`d R(s,a_actor) / d a_actor` is retained.

Replay action is included because it is the only action with an observed TD
residual. The policy action and perturbations define the policy-local geometry
that the actor is likely to compare. Local Gaussian draws use a dedicated
`torch.Generator`; they must not advance the base SAC/replay RNG stream.

## 6. Unified correction variants

All new residual variants use the same network, zero initialization,
optimizer, K, action set, perturbations, update schedule, and handoff schedule.

```text
raw:
    Delta_local = R_local

centered:
    Delta_local = R_local - mean_k(R_local)

full:
    Delta_local = R_local - mean_k(R_local)
    fitting weight = w(U)
```

For centered/full, `Delta_local.mean(dim=2)` is zero up to floating-point
roundoff for every critic and state. For raw, U is still computed for matched
diagnostics and local machinery, but `w=1`. Centered also uses `w=1`.

WSRL is a separate no-residual algorithm. Legacy shared-scalar Lancet-TD is
not one of these three v3 variants and must be named `Legacy Lancet-TD`.

## 7. TD supervision and stop-gradient boundary

The base critic computes its original target `y` and takes its normal
optimizer step. Lancet must use that exact target sample; it must not call
`_target_q()` again because stochastic target action sampling could produce a
different target.

The least invasive implementation is a v3-only `_td_loss` override that is
algebraically identical to `CQLCore._td_loss`, stores `target_y.detach()` for
the following hook, and returns the unchanged base loss. A parity test is
required. After the base critic step:

```python
with torch.no_grad():
    q_after = self._critic_forward(data.obs, data.actions, target=False)
    td_residual = target_y.unsqueeze(0) - q_after
```

Thus:

```text
target_y:       [B, 1]
q_after:        [N, B, 1]
td_residual:    [N, B, 1]
Delta_observed: [N, B, 1]
```

Both `target_y` and `q_after` are detached. For residual fitting, the replay
action is local-action index 0, so `Delta_observed = Delta_local[:, :, 0, :]`.
The residual receives separate
supervision for every critic head. The target critic and Bellman target remain
residual-free.

## 8. REDQ action-dependent uncertainty

For a fixed state b, use population variance over K actions:

```text
U_b = 2 / (N(N-1)) * sum_{i<j} Var_k(Q_i(b,k) - Q_j(b,k)).
```

Define action-centered critic values

```text
C_i,b,k = Q_i,b,k - mean_k Q_i,b,k.
```

Then `Var_k(Q_i-Q_j) = mean_k(C_i-C_j)^2`. Applying

```text
sum_{i<j} (x_i-x_j)^2 = N * sum_i (x_i-mean_i x)^2
```

at every `(b,k)` gives the exactly equivalent efficient form:

```text
U_b = (2N/(N-1)) * mean_k Var_i(C_i,b,k),
```

where `Var_i(..., unbiased=False)` uses denominator N. For `N=2`, ensemble
population variance is `(C_1-C_2)^2/4`; multiplication by 4 yields
`mean_k(C_1-C_2)^2 = Var_k(Q_1-Q_2)`, exactly the twin-critic formula.

Implementation:

```python
with torch.no_grad():
    centered_q = q_local - q_local.mean(dim=2, keepdim=True)
    u = (2 * n_critics / (n_critics - 1)) * (
        centered_q.var(dim=0, unbiased=False).mean(dim=1)
    )  # [B, 1]
```

U is a disagreement/risk proxy, not an error target. It is detached and only
changes the residual fitting weight. It never changes the base critic, target,
replay sampling, or correction direction.

Use a scalar EMA of the batch mean:

```text
ema_u <- 0.99 * ema_u + 0.01 * mean_b(U_b)
w_b = clip(1 + 1.0 * U_b / (stopgrad(ema_u) + 1e-8), 1, 3)
```

Initialize `ema_u` from the first active batch mean rather than zero. Save it
in checkpoint training state. This normalization is deliberately scalar and
simple; no learned uncertainty model or U-loss is introduced.

## 9. Residual objective

For `w=1` in raw/centered and the detached U weight in full:

```text
L_fit   = mean_{i,b} w_b * (Delta_observed_i,b - TD_residual_i,b)^2
L_small = mean_{i,b,k} Delta_local_i,b,k^2
L_R     = residual_fit_coef * L_fit + residual_small_coef * L_small
```

Recommended defaults are `residual_fit_coef=1.0` and
`residual_small_coef=1e-4`. In centered/full, the regularizer is explicitly on
centered `Delta`, never on raw `R`. In the raw ablation `Delta=R` by definition,
so the identical minimal-surgery term applies to the raw correction.

Only residual parameters receive gradients from `L_R`. Base critics, target,
U, local actions, and actor actions used to form the residual-training set are
detached.

## 10. Actor-facing repaired ensemble

Override `_actor_loss_from_batch(data)` rather than `_actor_loss(obs)` so the
actor path can include `data.actions` in its local set. Preserve SAC's existing
stochastic reparameterized action and entropy term. Use
`policy.q_values_all(features, action, target=False)` to retain the full
`[N,B,1]` ensemble before taking the minimum.

```text
Q_use_i = Q_base_i(s,a_actor) + lambda(t) * Delta_i(s,a_actor)
Q_actor = min_i Q_use_i
L_actor = mean(alpha * log_pi - Q_actor)
```

This is `min_i(Q_i + Delta_i)`, not `min_i(Q_i) + shared_R`. It preserves the
full-ensemble-min WSRL actor aggregation (`subsample_size=None`).

During this forward, temporarily set both base-critic and residual parameters
`requires_grad=False`, then restore their original flags afterward. Their
forward operations still preserve gradients with respect to `a_actor`, so
`dQ_use/da` reaches the policy while no parameter gradients are allocated for
either value branch. Zero the actor optimizer and step only it; base-critic and
residual optimizers are never involved in this path.

## 11. Online handoff lifecycle

Residual behavior is explicitly phase-gated:

- offline (`_online_start_step is None`): no residual update or actor effect;
- 5,000-step initial training phase: no residual update; actor/base critic are
  already frozen by WSRL's `TrainingUpdateMask`;
- first real post-warmup critic update: set
  `lancet_adaptation_start_step = current global_step` and adaptation step 0;
- active handoff: update residual after every base critic minibatch and apply
  actor correction;
- after the handoff window: skip residual updates and return `lambda=0`.

Use the piecewise-linear schedule:

```text
t = max(0, global_step - lancet_adaptation_start_step)
lambda(t) = max(0, 1 - t / 50_000).
```

It starts at exactly 1 on the first genuine adaptation update and reaches 0
smoothly after 50,000 post-warmup environment steps. This avoids a hard
policy-Q discontinuity while keeping Lancet an early-handoff mechanism. The
schedule is based on environment steps, not offline global steps or UTD
updates. Store both adaptation start step and residual update count.

At construction, exact-zero heads guarantee `R=Delta=0` and therefore
`Q_use=Q_base`. Offline and warmup inactivity preserve that equality until
online evidence produces the first residual update.

## 12. RNG ownership

RNG is an engineering constraint, not a new algorithm:

- construct the residual inside `torch.random.fork_rng` using
  `seed + 1_000_003`, restoring the base RNG afterward;
- use a dedicated device-local generator seeded with `seed + 2_000_003` for
  local perturbations and save/restore its state;
- use the existing SAC actor sample for actor loss and a deterministic actor
  action for residual fitting;
- evaluation must use its own deterministic seed stream in the runner.

This prevents Lancet-exclusive initialization and perturbations from shifting
base replay sampling or SAC action-noise streams.

## 13. Checkpoint contract

Lancet v3 checkpoint state must include:

- unchanged base policy, target critic, alpha, base optimizers, global step and
  global update through inherited checkpoint machinery;
- residual ensemble weights and residual optimizer;
- `variant`, architecture, K/noise, loss, U/EMA, and handoff-schedule config;
- scalar `ema_u`, EMA initialized flag;
- `lancet_adaptation_start_step` and residual update count;
- local-action generator state and deterministic seed offsets.

A WSRL offline checkpoint contains no residual state. Loading it into Lancet
v3 leaves the newly constructed zero-output residual and fresh optimizer
intact while restoring every base state. Loading a Lancet v3 checkpoint must
restore all residual/training state exactly. Legacy Lancet checkpoints are not
shape-compatible and must fail clearly rather than partially load.

## 14. Configuration API

Use a new `lancet_v3` registry entry and args dataclass. Do not silently change
the meaning of existing `lancet` archives.

```yaml
lancet_variant: full          # raw | centered | full
residual_hidden_dim: 128
residual_hidden_layers: 2
residual_lr: 0.001
residual_fit_coef: 1.0
residual_small_coef: 0.0001
local_action_count: 8
local_action_noise_std: 0.1
uncertainty_beta: 1.0
uncertainty_ema_decay: 0.99
uncertainty_weight_max: 3.0
uncertainty_eps: 1.0e-8
handoff_schedule: linear
handoff_window_steps: 50000
residual_init_seed_offset: 1000003
local_action_seed_offset: 2000003
```

There is no `lambda_u_variation` and no U prediction loss in v3. Exact-zero
output initialization is an invariant, not a tunable flag. The first v3
implementation should reject `use_compile=True` until target-capture and
stateful EMA behavior are explicitly tested under compilation.

## 15. Required logging

At the normal logging cadence, aggregate without changing update RNG:

- `lancet/lambda`, `lancet/adaptation_step`, `lancet/residual_updates`;
- raw R and centered Delta mean/std/absolute mean;
- `lancet/fit_loss`, `lancet/small_loss`, total residual loss;
- per-critic TD-residual mean/std/RMSE;
- `lancet/u_mean`, `u_std`, `u_ema`, weight mean/max;
- base-Q and corrected-Q mean/std;
- `mean(|Delta|) / (mean(|Q_base|)+eps)` and pointwise ratio mean;
- finite-state flags for actor/base critic/residual.

Action-wise ranking and U-to-error correlation are analysis diagnostics, not
training objectives.

## 16. Minimal implementation surface

Add or change in the implementation pass:

- add `rl_garden/algorithms/lancet_v3.py`;
- export `LancetV3`/`ResidualEnsemble` from `rl_garden/algorithms/__init__.py`;
- add `rl_garden/training/off2on/lancet_v3.py` with lazy builder and registry;
- add v3 configs under `configs/off2on/` and stage-specific shared-fork
  configs under the experiment protocol;
- add focused `tests/test_lancet_v3.py` and extend checkpoint/archive tests;
- add only the previously identified generic eval-seed/final-eval launcher
  patches needed by the benchmark protocol.

Do not modify the loss/update behavior in these baseline files:

- `rl_garden/algorithms/sac_core.py`
- `rl_garden/algorithms/cql.py`
- `rl_garden/algorithms/calql.py`
- `rl_garden/algorithms/wsrl.py`
- `rl_garden/policies/sac_policy.py`
- `rl_garden/algorithms/off2on.py`

The v3 subclass can use existing public hooks and a v3-only `_td_loss` parity
override. A baseline-file change requires separate justification and parity
evidence.

## 17. Unit-test plan

Required focused tests:

1. zero heads give `R=0`, `Delta=0`, and `Q_use==Q_base` exactly;
2. centered/full `Delta_local.mean(action_dim)` is approximately zero per
   critic/state; raw is not forcibly centered;
3. target `[B,1]` expands to per-critic TD target `[N,B,1]` without critic
   averaging;
4. actor uses `min_i(Q_i+Delta_i)` and differs from
   `min_i(Q_i)+mean/shared correction` on a constructed counterexample;
5. target-Q and base critic loss match WSRL and never use residual;
6. residual backward changes only residual parameters;
7. actor backward/step changes only actor parameters while retaining
   `dQ_use/da` through frozen critic/residual parameters;
8. residual is inactive offline and during warmup;
9. first post-warmup update sets adaptation step 0 and changes residual;
10. raw/centered/full share architecture/action samples and differ only in
    projection/weighting behavior;
11. pairwise and efficient U are equal, including N=2 reduction;
12. U and w are detached; `w=1` for raw/centered and clipped for full;
13. WSRL checkpoint fork has identical base state and zero Lancet output;
14. Lancet v3 checkpoint round-trip restores network, optimizer, EMA,
    counters, schedule, and local-generator state;
15. `utd=4` causes four base critic and four residual steps but one actor step;
16. Lancet-exclusive RNG does not alter the next base `torch.rand` sample;
17. all losses/Q/parameters are finite for one update.

## 18. Scientific limitations and non-goals

The projection theorem is oracle-level when TD residuals are known for every
local action. Model-free replay supervises only the observed action; practical
Lancet relies on function generalization plus the zero-mean architecture. U is
only ensemble disagreement and can be zero when all critics share the same
error. These are empirical questions, not implementation bugs.

V3 does not add a reference critic, FQE training objective, twin-consistency
loss, residual target critic, altered replay prioritization, or residual in the
Bellman target.
