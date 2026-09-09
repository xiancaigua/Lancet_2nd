# Lancet Implementation Specification

Status: frozen implementation contract; implementation follows this specification
Date: 2026-09-09
Theory source: `docs/design/Lancet_v3_技术实现思路与理论证明.pdf`
Code baseline audited at rl-garden commit `51d014d058605f92ecc823ea3b2b42ef25ab4dd3`

## 1. Scope and invariants

Lancet is an online-handoff extension of WSRL. It is not a replacement
critic and it does not participate in offline pretraining.

The following WSRL behavior is invariant:

- the Bellman target and target-critic REDQ subsampling;
- the base critic TD/CQL/Cal-QL loss and optimizer step;
- the target-critic Polyak update;
- the replay transition and reward/observation preprocessing;
- the offline checkpoint contents and the 5,000-step frozen warmup;
- the fact that the target critic never contains a Lancet correction.

The pre-migration executable in `rl_garden/algorithms/lancet.py` is the only
legacy implementation confirmed by Git history. It is designated **Lancet V1**
(shared scalar Raw Residual) and moves to the explicit `lancet_v1` identity.
No separate Lancet V2 executable is present in this repository. The current
method owns the formal `Lancet` class and `lancet` registry name.

## 2. Current update chain and Lancet insertion point

The real WSRL chain is:

```text
WSRL
  -> _CalQLRolloutTrainingShell
     -> CQLCore / SACCore
        -> _target_q(data)                    # target [B, 1]
        -> _critic_loss(data)                 # base Q [N, B, 1]
        -> q_optimizer.step()
        -> _post_critic_update(data, info)    # Lancet residual step
        -> _actor_loss_from_batch(data)       # Lancet actor Q
        -> actor_optimizer.step()
        -> _target_update()                   # base critic only
```

For integer `utd=4`, `SACCore.train_high_utd()` samples one batch of 1024,
splits it into four minibatches of 256, performs four critic updates and four
`_post_critic_update` calls, then performs one actor update on the full batch.
Lancet follows that schedule: one residual update per base-critic
minibatch, then one actor update.

## 3. Residual architecture decision

| Candidate | Theory alignment | Capacity/fairness | Decision |
|---|---|---|---|
| One shared scalar residual | Cannot fit critic-specific TD residuals; `min(Q)+R` commutes and does not repair ensemble members separately | smallest, but wrong current Lancet object | reject for current Lancet; legacy only |
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

Recommended Lancet defaults are two 128-wide hidden layers and residual LR
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

For the actor update, build the same detached local set from the replay
action, deterministic current actor action, and six perturbations. Compute its
residual mean under `torch.no_grad()` (or detach it explicitly), then evaluate
`R(s,a_actor)` separately on SAC's differentiable stochastic actor action:

```text
Delta_actor = R(s,a_actor) - stopgrad(mean_k R(s,a_k_local))
```

Thus the centering baseline has no actor or residual gradient, while
`d R(s,a_actor) / d a_actor` is retained.

Replay action is included because it is the only action with an observed TD
residual. The policy action and perturbations define the policy-local geometry
that the actor is likely to compare. Local Gaussian draws use a dedicated
`torch.Generator`; they must not advance the base SAC/replay RNG stream.

## 6. Unified correction variants

All new residual variants use the same network, zero initialization,
optimizer, K, action set, perturbations, update frequency, and handoff lifecycle.

```text
raw:
    Delta_local = R_local

centered:
    Delta_local = R_local - mean_k(R_local)

lancet:
    Delta_local = R_local - mean_k(R_local)
    fitting weight = w(U)
```

For centered/Lancet, `Delta_local.mean(dim=2)` is zero up to floating-point
roundoff for every critic and state. For raw, U is still computed for matched
diagnostics and local machinery, but `w=1`. Centered also uses `w=1`.

WSRL is a separate no-residual algorithm. Lancet V1 shared-scalar residual is
not one of these three current variants and must be named `Lancet V1`.

## 7. TD supervision and stop-gradient boundary

Each critic minibatch has one fixed order:

1. compute Bellman target `y` exactly once;
2. compute the unchanged WSRL base critic loss with this `y`;
3. call the base critic optimizer step;
4. forward `Q_i_updated(s,a)` with the updated base parameters;
5. form `stopgrad(y - Q_i_updated(s,a))`;
6. update only the residual branch.

Lancet must reuse that exact target sample and must not call `_target_q()`
again. Stochastic SAC actions or REDQ subsampling would otherwise define a
different target. The residual therefore means the Bellman error still left
after the normal base-critic update.

The least invasive implementation is a Lancet-only `_td_loss` override that is
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

For `w=1` in Raw/Centered and the detached U weight in Lancet:

```text
L_fit   = mean_{i,b} w_b * (Delta_observed_i,b - TD_residual_i,b)^2
L_small = mean_{i,b,k} Delta_local_i,b,k^2
L_R     = residual_fit_coef * L_fit + residual_small_coef * L_small
```

Recommended defaults are `residual_fit_coef=1.0` and
`residual_small_coef=1e-4`. In centered/Lancet, the regularizer is explicitly on
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

Use a constant active window:

```text
t = max(0, global_step - lancet_adaptation_start_step)
lambda(t) = 1  when 0 <= t < 50_000
lambda(t) = 0  when t >= 50_000
```

During the active window, residual fitting runs after every base-critic
minibatch and the actor uses corrected Q. At the boundary and afterward,
residual updates stop and the actor uses base Q exactly. The schedule uses
post-warmup environment steps, not offline global steps or UTD updates.
Linear or smooth decay is a future ablation, not part of the first method.
Store both adaptation start step and residual update count.

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

Lancet checkpoint state must include:

- unchanged base policy, target critic, alpha, base optimizers, global step and
  global update through inherited checkpoint machinery;
- residual ensemble weights and residual optimizer;
- `variant`, architecture, K/noise, loss, U/EMA, and active-window config;
- scalar `ema_u`, EMA initialized flag;
- `lancet_adaptation_start_step` and residual update count;
- local-action generator state and deterministic seed offsets.

A WSRL offline checkpoint contains no residual state. Loading it into Lancet
leaves the newly constructed zero-output residual and fresh optimizer
intact while restoring every base state. Loading a Lancet checkpoint must
restore all residual/training state exactly. Lancet V1 checkpoints are not
shape-compatible and must fail clearly rather than partially load.

## 14. Configuration API

Use the formal `Lancet` class and `lancet` registry entry. Move the confirmed
legacy executable to `LancetV1`/`lancet_v1`; old checkpoints carrying the
ambiguous class name `Lancet` must be detected from their legacy extra-state
schema and rejected by the new class before partial load.

```yaml
lancet_variant: lancet       # raw | centered | lancet
residual_hidden_dim: 128
residual_hidden_layers: 2
residual_lr: 0.001
residual_fit_coef: 1.0
residual_small_coef: 0.0001
local_action_count: 8
local_action_noise_scale: 0.1
uncertainty_beta: 1.0
uncertainty_ema_decay: 0.99
uncertainty_weight_max: 3.0
uncertainty_eps: 1.0e-8
handoff_window_steps: 50000
residual_init_seed_offset: 1000003
local_action_seed_offset: 2000003
```

There is no `lambda_u_variation` and no U prediction loss in Lancet. Exact-zero
output initialization is an invariant, not a tunable flag. The first Lancet
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

- move the confirmed legacy implementation to `rl_garden/algorithms/lancet_v1.py`;
- implement the current method in `rl_garden/algorithms/lancet.py`;
- export `Lancet`/`ResidualEnsemble` from `rl_garden/algorithms/__init__.py`;
- move legacy registration to `rl_garden/training/off2on/lancet_v1.py`;
- make `rl_garden/training/off2on/lancet.py` register the current method;
- add Lancet configs under `configs/off2on/` and stage-specific shared-fork
  configs under the experiment protocol;
- preserve legacy checks in `tests/test_lancet_v1.py`, implement focused
  current checks in `tests/test_lancet.py`, and extend checkpoint/archive tests;
- add only the previously identified generic eval-seed/final-eval launcher
  patches needed by the benchmark protocol.

Do not modify the loss/update behavior in these baseline files:

- `rl_garden/algorithms/sac_core.py`
- `rl_garden/algorithms/cql.py`
- `rl_garden/algorithms/calql.py`
- `rl_garden/algorithms/wsrl.py`
- `rl_garden/policies/sac_policy.py`
- `rl_garden/algorithms/off2on.py`

The Lancet subclass can use existing public hooks and a Lancet-only `_td_loss`
parity override. A baseline-file change requires separate justification and parity
evidence.

## 17. Unit-test plan

Required focused tests:

1. zero heads give `R=0`, `Delta=0`, and `Q_use==Q_base` exactly;
2. centered/Lancet `Delta_local.mean(action_dim)` is approximately zero per
   critic/state; Raw is not forcibly centered;
3. target `[B,1]` expands to per-critic TD target `[N,B,1]` without critic
   averaging;
4. residual supervision uses `Q_i_updated` after the base optimizer step but
   reuses the exact `y` captured by that step;
5. actor uses `min_i(Q_i+Delta_i)` on a constructed counterexample;
6. `R(s,a_actor)` carries action gradient while the local residual baseline
   is detached;
7. actor step changes only actor parameters, not critic/residual parameters;
8. residual step changes only residual parameters, not the base critic;
9. target-Q and base critic loss match WSRL and never use residual;
10. residual is inactive offline and during the 5k frozen warmup;
11. first post-warmup update sets adaptation step 0 and changes residual;
12. at adaptation step 50k, both correction and residual updates are off;
13. Raw/Centered/Lancet share architecture/action samples and differ only in
    projection/weighting behavior;
14. pairwise and efficient U are equal, including the N=2 reduction;
15. U is detached; `w=1` for Raw/Centered and clipped for Lancet;
16. first effective batch initializes EMA to exactly current batch mean;
17. `utd=4` causes four base critic and four residual steps but one actor step;
18. a WSRL checkpoint fork restores identical base state and zero correction;
19. Lancet checkpoint round-trip restores network, optimizer, EMA, counters,
    active-window state, and local generator;
20. all losses, Q values, weights, and parameters are finite for one update.

## 18. Scientific limitations and non-goals

The projection theorem is oracle-level when TD residuals are known for every
local action. Model-free replay supervises only the observed action; practical
Lancet relies on function generalization plus the zero-mean architecture. U is
only ensemble disagreement and can be zero when all critics share the same
error. These are empirical questions, not implementation bugs.

Lancet does not add a reference critic, FQE training objective, twin-consistency
loss, residual target critic, altered replay prioritization, or residual in the
Bellman target.
