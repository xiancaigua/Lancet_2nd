# Lancet Theory–Code Alignment

Status: current Lancet implemented and unit/regression validated on 2026-09-09;
real-data smoke and stability evidence remain separate gates.
Baseline anchor: rl-garden `51d014d058605f92ecc823ea3b2b42ef25ab4dd3`.

Git history contains one confirmed executable legacy implementation: the
shared-scalar Raw Residual scaffold now designated **Lancet V1**. No distinct
Lancet V2 executable is evidenced. The latest method is simply **Lancet**.

| Requirement | Lancet V1 | Current implementation / evidence | Status |
|---|---|---|---|
| Preserve WSRL Bellman target/base loss | inherited base loss | Lancet-only `_td_loss` is algebraically identical and target parity is tested | implemented, tested |
| Fit post-base-step remaining error | recomputed target and pre/post meaning ambiguous | capture one `y`, base step, recompute `Q_i_updated`, fit `stopgrad(y-Q_i_updated)` | implemented, target called once test |
| Inactive offline and warmup | residual also trained offline | gate on online state and inactive initial phase | implemented, lifecycle test |
| Per-critic correction | one scalar for all critics | shared backbone plus N exact-zero heads, `[N,B,1]` | implemented, shape/zero tests |
| Action-relative projection | absent | K=8 local action centering per critic/state | implemented, mean-zero test |
| Local policy action set | absent | replay, deterministic actor, six clipped Gaussian perturbations | implemented |
| Per-critic TD supervision | `y-mean_i(Q_i)` | `y-Q_i_updated` for every head | implemented, shape/target test |
| Repaired-ensemble actor | `min_i(Q_i)+R` | `min_i(Q_i+Delta_i)` | implemented, counterexample test |
| Actor centering gradient | absent | differentiable numerator minus detached local mean | implemented, action-gradient/isolation test |
| Residual optimizer isolation | base values detached | only residual optimizer steps in residual hook | implemented, parameter-isolation test |
| Actor optimizer isolation | shared residual path | critic/residual parameters frozen during actor backward | implemented, tested |
| U semantics | disabled placeholder loss | detached REDQ action disagreement only weights fitting | implemented; no U loss |
| REDQ U formula | absent | `2N/(N-1) * mean_k Var_i(C_ik)` | implemented, pairwise and N=2 tests |
| Minimal surgery | raw observed R | `mean(Delta_local^2)` on all local actions | implemented |
| Fork equality | random nonzero residual | exact-zero output gives `Q_use==Q_base` | implemented, fork test |
| Handoff lifecycle | correction permanent | constant one on adaptation `[0,50k)`, then update/correction off | implemented, boundary test |
| High UTD | hook follows critic updates | four residual steps and one actor step at UTD=4 | implemented, tested |
| Fair Raw/Centered/Lancet | separate legacy behavior | one architecture/action/optimizer/lifecycle framework with variant switch | implemented, variant tests |
| RNG isolation | global residual initialization | forked init RNG plus checkpointed local perturbation generator | implemented, round-trip tested |
| Checkpoint | basic residual state | schema, optimizer, variant, EMA, counters, local RNG | implemented, round-trip tested |
| Exact episode evaluation | not wired through WSRL registration | WSRL/Lancet builders pass the CalQL-shell episode target | implemented, dry-run validated |
| Adaptation/common endpoint evidence | absent | initial and final actual-step evaluation plus isolated eval seed counter | implemented, regression tested |

## Authoritative files

- `rl_garden/algorithms/sac_core.py`: unchanged base update ordering/high UTD.
- `rl_garden/algorithms/cql.py`: unchanged WSRL Bellman and conservative loss.
- `rl_garden/algorithms/calql.py`: unchanged offline Cal-QL/online override.
- `rl_garden/algorithms/wsrl.py`: only exact-episode evaluation passthrough added.
- `rl_garden/algorithms/lancet.py`: current correction implementation.
- `rl_garden/training/off2on/lancet.py`: registry/config plumbing.
- `tests/test_lancet.py`: tensor, gradient, lifecycle, UTD, and checkpoint evidence.

The generic evaluation patches do not alter rollout, target, loss, or optimizer
semantics. Acceptance for formal use still requires archived real-data smoke,
paired seed-0 stability pilots, evidence review, and a frozen clean commit.
