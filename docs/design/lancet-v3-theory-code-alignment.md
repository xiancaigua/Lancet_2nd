# Lancet v3 Theory–Code Alignment

Status: implementation plan against rl-garden commit
`ede0935942e94cc1a2b6f9889bb738b8f1e5b5ea`.

The current `Lancet` class is a tested legacy scaffold, not v3. This table is
the review checklist for the next coding pass.

| V3 requirement | Current implementation | Proposed implementation | Risk / status |
|---|---|---|---|
| Keep WSRL Bellman/base critic unchanged | inherited target/base update unchanged | preserve; capture the exact target in a parity-tested v3 override | low; parity test required |
| Residual only after online warmup | legacy hook runs after offline critic updates too | gate on online mode and inactive warmup; first active update defines adaptation step 0 | **mismatch** |
| Per-critic correction | one scalar shared by all critics | shared backbone + N zero-initialized heads, `[N,B,1]` | **mismatch** |
| Action-relative projection | none | subtract local-action mean per critic/state | **mismatch** |
| Local policy action set | none | K=8: replay, policy action, six clipped Gaussian perturbations | **mismatch** |
| Per-critic TD residual | `y - mean_i Q_i` | `stopgrad(y - Q_i)` for every head | **mismatch** |
| Exact base target sample | legacy recomputes `_target_q` after critic step | capture original target; never resample target action | **mismatch / important** |
| Repaired ensemble actor | `min_i(Q_i) + shared R` | `min_i(Q_i + lambda*Delta_i)` | **mismatch** |
| Base/target excluded from residual gradients | legacy uses no-grad target/base | retain; residual loss owns residual only | aligned |
| Actor action gradient through correction | present through shared R | freeze base-critic and residual parameters, retain gradient through action, and use full ensemble | partial |
| U meaning | disabled placeholder framed as a loss | detached REDQ action-dependent disagreement used only as fitting weight | **semantic replacement** |
| REDQ U calculation | absent | exact `2N/(N-1) * mean_k Var_i(C_ik)` | new; algebra proved in specification |
| Minimal surgery | penalizes raw shared R on observed action | penalize `Delta_local` across critics, states, and K actions | **mismatch** |
| Fork equality | residual final layer random | exact-zero output heads give `Q_use==Q_base` | **mismatch / critical** |
| Handoff schedule | correction permanent | linear 1→0 over 50k post-warmup env steps | **mismatch** |
| High UTD | hook executes once per critic minibatch | retain four residual steps for UTD=4 and one actor step | aligned mechanism |
| Fair Raw/Centered/Full | no unified variants | one framework/config, same architecture/action machinery | new |
| RNG isolation | residual init advances global RNG | forked init RNG + dedicated perturbation generator | **small engineering gap** |
| Checkpoint | legacy residual/optimizer/config saved | add EMA, adaptation counters, generator state, variant and schedule | partial |
| Compile behavior | legacy permits inherited compile flag | reject initially until stateful target capture/EMA compile tests exist | safety gate |

## Baseline files that remain authoritative

- `rl_garden/algorithms/sac_core.py`: update ordering and high-UTD semantics.
- `rl_garden/algorithms/cql.py`: WSRL base target/critic/actor math.
- `rl_garden/algorithms/calql.py`: offline Cal-QL and online CQL override.
- `rl_garden/algorithms/wsrl.py`: warm-start shell.
- `rl_garden/algorithms/off2on.py`: switch/replay/phase bookkeeping.
- `rl_garden/policies/sac_policy.py`: full ensemble Q and actor API.

V3 should extend these via subclass hooks rather than modifying their losses.

## Implementation acceptance gate

V3 is code-ready only when the focused tests in
`docs/design/lancet-v3-implementation.md` pass, the WSRL base-update parity
test passes, and a shared WSRL checkpoint produces exact initial policy-facing
Q equality. No pilot or formal benchmark may precede that gate.
