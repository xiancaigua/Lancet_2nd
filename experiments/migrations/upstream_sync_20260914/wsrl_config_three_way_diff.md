# WSRL AntMaze configuration three-way audit

Snapshots:

- A — pre-sync Lancet formal initializer at `db57e1e`;
- B — upstream WSRL correction `3c9c46a52a974737840a1188ced1800e2207c612`;
- C — merged upstream head `252d1e0948618a0cd3675b9a05e0bcf4c29b5afb`.

The generation-2 initializer is
`configs/off2on/wsrl_antmaze_medium_play_v2_offline_initializer.yaml`.

| Parameter | A: old formal | B: 3c9c46 | C: upstream HEAD | Classification | Generation-2 decision |
|---|---|---|---|---|---|
| `cql_alpha_lagrange_init` | implicit `1.0` | `e` | `e` | BUG FIX | adopt `2.718281828459045` |
| `cql_alpha_param` | implicit `softplus` | `exp_clip` | `exp_clip` | BUG FIX | adopt `exp_clip` |
| `cql_diff_clip_mode` | implicit `skip_when_autotune` | `always` | `always` | BUG FIX | adopt `always` |
| `cql_penalty_scale` | implicit `lagrange_only` | `lagrange_times_alpha` | `lagrange_times_alpha` | BUG FIX | adopt `lagrange_times_alpha` |
| `cql_importance_sample` | implicit `true` | explicit `true` | explicit `true` | EXPLICIT DOCUMENTATION ONLY | freeze `true` |
| `cql_max_target_backup` | implicit `true` | explicit `true` | explicit `true` | EXPLICIT DOCUMENTATION ONLY | freeze `true` |
| `cql_temp` | implicit `1.0` | explicit `1.0` | explicit `1.0` | EXPLICIT DOCUMENTATION ONLY | freeze `1.0` |
| `target_entropy` | `0.0` | `0.0` | `auto` | SCIENTIFIC HYPERPARAMETER CHANGE | retain `0.0` |
| `num_eval_episodes` | `20` | implicit `None` | implicit `None` | SCIENTIFIC EVALUATION CHANGE | retain protocol value `20` |
| observation form | legacy Box/state selector | legacy Box/state selector | canonical Dict `state` | FRAMEWORK REFACTOR | migrate representation; preserve values |
| `bootstrap_at_done` | implicit legacy behavior | implicit legacy behavior | framework default `truncated` after `9ce5e05` | BUG FIX / FRAMEWORK REFACTOR | explicit `truncated` |

## Evidence and boundary

The four substantive CQL changes were introduced together by `3c9c46a` to
correct the unstable AntMaze recipe. The three other CQL fields already had
the same effective values and are frozen explicitly for auditability.

`target_entropy: auto` was introduced later by `823f58b`, an unrelated
diffusion/observation refactor commit, with no evidence that it belongs to the
WSRL stability correction. Generation 2 therefore retains the old and 3c9c46
value `0.0`. Exact 20-episode evaluation is likewise retained from the frozen
Lancet protocol rather than inheriting an unrelated preset omission.

The upstream Dict-observation and `bootstrap_at_done=truncated` framework
changes are accepted. Dataset fingerprint comparison must demonstrate that
the state/action/reward/termination/MC-return numbers remain identical.
