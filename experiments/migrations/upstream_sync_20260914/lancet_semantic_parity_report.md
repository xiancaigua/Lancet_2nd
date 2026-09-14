# Lancet semantic parity report

Date: 2026-09-14

Integration branch: `integration/upstream-sync-20260914`

Upstream: `252d1e0948618a0cd3675b9a05e0bcf4c29b5afb`

## Result

**PASS** for the frozen Lancet semantics on synthetic state-only Dict observations.

The test starts WSRL and Lancet from identical base parameters and identical RNG
state. While Lancet is inactive, one complete update produces exactly equal Bellman
targets, critic and actor losses, alpha loss, policy state, target critic state, and
base optimizer states. The Lancet residual does not update.

During the active window, the base critic, target critic, and base critic optimizer
remain exactly equal to WSRL for the same batch and RNG. The only policy-facing
algorithmic addition is the actor correction `min_i(Q_i + Delta_i)`; the Bellman
target and target critic never include residual values.

## Required semantic checks

| Check | Result | Evidence |
|---|---|---|
| Exact-zero residual initialization | PASS | `test_residual_ensemble_shape_and_exact_zero_initialization` |
| Fork `Q_use == Q_base` | PASS | `test_zero_init_q_use_equals_base_q`, checkpoint fork test |
| Post-base-update residual and captured target reuse | PASS | `test_target_is_reused_and_residual_fits_post_step_per_critic_q` |
| Bellman target excludes residual | PASS | `test_target_q_is_identical_to_wsrl_and_never_uses_residual` |
| Centering and N-critic U formula | PASS | centered/U tests in `tests/test_lancet.py` |
| Offline/warmup/50k lifecycle | PASS | `test_offline_warmup_active_window_and_expiry` |
| Dict state extraction and B x K repeat | PASS | `test_dict_state_extraction_and_b_by_k_repeat_are_value_preserving` |
| Inactive WSRL numerical identity | PASS | `test_inactive_one_step_is_exact_wsrl_base_update_and_rng_parity` |
| Active base-critic identity | PASS | `test_active_lancet_preserves_base_critic_update_semantics` |
| Save/load/continuation and Lancet-local RNG | PASS | `test_wsrl_checkpoint_fork_and_lancet_round_trip` |

## Validation commands

```bash
./dev d4rl pytest -q tests/test_lancet.py tests/test_lancet_audit.py \
  tests/test_lancet_v1.py tests/test_lancet_email_notification.py \
  tests/test_lancet_experiment_analysis.py tests/test_lancet_experiment_archive.py \
  tests/test_lancet_training_lifecycle.py tests/test_training_registry.py
```

Result: `71 passed, 1 warning`.

```bash
./dev d4rl pytest -q tests/test_wsrl.py tests/test_wsrl_policy.py \
  tests/test_cql_calql.py tests/test_sac_core.py tests/test_checkpoint.py \
  tests/test_checkpoint_filtered_load.py tests/test_base_algorithm.py \
  tests/test_off_policy_episode_mode.py
```

Result: `174 passed, 2 warnings`.

No mathematical Lancet definition was changed by the upstream API migration.
