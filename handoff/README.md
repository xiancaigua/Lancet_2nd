# Lancet Engineering Handoff

Last audited: 2026-09-15

Active migration branch: `integration/upstream-sync-20260914`

Current main: `db57e1e9e1ee6e81ffafd99b1547a96e26389df1` (unchanged)

## Project goal and current state

Lancet is an offline-to-online critic-correction extension of rl-garden's
WSRL implementation. The persistent CUDA/D4RL runtime, one-checkout bind-mount
model, AntMaze dataset, current Lancet implementation, historical Lancet V1,
unit/regression tests, experiment archive launcher, and audits exist.

- Completed on the integration branch: full merge of upstream `252d1e0`,
  Lancet Dict-observation/policy migration, corrected WSRL AntMaze configs,
  resolved parity, dataset semantic comparison, numerical parity, checkpoint
  continuation, focused regressions, real AntMaze smoke, and SMTP test.
- The revised Ruff gate passes: clean upstream and integration have identical
  normalized sets of 2,610 inherited findings, while integration-added and
  Lancet-owned Python files are clean. Main promotion is the next step.
- Generation 1 is permanently `SUPERSEDED_PRE_UPSTREAM_SYNC`; it is retained
  for debugging and migration evidence and excluded from paper statistics.
- Current U semantics: detached REDQ action-disagreement weighting of residual
  fitting. There is intentionally no separate U loss or prediction target.
- Not started: corrected generation-2 initializer, baseline sanity, Lancet
  sanity, or formal v2.

## Read in this order

1. [Environment](01_environment.md)
2. [Docker and execution](03_docker_and_execution.md)
3. [rl-garden architecture](04_rlgarden_architecture.md)
4. [Lancet implementation](05_lancet_implementation.md)
5. [Testing](06_testing_and_validation.md)
6. [Experiment workflow](07_experiment_workflow.md)
7. [Known issues and next steps](09_known_issues_and_next_steps.md)

## Fast start

```bash
cd /home/zhaozihan/Lancet/rl-garden
./dev status
./dev d4rl python --version
scripts/audit/run_all.sh
```

Lancet code is in `rl_garden/algorithms/lancet.py` and its CLI registration is
in `rl_garden/training/off2on/lancet.py`. Experiment records are indexed in
[`experiments/README.md`](experiments/README.md); their directories under
`/home/zhaozihan/Lancet/data/runs/` are authoritative.

If the container disappears, rebuild/restart as described in
[Git and recovery](08_git_and_recovery.md). Source and data remain on the Host.
