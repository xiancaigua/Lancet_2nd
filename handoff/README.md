# Lancet Engineering Handoff

Last audited: 2026-09-10
Branch/implementation commit: `main` / `991991d1e6e77fbe42b95d2cc8a82faa376601b8`

## Project goal and current state

Lancet is an offline-to-online critic-correction extension of rl-garden's
WSRL implementation. The persistent CUDA/D4RL runtime, one-checkout bind-mount
model, AntMaze dataset, current Lancet implementation, historical Lancet V1,
unit/regression tests, experiment archive launcher, and audits exist.

- Completed: runtime, D4RL/MuJoCo, GPU/mount validation, current Lancet,
  continuity/archival tooling, current-method tiny smoke, and the paired
  seed-0 20k stability gate.
- Partial: formal configs/protocol exist, but the final evidence commit still
  needs human review/push and formal checkpoint hashes do not exist yet.
- Current U semantics: detached REDQ action-disagreement weighting of residual
  fitting. There is intentionally no separate U loss or prediction target.
- Not started: formal or multi-seed benchmark.

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
