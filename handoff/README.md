# Lancet Engineering Handoff

Last audited: 2026-09-08  
Branch/commit: `main` / `477abdac2f1e297c6ede82aafde20ce90b65de4b`

## Project goal and current state

Lancet is an offline-to-online critic-correction extension of rl-garden's
WSRL implementation. The persistent CUDA/D4RL runtime, one-checkout bind-mount
model, AntMaze dataset, Lancet v1 scaffold, unit tests, experiment archive
launcher, and independent audits exist.

- Completed: runtime, D4RL/MuJoCo, GPU/mount validation, Lancet residual path,
  continuity migration, run-type separation, archival/audit tooling, and archived real-data WSRL/Lancet tiny smokes.
- Partial: the fair formal checkpoint/config/metric/seed protocol is not frozen or committed.
- Not implemented: action-dependent U-variation loss; no formula was invented.
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
