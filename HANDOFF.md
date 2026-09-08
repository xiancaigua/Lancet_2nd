# Lancet / rl-garden Engineering Handoff

Last audited: 2026-09-07  
Branch: `feat/lancet`  
Base commit: `c45d89d4ed22dfe2db5e31679308cb4796834d4e`  
Working tree: environment, Lancet, and documentation changes are uncommitted.

## 1. Executive summary

### Completed

- A persistent D4RL legacy Docker runtime and generic `./dev` wrapper exist.
- Host source and data bind mounts, GPU access, host-network proxy access, Python 3.10.21, Torch 2.7.0+cu128, and `rl_garden` import were verified.
- D4RL imports; AntMaze and Kitchen environment construction/reset were verified.
- The complete AntMaze HDF5 now parses through `env.get_dataset()` (`(1000000, 29)` observations, `(1000000, 8)` actions).
- Lancet v1 is registered as a minimal WSRL extension with a shared state-action critic residual, checkpointing, logging, configs, and focused tests.
- Focused Lancet, WSRL, off2on-runner, and D4RL legacy tests passed.

### Partial

- AntMaze dataset download started, was interrupted, and is being resumed with `curl -C -` in the container. The file is not valid for training until complete and parsed.
- Lancet smoke is verified at unit/config-preflight level, not yet through a real D4RL offline-to-online run.

### Not completed

- No formal benchmark or paper-scale run was started.
- U-variation supervision is not implemented because its mathematical definition has not been provided.
- GitHub CLI/SSH authentication was not configured or changed.

## 2. Repository layout

```text
/home/zhaozihan/Lancet/
├── rl-garden/                       # only source checkout; Git working tree
│   ├── AGENTS.md                    # repository + continuity rules
│   ├── HANDOFF.md                   # this human-facing document
│   ├── .agent/                      # current state, memory index, templates
│   ├── dev                          # Docker lifecycle/command wrapper
│   ├── docker/lancet-d4rl/          # Dockerfile, runtime bootstrap, run script
│   ├── configs/off2on/              # WSRL and Lancet benchmark/smoke presets
│   ├── rl_garden/algorithms/        # algorithm implementations, including lancet.py
│   ├── rl_garden/training/off2on/   # registry, runner, Lancet registration
│   ├── scripts/smoke_lancet.sh      # download-free Lancet smoke
│   └── tests/test_lancet.py         # focused Lancet regression tests
└── data/                            # never commit datasets or large artifacts
    ├── datasets/d4rl/
    ├── checkpoints/
    ├── runs/
    ├── videos/
    └── cache/torch_extensions/
```

`rl-garden` is the only code copy. `data` is the persistent location for datasets, checkpoints, logs, videos, and compiled caches.

## 3. Host / Docker architecture

```mermaid
flowchart LR
    H[Server Host\nCodex, Git, one source tree] -->|bind mount| C[lancet-d4rl Docker]
    H -->|bind mount| D[/data/lancet]
    C --> S[/workspace/rl-garden]
    C --> D
    C -->|host network + proxy| P[127.0.0.1:7891]
```

- Host source: `/home/zhaozihan/Lancet/rl-garden`
- Container source: `/workspace/rl-garden`
- Host data: `/home/zhaozihan/Lancet/data`
- Container data: `/data/lancet`

The container runs `sleep infinity` detached. Deleting the container does not delete source or data because both reside on the Host.

## 4. Docker environment

| Property | Current value |
|---|---|
| Image | `lancet-d4rl:cu128` (`e4b5b87ed837`) |
| Container | `lancet-d4rl`, currently running |
| Base image | `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` |
| Python | 3.10.21 |
| PyTorch | 2.7.0+cu128 |
| GPU | NVIDIA RTX 4090 visible in container |
| Runtime | `--gpus all`, `--ipc=host`, `--shm-size=16g` |
| Network | `--network=host` |

`--network=host` deliberately makes `127.0.0.1:7891` inside the container refer to the Host proxy listener; Docker `-p` port publishing is neither needed nor relevant for outbound downloads. The proxy variables are injected by `run.sh`. It is useful for Git/PyPI/D4RL downloads, but running offline training must not depend on it staying available.

The Dockerfile installs system and MuJoCo 2.1 runtime requirements. `bootstrap.sh` runs against the bind-mounted source and installs editable `rl-garden`, pinned D4RL legacy dependencies, `mjrl`, and D4RL; source is never copied into the image.

## 5. How to use the environment

```bash
cd /home/zhaozihan/Lancet/rl-garden

./dev status
./dev start d4rl
./dev stop d4rl

./dev d4rl python --version
./dev d4rl nvidia-smi
./dev d4rl pytest -q tests/test_lancet.py
./dev d4rl python examples/train_off2on.py --help

# Only when pipes, redirects, or shell syntax are required:
./dev d4rl --shell 'python -m pytest -q tests/test_wsrl.py | tail -n 20'
```

`./dev d4rl <command>` is a general argv passthrough, not a command whitelist. It checks that `lancet-d4rl` is running, uses `/workspace/rl-garden`, and forwards stdout/stderr. `--shell` is intentionally explicit to avoid accidental quoting changes.

## 6. Git / GitHub

```text
Branch: feat/lancet
Base commit: c45d89d4ed22dfe2db5e31679308cb4796834d4e
origin: https://github.com/JaimeParker/rl-garden.git
upstream: not configured
```

No remote was changed, no token/key was copied into Docker, and no push was performed. At audit time `gh` was unavailable and GitHub SSH public-key authentication failed; this is an access setup issue, not a runtime dependency.

## 7. Relevant rl-garden architecture

- `rl_garden/algorithms/`: algorithm implementations. `WSRL` is the off2on backbone used by Lancet.
- `rl_garden/training/off2on/`: argument dataclasses, registry, builders, and the shared offline-to-online runner.
- `rl_garden/training/off2on/_runner.py`: phase orchestration and online switch.
- `rl_garden/training/off2on/_registry.py`: algorithm CLI registration; `examples/train_off2on.py` delegates here.
- Existing WSRL/SAC/Cal-QL policy, replay, target critic, checkpoint, and logging machinery remain the base implementation.
- `configs/off2on/`: strict YAML presets for real and tiny runs.

## 8. Lancet architecture

`Lancet` in `rl_garden/algorithms/lancet.py` subclasses `WSRL`; it does not fork or duplicate the WSRL implementation.

```mermaid
flowchart LR
    B[Replay batch] --> Q[WSRL base critic ensemble]
    B --> R[shared ResidualQNetwork R_phi(s,a)]
    Q --> T[unchanged WSRL TD target / base critic update]
    Q --> C[Q_i base + R_phi]
    R --> C
    C --> A[actor uses min_i Q_i base + R_phi]
    T --> L[detached residual TD fit]
    L --> R
```

### Actual behavior

- Base critic and target critic: created and updated through inherited WSRL/SAC/Cal-QL paths; Lancet intentionally leaves their TD target and base critic loss unchanged.
- Residual: one shared scalar `ResidualQNetwork(state, action) -> [batch, 1]`, rather than one network per REDQ critic.
- Corrected ensemble: `Q_corrected_i = Q_base_i + R_phi(s,a)`.
- Residual loss: fits detached `target_q - mean_i(Q_base_i)` and includes residual-magnitude regularization.
- Actor loss: replaces base actor Q term with `min_i(Q_base_i) + R_phi`.
- Logging: `lancet/*` residual statistics/losses plus `critic/td_error_*` are emitted when a logger exists.
- Checkpoint: Lancet adds residual network state to extra checkpoint state and residual optimizer to optimizer names; the focused round-trip test passed.
- Registry: `rl_garden/training/off2on/lancet.py` registers `lancet`; CLI help shows it.

### U-variation status

The action-dependent U-variation formula is intentionally **not invented**. `lambda_u_variation` defaults to `0`, and nonzero values raise an explicit error. A reviewed mathematical target is needed before adding this term.

## 9. Files changed in this working tree

| File / area | Purpose |
|---|---|
| `AGENTS.md` | Host/Docker rules plus Agent continuity protocol. |
| `dev` | Start/stop/status and generic container command wrapper. |
| `docker/lancet-d4rl/Dockerfile` | Reproducible Python 3.10, CUDA, MuJoCo legacy image definition. |
| `docker/lancet-d4rl/bootstrap.sh` | Pinned editable/runtime dependency installation after mounting source. |
| `docker/lancet-d4rl/run.sh` | Safe creation/start of the one scoped container and its mounts/env. |
| `rl_garden/algorithms/lancet.py` | WSRL subclass and shared residual correction. |
| `rl_garden/algorithms/__init__.py` | Lancet public exports. |
| `rl_garden/training/off2on/lancet.py` | Off2on arguments, builder, and registry registration. |
| `configs/off2on/lancet_antmaze_medium_play_v2.yaml` | Paper-budget Lancet comparison template. |
| `configs/off2on/lancet_antmaze_medium_play_smoke.yaml` | Tiny integration configuration. |
| `configs/off2on/wsrl_antmaze_medium_play_smoke.yaml` | Matched tiny WSRL baseline configuration. |
| `scripts/smoke_lancet.sh`, `tests/test_lancet.py` | Fast Lancet regression coverage. |
| `.agent/`, `HANDOFF.md` | Persistent Agent continuity and human handoff. |

## 10. Configuration

The initial target benchmark is `antmaze-medium-play-v2` with state observations.

- `lancet_antmaze_medium_play_v2.yaml`: 1,000,000 offline / 500,000 online steps, REDQ/WSRL parameters aligned with the existing WSRL AntMaze recipe; intended for later formal comparison, not run here.
- `lancet_antmaze_medium_play_smoke.yaml`: 2 offline / 4 online steps, small networks and buffer; intended only to validate wiring.
- `wsrl_antmaze_medium_play_smoke.yaml`: matching baseline smoke setup.

Formal WSRL/Lancet comparisons must share offline checkpoint, actor, base critics, dataset, seed, reward transform, and training budget. Lancet should differ only by its residual correction.

## 11. Tests performed

| Command | Result |
|---|---|
| `./dev d4rl bash scripts/smoke_lancet.sh` | 6 Lancet tests passed; registry/config preflight passed. |
| `./dev d4rl python -m pytest -q tests/test_wsrl.py` | 55 passed. |
| `./dev d4rl python -m pytest -q tests/test_off2on_runner.py` | 19 passed. |
| `./dev d4rl python -m pytest -q tests/test_d4rl_legacy_env.py tests/test_d4rl_legacy_dataset.py` | 19 passed. |
| D4RL import + `gym.make(...).reset()` for AntMaze and Kitchen | Passed; action shapes 8 and 9. |
| `./dev d4rl python -c 'import torch, rl_garden'` | Torch 2.7.0+cu128, CUDA true, `rl_garden` import passed. |

Not verified: a true dataset-backed WSRL/Lancet off2on smoke run. The required AntMaze HDF5 is now complete and parsed.

## 12. Known issues

1. The source-provided D4RL package emits optional Flow/CARLA/GymBullet warnings. These did not prevent AntMaze/Kitchen use.
2. U-variation supervision is a deliberate TODO, not a disabled bug.
3. GitHub CLI and SSH auth are not configured; do not create identity, keys, or tokens without user direction.
4. `tmux` is absent. For long formal runs, use an approved detached mechanism; do not attach a multi-hour job to Codex foreground.

## 13. Next steps

### P0

1. Run one WSRL and one Lancet tiny real-D4RL off2on smoke, checking the saved checkpoint and bounded log output.

### P1

1. Confirm fair offline checkpoint transfer for WSRL vs Lancet.
2. Define the U-variation target mathematically, review it, then implement it behind the existing hook.

### P2

1. Configure a user-approved GitHub auth method/fork before any push.
2. Install or use an approved scheduler/tmux workflow for formal experiments.

## 14. Recovery / rebuild

Source and data survive container deletion because they are Host bind mounts. If the scoped container alone needs recreation:

```bash
cd /home/zhaozihan/Lancet/rl-garden
docker build -f docker/lancet-d4rl/Dockerfile -t lancet-d4rl:cu128 .
./dev start d4rl
./dev status
```

`run.sh` never removes/replaces an existing container. If manual removal is truly necessary, inspect and act only on `lancet-d4rl`; never prune or modify unrelated server resources.

## 15. Handoff checklist

- [x] One Host source checkout and persistent data layout exist.
- [x] Long-lived `lancet-d4rl` runtime is running and GPU-visible.
- [x] Host/container source and data mounts are verified.
- [x] D4RL, AntMaze, Kitchen, WSRL, and Lancet focused tests are verified.
- [x] Lancet scaffold, registration, checkpoint state, logging, configs, and tests exist.
- [x] AntMaze HDF5 completes and parses successfully.
- [ ] Dataset-backed tiny WSRL and Lancet off2on runs complete.
- [ ] U-variation objective is specified and implemented.
