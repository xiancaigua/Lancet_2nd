# Agent Memory: Docker D4RL runtime

- Date: 2026-09-07
- Sequence: 002
- Agent: recovered retrospectively
- Branch: `feat/lancet`
- Commit before: unknown; recovery found `c45d89d`
- Commit after: `c45d89d` (working-tree changes remain uncommitted)
- Status: completed

## Goal

Provide a reproducible Python 3.10 CUDA/D4RL legacy runtime without changing Host Python.

## What changed

- Added `docker/lancet-d4rl/Dockerfile`, `bootstrap.sh`, and `run.sh`.
- Built `lancet-d4rl:cu128` from `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`.
- Installed pinned legacy D4RL/MuJoCo dependencies in the container bootstrap.

## Validation

Container reports Python 3.10.21, Torch 2.7.0+cu128, and CUDA available on an RTX 4090.

## Decisions

Image owns runtime; bind-mounted repository owns source. D4RL is installed from pinned commits because pip treats D4RL's floating `mjrl` URL as conflicting with the repository pin.

## Problems / caveats

Optional D4RL Flow/CARLA/GymBullet imports warn but AntMaze and Kitchen work.

## Resume hint

Use `./dev status`; rebuild from `docker/lancet-d4rl/Dockerfile` only when needed.
