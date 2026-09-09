# Environment

## Audited Host

| Item | Actual state |
|---|---|
| Host | `admin1-H3C-UniServer-R5300-G6` |
| OS | Ubuntu 24.04.4 LTS, kernel `7.0.0-28-generic` |
| GPU | 6 x NVIDIA GeForce RTX 4090 |
| Memory | 49140 MiB per GPU |
| Driver | 580.173.02 |
| Docker | 29.3.1 |
| Host tmux | 3.4 |

## Runtime

| Item | Actual state |
|---|---|
| Image | `lancet-d4rl:cu128` (`e4b5b87ed837`) |
| Base image | `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` |
| Container | `lancet-d4rl`, detached and running |
| Python | 3.10.21 |
| PyTorch | 2.7.0+cu128 |
| CUDA runtime | 12.8; CUDA available |
| MuJoCo | legacy 2.1 at `/opt/mujoco/mujoco210` |
| Network / IPC / shm | host / host / 16 GiB |

## Paths and lifecycle

- Host source: `/home/zhaozihan/Lancet/rl-garden`
- Container source: `/workspace/rl-garden`
- Host data: `/home/zhaozihan/Lancet/data`
- Container data: `/data/lancet`
- Dataset: `/data/lancet/datasets/d4rl`

Codex and Git run on the Host. Docker is a detached `sleep infinity` runtime
using Host bind mounts. Closing SSH does not stop that container. A foreground
`docker exec` child can still receive terminal/session termination; use Host
tmux or a scheduler for long runs. Deleting the container does not delete the
Host source or data.

## Proxy

The Host listener `127.0.0.1:7891` is injected into the host-networked
container as HTTP/HTTPS proxy variables. Host and container both established a
CONNECT tunnel to `chatgpt.com`; the site then returned an HTTP 403 Cloudflare
challenge. This proves proxy transport, not browser access. Offline training
must not require the tunnel to remain present.
