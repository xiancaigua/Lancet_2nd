# Docker and Execution

## Common commands

```bash
cd /home/zhaozihan/Lancet/rl-garden

./dev status
./dev start d4rl
./dev stop d4rl

./dev d4rl python --version
./dev d4rl nvidia-smi
./dev d4rl python -m pytest -q tests/test_lancet.py
./dev d4rl python examples/train_off2on.py lancet --help

./dev d4rl --shell 'python -m pytest -q tests/test_wsrl.py | tail -n 20'
```

`./dev d4rl <command>` forwards the argument array unchanged to `docker exec`
at `/workspace/rl-garden`. It is a generic passthrough, not a command whitelist.
Only `--shell` invokes `bash -lc`, for pipes, redirects, `&&`, or shell variable
expansion.

`./dev start d4rl` starts an existing stopped container or creates the named
container with `docker/lancet-d4rl/run.sh`, then runs the reproducible bootstrap.
The script never removes or replaces an existing container. It uses the Host
UID/GID, preventing root-owned source artifacts.

## Long-running work

Tiny smoke tests may run in the current foreground. Formal runs must use the
archive launcher plus a detached Host tmux/scheduler process. Example control
session only:

```bash
tmux new -s lancet
```

Do not use `docker system prune`, delete unrelated containers/images, or clone
a second source tree inside Docker.
