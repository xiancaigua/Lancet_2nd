# Lancet Current Agent State

Last updated: 2026-09-07  
Branch: `feat/lancet`  
Commit: `c45d89d4ed22dfe2db5e31679308cb4796834d4e` (working tree is uncommitted)

## Current objective

The v1 Lancet environment, residual-critic scaffold, and AntMaze dataset are ready. The immediate engineering priority is a tiny real off2on smoke; no paper-scale run is authorized.

## Repository

- Host source: `/home/zhaozihan/Lancet/rl-garden`
- Container source mount: `/workspace/rl-garden`
- Persistent data mount: `/data/lancet` (`/home/zhaozihan/Lancet/data` on Host)

## Runtime

- Image/container: `lancet-d4rl:cu128` / `lancet-d4rl` (running)
- Python/Torch: 3.10.21 / 2.7.0+cu128
- GPU: NVIDIA RTX 4090 visible; Docker uses `--gpus all`, host networking, host IPC, 16 GiB shm.
- Proxy: `127.0.0.1:7891` is injected into the container; it is optional for already-running offline work.

## Execution

```bash
cd /home/zhaozihan/Lancet/rl-garden
./dev d4rl <command>
./dev d4rl --shell '<complex shell expression>'
```

`dev` is an unrestricted argv passthrough; it is not a command whitelist.

## Code architecture

```text
WSRL / SAC / Cal-QL backbone
        ↓ unchanged base TD/CQL critic and target path
Lancet(WSRL)
  ├── shared ResidualQNetwork R_phi(s,a)
  ├── residual TD fit + magnitude regularizer
  └── actor uses min(Q_base) + R_phi
```

## Lancet implementation status

### Implemented

- Off2on registry entry `lancet`, flat Box state/action residual network, corrected-Q interface, residual optimizer, logging, and checkpoint state.
- Paper and tiny AntMaze configs; focused regression tests and smoke script.

### Partially implemented

- The smoke configuration has been preflighted and unit-level paths pass, but no real D4RL off2on run has completed because the dataset is incomplete.

### Not implemented

- U-variation supervision. `lambda_u_variation` must remain `0`; nonzero intentionally errors until a reviewed equation exists.

## Environment status

- D4RL 1.1 imports; AntMaze and Kitchen construct/reset successfully.
- The AntMaze HDF5 is complete (231,110,764 bytes) and `env.get_dataset()` parsed observations `(1000000, 29)` and actions `(1000000, 8)` successfully.
- Host/container source mount and proxy bridge were verified. Optional D4RL Flow/CARLA/GymBullet imports warn but are not required.

## Verified tests

- `./dev d4rl bash scripts/smoke_lancet.sh` — 6 passed.
- `./dev d4rl python -m pytest -q tests/test_wsrl.py` — 55 passed.
- `./dev d4rl python -m pytest -q tests/test_off2on_runner.py` — 19 passed.
- `./dev d4rl python -m pytest -q tests/test_d4rl_legacy_env.py tests/test_d4rl_legacy_dataset.py` — 19 passed.

## Known issues

1. No GitHub CLI authentication is configured; SSH public-key auth was unavailable during setup. No Git identity or credentials were changed.
2. `tmux` was not installed on the Host.

## Active decisions

- Host owns one checkout; Docker owns runtime only.
- Do not alter WSRL base TD target/update behavior for Lancet v1.
- Avoid formal runs in the foreground; output belongs under `/data/lancet`.

## Immediate next steps

1. Run the tiny WSRL then Lancet D4RL off2on smoke configs; inspect checkpoint and bounded logs.
2. Define and review the U-variation objective before enabling it.

## Read next

1. `AGENTS.md`
2. `.agent/CURRENT_STATE.md`
3. `.agent/memory/INDEX.md`
4. The latest relevant 3–5 memory entries
5. `rl_garden/algorithms/lancet.py` and `rl_garden/training/off2on/lancet.py`
