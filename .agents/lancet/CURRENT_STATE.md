# Lancet Current Agent State

Last updated: 2026-09-08 22:00 CST
Branch: `main`  
Commit: `477abdac2f1e297c6ede82aafde20ce90b65de4b` (working tree dirty)

## Current objective

Engineering consolidation and both archived tiny smokes are complete. The
first WSRL-vs-Lancet-TD formal protocol is drafted and awaiting human review
plus bounded enablement patches; no debug pilot, formal benchmark, or training
process is active.

## Repository and runtime

- Host path: `/home/zhaozihan/Lancet/rl-garden`
- Container path: `/workspace/rl-garden`
- Data: `/home/zhaozihan/Lancet/data` -> `/data/lancet`
- Image/container: `lancet-d4rl:cu128` / `lancet-d4rl` (running)
- Python/Torch/CUDA: 3.10.21 / 2.7.0+cu128 / 12.8
- GPU: 6 x NVIDIA GeForce RTX 4090, 49140 MiB each; CUDA visible

## Standard execution

```bash
./dev d4rl <command>
./dev d4rl --shell '<complex shell expression>'
```

## Lancet architecture

```text
WSRL/Cal-QL/SAC backbone (base critic and target unchanged)
  -> Lancet(WSRL)
     -> shared ResidualQNetwork R_phi(s,a)
     -> post-critic residual TD fit + magnitude regularizer
     -> actor uses min(Q_base) + R_phi
```

## Implementation status

### Implemented

- `lancet` off2on registry, shared residual network, corrected-Q interface,
  residual optimizer/logging/checkpoint hooks, AntMaze configs and unit tests.
- Persistent Docker runtime, Host/data bind mounts, D4RL/MuJoCo legacy stack.
- Lancet continuity has one source of truth under `.agents/lancet/`.
- Archived experiment launcher, strict smoke/debug/formal roots, formal
  clean-tree guard, and automatic post-run Memory/Handoff indexing.
- Runtime, bind-mount, D4RL 1M-dataset, and Lancet audits all pass.
- Split human Handoff is complete under `handoff/`; WSRL and Lancet real-data
  tiny-smoke archives are finalized.
- Planned formal protocol exists at
  `experiments/protocols/antmaze_wsrl_lancet_v1.md`.

### Partially implemented

- The protocol freezes the research question, paired shared-checkpoint
  strategy, 1M/500k budget, five seeds, metrics, statistics, failure policy,
  20k/20k debug gate, and readiness checklist.
- Current code is `NEEDS SMALL PATCH` for deterministic eval seeding, a forced
  final evaluation, stage-specific shared-fork configs, missing diagnostics,
  paired post-load RNG alignment, explicit GPU/lineage metadata, and periodic
  checkpoints.
- Tiny smokes finish before an AntMaze episode ends, so episodic return/success
  are explicitly not collected.

### Not implemented

- U-variation supervision. `lambda_u_variation` must remain `0`; a nonzero
  value intentionally errors because no reviewed mathematical target exists.

## Environment status

- D4RL: 1.1 imports; optional Flow/CARLA/GymBullet warnings are non-blocking.
- AntMaze: env reset and 1,000,000-transition dataset parse verified.
- Kitchen: env construction/reset previously verified; dataset not downloaded.
- Proxy: Host and container reach `127.0.0.1:7891`; ChatGPT returns an HTTP
  403 Cloudflare challenge after a successful CONNECT tunnel.
- Mount: source and data are bind-mounted read/write; container is detached.
- Host tmux: 3.4 installed.
- Hardware availability: six GPUs exist, but current occupancy is shared and
  dynamic; select an actually idle GPU before future pilots.

## Verified tests

- `./dev d4rl bash scripts/smoke_lancet.sh` -> 6 passed + config preflight.
- `./dev d4rl python -m pytest -q tests/test_wsrl.py` -> 55 passed.
- `./dev d4rl python -m pytest -q tests/test_off2on_runner.py` -> 19 passed.
- D4RL env/dataset focused tests -> 19 passed.
- Archived WSRL real-data smoke `20260908_123022` -> finished, all declared
  pipeline checks passed.
- Archived Lancet real-data smoke `20260908_123802` -> finished; finite
  residual losses, positive residual parameter deltas, all required
  TensorBoard tags, checkpoint reload passed.
- Final ruff check passed; combined Lancet/archive test selection: 10 passed.
- 2026-09-08 no-training checkpoint probe: strict WSRL-to-Lancet load, base
  tensor equality, reproducible residual initialization, and fresh residual
  optimizer state passed.
- Both full-scale WSRL/Lancet `--print-config` outputs match on base fields.

## Latest archived experiments

- `20260908_123802`: lancet smoke -> finished
  (`/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/lancet/seed_0/20260908_123802`)
- `20260908_123022`: wsrl smoke -> finished
  (`/home/zhaozihan/Lancet/data/runs/smoke/antmaze-medium-play-v2/wsrl/seed_0/20260908_123022`)

## Known issues

1. U-variation is deliberately unspecified and disabled.
2. The working tree contains this in-progress consolidation; no commit/push
   has been performed in this task.
3. The generic launcher does not yet record/select one GPU, and historical
   tiny-smoke archives did not measure peak GPU memory.

## Active decisions

- Host owns the only checkout; Docker is runtime only.
- Smoke/debug/formal outputs and checkpoints are strictly separated.
- Experiment directories are authoritative; Memory/Handoff only summarize.
- Lancet v1 does not change the inherited base TD target/critic update.
- The verified scientific label is Lancet-TD / Lancet w/o U; do not call it
  full Lancet.
- Formal v1 uses one seed-specific WSRL offline checkpoint forked into paired
  WSRL and Lancet-TD online branches.

## Immediate next steps

1. Human-review the protocol and current Lancet-TD implementation.
2. Apply/test only the protocol-enablement small patches listed in the protocol.
3. Commit/push a clean revision, then run the two archived seed-0 debug pilots.
4. Start formal work only if every readiness gate passes.

## Read next

1. `AGENTS.md`
2. Task-relevant `.agents/rules/` and `.agents/runbooks/`
3. `.agents/lancet/memory/INDEX.md`
4. Latest/task-relevant 3–5 memories
5. `handoff/README.md` and task-specific source
6. `experiments/protocols/antmaze_wsrl_lancet_v1.md` for benchmark work
