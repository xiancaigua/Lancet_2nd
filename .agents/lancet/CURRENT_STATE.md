# Lancet Current Agent State

Last updated: 2026-09-09
Branch: `main`
Commit: `ede0935942e94cc1a2b6f9889bb738b8f1e5b5ea` (working tree has v3 design work and the supplied PDF)

## Current objective

Lancet v3 now has a code-ready implementation design and a pre-registered
benchmark protocol against the current WSRL stack. The next task is to
implement it and its focused tests, then pass staged validation before any
formal benchmark. No v3 smoke, debug pilot, formal benchmark, or training
process has run.

## Repository and runtime

- Host path: `/home/zhaozihan/Lancet/rl-garden`
- Container path: `/workspace/rl-garden`
- Data: `/home/zhaozihan/Lancet/data` -> `/data/lancet`
- Image/container: `lancet-d4rl:cu128` / `lancet-d4rl` (running at last audit)
- Python/Torch/CUDA: 3.10.21 / 2.7.0+cu128 / 12.8
- GPU: 6 x NVIDIA GeForce RTX 4090, 49140 MiB each; CUDA visible

## Standard execution

```bash
./dev d4rl <command>
./dev d4rl --shell '<complex shell expression>'
```

## Lancet architecture

Current executable legacy scaffold:

```text
WSRL/Cal-QL/SAC backbone
  -> shared scalar R(s,a)
  -> target y - mean_i Q_i
  -> actor uses min_i Q_i + R
```

Planned v3 (designed, not implemented):

```text
WSRL base path unchanged
  -> shared residual backbone + N critic-aligned zero-output heads
  -> K=8 policy-local action centering
  -> per-critic TD-residual fit, optionally weighted by detached REDQ U
  -> actor uses min_i(Q_i + lambda(t) Delta_i)
```

## Implementation status

### Implemented and verified

- Legacy `lancet` registry, shared scalar residual, optimizer/logging/checkpoint
  hooks, configs, unit tests, audit, and historical real-data tiny smoke.
- WSRL real-data tiny smoke, Runtime/Mount/D4RL/Lancet audits, Docker runtime,
  experiment archive workflow, and split Human Handoff.
- AntMaze 1,000,000-transition dataset parse and environment reset.

### Designed, not implemented

- `docs/design/lancet-v3-implementation.md`: architecture, shapes, local
  actions, per-critic target, REDQ U proof, loss, actor path, zero init,
  lifecycle, RNG, checkpoint/config API, code surface, and tests.
- `docs/design/lancet-v3-theory-code-alignment.md`: every legacy/v3 mismatch.
- `experiments/protocols/antmaze_wsrl_lancet_v3.md`: shared initializer,
  WSRL-vs-Full main table, matched Raw/Centered ablations, early-AUC metrics,
  debug gate, statistics, and rerun policy.
- V3 is a new implementation identity; legacy smoke evidence does not validate
  it.

### Not implemented

- `LancetV3` / `lancet_v3` registry and configs.
- Per-critic centered residual heads, U weighting, 50k linear handoff, and v3
  checkpoint state.
- V3 unit tests, tiny smoke, stability pilot, and formal benchmark.

## Environment status

- D4RL: 1.1 imports; optional backend warnings are non-blocking.
- AntMaze: env reset and 1,000,000-transition dataset parse verified.
- Kitchen: env reset previously verified; dataset not locally validated.
- Proxy: Host/container proxy bridge previously verified; not relevant here.
- Mount: Host source/data bind mounts previously verified.
- Host tmux: 3.4 installed.

## Verified tests and evidence

- Historical legacy Lancet: 6 focused tests/config preflight, finite update,
  residual parameter change, logging, and checkpoint round-trip passed.
- Historical WSRL/off2on/D4RL tests and both archived tiny smokes passed.
- V3 design audit read the supplied 26-page PDF and current SACCore, CQL,
  CalQL, WSRL, off2on phase, SACPolicy, legacy Lancet, registry, config,
  checkpoint, and tests at commit `ede0935`.
- Pure Torch contract check: pairwise/efficient U agreed within 6.7e-16 for
  N=2 and N=10; centered-action mean was below 1e-16.
- No v3 code test or experiment has run because v3 is not implemented.

## Known issues

1. Legacy Lancet is scientifically and structurally different from v3.
2. Practical v3 has observed-action TD supervision only; oracle projection
   over counterfactual actions remains an empirical theory gap.
3. V3 implementation and protocol-enablement patches must add deterministic
   eval seeding, final endpoint evaluation, explicit GPU/lineage metadata, and
   stage configs before pilots.

## Active decisions

- Host owns the only checkout; Docker is runtime only.
- Smoke/debug/formal outputs and checkpoints remain strictly separated.
- Experiment directories are authoritative; Memory/Handoff only summarize.
- V3 leaves WSRL Bellman target/base critic/target critic unchanged.
- V3 uses shared backbone + N heads, exact-zero outputs, per-critic centered
  TD fitting, U only as detached weight, and a 50k post-warmup linear lambda.
- Existing `lancet` remains Legacy Lancet-TD; proposed v3 registry is
  `lancet_v3` to prevent silent semantic/checkpoint changes.
- Every seed pair shares one WSRL offline checkpoint.
- The formal main table is WSRL vs Full Lancet v3 at five paired seeds; Raw
  and Centered are pre-registered matched component ablations.
- Primary performance metric is normalized-score AUC over online steps 0-50k;
  endpoint and longer AUC metrics are secondary.

## Immediate next steps

1. Human-review the v3 design and revised protocol.
2. Implement `lancet_v3` and the focused unit tests without changing baseline
   WSRL loss/update behavior.
3. After unit tests, run an archived tiny v3 real-data smoke.
4. Only then run the seed-0 20k stability pilot; formal work remains blocked.

## Read next

1. `AGENTS.md`
2. Task-relevant `.agents/rules/` and `.agents/runbooks/`
3. `.agents/lancet/memory/INDEX.md`
4. `docs/design/Lancet_v3_技术实现思路与理论证明.pdf`
5. `docs/design/lancet-v3-implementation.md`
6. `docs/design/lancet-v3-theory-code-alignment.md`
7. Revised v3 benchmark protocol under `experiments/protocols/`
