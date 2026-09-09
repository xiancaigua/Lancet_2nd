# Agent Memory: Lancet specification finalized

- Date: 2026-09-09
- Sequence: 003
- Agent: Codex
- Branch: `main`
- Commit before: `51d014d058605f92ecc823ea3b2b42ef25ab4dd3`
- Commit after: working tree
- Status: completed

## Goal

Freeze the current Lancet name, implementation semantics, and benchmark contract before coding.

## What changed

- Renamed the current design/protocol paths to remove `v3` from the formal method identity.
- Defined the only confirmed legacy executable as Lancet V1; no V2 executable was invented.
- Froze post-base-step residual fitting with one reused target, detached actor centering baseline, constant 50k active window, adaptation-coordinate AUC, and common actual final endpoint.

## Key files

- `docs/design/lancet-implementation.md`
- `docs/design/lancet-theory-code-alignment.md`
- `experiments/protocols/antmaze_wsrl_lancet.md`

## Validation

Reviewed the current WSRL paper config, SAC/CQL/WSRL update chain, and legacy Git history at `51d014d`.

## Decisions

Current class/registry remain `Lancet`/`lancet`; legacy moves to `LancetV1`/`lancet_v1`. Formal primary metric is Adaptation AUC 0–50k.

## Problems / caveats

The implementation and all new-method runtime evidence are still pending.

## Next dependency

Migrate legacy identity and implement the frozen contract without changing WSRL base math.

## Resume hint

Read `docs/design/lancet-implementation.md`, then modify only the Lancet subclass/registration/config/test surface.
