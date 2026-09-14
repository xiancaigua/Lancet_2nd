# Upstream migration promotion gate

Date: 2026-09-14

## Verdict

**BLOCKED — `main` remains unchanged.**

The integration branch passes every migration-owned scientific/runtime gate,
but the user-specified repository-wide `ruff check .` gate does not pass. The
integration branch therefore must not be promoted to `main`, no
`FORMAL_IDENTITY_V2.json` is frozen, and generation-2 baseline/formal training
must not start.

## Gate table

| Gate | Result | Evidence |
|---|---|---|
| Full upstream merge | PASS | Merge commit `c1125de`; three conflicts resolved explicitly |
| Migration-owned Ruff | PASS | Focused check across Lancet, validators, migration and analysis scripts |
| Repository-wide `ruff check .` | **FAIL** | 2,611 findings; 2,321 reported auto-fixable |
| Lancet focused tests | PASS | 71 passed; final focused subset 45 passed |
| WSRL/SAC/CQL/CalQL/checkpoint | PASS | 174 passed |
| Observation/replay/off2on | PASS | 195 passed |
| Checkpoint continuation | PASS | Exact next-step save/load parity unit test |
| Dataset fingerprint | PASS | Raw and all semantic-prefix hashes identical |
| Resolved base config parity | PASS | 118 base parameters identical across four online variants |
| Scientific numerical parity | PASS | `lancet_semantic_parity_report.md` |
| Real AntMaze smoke | PASS | `real_antmaze_smoke_report.md` |
| Email infrastructure | PASS | `[Lancet] Upstream Merge Test` sent successfully |

## Ruff blocker classification

The representative failures start in imported upstream areas such as
`baselines/cal_ql`, `baselines/core`, `examples/generate.py`, and broad framework
modules. The dominant classes are 1,635 `UP045`, 212 import-order findings, 159
deprecated imports, and 137 unnecessary collection calls. The migration-owned
files pass focused Ruff. Mass-editing 2,611 unrelated findings solely to satisfy
the gate would violate the instruction not to alter unrelated upstream code for
cosmetic validation.

This is a promotion-policy blocker, not a Lancet numerical or semantic failure.
The safe next decision is either to explicitly accept a scoped Ruff gate for
the migration diff or to undertake a separately reviewed upstream-wide lint
cleanup. Until then, integration remains the only branch containing the resync.
