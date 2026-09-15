# Upstream migration promotion gate

Date: 2026-09-15

## Verdict

**PASS — eligible for `main` promotion.**

The repository-wide Ruff gate is evaluated as **no new lint regressions relative
to clean upstream/main, with integration-added and Lancet-owned Python files
clean**. This avoids rewriting unrelated upstream lint debt while retaining a
strict, reproducible regression gate for this migration.

## Gate table

| Gate | Result | Evidence |
|---|---|---|
| Full upstream merge | PASS | Merge commit `c1125de`; three conflicts resolved explicitly |
| No-new-regressions Ruff | PASS | Ruff 0.16.6: clean upstream `252d1e0` and integration both have 2,610 normalized findings; finding multisets match exactly |
| Integration-added Python Ruff | PASS | All 24 Python files added relative to upstream pass with zero findings |
| Lancet-owned Ruff | PASS | Lancet algorithms, runners, and core tests pass with zero findings |
| Lancet focused tests | PASS | 71 passed; final focused subset 45 passed |
| WSRL/SAC/CQL/CalQL/checkpoint | PASS | 174 passed |
| Observation/replay/off2on | PASS | 195 passed |
| Checkpoint continuation | PASS | Exact next-step save/load parity unit test |
| Dataset fingerprint | PASS | Raw and all semantic-prefix hashes identical |
| Resolved base config parity | PASS | 118 base parameters identical across four online variants |
| Scientific numerical parity | PASS | `lancet_semantic_parity_report.md` |
| Real AntMaze smoke | PASS | `real_antmaze_smoke_report.md` |
| Email infrastructure | PASS | `[Lancet] Upstream Merge Test` sent successfully |

## Ruff baseline method

The same Ruff 0.16.6 binary and repository configuration were run on:

- a clean detached worktree at upstream/main `252d1e0948618a0cd3675b9a05e0bcf4c29b5afb`;
- the integration worktree after migration.

Findings were normalized as `(repository-relative path, rule code, message)`.
This intentionally ignores line-number drift while retaining multiplicity.
Before correction, integration had one new `UP045` in the locally added
`num_eval_episodes` WSRL argument. That annotation was changed from
`Optional[int]` to `int | None`. After correction:

```text
upstream findings:                  2610
integration findings:               2610
normalized integration-only:           0
normalized upstream-only:              0
normalized multiset SHA256:
c9c3221b063ccb5aceb2268f4a1986701f22083652262bb92be7aa65440ea7a4
```

The full raw JSON files were retained only as local `/tmp` audit artifacts;
their hashes and the machine-readable summary are recorded in
`ruff_baseline_comparison.json`. A raw repository-wide zero-findings gate is
not meaningful for this merge because all remaining 2,610 findings reproduce
unchanged on clean upstream/main.

## Promotion rule

Future migration changes pass this gate only when:

1. the normalized integration finding multiset introduces no entries beyond
   the recorded upstream baseline; and
2. integration-added and Lancet-owned Python files remain individually clean.

Mass-fixing unrelated upstream findings is explicitly outside this migration.
