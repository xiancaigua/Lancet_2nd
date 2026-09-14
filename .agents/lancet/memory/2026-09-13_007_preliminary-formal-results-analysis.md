# Agent Memory: preliminary formal results analysis

- Date: 2026-09-13
- Sequence: 007
- Agent: Codex
- Branch: `main`
- Commit before: current infrastructure working tree
- Commit after: working tree; not committed
- Status: completed

## Goal

Create an evidence-backed preliminary analysis from the currently completed
formal TensorBoard archives without touching active training.

## What changed

- Added a read-only TensorBoard/scalar-to-SVG report generator.
- Created a timestamped formal analysis snapshot with Markdown, SVG figures,
  and a machine-readable evidence summary under `/data/lancet/runs/formal/analysis/`.

## Key files

- `scripts/analysis/generate_preliminary_formal_results.py`
- `/data/lancet/runs/formal/analysis/preliminary_2026-09-13/PRELIMINARY_FORMAL_RESULTS_ANALYSIS.md`

## Commands / validation

```bash
./dev d4rl python scripts/analysis/generate_preliminary_formal_results.py ...
./dev d4rl python -m py_compile scripts/analysis/generate_preliminary_formal_results.py
```

Result: two completed paired archives and raw running TensorBoard event files
were read; three SVG figures and the Markdown snapshot were produced.

## Decisions

The report separates observed post-adaptation scores from boundary-interpolated
AUC. The latter is flagged as an artifact when it inherits a warmup score.

## Problems / caveats

Only seeds 0 and 2 are complete; both have zero score at all observed
post-adaptation evaluations. Seeds 3/4 are interim only; seed 1 is remote.

## Next dependency

Complete the frozen paired matrix before final statistical comparison.

## Resume hint

Read the generated Markdown snapshot and its `evidence_snapshot.json` first.
