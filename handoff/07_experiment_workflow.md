# Experiment Archival Workflow

## Current protocol identities

The current planned scientific entry point is
[`experiments/protocols/antmaze_wsrl_lancet_v3.md`](../experiments/protocols/antmaze_wsrl_lancet_v3.md).
It covers WSRL, capacity-matched Raw and Centered residual ablations, and Full
Lancet v3 on `antmaze-medium-play-v2`. It is a pre-registered design: v3 is not
implemented and no v3 pilot or formal run has started.

[`antmaze_wsrl_lancet_v1.md`](../experiments/protocols/antmaze_wsrl_lancet_v1.md)
is retained only as the never-executed Legacy Lancet-TD/shared-scalar protocol.
It must not be used as the contract or evidence for v3.

Every smoke, debug, or formal run must be launched through
`scripts/experiments/archive_run.py` on the Host. The experiment directory is
authoritative; Agent Memory and this Handoff contain summaries and indexes.

## Isolation and protection

```text
/home/zhaozihan/Lancet/data/runs/{smoke,debug,formal}/
/home/zhaozihan/Lancet/data/checkpoints/{smoke,debug,formal}/
```

The launcher validates run-type intent, generates only paths under the matching
root, verifies those paths in rl-garden's resolved config, rejects non-smoke
configs for smoke, rejects smoke configs for formal, and requires a clean Git
tree for formal unless an explicit dirty-run override captures `git.diff`.
The v3 protocol is stricter: formal runs may not use that override.

## Required order

```text
purpose + hypothesis + success criteria
  -> archive directory
  -> README/config/command/metadata/resolved config
  -> prepared -> running
  -> process log + checkpoints/metrics
  -> finished/failed/stopped
  -> evidence-based analysis
  -> Agent Memory + CURRENT_STATE + human experiment index
```

Before execution the archive contains:

- `README.md`: purpose, hypothesis, type, algorithm/env/seed, Git revision,
  comparison, exact command, expected outputs, success/failure criteria;
- `config.yaml`: frozen source config;
- `resolved_config.json`: effective config emitted by rl-garden;
- `command.txt`: actual `./dev d4rl ...` argv rendered shell-safely;
- `metadata.json`: status, timestamps, Git state, paths, and hardware;
- `metrics/` and `plots/` placeholders.

After execution it contains `train.log`, updated metadata, and `analysis.md`.
A keyboard interruption is finalized as `stopped` (return code 130), not left
at `running`. Analysis must cite process/log/checkpoint/metric evidence and
mark uncollected metrics as `not collected`; it must never infer results from
configuration alone.

## Smoke, debug, and formal roles

- **Smoke** validates plumbing, finite updates, switch, checkpoint, and shapes;
  it is not a performance claim.
- **Debug pilot** is an engineering stability gate and never enters a formal
  result table.
- **Formal** requires a clean frozen commit and config, shared checkpoint
  lineage, seed, dataset, budget, selected GPU, timestamps, metrics, and final
  checkpoint. It runs detached through Host tmux or a scheduler.

## Launcher example

```bash
python3 scripts/experiments/archive_run.py \
  --run-type smoke \
  --algorithm wsrl \
  --environment antmaze-medium-play-v2 \
  --seed 0 \
  --config configs/off2on/wsrl_antmaze_medium_play_smoke.yaml \
  --purpose 'Validate real-data WSRL offline-to-online plumbing.' \
  --hypothesis 'The tiny run completes with checkpoints.' \
  --success-criteria 'Process exits zero.' \
  --success-criteria 'Offline and final checkpoints can be reloaded.'
```

Arguments after `--` are forwarded to the training CLI and frozen in the
command record. See `scripts/experiments/README.md`.

## Existing outputs

Nine older setup/test/download logs lacked pre-run intent/config/metadata.
They remain preserved under `runs/legacy_unclassified/`; none was relabeled
formal. No existing checkpoint required classification.

## V3 implementation-to-benchmark gates

The required order is:

1. human review of `docs/design/lancet-v3-implementation.md` and the v3
   protocol;
2. implement `lancet_v3` plus focused unit tests without changing WSRL's base
   target or critic update;
3. run one archived tiny real-data v3 smoke;
4. add/review shared-initializer and online-fork configs, deterministic eval
   seed, forced endpoint evaluation, diagnostics, GPU/lineage metadata, and
   periodic checkpoint cadence;
5. run the archived seed-0 debug pilot from the v3 protocol;
6. freeze a clean pushed commit and only then authorize formal execution.

Do not run the legacy v1 pilot as a substitute for v3 validation.
