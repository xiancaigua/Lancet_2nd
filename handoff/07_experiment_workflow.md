# Experiment Archival Workflow

## Current protocol identities

The current planned scientific entry point is
[`experiments/protocols/antmaze_wsrl_lancet.md`](../experiments/protocols/antmaze_wsrl_lancet.md).
It covers WSRL, capacity-matched Raw and Centered residual ablations, and
current Lancet on `antmaze-medium-play-v2`. Current-method tiny smoke and the
paired seed-0 20k debug pilot passed; no formal run has started.

[`antmaze_wsrl_lancet_v1.md`](../experiments/protocols/antmaze_wsrl_lancet_v1.md)
is retained only as the never-executed Lancet V1 shared-scalar protocol.
It must not be used as the contract or evidence for current Lancet.

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
The current protocol is stricter: formal runs may not use that override.

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

## Lancet implementation-to-benchmark gates

The required order is:

1. current Lancet implementation and baseline regression validation (done);
2. archived tiny real-data shared-fork smoke (done);
3. archived seed-0 20k shared initializer and paired online pilot (done);
4. evidence-based checkpoint/scalar/paired-AUC analysis (done);
5. human-review and push a clean frozen evidence commit (pending);
6. recheck formal roots and explicitly authorize the five archived 1M shared
   offline initializers before any online formal branch (pending).

Do not run the Lancet V1 pilot as a substitute for current Lancet validation.
