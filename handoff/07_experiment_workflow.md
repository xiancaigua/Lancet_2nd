# Experiment Archival Workflow

## Planned first formal protocol

The frozen design entry point is
[`experiments/protocols/antmaze_wsrl_lancet_v1.md`](../experiments/protocols/antmaze_wsrl_lancet_v1.md).
It compares WSRL with **Lancet-TD / Lancet w/o U** on
`antmaze-medium-play-v2`; `lambda_u_variation` remains `0.0`. The protocol is
planned only. Formal execution is blocked until its small implementation
patches and two seed-0 debug pilots pass.

Every smoke, debug, or formal run must be launched through
`scripts/experiments/archive_run.py` on the Host. The experiment directory is
authoritative; Agent Memory and this Handoff contain only summaries/indexes.

## Isolation and protection

```text
/home/zhaozihan/Lancet/data/runs/{smoke,debug,formal}/
/home/zhaozihan/Lancet/data/checkpoints/{smoke,debug,formal}/
```

The launcher validates run-type intent, generates only paths under the matching
root, verifies those paths in rl-garden's resolved config, rejects non-smoke
configs for smoke, rejects smoke configs for formal, and requires a clean Git
tree for formal unless an explicit dirty-run override captures `git.diff`.

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
- `command.txt`: the actual `./dev d4rl ...` argv rendered shell-safely;
- `metadata.json`: status, timestamps, Git state, paths, hardware;
- `metrics/` and `plots/` placeholders.

After execution it contains `train.log`, updated metadata, and `analysis.md`.
A keyboard interruption is finalized as `stopped` (return code 130) rather than leaving metadata at `running`.
The automatic analysis reports only process/log/checkpoint evidence and marks
uncollected metrics as such. Humans/Agents may enrich it from real metrics,
evaluation, checkpoints, or WandB, but must never infer results from config.

## Smoke versus formal

Smoke validates plumbing: data load, updates, switch, env steps, logging,
checkpoints, finite values, and shapes. It is not a performance claim. Formal
runs additionally require frozen commit/config, dataset, seed, baseline,
budget, hardware, timestamps, final metrics, and final checkpoint. Run formal
work detached through Host tmux or a scheduler, never as an unarchived Codex
foreground command.

## Launcher example

```bash
python3 scripts/experiments/archive_run.py \
  --run-type smoke \
  --algorithm wsrl \
  --environment antmaze-medium-play-v2 \
  --seed 0 \
  --config configs/off2on/wsrl_antmaze_medium_play_smoke.yaml \
  --purpose 'Validate real-data WSRL offline-to-online plumbing.' \
  --hypothesis 'The 2-offline/4-online-step run completes with checkpoints.' \
  --success-criteria 'Process exits zero.' \
  --success-criteria 'Offline and final checkpoints can be reloaded.'
```

Arguments after `--` are forwarded to the training CLI and frozen in the
command record. See `scripts/experiments/README.md`.

## Existing outputs

Nine older setup/test/download logs lacked pre-run intent/config/metadata.
They were preserved under `runs/legacy_unclassified/`; none was relabeled
formal. The preserved files are `lancet_d4rl_bootstrap.log`,
`antmaze_dataset_resume.log`, three D4RL import/env/dataset logs, two
config-preflight logs, and two WSRL pytest logs. No existing checkpoint
required classification.

## Protocol-specific preflight

Before the first AntMaze v1 pilot, create/review stage-specific configs for a
shared WSRL offline initializer and two online forks. Add deterministic eval
seeding, a forced final-endpoint evaluation, required Q/residual diagnostics,
paired post-load RNG alignment, explicit GPU selection, checkpoint-lineage
metadata, and periodic checkpoint frequency. Do not use the current full WSRL and Lancet YAML files directly for
the paired comparison: both currently include their own offline phase.
