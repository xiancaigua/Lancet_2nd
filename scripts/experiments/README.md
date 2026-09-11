# Lancet Experiment Launcher

Run `archive_run.py` on the Host, not inside Docker. It prepares an archive,
materializes the env/agent and effective config through `./dev d4rl --dry-run`, enforces run-type paths,
executes the container command, finalizes metadata/analysis, writes one Agent
Memory, and indexes the result in the human Handoff.

Every run requires an explicit purpose, hypothesis, and success criteria.
Formal runs reject smoke configs and dirty worktrees by default.
They additionally require `--protocol` and `--dataset-path`; the launcher
stores their SHA256 values together with source/resolved config hashes.
Online formal forks must also pass `--lineage-checkpoint` so the exact shared
initializer path and hash are frozen in metadata.

Legacy direct runs with `--gpu-id N` acquire the Host lock
`/home/zhaozihan/Lancet/data/locks/gpu_N.lock` before training. Multiple
pre-registered runs assigned to one GPU therefore queue safely; metadata shows
`gpu_lock_state` as `waiting`, `acquired`, or `released`. Completion-time Agent Memory,
current-state, and Handoff updates are serialized with a separate continuity
lock.

Do not use the old fixed-GPU/45 GiB waiting pattern for queued formal work.
Formal capacity is now managed by one Host lifecycle controller:

```bash
python3 scripts/experiments/run_training.py status
```

The controller keeps all waiting jobs in one atomic JSON state file and checks
capacity at most once per five-hour normal cycle. It requires stable free
memory, low utilization, a free Lancet GPU lock, no existing Lancet job, and
only a small dormant foreign context. The current initializer gate is derived
from the observed 5,564 MiB footprint plus an 8 GiB safety margin (13,756 MiB),
not a nearly-empty 49 GiB card. It never stops or modifies foreign processes.

`run_training.py` also owns lifecycle transitions (`queued`, `starting`,
`running`, `completed`, `failed`, `blocked`), writes `progress.json`, sends
idempotent state-change email, and validates a completed initializer before
preparing its seed-matched WSRL/Lancet online pair. Existing jobs use its
non-invasive attach mode; they are never restarted. A managed worker exists
only while an assigned training command is running, and training still enters
through the exact archived `./dev d4rl ...` command.

```bash
python3 scripts/experiments/archive_run.py \
  --run-type smoke --algorithm wsrl \
  --environment antmaze-medium-play-v2 --seed 0 --gpu-id 0 \
  --config configs/off2on/wsrl_antmaze_medium_play_lancet_smoke_initializer.yaml \
  --purpose "Validate the real-data WSRL off2on pipeline." \
  --hypothesis "The tiny pipeline completes with finite updates and checkpoints." \
  --success-criteria "Process exits zero."
```

Formal identity example:

```bash
python3 scripts/experiments/archive_run.py \
  --run-type formal --algorithm wsrl --environment antmaze-medium-play-v2 \
  --seed 0 --gpu-id 0 \
  --config configs/off2on/wsrl_antmaze_medium_play_v2_offline_initializer.yaml \
  --protocol experiments/protocols/antmaze_wsrl_lancet.md \
  --dataset-path /home/zhaozihan/Lancet/data/datasets/d4rl/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5 \
  --purpose "Create the seed-0 shared formal initializer." \
  --hypothesis "The frozen WSRL initializer completes and is reloadable." \
  --success-criteria "One million updates and offline_final.pt validation pass."
```

`--gpu-id N` records the physical GPU and executes the container command with only that GPU visible. `--method-label raw-residual` (or `centered-residual`) keeps ablation archives separate while the executable registry remains `lancet`.

Arguments after `--` are forwarded to the training CLI and recorded verbatim.

After a run, validate its final checkpoint and merge every TensorBoard event
file (including the end-of-run hparam file) with:

```bash
./dev d4rl python scripts/experiments/analyze_run.py \
  --run-dir /data/lancet/runs/<type>/<run> \
  --checkpoint /data/lancet/checkpoints/<type>/<run>/final.pt \
  --reload-verified
```

Use `--reload-verified` only after a separate training-CLI `--dry-run` has
successfully constructed the agent and loaded that checkpoint. The analyzer
writes `metrics/{summary,scalars}.json` and evidence-based `analysis.md`.

For a paired debug/formal archive whose Lancet summary contains the adaptation
start coordinate:

```bash
./dev d4rl python scripts/experiments/compare_runs.py \
  --baseline-run /data/lancet/runs/<type>/<wsrl-run> \
  --candidate-run /data/lancet/runs/<type>/<lancet-run>
```

This writes `paired_comparison.{json,md}` under the candidate's `metrics/` and
explicitly reports whether the formal 0–50k adaptation horizon is complete.

The notification implementation reads only the ignored local file
`configs/local/lancet_email.env`. SMTP failures are recorded as notification
errors and never change a training result. No periodic progress email is sent.
