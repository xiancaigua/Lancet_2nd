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

Runs with `--gpu-id N` acquire the Host lock
`/home/zhaozihan/Lancet/data/locks/gpu_N.lock` before training. Multiple
pre-registered runs assigned to one GPU therefore queue safely; metadata shows
`gpu_lock_state` as `waiting`, `acquired`, or `released`. Completion-time Agent Memory,
current-state, and Handoff updates are serialized with a separate continuity
lock.

When other projects already occupy the selected devices, add
`--min-free-gpu-mib 45000`. The archive is created immediately, but training
remains `prepared` with `gpu_capacity_state=waiting` until the selected GPU has
enough free memory. This check never stops or modifies the existing process.

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
