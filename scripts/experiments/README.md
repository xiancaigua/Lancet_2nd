# Lancet Experiment Launcher

Run `archive_run.py` on the Host, not inside Docker. It prepares an archive,
resolves the effective config through `./dev d4rl`, enforces run-type paths,
executes the container command, finalizes metadata/analysis, writes one Agent
Memory, and indexes the result in the human Handoff.

Every run requires an explicit purpose, hypothesis, and success criteria.
Formal runs reject smoke configs and dirty worktrees by default.

```bash
python3 scripts/experiments/archive_run.py \
  --run-type smoke --algorithm wsrl \
  --environment antmaze-medium-play-v2 --seed 0 \
  --config configs/off2on/wsrl_antmaze_medium_play_smoke.yaml \
  --purpose "Validate the real-data WSRL off2on pipeline." \
  --hypothesis "The tiny pipeline completes with finite updates and checkpoints." \
  --success-criteria "Process exits zero."
```

Arguments after `--` are forwarded to the training CLI and recorded verbatim.
