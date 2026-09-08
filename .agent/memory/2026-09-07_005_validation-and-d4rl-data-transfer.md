# Agent Memory: validation and D4RL data transfer

- Date: 2026-09-07
- Sequence: 005
- Agent: recovered retrospectively
- Branch: `feat/lancet`
- Commit before: unknown; recovery found `c45d89d`
- Commit after: `c45d89d` (working-tree changes remain uncommitted)
- Status: partial

## Goal

Verify D4RL/WSRL/Lancet short paths and obtain the AntMaze dataset needed for real off2on smoke.

## Validation

- AntMaze and Kitchen construct/reset successfully.
- `test_wsrl.py`: 55 passed; `test_off2on_runner.py`: 19 passed; legacy D4RL tests: 19 passed.
- Lancet/WSRL smoke configs pass CLI preflight; no formal training was started.

## Problems / caveats

The 231,110,764-byte AntMaze HDF5 transfer was interrupted. A container-side `curl -C -` resume is active, but the source has closed transfers repeatedly. The file must not be used until complete.

## Resume hint

Check `ps`/`antmaze_dataset_resume.log`, verify exact file size, then run a true tiny D4RL off2on smoke.
