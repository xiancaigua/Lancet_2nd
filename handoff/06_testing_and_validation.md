# Testing and Validation

Only commands actually run are listed.

| Command | Result |
|---|---|
| `./dev d4rl bash scripts/smoke_lancet.sh` | 6 tests passed; registry/config preflight passed |
| `./dev d4rl python -m pytest -q tests/test_wsrl.py` | 55 passed |
| `./dev d4rl python -m pytest -q tests/test_off2on_runner.py` | 19 passed |
| focused D4RL env/dataset pytest | 19 passed |
| `./dev d4rl python -m pytest -q tests/test_lancet_experiment_archive.py` | 3 passed |
| `scripts/audit/run_all.sh` | all audits passed; Lancet audit set 7 passed |
| final ruff check on new Python tools/tests | passed |
| combined Lancet/archive pytest selection | 10 passed |

The audit suite verifies:

- container/Python/Torch/CUDA/GPU/`rl_garden` runtime;
- bidirectional Host/container bind-mount behavior and cleanup;
- D4RL import, AntMaze reset/action space, required dataset keys, exactly
  1,000,000 transitions, and finite observations/actions/rewards;
- Lancet registry/config, residual/corrected-Q shapes, one finite update,
  nonzero residual parameter delta, safe `lambda_u_variation=0`, checkpoint
  residual state/optimizer presence, and reload equality.

Archived WSRL smoke `20260908_123022` loaded 1M transitions, completed 2 offline and 4 online updates, changed actor/critic/target tensors, saved eight checkpoints, and reloaded `final.pt`. Archived Lancet smoke `20260908_123802` additionally recorded finite residual losses, positive offline and online residual deltas, all required TensorBoard tags, residual optimizer state, and successful reload. Its four-step episodic return/success are not collected (printed as `nan` because no episode ended), while losses and saved tensors are finite.
Kitchen environment reset was previously verified; its dataset was not fetched.
