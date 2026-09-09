#!/usr/bin/env bash
# Download-free algorithm smoke. The optional D4RL integration command is
# intentionally separate because it may fetch the legacy AntMaze HDF5 once.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q tests/test_lancet_v1.py
python examples/train_off2on.py lancet \
  --config configs/off2on/lancet_v1_antmaze_medium_play_smoke.yaml --print-config >/dev/null
echo "Lancet V1 unit/registry smoke passed."
