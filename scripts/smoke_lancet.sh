#!/usr/bin/env bash
# Current Lancet unit/config smoke. Real-data runs must use the experiment
# archive launcher and a seed-matched shared WSRL offline checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q tests/test_lancet.py tests/test_lancet_audit.py
python examples/train_off2on.py lancet \
  --config configs/off2on/lancet_antmaze_medium_play_smoke.yaml \
  --print-config >/dev/null
echo "Lancet unit/registry/config smoke passed."
