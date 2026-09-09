#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo"

./dev d4rl python -m pytest -q tests/test_lancet.py tests/test_lancet_audit.py
./dev d4rl python examples/train_off2on.py lancet \
  --config configs/off2on/lancet_antmaze_medium_play_smoke.yaml \
  --print-config >/dev/null
printf '%s\n' 'lancet_registry_config_audit=passed'
