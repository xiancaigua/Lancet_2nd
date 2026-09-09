#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo"

./dev status
./dev d4rl python --version
./dev d4rl python -c 'import rl_garden, torch; print("torch=" + torch.__version__); print("cuda_runtime=" + str(torch.version.cuda)); print("cuda=" + str(torch.cuda.is_available())); print("gpu=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")); print("rl_garden=ok")'
./dev d4rl nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
