#!/usr/bin/env bash
set -euo pipefail
cd /workspace/rl-garden
python -m pip install --user --upgrade "pip<25"
python -m pip install --user -e '.[wandb,dev]'
# D4RL metadata names the same mjrl commit as a floating master URL. Install
# the repository-pinned legacy stack explicitly to avoid pip treating equal
# commits at two URLs as a resolver conflict.
python -m pip install --user gym==0.23.1 numpy==1.23.5 scipy==1.10.1 Cython==0.29.37 \
    mujoco-py==2.1.2.14 dm-control==1.0.14 mujoco==2.3.7 six==1.17.0 termcolor==2.5.0 \
    pybullet click
python -m pip install --user --no-deps \
    "mjrl @ git+https://github.com/aravindr93/mjrl.git@3871d93763d3b49c4741e6daeaebbc605fe140dc"
python -m pip install --user --no-deps \
    "d4rl @ git+https://github.com/nakamotoo/D4RL.git@d234792d5562b738b12b78e9074a2d6c7e23b01f"
python - <<'PY'
import rl_garden
import torch
print(f"bootstrap: rl_garden={rl_garden.__file__}")
print(f"bootstrap: torch={torch.__version__}, cuda={torch.cuda.is_available()}")
PY
