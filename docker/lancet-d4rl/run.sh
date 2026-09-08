#!/usr/bin/env bash
# Creates/starts only lancet-d4rl. It never removes or replaces containers.
set -euo pipefail
readonly CONTAINER=lancet-d4rl IMAGE=lancet-d4rl:cu128 HOST_ROOT=/home/zhaozihan/Lancet
readonly HOST_UID="$(id -u)" HOST_GID="$(id -g)" PROXY=http://127.0.0.1:7891
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != true ]; then docker start "$CONTAINER" >/dev/null; fi
else
  docker run --gpus all -dit --name "$CONTAINER" --network=host --ipc=host \
    --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    --user "${HOST_UID}:${HOST_GID}" \
    -e MPLBACKEND=Agg -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/opt/lancet-home -e USER=lancet -e LOGNAME=lancet -e PATH=/opt/lancet-home/.local/bin:/opt/conda/envs/lancet-py310/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e http_proxy="$PROXY" -e https_proxy="$PROXY" -e HTTP_PROXY="$PROXY" -e HTTPS_PROXY="$PROXY" \
    -e D4RL_DATASET_DIR=/data/lancet/datasets/d4rl \
    -e TORCH_EXTENSIONS_DIR=/data/lancet/cache/torch_extensions/d4rl \
    -e WANDB_DIR=/data/lancet/runs/wandb -e XDG_CACHE_HOME=/data/lancet/cache \
    -v "${HOST_ROOT}/rl-garden:/workspace/rl-garden" -v "${HOST_ROOT}/data:/data/lancet" \
    -w /workspace/rl-garden "$IMAGE" sleep infinity >/dev/null
fi
docker exec -e USER=lancet -e LOGNAME=lancet -e PATH=/opt/lancet-home/.local/bin:/opt/conda/envs/lancet-py310/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin -u "${HOST_UID}:${HOST_GID}" -w /workspace/rl-garden "$CONTAINER" bash docker/lancet-d4rl/bootstrap.sh
