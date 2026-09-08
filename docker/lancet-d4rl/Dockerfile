FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG MUJOCO_VERSION=2.1.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git patchelf pkg-config \
    libegl1 libgl1 libglew-dev libglfw3 libosmesa6-dev \
    libx11-6 libxext6 libxrender1 libxrandr2 libxcursor1 libxi6 libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

# mujoco-py==2.1.2.14 requires this fixed legacy runtime.
RUN mkdir -p /opt/mujoco \
    && curl --fail --location --retry 3 \
       "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz" \
       -o /tmp/mujoco210.tar.gz \
    && tar -xzf /tmp/mujoco210.tar.gz -C /opt/mujoco \
    && rm /tmp/mujoco210.tar.gz

RUN conda create -y -n lancet-py310 python=3.10 pip \
    && conda clean -afy
RUN /opt/conda/envs/lancet-py310/bin/python -m pip install --no-cache-dir \
    torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128

ENV PATH=/opt/conda/envs/lancet-py310/bin:${PATH} \
    MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210 \
    LD_LIBRARY_PATH=/opt/mujoco/mujoco210/bin:${LD_LIBRARY_PATH} \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/opt/lancet-home

# Runtime dependencies come from the bind-mounted bootstrap script; no source
# checkout is copied into the image. 1777 supports the host UID/GID safely.
RUN mkdir -p /opt/lancet-home /data/lancet/cache \
    && chmod 1777 /opt/lancet-home /data/lancet/cache

WORKDIR /workspace/rl-garden
