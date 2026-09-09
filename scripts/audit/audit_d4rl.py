#!/usr/bin/env python3
"""Validate the local AntMaze environment and complete 1M-transition dataset."""

from __future__ import annotations

import os

import d4rl
import gym
import numpy as np

dataset_dir = os.environ.get("D4RL_DATASET_DIR")
assert dataset_dir == "/data/lancet/datasets/d4rl", dataset_dir
env = gym.make("antmaze-medium-play-v2")
try:
    reset = env.reset()
    observation = reset[0] if isinstance(reset, tuple) else reset
    dataset = env.get_dataset()
    required = {"observations", "actions", "rewards", "terminals"}
    missing = required.difference(dataset)
    assert not missing, missing
    assert dataset["observations"].shape[0] == 1_000_000
    for key in ("observations", "actions", "rewards"):
        assert np.isfinite(dataset[key]).all(), key
    assert np.isfinite(observation).all()
    print(f"d4rl_version={getattr(d4rl, '__version__', 'unknown')}")
    print(f"dataset_dir={dataset_dir}")
    print(f"observations={dataset['observations'].shape}")
    print(f"actions={dataset['actions'].shape}")
    print(f"action_space={env.action_space}")
    print("d4rl_audit=passed")
finally:
    env.close()
