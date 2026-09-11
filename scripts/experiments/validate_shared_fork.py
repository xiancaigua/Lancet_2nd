#!/usr/bin/env python3
"""Verify one WSRL checkpoint forks identically into WSRL and Lancet."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import WSRL, Lancet


class _ShapeOnlyVecEnv:
    num_envs = 1

    def __init__(self, observation_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        self.single_observation_space = spaces.Box(
            -np.inf, np.inf, shape=observation_shape, dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            -1.0, 1.0, shape=action_shape, dtype=np.float32
        )


def _equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _constructor_kwargs(checkpoint: dict, env) -> dict:
    hyperparameters = checkpoint["metadata"]["hyperparameters"]
    accepted = {
        name
        for name, parameter in inspect.signature(WSRL.__init__).parameters.items()
        if name not in {"self", "env", "eval_env"}
        and parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
    }
    kwargs = {name: value for name, value in hyperparameters.items() if name in accepted}
    kwargs.update(
        env=env,
        device="cpu",
        buffer_device="cpu",
        buffer_size=8,
        learning_starts=0,
        eval_freq=0,
        checkpoint_dir=None,
    )
    return kwargs


def validate(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    observation_shape = tuple(metadata["observation_space"]["shape"])
    action_shape = tuple(metadata["action_space"]["shape"])
    env = _ShapeOnlyVecEnv(observation_shape, action_shape)
    kwargs = _constructor_kwargs(checkpoint, env)
    wsrl = WSRL(**kwargs)
    lancet = Lancet(**kwargs)
    wsrl.load(path, load_replay_buffer=False)
    lancet.load(path, load_replay_buffer=False)

    policy_equal = _equal(wsrl.policy.state_dict(), lancet.policy.state_dict())
    optimizer_equal = all(
        _equal(getattr(wsrl, name).state_dict(), getattr(lancet, name).state_dict())
        for name in ("q_optimizer", "actor_optimizer", "alpha_optimizer", "cql_alpha_optimizer")
    )
    counters_equal = (
        wsrl._global_step == lancet._global_step
        and wsrl._global_update == lancet._global_update
        and wsrl._online_start_step == lancet._online_start_step
    )
    generator = torch.Generator().manual_seed(947)
    obs = torch.randn(4, *observation_shape, generator=generator)
    actions = torch.empty(4, *action_shape).uniform_(-1.0, 1.0, generator=generator)
    residual = lancet.residual(obs, actions)
    lancet._online_start_step = 0
    lancet._lancet_adaptation_start_step = 0
    lancet._global_step = 0
    base_q = lancet._critic_forward(obs, actions, target=False)
    corrected_q = lancet.corrected_q_values(obs, actions)
    zero_residual = torch.equal(residual, torch.zeros_like(residual))
    q_use_equal = torch.equal(base_q, corrected_q)
    passed = policy_equal and optimizer_equal and counters_equal and zero_residual and q_use_equal
    return {
        "checkpoint": str(path),
        "source_algorithm": metadata.get("algorithm_class"),
        "policy_actor_critic_target_alpha_equal": policy_equal,
        "base_optimizer_states_equal": optimizer_equal,
        "global_phase_counters_equal": counters_equal,
        "lancet_residual_exact_zero": zero_residual,
        "fork_q_use_equals_q_base": q_use_equal,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.checkpoint)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
