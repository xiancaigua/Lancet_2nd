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

    def __init__(self, observation_space, action_space):
        self.single_observation_space = observation_space
        self.single_action_space = action_space


def _space_from_summary(summary: dict, *, low=-np.inf, high=np.inf):
    if summary["type"] == "Box":
        return spaces.Box(
            low,
            high,
            shape=tuple(summary["shape"]),
            dtype=np.dtype(summary["dtype"]),
        )
    if summary["type"] == "Dict":
        return spaces.Dict(
            {
                key: _space_from_summary(value)
                for key, value in summary["spaces"].items()
            }
        )
    raise ValueError(f"unsupported checkpoint space type: {summary['type']}")


def _random_observation(space, batch: int, generator: torch.Generator):
    if isinstance(space, spaces.Dict):
        return {
            key: _random_observation(value, batch, generator)
            for key, value in space.spaces.items()
        }
    return torch.randn(batch, *space.shape, generator=generator)


def _equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _constructor_kwargs(checkpoint: dict, env) -> dict:
    hyperparameters = checkpoint["metadata"]["hyperparameters"]
    accepted = {
        name
        for name, parameter in inspect.signature(WSRL.__init__).parameters.items()
        if name not in {"self", "env", "eval_env"}
        and parameter.kind not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
    }
    kwargs = {
        name: value for name, value in hyperparameters.items() if name in accepted
    }
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
    observation_space = _space_from_summary(metadata["observation_space"])
    hyperparameters = metadata.get("hyperparameters", {})
    action_space = _space_from_summary(
        metadata["action_space"],
        low=hyperparameters.get("action_low", -1.0),
        high=hyperparameters.get("action_high", 1.0),
    )
    env = _ShapeOnlyVecEnv(observation_space, action_space)
    kwargs = _constructor_kwargs(checkpoint, env)
    wsrl = WSRL(**kwargs)
    lancet = Lancet(**kwargs)
    wsrl.load(path, load_replay_buffer=False)
    lancet.load(path, load_replay_buffer=False)

    policy_equal = _equal(wsrl.policy.state_dict(), lancet.policy.state_dict())
    optimizer_equal = True
    for name in (
        "q_optimizer",
        "actor_optimizer",
        "alpha_optimizer",
        "cql_alpha_optimizer",
    ):
        wsrl_optimizer = getattr(wsrl, name)
        lancet_optimizer = getattr(lancet, name)
        if (wsrl_optimizer is None) != (lancet_optimizer is None):
            optimizer_equal = False
        elif wsrl_optimizer is not None:
            optimizer_equal = optimizer_equal and _equal(
                wsrl_optimizer.state_dict(), lancet_optimizer.state_dict()
            )
    counters_equal = (
        wsrl._global_step == lancet._global_step
        and wsrl._global_update == lancet._global_update
        and wsrl._online_start_step == lancet._online_start_step
    )
    generator = torch.Generator().manual_seed(947)
    obs = _random_observation(observation_space, 4, generator)
    actions = torch.empty(4, *action_space.shape).uniform_(
        -1.0, 1.0, generator=generator
    )
    residual = lancet.residual(obs, actions)
    lancet._online_start_step = 0
    lancet._lancet_adaptation_start_step = 0
    lancet._global_step = 0
    base_q = lancet._critic_forward(obs, actions, target=False)
    corrected_q = lancet.corrected_q_values(obs, actions)
    zero_residual = torch.equal(residual, torch.zeros_like(residual))
    q_use_equal = torch.equal(base_q, corrected_q)
    q_use_max_abs_diff = float((base_q - corrected_q).abs().max().item())
    passed = (
        policy_equal
        and optimizer_equal
        and counters_equal
        and zero_residual
        and q_use_equal
    )
    return {
        "checkpoint": str(path),
        "source_algorithm": metadata.get("algorithm_class"),
        "policy_actor_critic_target_alpha_equal": policy_equal,
        "base_optimizer_states_equal": optimizer_equal,
        "global_phase_counters_equal": counters_equal,
        "lancet_residual_exact_zero": zero_residual,
        "fork_q_use_equals_q_base": q_use_equal,
        "fork_q_use_max_abs_diff": q_use_max_abs_diff,
        "fork_base_q_finite": bool(torch.isfinite(base_q).all().item()),
        "fork_corrected_q_finite": bool(torch.isfinite(corrected_q).all().item()),
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
