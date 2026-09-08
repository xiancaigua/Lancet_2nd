"""Focused, download-free Lancet v1 regression tests."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import Lancet, ResidualQNetwork


class DummyVecEnv:
    num_envs = 2
    single_observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    single_action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def _agent(*, use_lancet: bool = True) -> Lancet:
    return Lancet(
        env=DummyVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=8,
        learning_starts=0,
        training_freq=1,
        utd=1.0,
        eval_freq=0,
        net_arch={"pi": [16], "qf": [16]},
        n_critics=3,
        critic_subsample_size=2,
        use_cql_loss=False,
        use_calql=False,
        use_lancet=use_lancet,
        residual_hidden_dim=16,
        residual_hidden_layers=1,
        lambda_u_variation=0.0,
    )


def _fill(agent: Lancet, steps: int = 12) -> None:
    for step in range(steps):
        obs = torch.randn(2, 4)
        next_obs = torch.randn(2, 4)
        actions = torch.randn(2, 2).clamp(-1, 1)
        rewards = torch.randn(2)
        dones = torch.ones(2) if step == steps - 1 else torch.zeros(2)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_residual_network_shape():
    net = ResidualQNetwork(4, 2, [8, 8])
    assert net(torch.randn(5, 4), torch.randn(5, 2)).shape == (5, 1)


def test_disabled_mode_returns_zero_residual():
    agent = _agent(use_lancet=False)
    actions = torch.randn(3, 2)
    assert agent.residual(torch.randn(3, 4), actions).equal(torch.zeros(3, 1))
    assert agent.residual_optimizer is None


def test_corrected_q_and_one_update_are_finite():
    agent = _agent()
    _fill(agent)
    batch = agent.replay_buffer.sample(8)
    assert agent.corrected_q_values(batch.obs, batch.actions).shape == (3, 8, 1)
    info = agent.train(gradient_steps=1, compute_info=True)
    for key in (
        "lancet_residual_loss",
        "lancet_td_residual_loss",
        "lancet_u_variation_loss",
        "lancet_reg_loss",
        "critic_td_error_abs_mean",
    ):
        assert key in info
        assert math.isfinite(info[key])


def test_checkpoint_round_trip_and_online_switch(tmp_path):
    agent = _agent()
    _fill(agent)
    agent.train(gradient_steps=1, compute_info=True)
    before = {k: v.detach().clone() for k, v in agent.residual_network.state_dict().items()}
    checkpoint = agent.save(tmp_path / "lancet.pt")
    restored = _agent()
    restored.load(checkpoint)
    for key, value in before.items():
        assert torch.equal(value, restored.residual_network.state_dict()[key])
    restored.switch_to_online_mode(online_replay_mode="empty")
    assert restored._online_start_step is not None


def test_u_variation_requires_an_approved_formula():
    with pytest.raises(ValueError, match="U-variation"):
        Lancet(env=DummyVecEnv(), device="cpu", buffer_device="cpu", lambda_u_variation=1.0)


def test_lancet_registry_discovery():
    from rl_garden.training.off2on._registry import registry

    registry.discover()
    assert "lancet" in registry.entries()
