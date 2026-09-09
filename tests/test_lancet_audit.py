"""Evidence-oriented current Lancet checks used by the audit script."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import Lancet


class _Env:
    num_envs = 2
    single_observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    single_action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def _agent(tmp_path=None) -> Lancet:
    return Lancet(
        env=_Env(),
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
        residual_hidden_dim=16,
        residual_hidden_layers=1,
        checkpoint_dir=None if tmp_path is None else str(tmp_path),
    )


def _fill(agent: Lancet) -> None:
    generator = torch.Generator().manual_seed(17)
    for step in range(12):
        agent.replay_buffer.add(
            torch.randn(2, 4, generator=generator),
            torch.randn(2, 4, generator=generator),
            torch.randn(2, 2, generator=generator).clamp(-1, 1),
            torch.randn(2, generator=generator),
            torch.ones(2) if step == 11 else torch.zeros(2),
        )


def test_residual_step_changes_parameters_and_checkpoint_contains_optimizer(tmp_path):
    agent = _agent(tmp_path)
    _fill(agent)
    agent._online_start_step = 0
    before = [
        parameter.detach().clone() for parameter in agent.residual_network.parameters()
    ]

    metrics = agent.train(gradient_steps=1, compute_info=True)
    parameter_delta = sum(
        (after.detach() - old).square().sum().item()
        for after, old in zip(agent.residual_network.parameters(), before)
    )
    assert parameter_delta > 0.0
    assert agent._residual_update_count == 1
    assert all(np.isfinite(value) for value in metrics.values())

    checkpoint_path = agent.save(tmp_path / "audit.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "residual_optimizer" in checkpoint["state"]["optimizers"]
    assert checkpoint["state"]["extra"]["lancet"]["schema_version"] == 1

    restored = _agent(tmp_path)
    restored.load(checkpoint_path)
    for expected, actual in zip(
        agent.residual_network.parameters(), restored.residual_network.parameters()
    ):
        assert torch.equal(expected, actual)
    assert restored._u_ema_initialized
