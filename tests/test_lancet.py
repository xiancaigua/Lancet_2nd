"""Focused current-Lancet tensor, gradient, lifecycle, and checkpoint tests."""

from __future__ import annotations

import copy
import math
from types import MethodType

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import WSRL, Lancet, ResidualEnsemble
from rl_garden.common.training_phase import InitialTrainingPhase


class DummyVecEnv:
    num_envs = 2
    single_observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    single_action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def _base_kwargs(**overrides):
    kwargs = {
        "env": DummyVecEnv(),
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 128,
        "batch_size": 8,
        "learning_starts": 0,
        "training_freq": 1,
        "utd": 1.0,
        "eval_freq": 0,
        "net_arch": {"pi": [16], "qf": [16]},
        "n_critics": 3,
        "critic_subsample_size": 2,
        "use_cql_loss": False,
        "use_calql": False,
        "seed": 7,
    }
    kwargs.update(overrides)
    return kwargs


def _agent(variant: str = "lancet", **overrides) -> Lancet:
    return Lancet(
        **_base_kwargs(**overrides),
        lancet_variant=variant,
        residual_hidden_dim=16,
        residual_hidden_layers=1,
        residual_lr=1e-3,
        local_action_count=8,
    )


def _fill(agent, steps: int = 12) -> None:
    generator = torch.Generator().manual_seed(17)
    for step in range(steps):
        agent.replay_buffer.add(
            torch.randn(2, 4, generator=generator),
            torch.randn(2, 4, generator=generator),
            torch.randn(2, 2, generator=generator).clamp(-1, 1),
            torch.randn(2, generator=generator),
            torch.ones(2) if step == steps - 1 else torch.zeros(2),
        )


def _activate(agent: Lancet, step: int = 0) -> None:
    agent._online_start_step = 0
    agent._initial_phase_start_step = None
    agent._lancet_adaptation_start_step = 0
    agent._global_step = step


def _parameter_state(module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before, module) -> bool:
    return any(
        not torch.equal(old, current.detach())
        for old, current in zip(before, module.parameters())
    )


def test_residual_ensemble_shape_and_exact_zero_initialization():
    network = ResidualEnsemble(4, 2, 3, [8, 8])
    output = network(torch.randn(5, 4), torch.randn(5, 2))
    assert output.shape == (3, 5, 1)
    assert torch.equal(output, torch.zeros_like(output))


def test_zero_init_q_use_equals_base_q():
    agent = _agent()
    _activate(agent)
    obs = torch.randn(5, 4)
    actions = torch.randn(5, 2).clamp(-1, 1)
    base = agent._critic_forward(obs, actions, target=False)
    corrected = agent.corrected_q_values(obs, actions)
    assert torch.equal(corrected, base)


def test_centered_delta_has_zero_action_mean_and_raw_is_unprojected():
    values = torch.randn(3, 4, 8, 1)
    centered = _agent("centered").delta_local(values)
    assert torch.allclose(centered.mean(dim=2), torch.zeros(3, 4, 1), atol=1e-6)
    assert torch.equal(_agent("raw").delta_local(values), values)


def test_uncertainty_efficient_matches_pairwise_and_twin_case():
    for n_critics in (2, 5):
        q_local = torch.randn(n_critics, 7, 8, 1, dtype=torch.float64)
        efficient = Lancet.uncertainty_from_q_local(q_local)
        pairwise = []
        for i in range(n_critics):
            for j in range(i + 1, n_critics):
                difference = q_local[i] - q_local[j]
                pairwise.append(difference.var(dim=1, unbiased=False))
        expected = torch.stack(pairwise).mean(dim=0)
        assert torch.allclose(efficient, expected, atol=1e-12, rtol=1e-12)


def test_u_weight_ema_initializes_from_first_active_batch():
    agent = _agent()
    uncertainty = torch.tensor([[1.0], [3.0]])
    weights = agent.uncertainty_weights(uncertainty, update_ema=True)
    assert agent._u_ema == 2.0
    assert agent._u_ema_initialized
    assert torch.allclose(weights, torch.tensor([[1.5], [2.5]]), atol=1e-6)
    for variant in ("raw", "centered"):
        other = _agent(variant)
        assert torch.equal(
            other.uncertainty_weights(uncertainty, update_ema=True),
            torch.ones_like(uncertainty),
        )


def test_target_is_reused_and_residual_fits_post_step_per_critic_q():
    agent = _agent("raw")
    _fill(agent)
    _activate(agent)
    batch = agent.replay_buffer.sample(8)
    agent._sample_train_batch = lambda _: batch
    calls = 0
    target = torch.linspace(-1.0, 1.0, 8).unsqueeze(1)

    def fixed_target(self, data):
        nonlocal calls
        calls += 1
        return target

    agent._target_q = MethodType(fixed_target, agent)
    metrics = agent.train(gradient_steps=1, compute_info=True)
    with torch.no_grad():
        q_updated = agent._critic_forward(batch.obs, batch.actions, target=False)
        expected = (target.unsqueeze(0).expand_as(q_updated) - q_updated).mean().item()
    assert calls == 1
    assert math.isclose(metrics["lancet_td_residual_mean"], expected, abs_tol=1e-6)
    assert agent._residual_update_count == 1


def test_residual_step_changes_only_residual_after_base_step():
    agent = _agent("raw")
    _fill(agent)
    _activate(agent)
    batch = agent.replay_buffer.sample(8)
    agent._captured_target_y = agent._target_q(batch).detach()
    critic_before = _parameter_state(agent.policy.critic)
    residual_before = _parameter_state(agent.residual_network)
    agent._post_critic_update(batch, {})
    assert not _changed(critic_before, agent.policy.critic)
    assert _changed(residual_before, agent.residual_network)


def test_actor_uses_min_of_per_critic_repaired_q():
    agent = _agent("raw")
    _activate(agent)
    batch = (
        agent.replay_buffer.sample(8) if agent.replay_buffer.sampleable_size else None
    )
    if batch is None:
        _fill(agent)
        batch = agent.replay_buffer.sample(8)
    action = torch.zeros(8, 2, requires_grad=True)
    log_prob = torch.zeros(8, 1)
    features = torch.zeros(8, 4)
    q_all = torch.tensor([1.0, 3.0, 2.0]).reshape(3, 1, 1).expand(3, 8, 1)
    correction = torch.tensor([4.0, -4.0, 0.5]).reshape(3, 1, 1).expand(3, 8, 1)
    agent.policy.actor_action_log_prob = lambda *args, **kwargs: (
        action,
        log_prob,
        features,
    )
    agent.policy.critic_features_for = lambda *args, **kwargs: features
    agent.policy.q_values_all = lambda *args, **kwargs: q_all
    agent._local_actions = lambda obs, replay: replay.unsqueeze(1).expand(-1, 8, -1)
    agent.residual_local = lambda obs, local: torch.zeros(3, 8, 8, 1)
    agent.residual = lambda obs, actions: correction
    loss, _ = agent._actor_loss_from_batch(batch)
    expected_q = (q_all + correction).min(dim=0).values
    assert torch.allclose(loss, -expected_q.mean())
    assert not torch.equal(expected_q, q_all.min(dim=0).values)


def test_actor_centering_keeps_action_gradient_but_not_q_parameter_gradients():
    agent = _agent("centered")
    _activate(agent)
    with torch.no_grad():
        final = next(
            module
            for module in reversed(list(agent.residual_network.modules()))
            if isinstance(module, torch.nn.Linear)
        )
        final.weight.fill_(0.2)
    action = torch.randn(6, 2, requires_grad=True)
    obs = torch.randn(6, 4)
    baseline = torch.randn(3, 6, 1).detach()
    with agent._frozen_q_parameters():
        delta = agent.residual(obs, action) - baseline
        delta.sum().backward()
    assert action.grad is not None and action.grad.abs().sum() > 0
    assert all(
        parameter.grad is None for parameter in agent.residual_network.parameters()
    )
    assert all(parameter.grad is None for parameter in agent.policy.critic.parameters())


def test_target_q_is_identical_to_wsrl_and_never_uses_residual():
    wsrl = WSRL(**_base_kwargs())
    agent = _agent()
    wsrl.policy.load_state_dict(copy.deepcopy(agent.policy.state_dict()))
    _fill(agent)
    batch = agent.replay_buffer.sample(8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        expected = wsrl._target_q(batch)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        actual = agent._target_q(batch)
    assert torch.equal(actual, expected)


def test_offline_warmup_active_window_and_expiry():
    offline = _agent()
    _fill(offline)
    offline.train(gradient_steps=1)
    assert offline._residual_update_count == 0

    phase = InitialTrainingPhase(
        duration_steps=5,
        update_actor=False,
        update_critic=False,
        update_encoder=False,
    )
    warmup = _agent(initial_training_phase=phase)
    _fill(warmup)
    warmup.switch_to_online_mode("append")
    warmup._global_step = 4
    assert warmup.train(gradient_steps=1, compute_info=True) == {}
    assert warmup._residual_update_count == 0

    warmup._global_step = 5
    warmup.train(gradient_steps=1)
    assert warmup.adaptation_step == 0
    assert warmup._residual_update_count == 1
    assert warmup.correction_scale() == 1.0

    warmup._global_step = 5 + warmup.handoff_window_steps
    count = warmup._residual_update_count
    warmup.train(gradient_steps=1)
    assert warmup.correction_scale() == 0.0
    assert warmup._residual_update_count == count


def test_utd_four_runs_four_residual_steps_and_one_actor_step():
    agent = _agent(utd=4.0, batch_size=16)
    _fill(agent, 20)
    _activate(agent)
    q_steps = 0
    actor_steps = 0
    original_q_step = agent.q_optimizer.step
    original_actor_step = agent.actor_optimizer.step

    def q_step(*args, **kwargs):
        nonlocal q_steps
        q_steps += 1
        return original_q_step(*args, **kwargs)

    def actor_step(*args, **kwargs):
        nonlocal actor_steps
        actor_steps += 1
        return original_actor_step(*args, **kwargs)

    agent.q_optimizer.step = q_step
    agent.actor_optimizer.step = actor_step
    agent.train(gradient_steps=4)
    assert q_steps == 4
    assert agent._residual_update_count == 4
    assert actor_steps == 1


def test_wsrl_checkpoint_fork_and_lancet_round_trip(tmp_path):
    wsrl = WSRL(**_base_kwargs(checkpoint_dir=str(tmp_path)))
    _fill(wsrl)
    wsrl.train(gradient_steps=1)
    shared_path = wsrl.save(tmp_path / "offline_final.pt")

    agent = _agent(checkpoint_dir=str(tmp_path))
    agent.load(shared_path, load_replay_buffer=False)
    for expected, actual in zip(wsrl.policy.parameters(), agent.policy.parameters()):
        assert torch.equal(expected, actual)
    obs = torch.randn(4, 4)
    actions = torch.randn(4, 2).clamp(-1, 1)
    base = agent._critic_forward(obs, actions, target=False)
    _activate(agent)
    assert torch.equal(agent.corrected_q_values(obs, actions), base)

    _fill(agent)
    agent.train(gradient_steps=1)
    saved = agent.save(tmp_path / "lancet.pt")
    checkpoint = torch.load(saved, map_location="cpu", weights_only=False)
    assert "residual_optimizer" in checkpoint["state"]["optimizers"]
    assert checkpoint["state"]["extra"]["lancet"]["schema_version"] == 1
    restored = _agent(checkpoint_dir=str(tmp_path))
    restored.load(saved, load_replay_buffer=False)
    for expected, actual in zip(
        agent.residual_network.parameters(), restored.residual_network.parameters()
    ):
        assert torch.equal(expected, actual)
    assert restored._residual_update_count == agent._residual_update_count
    assert restored._u_ema == agent._u_ema
    assert torch.equal(
        restored._local_action_generator.get_state(),
        agent._local_action_generator.get_state(),
    )


def test_registry_exposes_current_and_legacy_names():
    from rl_garden.training.off2on._registry import registry

    registry.discover()
    assert {"lancet", "lancet_v1"} <= set(registry.entries())
