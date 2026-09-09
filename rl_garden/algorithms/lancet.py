"""Lancet: action-relative, critic-aligned correction for WSRL handoff.

Lancet leaves the WSRL Bellman target, base critic update, REDQ target
subsampling, and target critic unchanged.  During a bounded post-warmup
window, a separate residual ensemble fits the per-critic Bellman error that
remains *after* the normal base critic optimizer step.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn

from rl_garden.algorithms.wsrl import WSRL
from rl_garden.common.optim import make_optimizer
from rl_garden.networks.mlp import create_mlp

LancetVariant = Literal["raw", "centered", "lancet"]


class ResidualEnsemble(nn.Module):
    """Shared state-action backbone with one scalar head per base critic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_critics: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        if n_critics < 1:
            raise ValueError("n_critics must be >= 1.")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.n_critics = int(n_critics)
        self.network = create_mlp(
            self.state_dim + self.action_dim,
            self.n_critics,
            list(hidden_dims),
            use_layer_norm=True,
        )
        output = next(
            module
            for module in reversed(list(self.network.modules()))
            if isinstance(module, nn.Linear)
        )
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = state.reshape(state.shape[0], self.state_dim)
        action = action.reshape(action.shape[0], self.action_dim)
        values = self.network(torch.cat([state, action], dim=-1))
        return values.transpose(0, 1).unsqueeze(-1)  # [N, B, 1]


class Lancet(WSRL):
    """WSRL plus a bounded action-relative residual correction branch."""

    _compatible_checkpoint_algorithms = ("Lancet", "WSRL", "CalQL", "CQL")
    _CHECKPOINT_SCHEMA_VERSION = 1

    def __init__(
        self,
        *args: Any,
        lancet_variant: LancetVariant = "lancet",
        residual_hidden_dim: int = 128,
        residual_hidden_layers: int = 2,
        residual_lr: float = 1e-3,
        residual_fit_coef: float = 1.0,
        residual_small_coef: float = 1e-4,
        local_action_count: int = 8,
        local_action_noise_scale: float = 0.1,
        uncertainty_beta: float = 1.0,
        uncertainty_ema_decay: float = 0.99,
        uncertainty_weight_max: float = 3.0,
        uncertainty_eps: float = 1e-8,
        handoff_window_steps: int = 50_000,
        residual_init_seed_offset: int = 1_000_003,
        local_action_seed_offset: int = 2_000_003,
        **kwargs: Any,
    ) -> None:
        if lancet_variant not in {"raw", "centered", "lancet"}:
            raise ValueError(f"Unknown lancet_variant: {lancet_variant!r}.")
        if residual_hidden_dim < 1 or residual_hidden_layers < 1:
            raise ValueError("Residual hidden dimensions/layers must be >= 1.")
        if residual_lr <= 0 or residual_fit_coef < 0 or residual_small_coef < 0:
            raise ValueError(
                "Residual LR must be positive and loss coefficients non-negative."
            )
        if local_action_count < 2:
            raise ValueError("local_action_count must be >= 2.")
        if local_action_noise_scale < 0:
            raise ValueError("local_action_noise_scale must be non-negative.")
        if uncertainty_beta < 0 or not 0 <= uncertainty_ema_decay < 1:
            raise ValueError("Invalid uncertainty beta/EMA decay.")
        if uncertainty_weight_max < 1 or uncertainty_eps <= 0:
            raise ValueError("Uncertainty cap must be >= 1 and epsilon positive.")
        if handoff_window_steps < 1:
            raise ValueError("handoff_window_steps must be >= 1.")
        if kwargs.get("use_compile", False):
            raise ValueError("Lancet currently requires use_compile=False.")

        self.lancet_variant = lancet_variant
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.residual_hidden_layers = int(residual_hidden_layers)
        self.residual_lr = float(residual_lr)
        self.residual_fit_coef = float(residual_fit_coef)
        self.residual_small_coef = float(residual_small_coef)
        self.local_action_count = int(local_action_count)
        self.local_action_noise_scale = float(local_action_noise_scale)
        self.uncertainty_beta = float(uncertainty_beta)
        self.uncertainty_ema_decay = float(uncertainty_ema_decay)
        self.uncertainty_weight_max = float(uncertainty_weight_max)
        self.uncertainty_eps = float(uncertainty_eps)
        self.handoff_window_steps = int(handoff_window_steps)
        self.residual_init_seed_offset = int(residual_init_seed_offset)
        self.local_action_seed_offset = int(local_action_seed_offset)

        self.residual_network: ResidualEnsemble | None = None
        self.residual_optimizer = None
        self._captured_target_y: torch.Tensor | None = None
        self._lancet_adaptation_start_step: int | None = None
        self._residual_update_count = 0
        self._u_ema = 0.0
        self._u_ema_initialized = False
        self._local_action_generator: torch.Generator | None = None
        super().__init__(*args, **kwargs)
        if not self.use_td_loss:
            raise ValueError(
                "Lancet requires the unchanged WSRL TD loss to be enabled."
            )

    def _setup_model(self) -> None:
        super()._setup_model()
        obs_space = self.env.single_observation_space
        action_space = self.env.single_action_space
        if not isinstance(obs_space, spaces.Box) or not isinstance(
            action_space, spaces.Box
        ):
            raise TypeError(
                "Lancet currently supports flat Box observations/actions only."
            )
        state_dim = int(np.prod(obs_space.shape))
        action_dim = int(np.prod(action_space.shape))
        devices: list[int] = []
        if self.device.type == "cuda":
            devices = [
                torch.cuda.current_device()
                if self.device.index is None
                else self.device.index
            ]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed + self.residual_init_seed_offset)
            self.residual_network = ResidualEnsemble(
                state_dim,
                action_dim,
                self.n_critics,
                [self.residual_hidden_dim] * self.residual_hidden_layers,
            ).to(self.device)
        self.residual_optimizer = make_optimizer(
            self.residual_network.parameters(),
            lr=self.residual_lr,
            weight_decay=0.0,
            use_adamw=False,
        )
        self._local_action_generator = torch.Generator(device=self.device)
        self._local_action_generator.manual_seed(
            self.seed + self.local_action_seed_offset
        )
        self._action_low = torch.as_tensor(
            action_space.low, device=self.device, dtype=torch.float32
        )
        self._action_high = torch.as_tensor(
            action_space.high, device=self.device, dtype=torch.float32
        )
        self._action_noise_std = (
            self.local_action_noise_scale * (self._action_high - self._action_low) / 2
        )

    @property
    def adaptation_step(self) -> int | None:
        if self._lancet_adaptation_start_step is None:
            return None
        return max(0, self._global_step - self._lancet_adaptation_start_step)

    def correction_scale(self) -> float:
        step = self.adaptation_step
        return 1.0 if step is not None and step < self.handoff_window_steps else 0.0

    def residual(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        assert self.residual_network is not None
        return self.residual_network(obs, actions)

    def _flat_local_inputs(
        self, obs: torch.Tensor, local_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count = local_actions.shape[:2]
        obs_flat = (
            obs.reshape(batch, -1)
            .unsqueeze(1)
            .expand(batch, count, -1)
            .reshape(batch * count, -1)
        )
        return obs_flat, local_actions.reshape(batch * count, -1)

    def residual_local(
        self, obs: torch.Tensor, local_actions: torch.Tensor
    ) -> torch.Tensor:
        batch, count = local_actions.shape[:2]
        obs_flat, action_flat = self._flat_local_inputs(obs, local_actions)
        return self.residual(obs_flat, action_flat).reshape(
            self.n_critics, batch, count, 1
        )

    def _local_actions(
        self, obs: torch.Tensor, replay_actions: torch.Tensor
    ) -> torch.Tensor:
        assert self._local_action_generator is not None
        with torch.no_grad():
            actor_action = self.policy.predict(obs, deterministic=True).detach()
            actions = [replay_actions.detach(), actor_action]
            if self.local_action_count > 2:
                shape = (
                    replay_actions.shape[0],
                    self.local_action_count - 2,
                    replay_actions.shape[-1],
                )
                noise = (
                    torch.randn(
                        shape,
                        device=self.device,
                        dtype=replay_actions.dtype,
                        generator=self._local_action_generator,
                    )
                    * self._action_noise_std
                )
                perturbed = actor_action.unsqueeze(1) + noise
                perturbed = torch.maximum(
                    torch.minimum(perturbed, self._action_high), self._action_low
                )
                actions.extend(perturbed.unbind(dim=1))
            return torch.stack(actions, dim=1)

    def delta_local(self, residual_local: torch.Tensor) -> torch.Tensor:
        if self.lancet_variant == "raw":
            return residual_local
        return residual_local - residual_local.mean(dim=2, keepdim=True)

    @staticmethod
    def uncertainty_from_q_local(q_local: torch.Tensor) -> torch.Tensor:
        """Efficient mean pairwise action-dependent REDQ disagreement [B,1]."""
        n_critics = q_local.shape[0]
        if n_critics < 2:
            return torch.zeros(
                q_local.shape[1], 1, device=q_local.device, dtype=q_local.dtype
            )
        centered = q_local - q_local.mean(dim=2, keepdim=True)
        critic_variance = (
            centered.squeeze(-1).var(dim=0, unbiased=False).mean(dim=1, keepdim=True)
        )
        return (2.0 * n_critics / (n_critics - 1) * critic_variance).detach()

    def uncertainty_weights(
        self, uncertainty: torch.Tensor, *, update_ema: bool
    ) -> torch.Tensor:
        batch_mean = float(uncertainty.detach().mean().item())
        if update_ema:
            if not self._u_ema_initialized:
                self._u_ema = batch_mean
                self._u_ema_initialized = True
            else:
                self._u_ema = (
                    self.uncertainty_ema_decay * self._u_ema
                    + (1.0 - self.uncertainty_ema_decay) * batch_mean
                )
        if self.lancet_variant != "lancet":
            return torch.ones_like(uncertainty)
        denominator = self._u_ema + self.uncertainty_eps
        return torch.clamp(
            1.0 + self.uncertainty_beta * uncertainty.detach() / denominator,
            min=1.0,
            max=self.uncertainty_weight_max,
        )

    def corrected_q_values(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        local_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_q = self._critic_forward(obs, actions, target=False)
        scale = self.correction_scale()
        if scale == 0.0:
            return base_q
        if self.lancet_variant == "raw":
            delta = self.residual(obs, actions)
        else:
            if local_actions is None:
                local_actions = self._local_actions(obs, actions)
            baseline = self.residual_local(obs, local_actions).mean(dim=2)
            delta = self.residual(obs, actions) - baseline
        return base_q + scale * delta

    def _critic_loss(self, data) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._captured_target_y = None
        return super()._critic_loss(data)

    def _td_loss(
        self, data, q_pred: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_y = self._target_q(data)
        self._captured_target_y = target_y.detach()
        expanded = target_y.unsqueeze(0).expand_as(q_pred)
        td_loss = F.mse_loss(q_pred, expanded)
        return td_loss, {
            "td_loss": td_loss.detach(),
            "target_q": target_y.mean().detach(),
        }

    def _post_critic_update(self, data, critic_info: dict[str, torch.Tensor]) -> None:
        super()._post_critic_update(data, critic_info)
        target_y = self._captured_target_y
        self._captured_target_y = None
        if (
            self._online_start_step is None
            or self._active_initial_training_phase() is not None
        ):
            return
        if self._lancet_adaptation_start_step is None:
            self._lancet_adaptation_start_step = self._global_step
        if self.correction_scale() == 0.0:
            return
        if target_y is None:
            raise RuntimeError(
                "Lancet residual update did not receive the base critic target."
            )
        assert self.residual_network is not None and self.residual_optimizer is not None

        with torch.no_grad():
            q_updated = self._critic_forward(data.obs, data.actions, target=False)
            td_residual = target_y.unsqueeze(0).expand_as(q_updated) - q_updated
            local_actions = self._local_actions(data.obs, data.actions)
            obs_flat, action_flat = self._flat_local_inputs(data.obs, local_actions)
            q_local = self._critic_forward(obs_flat, action_flat, target=False).reshape(
                self.n_critics, data.actions.shape[0], self.local_action_count, 1
            )
            uncertainty = self.uncertainty_from_q_local(q_local)
            weights = self.uncertainty_weights(uncertainty, update_ema=True)

        residual_local = self.residual_local(data.obs, local_actions)
        delta_local = self.delta_local(residual_local)
        delta_observed = delta_local[:, :, 0, :]
        fit_loss = (
            weights.unsqueeze(0) * (delta_observed - td_residual).square()
        ).mean()
        small_loss = delta_local.square().mean()
        residual_loss = (
            self.residual_fit_coef * fit_loss + self.residual_small_coef * small_loss
        )

        self.residual_optimizer.zero_grad()
        residual_loss.backward()
        residual_params = list(self.residual_network.parameters())
        self._sync_ddp_grads(residual_params)
        self._clip_grad_norm(residual_params)
        self.residual_optimizer.step()
        self._residual_update_count += 1

        with torch.no_grad():
            base_abs = q_updated.abs().mean().clamp_min(self.uncertainty_eps)
            delta_abs = delta_observed.abs().mean()
            point_ratio = (
                delta_observed.abs() / (q_updated.abs() + self.uncertainty_eps)
            ).mean()
            corrected = q_updated + delta_observed
            critic_info.update(
                lancet_lambda=torch.tensor(self.correction_scale(), device=self.device),
                lancet_adaptation_step=torch.tensor(
                    float(self.adaptation_step or 0), device=self.device
                ),
                lancet_residual_updates=torch.tensor(
                    float(self._residual_update_count), device=self.device
                ),
                lancet_residual_mean=residual_local.mean().detach(),
                lancet_residual_std=residual_local.std(unbiased=False).detach(),
                lancet_residual_abs_mean=residual_local.abs().mean().detach(),
                lancet_delta_mean=delta_local.mean().detach(),
                lancet_delta_std=delta_local.std(unbiased=False).detach(),
                lancet_delta_abs_mean=delta_abs.detach(),
                lancet_residual_loss=residual_loss.detach(),
                lancet_fit_loss=fit_loss.detach(),
                lancet_small_loss=small_loss.detach(),
                lancet_td_residual_mean=td_residual.mean().detach(),
                lancet_td_residual_std=td_residual.std(unbiased=False).detach(),
                lancet_td_residual_rmse=td_residual.square().mean().sqrt().detach(),
                lancet_u_mean=uncertainty.mean().detach(),
                lancet_u_std=uncertainty.std(unbiased=False).detach(),
                lancet_u_ema=torch.tensor(self._u_ema, device=self.device),
                lancet_weight_mean=weights.mean().detach(),
                lancet_weight_max=weights.max().detach(),
                lancet_base_q_mean=q_updated.mean().detach(),
                lancet_base_q_std=q_updated.std(unbiased=False).detach(),
                lancet_corrected_q_mean=corrected.mean().detach(),
                lancet_corrected_q_std=corrected.std(unbiased=False).detach(),
                lancet_delta_q_ratio=(delta_abs / base_abs).detach(),
                lancet_pointwise_delta_q_ratio=point_ratio.detach(),
            )

    @contextmanager
    def _frozen_q_parameters(self) -> Iterator[None]:
        parameters = list(self.policy.critic.parameters())
        assert self.residual_network is not None
        parameters.extend(self.residual_network.parameters())
        previous = [parameter.requires_grad for parameter in parameters]
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            yield
        finally:
            for parameter, required in zip(parameters, previous):
                parameter.requires_grad_(required)

    def _actor_loss_from_batch(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        if self.correction_scale() == 0.0:
            return super()._actor_loss_from_batch(data)
        assert self.residual_network is not None
        alpha = self._current_alpha().detach()
        with self._frozen_q_parameters():
            action, log_prob, actor_features = self.policy.actor_action_log_prob(
                data.obs, stop_gradient=self._actor_stop_gradient()
            )
            critic_features = self.policy.critic_features_for(
                data.obs, actor_features, stop_gradient=True
            )
            q_all = self.policy.q_values_all(critic_features, action, target=False)
            local_actions = self._local_actions(data.obs, data.actions)
            with torch.no_grad():
                baseline = self.residual_local(data.obs, local_actions).mean(dim=2)
            residual_actor = self.residual(data.obs, action)
            delta_actor = (
                residual_actor
                if self.lancet_variant == "raw"
                else residual_actor - baseline
            )
            q_actor = (q_all + delta_actor).min(dim=0).values
            loss = (alpha * log_prob - q_actor).mean()
        return loss, log_prob.detach()

    def _optimizer_names(self) -> tuple[str, ...]:
        return (*super()._optimizer_names(), "residual_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "lancet_variant": self.lancet_variant,
            "residual_hidden_dim": self.residual_hidden_dim,
            "residual_hidden_layers": self.residual_hidden_layers,
            "residual_lr": self.residual_lr,
            "residual_fit_coef": self.residual_fit_coef,
            "residual_small_coef": self.residual_small_coef,
            "local_action_count": self.local_action_count,
            "local_action_noise_scale": self.local_action_noise_scale,
            "uncertainty_beta": self.uncertainty_beta,
            "uncertainty_ema_decay": self.uncertainty_ema_decay,
            "uncertainty_weight_max": self.uncertainty_weight_max,
            "uncertainty_eps": self.uncertainty_eps,
            "handoff_window_steps": self.handoff_window_steps,
        }

    def _training_state_dict(self) -> dict[str, Any]:
        state = super()._training_state_dict()
        state.update(
            lancet_adaptation_start_step=self._lancet_adaptation_start_step,
            lancet_residual_update_count=self._residual_update_count,
            lancet_u_ema=self._u_ema,
            lancet_u_ema_initialized=self._u_ema_initialized,
            lancet_local_action_rng=(
                self._local_action_generator.get_state()
                if self._local_action_generator is not None
                else None
            ),
        )
        return state

    def _load_training_state_dict(self, state: dict[str, Any]) -> None:
        super()._load_training_state_dict(state)
        start = state.get("lancet_adaptation_start_step")
        self._lancet_adaptation_start_step = None if start is None else int(start)
        self._residual_update_count = int(state.get("lancet_residual_update_count", 0))
        self._u_ema = float(state.get("lancet_u_ema", 0.0))
        self._u_ema_initialized = bool(state.get("lancet_u_ema_initialized", False))
        rng_state = state.get("lancet_local_action_rng")
        if rng_state is not None and self._local_action_generator is not None:
            self._local_action_generator.set_state(rng_state.cpu())

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        state = super()._extra_checkpoint_state()
        assert self.residual_network is not None
        state["lancet"] = {
            "schema_version": self._CHECKPOINT_SCHEMA_VERSION,
            "variant": self.lancet_variant,
            "residual_network": self.residual_network.state_dict(),
        }
        return state

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        lancet_state = state.get("lancet")
        if lancet_state is None:
            return
        if lancet_state.get("schema_version") != self._CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "This checkpoint contains Lancet V1 residual state; use lancet_v1."
            )
        if lancet_state.get("variant") != self.lancet_variant:
            raise ValueError(
                f"Checkpoint variant {lancet_state.get('variant')!r} does not match "
                f"constructed variant {self.lancet_variant!r}."
            )
        assert self.residual_network is not None
        self.residual_network.load_state_dict(lancet_state["residual_network"])

    def load_state_dict(
        self, sd: dict[str, Any], strict: bool = True, load_optimizers: bool = True
    ) -> None:
        lancet_state = sd.get("extra", {}).get("lancet")
        if lancet_state is not None and "schema_version" not in lancet_state:
            raise ValueError(
                "Lancet V1 checkpoint detected; load it with algorithm lancet_v1."
            )
        super().load_state_dict(sd, strict=strict, load_optimizers=load_optimizers)

    def _log_update_metrics(self, metrics: dict[str, float], step: int) -> None:
        super()._log_update_metrics(metrics, step)
        if self.logger is None:
            return
        # Lifecycle coordinates remain observable after the bounded residual
        # window closes, when no residual-loss metrics are produced.
        self.logger.add_scalar("lancet/lambda", self.correction_scale(), step)
        adaptation_step = self.adaptation_step
        self.logger.add_scalar(
            "lancet/adaptation_step",
            float(adaptation_step if adaptation_step is not None else -1),
            step,
        )
        self.logger.add_scalar(
            "lancet/residual_updates", float(self._residual_update_count), step
        )
        for name, module in (
            ("actor_parameters_finite", self.policy.actor),
            ("critic_parameters_finite", self.policy.critic),
            ("residual_parameters_finite", self.residual_network),
        ):
            finite = all(
                bool(torch.isfinite(parameter).all().item())
                for parameter in module.parameters()
            )
            self.logger.add_scalar(f"lancet/{name}", float(finite), step)
        lifecycle_keys = {
            "lancet_lambda",
            "lancet_adaptation_step",
            "lancet_residual_updates",
        }
        for key, value in metrics.items():
            if key.startswith("lancet_") and key not in lifecycle_keys:
                self.logger.add_scalar(
                    "lancet/" + key.removeprefix("lancet_"), value, step
                )
