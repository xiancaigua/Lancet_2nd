"""Lancet: WSRL with a shared state-action critic residual correction.

The base WSRL/Cal-QL critic and its TD target are deliberately unchanged.
``R_phi(s, a)`` is trained after each base critic step against the detached
base TD residual.  It is shared by every REDQ member, so it does not change
the ensemble's epistemic capacity.  The actor sees ``min_i Q_i + R_phi``.

The planned action-dependent U-variation supervision has no settled formula
in this repository.  Its explicit hook is therefore disabled by default and
raises when enabled rather than quietly inventing a surrogate objective.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn

from rl_garden.algorithms.wsrl import WSRL
from rl_garden.common.optim import make_optimizer
from rl_garden.networks.mlp import create_mlp


class ResidualQNetworkV1(nn.Module):
    """Shared scalar correction ``R_phi(state, action)`` for a Box state space."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.network = create_mlp(
            self.state_dim + self.action_dim,
            1,
            list(hidden_dims),
            use_layer_norm=True,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = state.reshape(state.shape[0], self.state_dim)
        action = action.reshape(action.shape[0], self.action_dim)
        return self.network(torch.cat([state, action], dim=-1))


class LancetV1(WSRL):
    """WSRL preserving base TD updates with a learned shared Q residual.

    TD target: unchanged WSRL target critic.  Base critic: unchanged WSRL
    TD/CQL update.  Residual target: ``stopgrad(target_q - mean_i Q_i)``.
    Actor: maximizes ``min_i Q_i(s,a) + R_phi(s,a)``.
    """

    _compatible_checkpoint_algorithms = ("LancetV1", "Lancet", "WSRL", "CalQL", "CQL")

    def __init__(
        self,
        *args: Any,
        use_lancet: bool = True,
        residual_hidden_dim: int = 256,
        residual_hidden_layers: int = 2,
        residual_lr: float = 3e-4,
        lambda_td_residual: float = 1.0,
        lambda_u_variation: float = 0.0,
        lambda_residual_reg: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        if residual_hidden_dim < 1 or residual_hidden_layers < 1:
            raise ValueError(
                "residual_hidden_dim and residual_hidden_layers must be >= 1."
            )
        if residual_lr <= 0:
            raise ValueError("residual_lr must be > 0.")
        if lambda_td_residual < 0 or lambda_u_variation < 0 or lambda_residual_reg < 0:
            raise ValueError("Lancet loss weights must be non-negative.")
        if lambda_u_variation != 0:
            raise ValueError(
                "Lancet U-variation supervision is an explicit TODO: its formula is "
                "not yet specified, so lambda_u_variation must remain 0."
            )
        self.use_lancet = use_lancet
        self.residual_hidden_dim = residual_hidden_dim
        self.residual_hidden_layers = residual_hidden_layers
        self.residual_lr = residual_lr
        self.lambda_td_residual = lambda_td_residual
        self.lambda_u_variation = lambda_u_variation
        self.lambda_residual_reg = lambda_residual_reg
        self.residual_network: ResidualQNetworkV1 | None = None
        self.residual_optimizer = None
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if not self.use_lancet:
            return
        obs_space = self.env.single_observation_space
        action_space = self.env.single_action_space
        if not isinstance(obs_space, spaces.Box) or not isinstance(
            action_space, spaces.Box
        ):
            raise TypeError(
                "Lancet V1 supports flat Box state/action observations only."
            )
        state_dim = int(np.prod(obs_space.shape))
        action_dim = int(np.prod(action_space.shape))
        self.residual_network = ResidualQNetworkV1(
            state_dim,
            action_dim,
            [self.residual_hidden_dim] * self.residual_hidden_layers,
        ).to(self.device)
        self.residual_optimizer = make_optimizer(
            self.residual_network.parameters(),
            lr=self.residual_lr,
            weight_decay=0.0,
            use_adamw=False,
        )

    def _optimizer_names(self) -> tuple[str, ...]:
        return (*super()._optimizer_names(), "residual_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "use_lancet": self.use_lancet,
            "residual_hidden_dim": self.residual_hidden_dim,
            "residual_hidden_layers": self.residual_hidden_layers,
            "residual_lr": self.residual_lr,
            "lambda_td_residual": self.lambda_td_residual,
            "lambda_u_variation": self.lambda_u_variation,
            "lambda_residual_reg": self.lambda_residual_reg,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        state = super()._extra_checkpoint_state()
        state["lancet"] = {
            "enabled": self.use_lancet,
            "residual_network": (
                self.residual_network.state_dict()
                if self.residual_network is not None
                else None
            ),
        }
        return state

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        lancet_state = state.get("lancet", {})
        network_state = lancet_state.get("residual_network")
        if network_state is not None and self.residual_network is not None:
            self.residual_network.load_state_dict(network_state)

    def residual(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return ``R_phi(s,a)`` or exact zeros when Lancet is disabled."""
        if self.residual_network is None:
            return torch.zeros(
                (actions.shape[0], 1), device=actions.device, dtype=actions.dtype
            )
        if not isinstance(obs, torch.Tensor):
            raise TypeError(
                "Lancet V1 residual expects flat tensor state observations."
            )
        return self.residual_network(obs, actions)

    def corrected_q_values(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Return all base ensemble Qs plus the shared residual correction."""
        return self._critic_forward(obs, actions, target=False) + self.residual(
            obs, actions
        ).unsqueeze(0)

    def _u_variation_loss(self, data: Any) -> torch.Tensor:
        del data
        # TODO(lancet): implement only after the action-dependent U-variation
        # target is mathematically specified and reviewed.
        if self.lambda_u_variation != 0:
            raise RuntimeError(
                "Lancet U-variation loss has no approved definition yet."
            )
        return torch.zeros((), device=self.device)

    def _post_critic_update(
        self, data: Any, critic_info: dict[str, torch.Tensor]
    ) -> None:
        super()._post_critic_update(data, critic_info)
        if self.residual_network is None or self.residual_optimizer is None:
            return
        with torch.no_grad():
            base_q = self._critic_forward(data.obs, data.actions, target=False).mean(
                dim=0
            )
            residual_target = self._target_q(data) - base_q
            td_error = residual_target.reshape(-1)
        residual_pred = self.residual(data.obs, data.actions)
        td_residual_loss = F.mse_loss(residual_pred, residual_target)
        reg_loss = residual_pred.square().mean()
        u_variation_loss = self._u_variation_loss(data)
        residual_loss = (
            self.lambda_td_residual * td_residual_loss
            + self.lambda_u_variation * u_variation_loss
            + self.lambda_residual_reg * reg_loss
        )
        self.residual_optimizer.zero_grad()
        residual_loss.backward()
        self._clip_grad_norm(self.residual_network.parameters())
        self.residual_optimizer.step()
        base_scale = base_q.detach().abs().mean().clamp_min(1e-8)
        critic_info.update(
            lancet_residual_mean=residual_pred.detach().mean(),
            lancet_residual_abs_mean=residual_pred.detach().abs().mean(),
            lancet_residual_std=residual_pred.detach().std(unbiased=False),
            lancet_residual_loss=residual_loss.detach(),
            lancet_td_residual_loss=td_residual_loss.detach(),
            lancet_u_variation_loss=u_variation_loss.detach(),
            lancet_reg_loss=reg_loss.detach(),
            lancet_residual_q_ratio=(residual_pred.detach().abs().mean() / base_scale),
            critic_td_error_mean=td_error.mean(),
            critic_td_error_abs_mean=td_error.abs().mean(),
        )

    def _actor_loss(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self._current_alpha().detach()
        action, log_prob, actor_features = self._actor_action_log_prob(
            obs, stop_gradient=self._actor_stop_gradient()
        )
        critic_features = self.policy.critic_features_for(
            obs, actor_features, stop_gradient=True
        )
        base_min_q = self.policy.min_q_value(
            critic_features, action, subsample_size=None, target=False
        )
        corrected_min_q = base_min_q + self.residual(obs, action)
        return (alpha * log_prob - corrected_min_q).mean(), log_prob.detach()

    def _log_update_metrics(self, metrics: dict[str, float], step: int) -> None:
        super()._log_update_metrics(metrics, step)
        if self.logger is None:
            return
        tags = {
            "lancet_residual_mean": "lancet/residual_mean",
            "lancet_residual_abs_mean": "lancet/residual_abs_mean",
            "lancet_residual_std": "lancet/residual_std",
            "lancet_residual_loss": "lancet/residual_loss",
            "lancet_td_residual_loss": "lancet/td_residual_loss",
            "lancet_u_variation_loss": "lancet/u_variation_loss",
            "lancet_reg_loss": "lancet/reg_loss",
            "lancet_residual_q_ratio": "lancet/residual_q_ratio",
            "critic_td_error_mean": "critic/td_error_mean",
            "critic_td_error_abs_mean": "critic/td_error_abs_mean",
        }
        for key, tag in tags.items():
            if key in metrics:
                self.logger.add_scalar(tag, metrics[key], step)
