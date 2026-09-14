"""Lancet offline-to-online registration (flat state observations)."""

from dataclasses import dataclass
from typing import Literal

from rl_garden.common.cli_args import (
    resolve_critic_encoder_config,
    resolve_obs_groups_config,
)
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.inspection import construct_agent
from rl_garden.training.off2on._args import (
    VisionWSRLTrainingArgs,
    initial_training_phase_from_args,
)
from rl_garden.training.off2on._registry import registry


@dataclass
class LancetOff2OnArgs(VisionWSRLTrainingArgs, EnvBackendArgs):
    """Current Lancet and its capacity-matched residual ablations."""

    hidden_dim: int = 256
    actor_hidden_layers: int = 2
    critic_hidden_layers: int = 4
    target_entropy: float | str = "auto"
    num_eval_episodes: int | None = None
    policy_log_std_multiplier: float | None = None
    policy_log_std_offset: float | None = None
    bootstrap_at_done: Literal["always", "never", "truncated"] = "truncated"
    lancet_variant: Literal["raw", "centered", "lancet"] = "lancet"
    residual_hidden_dim: int = 128
    residual_hidden_layers: int = 2
    residual_lr: float = 1e-3
    residual_fit_coef: float = 1.0
    residual_small_coef: float = 1e-4
    local_action_count: int = 8
    local_action_noise_scale: float = 0.1
    uncertainty_beta: float = 1.0
    uncertainty_ema_decay: float = 0.99
    uncertainty_weight_max: float = 3.0
    uncertainty_eps: float = 1e-8
    handoff_window_steps: int = 50_000
    residual_init_seed_offset: int = 1_000_003
    local_action_seed_offset: int = 2_000_003


def build_lancet(args: LancetOff2OnArgs, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Lancet

    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
    }
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing

    agent = construct_agent(
        Lancet,
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        training_freq=args.training_freq,
        utd=args.utd,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        alpha_lr=args.alpha_lr,
        cql_alpha_lr=args.cql_alpha_lr,
        policy_frequency=args.policy_frequency,
        target_network_frequency=args.target_network_frequency,
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        use_cql_loss=args.use_cql_loss,
        cql_n_actions=args.cql_n_actions,
        cql_alpha=args.cql_alpha,
        cql_autotune_alpha=args.cql_autotune_alpha,
        cql_alpha_lagrange_init=args.cql_alpha_lagrange_init,
        cql_target_action_gap=args.cql_target_action_gap,
        cql_importance_sample=args.cql_importance_sample,
        cql_max_target_backup=args.cql_max_target_backup,
        cql_temp=args.cql_temp,
        cql_clip_diff_min=args.cql_clip_diff_min,
        cql_clip_diff_max=args.cql_clip_diff_max,
        cql_action_sample_method=args.cql_action_sample_method,
        cql_penalty_scale=args.cql_penalty_scale,
        cql_diff_clip_mode=args.cql_diff_clip_mode,
        cql_alpha_param=args.cql_alpha_param,
        backup_entropy=args.backup_entropy,
        use_calql=args.use_calql,
        calql_bound_random_actions=args.calql_bound_random_actions,
        actor_use_layer_norm=args.actor_use_layer_norm,
        critic_use_layer_norm=args.critic_use_layer_norm,
        actor_use_group_norm=args.actor_use_group_norm,
        critic_use_group_norm=args.critic_use_group_norm,
        num_groups=args.num_groups,
        actor_dropout_rate=args.actor_dropout_rate,
        critic_dropout_rate=args.critic_dropout_rate,
        kernel_init=args.kernel_init,
        backbone_type=args.backbone_type,
        std_parameterization=args.std_parameterization,
        net_arch={
            "pi": [args.hidden_dim] * args.actor_hidden_layers,
            "qf": [args.hidden_dim] * args.critic_hidden_layers,
        },
        target_entropy=args.target_entropy,
        policy_log_std_multiplier=args.policy_log_std_multiplier,
        policy_log_std_offset=args.policy_log_std_offset,
        bootstrap_at_done=args.bootstrap_at_done,
        online_cql_alpha=args.online_cql_alpha,
        online_use_cql_loss=args.online_use_cql_loss,
        initial_training_phase=initial_training_phase_from_args(args),
        offline_sampling=args.offline_sampling,
        sparse_reward_mc=args.sparse_reward_mc,
        sparse_negative_reward=args.sparse_negative_reward,
        success_threshold=args.success_threshold,
        use_sarsa_reference=args.use_sarsa_reference,
        sarsa_hidden_dims=args.sarsa_hidden_dims,
        sarsa_lr=args.sarsa_lr,
        lancet_variant=args.lancet_variant,
        residual_hidden_dim=args.residual_hidden_dim,
        residual_hidden_layers=args.residual_hidden_layers,
        residual_lr=args.residual_lr,
        residual_fit_coef=args.residual_fit_coef,
        residual_small_coef=args.residual_small_coef,
        local_action_count=args.local_action_count,
        local_action_noise_scale=args.local_action_noise_scale,
        uncertainty_beta=args.uncertainty_beta,
        uncertainty_ema_decay=args.uncertainty_ema_decay,
        uncertainty_weight_max=args.uncertainty_weight_max,
        uncertainty_eps=args.uncertainty_eps,
        handoff_window_steps=args.handoff_window_steps,
        residual_init_seed_offset=args.residual_init_seed_offset,
        local_action_seed_offset=args.local_action_seed_offset,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        num_eval_episodes=args.num_eval_episodes,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
        **image_kwargs,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_lancet(args: LancetOff2OnArgs) -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_lancet, algorithm="lancet")


def _lancet_algorithm_cls() -> type:
    from rl_garden.algorithms import Lancet

    return Lancet


registry.register(
    "lancet", LancetOff2OnArgs, run_lancet, algorithm_cls=_lancet_algorithm_cls
)
