"""Lancet V1 offline-to-online registration (state observations in v1)."""

from dataclasses import dataclass
from typing import Literal

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.inspection import construct_agent
from rl_garden.training.off2on._args import (
    VisionWSRLTrainingArgs,
    initial_training_phase_from_args,
)
from rl_garden.training.off2on._registry import registry


@dataclass
class LancetV1Off2OnArgs(VisionWSRLTrainingArgs, EnvBackendArgs):
    """Lancet V1 critic correction; it is intentionally flat-state only."""

    obs_mode: str = "state"
    hidden_dim: int = 256
    actor_hidden_layers: int = 2
    critic_hidden_layers: int = 4
    target_entropy: float | str = "auto"
    num_eval_episodes: int | None = None
    bootstrap_at_done: Literal["always", "never", "truncated"] = "always"
    use_lancet: bool = True
    residual_hidden_dim: int = 256
    residual_hidden_layers: int = 2
    residual_lr: float = 3e-4
    lambda_td_residual: float = 1.0
    # Explicitly disabled until Lancet's U-variation equation is specified.
    lambda_u_variation: float = 0.0
    lambda_residual_reg: float = 1e-4


def build_lancet_v1(args: LancetV1Off2OnArgs, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms.lancet_v1 import LancetV1

    agent = construct_agent(
        LancetV1,
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
        bootstrap_at_done=args.bootstrap_at_done,
        online_cql_alpha=args.online_cql_alpha,
        online_use_cql_loss=args.online_use_cql_loss,
        initial_training_phase=initial_training_phase_from_args(args),
        offline_sampling=args.offline_sampling,
        sparse_reward_mc=args.sparse_reward_mc,
        sparse_negative_reward=args.sparse_negative_reward,
        success_threshold=args.success_threshold,
        use_lancet=args.use_lancet,
        residual_hidden_dim=args.residual_hidden_dim,
        residual_hidden_layers=args.residual_hidden_layers,
        residual_lr=args.residual_lr,
        lambda_td_residual=args.lambda_td_residual,
        lambda_u_variation=args.lambda_u_variation,
        lambda_residual_reg=args.lambda_residual_reg,
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
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_lancet_v1(args: LancetV1Off2OnArgs) -> None:
    from rl_garden.training.off2on._runner import run_off2on

    run_off2on(args, build_agent=build_lancet_v1, algorithm="lancet_v1")


registry.register("lancet_v1", LancetV1Off2OnArgs, run_lancet_v1)
