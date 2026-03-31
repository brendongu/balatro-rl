"""Shared environment factory with configurable wrapper composition.

Both train_rl.py and evaluate.py use this to construct identically-configured
environments, avoiding drift between training and evaluation.
"""

from __future__ import annotations

import gymnasium

from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv

from balatro_rl.config import Config
from balatro_rl.env.wrappers import (
    CurriculumWrapper,
    ExpertRewardWrapper,
    ObservationAugmentWrapper,
    RewardConfig as WrapperRewardConfig,
)


def make_env(
    cfg: Config,
    *,
    seed: int = 0,
    seed_prefix: str = "PPO",
    reward_wrapper: bool = False,
    curriculum: bool = False,
    augment_obs: bool = False,
) -> gymnasium.Env:
    """Create a BalatroGymnasiumEnv with an optional wrapper stack.

    Wrapper composition order (innermost to outermost):
        BalatroGymnasiumEnv -> ExpertRewardWrapper -> CurriculumWrapper -> ObservationAugmentWrapper

    Args:
        cfg: Loaded Config from TOML.
        seed: RNG seed for the environment.
        seed_prefix: Prefix for jackdaw's seed string.
        reward_wrapper: Apply ExpertRewardWrapper with coefficients from cfg.reward.
        curriculum: Apply CurriculumWrapper with stages from cfg.curriculum.
        augment_obs: Apply ObservationAugmentWrapper (+10 derived features).
    """
    env: gymnasium.Env = BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter,
        max_steps=cfg.env.max_steps,
        seed_prefix=f"{seed_prefix}_{seed}",
        reward_shaping=cfg.env.reward_shaping,
    )

    if reward_wrapper:
        env = ExpertRewardWrapper(
            env,
            config=WrapperRewardConfig(
                ante_progress_exp=cfg.reward.ante_progress_exp,
                ante_progress_scale=cfg.reward.ante_progress_scale,
                discard_efficiency_bonus=cfg.reward.discard_efficiency_bonus,
                hand_diversity_penalty=cfg.reward.hand_diversity_penalty,
                diversity_window=cfg.reward.diversity_window,
            ),
        )

    if curriculum:
        env = CurriculumWrapper(
            env,
            window_size=cfg.curriculum.window_size,
            start_stage=cfg.curriculum.start_stage,
        )

    if augment_obs:
        env = ObservationAugmentWrapper(env)

    return env
