"""Tests for jackdaw integration and custom wrappers."""

import numpy as np
import pytest
from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter

from balatro_rl.env.wrappers import (
    CurriculumStage,
    CurriculumWrapper,
    ExpertRewardWrapper,
    ObservationAugmentWrapper,
    RewardConfig,
)


@pytest.fixture
def base_env():
    e = BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter, max_steps=500, reward_shaping=True
    )
    yield e
    e.close()


# --- Jackdaw smoke tests ---


def test_reset_returns_valid_observation(base_env):
    obs, info = base_env.reset(seed=42)
    assert isinstance(obs, dict)
    expected_keys = {
        "global", "hand_card", "joker", "consumable",
        "shop_item", "pack_card", "entity_counts",
    }
    assert set(obs.keys()) == expected_keys
    assert obs["global"].shape == (235,)
    assert obs["global"].dtype == np.float32
    assert "action_mask" in info


def test_action_masks_matches_info(base_env):
    _, info = base_env.reset(seed=42)
    np.testing.assert_array_equal(base_env.action_masks(), info["action_mask"])


def test_full_random_episode(base_env):
    base_env.reset(seed=42)
    rng = np.random.default_rng(42)
    steps = 0
    while True:
        mask = base_env.action_masks()
        legal = np.where(mask)[0]
        assert len(legal) > 0
        obs, _, terminated, truncated, info = base_env.step(int(rng.choice(legal)))
        steps += 1
        if terminated or truncated:
            break
    assert steps > 0
    assert "balatro/ante_reached" in info


# --- ExpertRewardWrapper ---


def test_expert_reward_wrapper_modifies_reward(base_env):
    env = ExpertRewardWrapper(base_env)
    env.reset(seed=99)

    mask = env.action_masks()
    legal = np.where(mask)[0]
    _, reward_wrapped, _, _, _ = env.step(int(legal[0]))

    assert isinstance(reward_wrapped, float)


def test_expert_reward_config():
    cfg = RewardConfig(ante_progress_scale=0.5)
    assert cfg.ante_progress_scale == 0.5


# --- ObservationAugmentWrapper ---


def test_obs_augment_extends_global(base_env):
    env = ObservationAugmentWrapper(base_env)
    obs, _ = env.reset(seed=42)
    assert obs["global"].shape == (235 + ObservationAugmentWrapper.N_AUGMENTED,)
    assert obs["global"].dtype == np.float32


def test_obs_augment_step(base_env):
    env = ObservationAugmentWrapper(base_env)
    env.reset(seed=42)
    mask = env.action_masks()
    legal = np.where(mask)[0]
    obs, _, _, _, _ = env.step(int(legal[0]))
    assert obs["global"].shape == (235 + ObservationAugmentWrapper.N_AUGMENTED,)


# --- CurriculumWrapper ---


def test_curriculum_starts_at_stage_0(base_env):
    env = CurriculumWrapper(base_env)
    _, info = env.reset(seed=42)
    assert info["curriculum/stage"] == 0
    assert "curriculum/label" in info


def test_curriculum_full_episode(base_env):
    stages = [
        CurriculumStage(ante_cap=1, stake=1, success_threshold=0.5, label="test"),
    ]
    env = CurriculumWrapper(base_env, stages=stages)
    env.reset(seed=42)
    rng = np.random.default_rng(42)

    while True:
        mask = env.action_masks()
        legal = np.where(mask)[0]
        _, _, terminated, truncated, info = env.step(int(rng.choice(legal)))
        if terminated or truncated:
            break

    assert "curriculum/stage" in info


# --- Wrapper composition ---


def test_wrapper_stacking(base_env):
    env = ObservationAugmentWrapper(
        CurriculumWrapper(
            ExpertRewardWrapper(base_env)
        )
    )
    obs, info = env.reset(seed=42)
    assert obs["global"].shape == (235 + ObservationAugmentWrapper.N_AUGMENTED,)
    assert "curriculum/stage" in info

    mask = env.action_masks()
    legal = np.where(mask)[0]
    obs, reward, _, _, _ = env.step(int(legal[0]))
    assert isinstance(reward, float)
    assert obs["global"].shape == (235 + ObservationAugmentWrapper.N_AUGMENTED,)
