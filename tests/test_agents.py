"""Tests for agent dispatch and phase-specific policies."""

import numpy as np
import pytest
from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter

from balatro_rl.agents.blind import BlindPolicy
from balatro_rl.agents.dispatch import Phase, PhaseDispatchAgent, _detect_phase
from balatro_rl.agents.hand import HandPolicy
from balatro_rl.agents.shop import ShopPolicy


@pytest.fixture
def env():
    e = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=500)
    yield e
    e.close()


def test_detect_phase_from_obs(env):
    obs, _ = env.reset(seed=42)
    phase = _detect_phase(obs)
    assert isinstance(phase, Phase)
    assert phase in Phase


def test_dispatch_agent_selects_legal_action(env):
    agent = PhaseDispatchAgent()
    obs, _ = env.reset(seed=42)
    mask = env.action_masks()
    action = agent.select_action(obs, mask)
    assert mask[action], "Dispatch agent selected an illegal action"


def test_dispatch_agent_full_episode(env):
    agent = PhaseDispatchAgent()
    obs, _ = env.reset(seed=42)
    steps = 0

    while True:
        mask = env.action_masks()
        action = agent.select_action(obs, mask)
        assert mask[action]
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    assert steps > 0


def test_hand_policy_random(env):
    policy = HandPolicy()
    obs, _ = env.reset(seed=42)
    mask = env.action_masks()
    action = policy.select_action(obs, mask)
    assert mask[action]


def test_shop_policy_random():
    policy = ShopPolicy()
    mask = np.zeros(500, dtype=bool)
    mask[5] = True
    mask[10] = True
    obs = {"global": np.zeros(235, dtype=np.float32)}
    action = policy.select_action(obs, mask)
    assert action in (5, 10)


def test_blind_policy_random():
    policy = BlindPolicy()
    mask = np.zeros(500, dtype=bool)
    mask[0] = True
    obs = {"global": np.zeros(235, dtype=np.float32)}
    action = policy.select_action(obs, mask)
    assert action == 0


def test_get_policy_for_phase():
    agent = PhaseDispatchAgent()
    assert agent.get_policy_for_phase(Phase.SELECTING_HAND) is agent.hand_policy
    assert agent.get_policy_for_phase(Phase.SHOP) is agent.shop_policy
    assert agent.get_policy_for_phase(Phase.BLIND_SELECT) is agent.blind_policy
    assert agent.get_policy_for_phase(Phase.ROUND_EVAL) is None
