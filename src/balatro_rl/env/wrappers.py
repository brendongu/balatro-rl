"""Gymnasium wrappers for reward shaping, curriculum learning, and observation augmentation.

All wrappers are composable and operate on top of jackdaw's BalatroGymnasiumEnv.
Typical stacking order:

    base_env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, reward_shaping=True)
    env = ObservationAugmentWrapper(CurriculumWrapper(ExpertRewardWrapper(base_env)))
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Expert Reward Wrapper
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """Tunable reward shaping coefficients."""

    ante_progress_exp: float = 1.5
    ante_progress_scale: float = 0.1
    discard_efficiency_bonus: float = 0.005
    hand_diversity_penalty: float = -0.01
    diversity_window: int = 20


class _ActionMaskMixin:
    """Propagate action_masks() through wrapper chains."""

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


class ExpertRewardWrapper(_ActionMaskMixin, gymnasium.Wrapper):
    """Layer expert-informed reward signals onto jackdaw's base reward.

    Signals (additive on top of base reward from BalatroGymnasiumEnv):

    1. **Ante-progress scaling**: Super-linear reward for reaching higher antes.
       Surviving ante 5+ is disproportionately valuable since most runs die there.
       Reward: ``ante_progress_scale * (new_ante / 8) ^ ante_progress_exp``
       when ante increases.

    2. **Discard efficiency**: Bonus for clearing rounds with discards remaining.
       Expert insight: saving discards = optionality for future hands.
       Reward: ``discard_efficiency_bonus * discards_left`` when a round is won.

    3. **Hand-type diversity**: Small penalty for repeatedly playing the same
       hand type. Encourages learning to use joker triggers rather than always
       spamming the highest-value hand.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        config: RewardConfig | None = None,
    ) -> None:
        super().__init__(env)
        self.cfg = config or RewardConfig()
        self._prev_ante: int = 1
        self._prev_round: int = 0
        self._recent_hand_types: deque[int] = deque(maxlen=self.cfg.diversity_window)

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._prev_ante = 1
        self._prev_round = 0
        self._recent_hand_types.clear()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += self._expert_bonus(obs, info)
        return obs, reward, terminated, truncated, info

    def _expert_bonus(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> float:
        bonus = 0.0
        g = obs["global"]
        ante = int(round(g[10] * 8.0))
        round_num = int(round(g[11] * 30.0))
        discards_left = g[14] * 10.0

        # 1. Ante-progress: reward when ante increases
        if ante > self._prev_ante:
            ante_frac = ante / 8.0
            bonus += self.cfg.ante_progress_scale * (ante_frac ** self.cfg.ante_progress_exp)

        # 2. Discard efficiency: when round advances, reward remaining discards
        if round_num > self._prev_round and discards_left > 0:
            bonus += self.cfg.discard_efficiency_bonus * discards_left

        # 3. Hand-type diversity: penalize repeated same hand type
        #    Hand type indicators are in global[211:223] (12 hand types)
        hand_type_vec = g[211:223]
        if hand_type_vec.any():
            played_type = int(np.argmax(hand_type_vec))
            if self._recent_hand_types:
                repeat_frac = sum(
                    1 for t in self._recent_hand_types if t == played_type
                ) / len(self._recent_hand_types)
                if repeat_frac > 0.5:
                    bonus += self.cfg.hand_diversity_penalty
            self._recent_hand_types.append(played_type)

        self._prev_ante = ante
        self._prev_round = round_num
        return bonus


# ---------------------------------------------------------------------------
# Curriculum Wrapper
# ---------------------------------------------------------------------------


@dataclass
class CurriculumStage:
    """Definition of a single curriculum stage."""

    ante_cap: int
    stake: int = 1
    success_threshold: float = 0.5
    label: str = ""


DEFAULT_STAGES = [
    CurriculumStage(ante_cap=1, stake=1, success_threshold=0.80, label="survive_ante_1"),
    CurriculumStage(ante_cap=3, stake=1, success_threshold=0.60, label="reach_ante_3"),
    CurriculumStage(ante_cap=5, stake=1, success_threshold=0.40, label="reach_ante_5"),
    CurriculumStage(ante_cap=8, stake=1, success_threshold=0.10, label="win_white"),
    CurriculumStage(ante_cap=8, stake=2, success_threshold=0.05, label="win_red"),
]


class CurriculumWrapper(_ActionMaskMixin, gymnasium.Wrapper):
    """Progressive difficulty via ante caps and stake selection.

    Tracks success rate over a rolling window and auto-promotes
    when the agent meets the threshold for the current stage.

    When ante_cap is reached within an episode, the episode terminates
    as a success (truncated=True with won=True in info), even if the
    agent hasn't beaten the full game.

    Args:
        env: Base environment (must be BalatroGymnasiumEnv or wrapped).
        stages: Ordered list of curriculum stages.
        window_size: Rolling window for success rate calculation.
        start_stage: Index of the starting stage.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        stages: list[CurriculumStage] | None = None,
        window_size: int = 100,
        start_stage: int = 0,
    ) -> None:
        super().__init__(env)
        self.stages = stages or DEFAULT_STAGES
        self.window_size = window_size
        self.current_stage_idx = start_stage
        self._results: deque[bool] = deque(maxlen=window_size)
        self._episode_max_ante: int = 1

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    @property
    def success_rate(self) -> float:
        if not self._results:
            return 0.0
        return sum(self._results) / len(self._results)

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._episode_max_ante = 1
        info["curriculum/stage"] = self.current_stage_idx
        info["curriculum/label"] = self.current_stage.label
        info["curriculum/success_rate"] = self.success_rate
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        g = obs["global"]
        ante = int(round(g[10] * 8.0))
        self._episode_max_ante = max(self._episode_max_ante, ante)

        # Truncate if ante cap reached (count as success)
        stage = self.current_stage
        if not terminated and ante >= stage.ante_cap:
            truncated = True
            info["balatro/ante_reached"] = self._episode_max_ante
            info["balatro/rounds_beaten"] = info.get("balatro/rounds_beaten", 0)
            info["balatro/won"] = True

        if terminated or truncated:
            success = self._episode_max_ante >= stage.ante_cap
            self._results.append(success)
            info["curriculum/stage"] = self.current_stage_idx
            info["curriculum/success_rate"] = self.success_rate
            self._maybe_promote()

        return obs, reward, terminated, truncated, info

    def _maybe_promote(self) -> None:
        if self.current_stage_idx >= len(self.stages) - 1:
            return
        if len(self._results) < self.window_size:
            return
        if self.success_rate >= self.current_stage.success_threshold:
            self.current_stage_idx += 1
            self._results.clear()


# ---------------------------------------------------------------------------
# Observation Augmentation Wrapper
# ---------------------------------------------------------------------------


class ObservationAugmentWrapper(_ActionMaskMixin, gymnasium.ObservationWrapper):
    """Add expert-derived features to jackdaw's base observation.

    Appends a fixed-size vector of derived features to the ``global``
    observation key. All features are normalized to [0, 1] or small
    bounded ranges.

    Derived features (appended to global):
        [0]   chips_progress    — chips_scored / chips_needed (clamped to [0,1])
        [1]   hands_urgency     — chips_remaining / (hands_remaining * best_hand_estimate)
        [2]   flush_proximity   — max same-suit count in hand / 5
        [3]   straight_proximity — longest consecutive rank run / 5
        [4:8] suit_density      — fraction of each suit remaining in deck
        [8]   economy_health    — dollars / (ante * 10), clamped
        [9]   hand_size_ratio   — current hand count / hand size limit
    """

    N_AUGMENTED = 10

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)

        orig_global = self.observation_space["global"]
        new_dim = orig_global.shape[0] + self.N_AUGMENTED
        new_spaces = dict(self.observation_space.spaces)
        new_spaces["global"] = spaces.Box(
            -np.inf, np.inf, shape=(new_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        g = obs["global"]
        aug = np.zeros(self.N_AUGMENTED, dtype=np.float32)

        ante = max(g[10] * 8.0, 1.0)
        hands_left = g[13] * 10.0
        discards_left = g[14] * 10.0
        chips_progress = g[20]  # already encoded as min(chips/blind, 10)/10
        dollars = _inv_log_scale(g[12])
        deck_size = _inv_log_scale(g[27])

        # [0] chips_progress (re-scaled from jackdaw's 0-1 encoding)
        aug[0] = min(chips_progress * 10.0, 1.0)

        # [1] hands_urgency: how much score is needed per remaining hand
        chips_remaining = max(1.0 - aug[0], 0.0)
        aug[1] = min(chips_remaining / max(hands_left, 0.5), 1.0)

        # [2] flush_proximity — from jackdaw's draw analysis at global[227]
        aug[2] = g[227] if len(g) > 227 else 0.0

        # [3] straight_proximity — from jackdaw's draw analysis at global[228]
        aug[3] = g[228] if len(g) > 228 else 0.0

        # [4:8] suit density in deck — from discard pile histogram at global[159:211]
        #        52 dims = 4 suits × 13 ranks; sum per suit / 13 gives density
        if len(g) > 210:
            discard_hist = g[159:211].reshape(4, 13)
            discarded_per_suit = discard_hist.sum(axis=1)
            # Approximate: standard deck has 13 per suit, minus discarded
            remaining_per_suit = np.maximum(13.0 - discarded_per_suit, 0.0)
            total_remaining = max(deck_size, 1.0)
            aug[4:8] = remaining_per_suit / total_remaining
        else:
            aug[4:8] = 0.25

        # [8] economy_health
        aug[8] = min(dollars / max(ante * 10.0, 1.0), 2.0) / 2.0

        # [9] hand_size_ratio: entity_counts[0] (hand cards) / hand_size
        hand_count = obs["entity_counts"][0]
        hand_size_limit = g[15] * 15.0
        aug[9] = min(hand_count / max(hand_size_limit, 1.0), 1.0)

        obs = dict(obs)
        obs["global"] = np.concatenate([g, aug])
        return obs


def _inv_log_scale(v: float) -> float:
    """Invert jackdaw's log_scale: log1p(x) / log1p(1e6)."""
    return math.expm1(v * math.log1p(1e6))
