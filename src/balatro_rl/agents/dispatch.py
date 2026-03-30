"""Phase-aware dispatch agent.

Routes decisions to phase-specific policies based on the game phase
encoded in the observation vector. Phases are one-hot encoded in
global[0:6] by jackdaw's observation encoder:

    0: BLIND_SELECT
    1: SELECTING_HAND
    2: ROUND_EVAL
    3: SHOP
    4: PACK_OPENING
    5: GAME_OVER
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

from balatro_rl.agents.base import PhasePolicy
from balatro_rl.agents.blind import BlindPolicy
from balatro_rl.agents.hand import HandPolicy
from balatro_rl.agents.shop import ShopPolicy


class Phase(IntEnum):
    BLIND_SELECT = 0
    SELECTING_HAND = 1
    ROUND_EVAL = 2
    SHOP = 3
    PACK_OPENING = 4
    GAME_OVER = 5


def _detect_phase(obs: dict[str, np.ndarray]) -> Phase:
    """Extract the current game phase from the observation's one-hot encoding."""
    phase_onehot = obs["global"][:6]
    return Phase(int(np.argmax(phase_onehot)))


class PhaseDispatchAgent:
    """Meta-agent that dispatches to phase-specific policies.

    For phases without a dedicated policy (ROUND_EVAL, PACK_OPENING),
    falls back to random legal action selection. ROUND_EVAL is always
    trivial (only CashOut is legal), and PACK_OPENING has a small
    action space that the hand policy can handle.

    Args:
        hand_policy: Policy for SELECTING_HAND phase.
        shop_policy: Policy for SHOP phase.
        blind_policy: Policy for BLIND_SELECT phase.
        fallback: Optional policy for all other phases.
    """

    def __init__(
        self,
        hand_policy: PhasePolicy | None = None,
        shop_policy: PhasePolicy | None = None,
        blind_policy: PhasePolicy | None = None,
        fallback: PhasePolicy | None = None,
    ) -> None:
        self.hand_policy = hand_policy or HandPolicy()
        self.shop_policy = shop_policy or ShopPolicy()
        self.blind_policy = blind_policy or BlindPolicy()
        self._fallback = fallback

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        action_mask: np.ndarray,
    ) -> int:
        phase = _detect_phase(obs)

        if phase == Phase.SELECTING_HAND:
            return self.hand_policy.select_action(obs, action_mask)
        elif phase == Phase.SHOP:
            return self.shop_policy.select_action(obs, action_mask)
        elif phase == Phase.BLIND_SELECT:
            return self.blind_policy.select_action(obs, action_mask)
        elif self._fallback is not None:
            return self._fallback.select_action(obs, action_mask)

        legal = np.where(action_mask)[0]
        return int(np.random.choice(legal))

    def get_policy_for_phase(self, phase: Phase) -> PhasePolicy | None:
        """Return the policy handling a given phase, or None."""
        mapping: dict[Phase, Any] = {
            Phase.SELECTING_HAND: self.hand_policy,
            Phase.SHOP: self.shop_policy,
            Phase.BLIND_SELECT: self.blind_policy,
        }
        return mapping.get(phase)
