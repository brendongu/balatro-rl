"""Shop policy (SHOP phase).

Responsible for BuyCard, SellJoker, SellConsumable, Reroll, RedeemVoucher,
OpenBooster, and NextRound decisions. This is where long-term build strategy
matters most — joker synergies, economy management, and build coherence.
"""

from __future__ import annotations

import numpy as np


class ShopPolicy:
    """Policy for shop-phase decisions.

    Initial implementation: wraps a MaskablePPO model.
    Shop decisions are the strongest candidate for imitation learning
    since expert knowledge about build construction is hard to learn
    from sparse rewards alone.
    """

    def __init__(self, model: object | None = None) -> None:
        self._model = model

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        action_mask: np.ndarray,
    ) -> int:
        if self._model is not None:
            return self._predict_from_model(obs, action_mask)
        return self._random_legal(action_mask)

    def _predict_from_model(
        self,
        obs: dict[str, np.ndarray],
        action_mask: np.ndarray,
    ) -> int:
        action, _ = self._model.predict(obs, action_masks=action_mask, deterministic=True)
        return int(action)

    def _random_legal(self, action_mask: np.ndarray) -> int:
        legal = np.where(action_mask)[0]
        return int(np.random.choice(legal))

    def update(self, **kwargs: object) -> None:
        pass
