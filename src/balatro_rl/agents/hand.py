"""Hand selection policy (SELECTING_HAND phase).

Responsible for PlayHand and Discard decisions. This is the most
combinatorially complex phase — the action space includes all legal
card combinations for both playing and discarding.
"""

from __future__ import annotations

import numpy as np


class HandPolicy:
    """Policy for the hand-play phase.

    Initial implementation: wraps a MaskablePPO model.
    Can be swapped for a heuristic, IL-trained model, or custom architecture.
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
