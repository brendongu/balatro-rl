"""Blind selection policy (BLIND_SELECT phase).

Responsible for SelectBlind and SkipBlind decisions. Small action space
(2 actions when skipping is possible, 1 otherwise) but meaningful strategic
impact — skipping blinds trades score opportunity for tags/tempo.
"""

from __future__ import annotations

import numpy as np

_SELECT_BLIND = 2


class BlindPolicy:
    """Policy for blind select/skip decisions.

    Initial implementation: wraps a MaskablePPO model.
    Can be swapped for a heuristic, IL-trained model, or custom architecture.
    """

    def __init__(self, model: object | None = None) -> None:
        self._model = model

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        action_mask: np.ndarray,
        action_table: list | None = None,
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


class HeuristicBlindPolicy:
    """Always select the blind. Never skip."""

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        action_mask: np.ndarray,
        action_table: list | None = None,
    ) -> int:
        if action_table is not None:
            for idx, fa in enumerate(action_table):
                if action_mask[idx] and fa.action_type == _SELECT_BLIND:
                    return idx
        # Fallback: first legal action (should be SelectBlind)
        return int(np.where(action_mask)[0][0])

    def update(self, **kwargs: object) -> None:
        pass
