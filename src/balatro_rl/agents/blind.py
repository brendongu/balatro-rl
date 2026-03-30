"""Blind selection policy (BLIND_SELECT phase).

Responsible for SelectBlind and SkipBlind decisions. Small action space
(2 actions when skipping is possible, 1 otherwise) but meaningful strategic
impact — skipping blinds trades score opportunity for tags/tempo.
"""

from __future__ import annotations

import numpy as np


class BlindPolicy:
    """Policy for blind select/skip decisions.

    Initial implementation: wraps a MaskablePPO model.
    Could also be a simple heuristic (e.g. always select, or skip
    specific boss blinds based on current build).
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
