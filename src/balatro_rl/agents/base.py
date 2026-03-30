"""Base protocol for phase-specific agents."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PhasePolicy(Protocol):
    """Interface for a phase-specific policy.

    Each implementation decides how to select an action given the current
    observation and action mask, but only for its own game phase.
    """

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        action_mask: np.ndarray,
    ) -> int:
        """Choose an action index from the legal actions.

        Args:
            obs: Dictionary observation from BalatroGymnasiumEnv.
            action_mask: Boolean mask of shape (MAX_ACTIONS,).

        Returns:
            Integer action index (must be legal per action_mask).
        """
        ...

    def update(self, **kwargs: object) -> None:
        """Optional hook for updating policy parameters (e.g. from training)."""
        ...
