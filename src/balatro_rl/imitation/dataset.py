"""PyTorch Dataset for behavioral cloning from collected demonstrations.

Loads episodes saved by :class:`DemoCollector` and presents them as
(observation, action_mask, action) tuples for supervised training.

Supports optional phase filtering so you can train phase-specific
policies on only the relevant transitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class DemoDataset:
    """Numpy-backed demonstration dataset (no torch dependency required).

    Loads all episodes from a directory into memory and concatenates
    them into flat arrays. Supports indexing and phase filtering.

    Args:
        data_dir: Directory containing ``.npz`` episode files.
        phase_filter: If set, only include transitions from this phase index.
            Phase indices: 0=BLIND_SELECT, 1=SELECTING_HAND, 2=ROUND_EVAL,
            3=SHOP, 4=PACK_OPENING.
        max_episodes: Maximum number of episodes to load (None = all).
    """

    def __init__(
        self,
        data_dir: str | Path,
        phase_filter: int | None = None,
        max_episodes: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.phase_filter = phase_filter

        files = sorted(self.data_dir.glob("episode_*.npz"))
        if max_episodes is not None:
            files = files[:max_episodes]

        all_obs_global: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_phases: list[np.ndarray] = []

        for f in files:
            ep = np.load(f)
            all_obs_global.append(ep["obs_global"])
            all_masks.append(ep["action_masks"])
            all_actions.append(ep["actions"])
            all_phases.append(ep["phases"])

        if not all_obs_global:
            self.obs_global = np.empty((0, 0), dtype=np.float32)
            self.action_masks = np.empty((0, 0), dtype=bool)
            self.actions = np.empty((0,), dtype=np.int32)
            self.phases = np.empty((0,), dtype=np.int8)
            return

        self.obs_global = np.concatenate(all_obs_global)
        self.action_masks = np.concatenate(all_masks)
        self.actions = np.concatenate(all_actions)
        self.phases = np.concatenate(all_phases)

        if phase_filter is not None:
            mask = self.phases == phase_filter
            self.obs_global = self.obs_global[mask]
            self.action_masks = self.action_masks[mask]
            self.actions = self.actions[mask]
            self.phases = self.phases[mask]

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "obs_global": self.obs_global[idx],
            "action_mask": self.action_masks[idx],
            "action": self.actions[idx],
            "phase": self.phases[idx],
        }

    def summary(self) -> dict[str, Any]:
        """Return summary statistics about the dataset."""
        if len(self) == 0:
            return {"total_transitions": 0}
        phase_names = {
            0: "BLIND_SELECT",
            1: "SELECTING_HAND",
            2: "ROUND_EVAL",
            3: "SHOP",
            4: "PACK_OPENING",
        }
        unique, counts = np.unique(self.phases, return_counts=True)
        phase_dist = {
            phase_names.get(int(p), f"UNKNOWN_{p}"): int(c) for p, c in zip(unique, counts)
        }
        return {
            "total_transitions": len(self),
            "phase_distribution": phase_dist,
            "action_range": (int(self.actions.min()), int(self.actions.max())),
        }
