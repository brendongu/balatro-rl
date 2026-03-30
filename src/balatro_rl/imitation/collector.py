"""Demonstration data collector.

Records (observation, action, phase, metadata) tuples from any agent
interacting with the jackdaw environment. Data is stored as compressed
numpy archives for efficient loading.

Usage::

    env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)
    collector = DemoCollector(save_dir="data/demos")

    obs, info = env.reset()
    collector.begin_episode()

    while not done:
        action = expert_agent.select_action(obs, env.action_masks())
        collector.record(obs, env.action_masks(), action, info)
        obs, reward, terminated, truncated, info = env.step(action)

    collector.end_episode(info)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _detect_phase(obs: dict[str, np.ndarray]) -> int:
    """Extract phase index from observation one-hot encoding."""
    return int(np.argmax(obs["global"][:6]))


class DemoCollector:
    """Collects demonstration episodes and saves them to disk.

    Each episode is saved as a separate ``.npz`` file containing:
        - ``obs_global``: (T, D_global) float32
        - ``obs_hand_card``: (T, max_hand, D_card) float32
        - ``obs_joker``: (T, max_joker, D_joker) float32
        - ``obs_entity_counts``: (T, N_entity_types) float32
        - ``action_masks``: (T, MAX_ACTIONS) bool
        - ``actions``: (T,) int32
        - ``phases``: (T,) int8
        - ``rewards``: (T,) float32
        - ``metadata``: dict with episode-level info

    Args:
        save_dir: Directory to save episode files.
        compress: Use compressed npz format.
    """

    def __init__(self, save_dir: str | Path, compress: bool = True) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._compress = compress
        self._episode_count = 0
        self._buffer: dict[str, list] = {}
        self._in_episode = False

    def begin_episode(self) -> None:
        """Start recording a new episode."""
        self._buffer = {
            "obs_global": [],
            "obs_hand_card": [],
            "obs_joker": [],
            "obs_entity_counts": [],
            "action_masks": [],
            "actions": [],
            "phases": [],
            "rewards": [],
        }
        self._in_episode = True

    def record(
        self,
        obs: dict[str, np.ndarray],
        action_mask: np.ndarray,
        action: int,
        reward: float = 0.0,
    ) -> None:
        """Record a single transition."""
        assert self._in_episode, "Call begin_episode() first"

        self._buffer["obs_global"].append(obs["global"].copy())
        self._buffer["obs_hand_card"].append(obs["hand_card"].copy())
        self._buffer["obs_joker"].append(obs["joker"].copy())
        self._buffer["obs_entity_counts"].append(obs["entity_counts"].copy())
        self._buffer["action_masks"].append(action_mask.copy())
        self._buffer["actions"].append(action)
        self._buffer["phases"].append(_detect_phase(obs))
        self._buffer["rewards"].append(reward)

    def end_episode(self, info: dict[str, Any] | None = None) -> Path:
        """Finalize and save the current episode. Returns the save path."""
        assert self._in_episode, "No episode in progress"
        self._in_episode = False

        arrays = {
            "obs_global": np.array(self._buffer["obs_global"], dtype=np.float32),
            "obs_hand_card": np.array(self._buffer["obs_hand_card"], dtype=np.float32),
            "obs_joker": np.array(self._buffer["obs_joker"], dtype=np.float32),
            "obs_entity_counts": np.array(self._buffer["obs_entity_counts"], dtype=np.float32),
            "action_masks": np.array(self._buffer["action_masks"], dtype=bool),
            "actions": np.array(self._buffer["actions"], dtype=np.int32),
            "phases": np.array(self._buffer["phases"], dtype=np.int8),
            "rewards": np.array(self._buffer["rewards"], dtype=np.float32),
        }

        if info:
            arrays["meta_ante"] = np.array([info.get("balatro/ante_reached", 0)])
            arrays["meta_won"] = np.array([info.get("balatro/won", False)])
            arrays["meta_rounds"] = np.array([info.get("balatro/rounds_beaten", 0)])

        filename = f"episode_{self._episode_count:06d}.npz"
        path = self.save_dir / filename
        save_fn = np.savez_compressed if self._compress else np.savez
        save_fn(path, **arrays)

        self._episode_count += 1
        self._buffer = {}
        return path

    @property
    def episodes_saved(self) -> int:
        return self._episode_count
