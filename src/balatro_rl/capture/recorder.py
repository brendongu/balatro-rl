"""JSONL session recorder for expert gameplay capture.

Records raw balatrobot gamestates and actions to JSONL files for later
conversion to the NPZ demo format used by DemoDataset.

Each session produces one ``.jsonl`` file with three record types:

- ``session_start``: mode, scenario, timestamp
- ``transition``: gamestate JSON + action (method/params) + metadata
- ``session_end``: ante reached, win status, stats

Usage::

    recorder = SessionRecorder(save_dir="data/captures")
    recorder.begin_session(mode="observe")
    recorder.record_transition(state_json, action={"method": "play", "params": {"cards": [0,2]}})
    recorder.end_session(ante_reached=3, won=False)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class SessionRecorder:
    """Records capture sessions as JSONL files.

    Args:
        save_dir: Directory to store session files.
    """

    def __init__(self, save_dir: str | Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._file: Any = None
        self._path: Path | None = None
        self._session_count = 0
        self._transition_count = 0

    def begin_session(
        self,
        mode: str = "observe",
        scenario: str | None = None,
    ) -> Path:
        """Start a new recording session. Returns the file path."""
        timestamp = int(time.time())
        filename = f"session_{timestamp}_{self._session_count:04d}.jsonl"
        self._path = self.save_dir / filename
        self._file = open(self._path, "w")
        self._transition_count = 0

        self._write({
            "type": "session_start",
            "ts": timestamp,
            "mode": mode,
            "scenario": scenario,
        })
        return self._path

    def record_transition(
        self,
        gamestate: dict[str, Any],
        action: dict[str, Any] | None = None,
        *,
        inferred: bool = False,
    ) -> None:
        """Record a single game state + action transition.

        Args:
            gamestate: Raw balatrobot gamestate JSON.
            action: Action dict with ``method`` and ``params`` keys.
                None for the final state (no action taken).
            inferred: True if the action was inferred from state diffs
                (observer mode) rather than explicitly known.
        """
        assert self._file is not None, "Call begin_session() first"

        record: dict[str, Any] = {
            "type": "transition",
            "ts": int(time.time() * 1000),
            "state": gamestate,
        }
        if action is not None:
            record["action"] = action
            record["inferred"] = inferred
        self._write(record)
        self._transition_count += 1

    def end_session(
        self,
        ante_reached: int = 0,
        won: bool = False,
    ) -> Path:
        """Finalize and close the current session. Returns the file path."""
        assert self._file is not None, "No session in progress"

        self._write({
            "type": "session_end",
            "ts": int(time.time()),
            "ante_reached": ante_reached,
            "won": won,
            "transitions": self._transition_count,
        })

        self._file.close()
        self._file = None
        self._session_count += 1

        path = self._path
        self._path = None
        return path

    @property
    def in_session(self) -> bool:
        return self._file is not None

    @property
    def sessions_saved(self) -> int:
        return self._session_count

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._file.flush()
