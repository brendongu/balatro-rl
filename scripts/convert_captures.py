"""Convert JSONL capture sessions to NPZ demo files.

Reads recorded sessions from the capture harness, converts raw balatrobot
gamestates to observations using the jackdaw encoder, and produces .npz
files compatible with DemoDataset.

Usage::

    # Convert all sessions in a directory
    uv run python scripts/convert_captures.py --input data/captures --output data/demos

    # Convert a single session
    uv run python scripts/convert_captures.py --input data/captures/session_001.jsonl --output data/demos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.observation import encode_observation

from balatro_rl.capture.state_builder import build_game_state

_SPEC = balatro_game_spec()
_ENTITY_INFO: list[tuple[str, int, int]] = [
    (et.name, et.max_count, et.feature_dim) for et in _SPEC.entity_types
]

_METHOD_TO_ACTION_TYPE: dict[str, int] = {
    "play": 0,
    "discard": 1,
    "select": 2,
    "skip": 3,
    "cash_out": 4,
    "reroll": 5,
    "next_round": 6,
    "pack": 7,  # skip or pick — disambiguated by params
    "buy": 8,   # card, voucher, or pack — disambiguated by params
    "sell": 9,  # joker or consumable — disambiguated by params
    "use": 11,
}


def _obs_to_arrays(gs_dict: dict) -> dict[str, np.ndarray]:
    """Encode a game_state dict to padded observation arrays."""
    obs = encode_observation(gs_dict)

    result: dict[str, np.ndarray] = {
        "global": obs.global_context.astype(np.float32),
    }

    counts: list[int] = []
    for name, max_count, feat_dim in _ENTITY_INFO:
        arr = obs.entities.get(name)
        padded = np.zeros((max_count, feat_dim), dtype=np.float32)
        if arr is not None and arr.shape[0] > 0:
            n = min(arr.shape[0], max_count)
            padded[:n] = arr[:n]
            counts.append(n)
        else:
            counts.append(0)
        result[name] = padded

    result["entity_counts"] = np.array(counts, dtype=np.float32)
    return result


def _detect_phase(obs_arrays: dict[str, np.ndarray]) -> int:
    """Extract phase index from the one-hot encoding in global."""
    return int(np.argmax(obs_arrays["global"][:6]))


def _action_to_record(
    method: str,
    params: dict,
) -> dict:
    """Convert a semantic action to a structured record for storage.

    Returns a dict with action_type (int) and optional targets.
    We store the semantic action as-is alongside the encoded observation;
    the flat action index mapping requires the full action table which
    depends on the specific game state and is expensive to reconstruct
    offline. Instead, we store enough info for the BC training loop
    to match against action tables at training time.
    """
    action_type = _METHOD_TO_ACTION_TYPE.get(method, -1)

    # Disambiguate overloaded methods
    if method == "pack":
        if params.get("skip"):
            action_type = 7  # SkipPack
        else:
            action_type = 14  # PickPackCard

    if method == "buy":
        if "voucher" in params:
            action_type = 12  # RedeemVoucher
        elif "pack" in params:
            action_type = 13  # OpenBooster
        else:
            action_type = 8  # BuyCard

    if method == "sell":
        if "consumable" in params:
            action_type = 10  # SellConsumable
        else:
            action_type = 9  # SellJoker

    return {
        "action_type": action_type,
        "method": method,
        "params": params,
    }


def convert_session(
    jsonl_path: Path,
    output_dir: Path,
    episode_offset: int = 0,
) -> int:
    """Convert a single JSONL session to NPZ demo file(s).

    Returns the number of transitions converted.
    """
    transitions: list[dict] = []

    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") == "transition":
                transitions.append(record)

    if not transitions:
        return 0

    obs_global_list: list[np.ndarray] = []
    obs_hand_card_list: list[np.ndarray] = []
    obs_joker_list: list[np.ndarray] = []
    obs_entity_counts_list: list[np.ndarray] = []
    actions_list: list[int] = []
    phases_list: list[int] = []

    converted = 0
    for t in transitions:
        state_json = t.get("state")
        action_json = t.get("action")

        if state_json is None:
            continue

        # Skip transitions without actions (terminal states)
        if action_json is None:
            continue

        try:
            gs_dict = build_game_state(state_json)
            obs_arrays = _obs_to_arrays(gs_dict)
        except Exception as e:
            print(f"  Warning: failed to encode state: {e}")
            continue

        phase = _detect_phase(obs_arrays)
        action_record = _action_to_record(
            action_json.get("method", ""),
            action_json.get("params", {}),
        )

        obs_global_list.append(obs_arrays["global"])
        obs_hand_card_list.append(obs_arrays["hand_card"])
        obs_joker_list.append(obs_arrays["joker"])
        obs_entity_counts_list.append(obs_arrays["entity_counts"])

        # Store the action_type as the action index.
        # For BC training, the action_type plus params can be matched
        # against the action table generated at training time.
        actions_list.append(action_record["action_type"])
        phases_list.append(phase)
        converted += 1

    if converted == 0:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"episode_{episode_offset:06d}.npz"
    path = output_dir / filename

    np.savez_compressed(
        path,
        obs_global=np.array(obs_global_list, dtype=np.float32),
        obs_hand_card=np.array(obs_hand_card_list, dtype=np.float32),
        obs_joker=np.array(obs_joker_list, dtype=np.float32),
        obs_entity_counts=np.array(obs_entity_counts_list, dtype=np.float32),
        action_masks=np.ones((converted, 500), dtype=bool),
        actions=np.array(actions_list, dtype=np.int32),
        phases=np.array(phases_list, dtype=np.int8),
        rewards=np.zeros(converted, dtype=np.float32),
    )

    print(f"  Saved {converted} transitions to {path}")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JSONL capture sessions to NPZ demo files"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file or directory of JSONL files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/demos",
        help="Output directory for NPZ files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        jsonl_files = [input_path]
    else:
        jsonl_files = sorted(input_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {input_path}")
        return

    total_transitions = 0
    for i, jsonl_path in enumerate(jsonl_files):
        print(f"Converting {jsonl_path.name}...")
        n = convert_session(jsonl_path, output_dir, episode_offset=i)
        total_transitions += n

    print(f"\nConverted {len(jsonl_files)} sessions, {total_transitions} total transitions")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
