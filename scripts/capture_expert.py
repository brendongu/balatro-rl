"""Capture expert Balatro gameplay via balatrobot.

Two modes:
  - observe:     passively poll the game while the expert plays in the UI
  - interactive:  terminal interface where the expert picks from legal actions

Both modes support custom gamestate scenarios via --scenario.

Usage::

    # Passive observation while playing in the Balatro UI
    uv run python scripts/capture_expert.py --mode observe --save-dir data/captures

    # Interactive terminal play
    uv run python scripts/capture_expert.py --mode interactive --save-dir data/captures

    # Load a scenario first, then observe
    uv run python scripts/capture_expert.py --mode observe --scenario scenarios/late_game.toml

    # Custom host/port
    uv run python scripts/capture_expert.py --mode interactive --host 127.0.0.1 --port 12346
"""

from __future__ import annotations

import argparse
import sys

from balatro_rl.capture.recorder import SessionRecorder
from balatro_rl.client import BalatroClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture expert Balatro gameplay via balatrobot"
    )
    parser.add_argument(
        "--mode",
        choices=["observe", "interactive"],
        default="observe",
        help="Capture mode: observe (passive polling) or interactive (terminal UI)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/captures",
        help="Directory to save capture session files",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Path to a TOML scenario file to load before capture",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="balatrobot API host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12346,
        help="balatrobot API port",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.2,
        help="Polling interval in seconds (observe mode only)",
    )
    args = parser.parse_args()

    client = BalatroClient(host=args.host, port=args.port)

    print(f"Connecting to balatrobot at {client.url}...")
    try:
        client.health()
    except Exception as e:
        print(f"Failed to connect: {e}", file=sys.stderr)
        sys.exit(1)
    print("Connected.")

    # Apply scenario if provided
    scenario_name = None
    if args.scenario:
        from balatro_rl.capture.scenarios import apply_scenario, load_scenario

        print(f"Loading scenario: {args.scenario}")
        scenario = load_scenario(args.scenario)
        scenario_name = scenario.name
        apply_scenario(client, scenario)
        print(f"Scenario applied: {scenario.name}")
        if scenario.description:
            print(f"  {scenario.description}")

    recorder = SessionRecorder(save_dir=args.save_dir)

    if args.mode == "observe":
        from balatro_rl.capture.observer import GameObserver

        print(f"\nObserving gameplay (poll interval: {args.poll_interval}s)")
        print("Play in the Balatro UI. Press Ctrl-C to stop.\n")

        observer = GameObserver(
            client, recorder, poll_interval=args.poll_interval
        )
        result = observer.run(scenario=scenario_name)

    elif args.mode == "interactive":
        from balatro_rl.capture.interactive import InteractiveHarness

        print("\nInteractive capture mode. Select actions by number.")
        print("Enter 'q' to quit.\n")

        harness = InteractiveHarness(client, recorder)
        result = harness.run(scenario=scenario_name)

    print(f"\nSession saved: {result['path']}")
    print(f"  Transitions: {result['transitions']}")
    print(f"  Ante reached: {result['ante_reached']}")
    print(f"  Won: {result['won']}")


if __name__ == "__main__":
    main()
