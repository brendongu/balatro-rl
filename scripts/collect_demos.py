"""Collect demonstration episodes using a heuristic or trained agent.

Runs the agent for N episodes and saves transitions to disk for
behavioral cloning training.

Usage::

    uv run python scripts/collect_demos.py --n-episodes 500 --save-dir data/demos
    uv run python scripts/collect_demos.py --model runs/baseline/balatro_ppo.zip -n 100
"""

from __future__ import annotations

import argparse

import numpy as np
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv

from balatro_rl.imitation.collector import DemoCollector


def collect(
    n_episodes: int,
    save_dir: str,
    model_path: str | None = None,
    max_steps: int = 10_000,
    seed: int = 0,
) -> None:
    env = BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter,
        max_steps=max_steps,
        seed_prefix=f"DEMO_{seed}",
        reward_shaping=True,
    )
    collector = DemoCollector(save_dir=save_dir)

    model = None
    if model_path:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(model_path)

    rng = np.random.default_rng(seed)
    antes_reached: list[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        collector.begin_episode()

        while True:
            mask = env.action_masks()

            if model is not None:
                action, _ = model.predict(obs, action_masks=mask, deterministic=False)
                action = int(action)
            else:
                legal = np.where(mask)[0]
                action = int(rng.choice(legal))

            collector.record(obs, mask, action)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        path = collector.end_episode(info)
        ante = info.get("balatro/ante_reached", 1)
        antes_reached.append(ante)

        if (ep + 1) % 50 == 0:
            print(
                f"  Episode {ep + 1}/{n_episodes} | "
                f"mean_ante={np.mean(antes_reached[-50:]):.2f} | "
                f"saved to {path}"
            )

    env.close()
    print(f"\nCollected {collector.episodes_saved} episodes to {save_dir}")
    print(f"  mean_ante_reached: {np.mean(antes_reached):.2f}")
    print(f"  max_ante_reached:  {max(antes_reached)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Balatro demonstration data")
    parser.add_argument("-n", "--n-episodes", type=int, default=500)
    parser.add_argument("--save-dir", type=str, default="data/demos")
    parser.add_argument("--model", type=str, default=None, help="Path to model .zip (or random)")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Collecting {args.n_episodes} episodes...")
    if args.model:
        print(f"  Using model: {args.model}")
    else:
        print("  Using random policy")

    collect(
        n_episodes=args.n_episodes,
        save_dir=args.save_dir,
        model_path=args.model,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
