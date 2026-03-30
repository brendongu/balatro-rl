"""Evaluate a trained model on Balatro.

Loads a saved MaskablePPO checkpoint and runs N episodes, reporting
aggregate metrics.

Usage::

    uv run python scripts/evaluate.py --model runs/balatro_ppo/balatro_ppo.zip
    uv run python scripts/evaluate.py --model runs/balatro_ppo/balatro_ppo.zip -n 200
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv
from sb3_contrib import MaskablePPO


def evaluate(
    model_path: str,
    n_episodes: int = 100,
    max_steps: int = 10_000,
    deterministic: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    env = BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter,
        max_steps=max_steps,
        seed_prefix=f"EVAL_{seed}",
        reward_shaping=False,
    )
    model = MaskablePPO.load(model_path)

    metrics: dict[str, list] = defaultdict(list)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0

        while True:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        metrics["ante_reached"].append(info.get("balatro/ante_reached", 1))
        metrics["rounds_beaten"].append(info.get("balatro/rounds_beaten", 0))
        metrics["won"].append(info.get("balatro/won", False))
        metrics["episode_length"].append(steps)
        metrics["total_reward"].append(total_reward)

    env.close()

    results = {}
    for key, values in metrics.items():
        arr = np.array(values, dtype=np.float64)
        results[f"mean_{key}"] = float(arr.mean())
        results[f"std_{key}"] = float(arr.std())
        if key == "won":
            results["win_rate"] = float(arr.mean())
        if key == "ante_reached":
            results["max_ante_reached"] = float(arr.max())

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Balatro agent")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model .zip")
    parser.add_argument("-n", "--n-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    print(f"Evaluating {args.model} over {args.n_episodes} episodes...")
    results = evaluate(
        model_path=args.model,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        seed=args.seed,
    )

    print("\nResults:")
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-' * 25} {'-' * 10}")
    for key, val in sorted(results.items()):
        print(f"  {key:<25} {val:>10.4f}")


if __name__ == "__main__":
    main()
