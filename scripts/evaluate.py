"""Evaluate a trained model on Balatro.

Loads a saved MaskablePPO checkpoint and runs N episodes, reporting
aggregate metrics.

Usage::

    uv run python scripts/evaluate.py --model runs/balatro_ppo/balatro_ppo.zip
    uv run python scripts/evaluate.py --model runs/balatro_ppo/balatro_ppo.zip -n 200
    uv run python scripts/evaluate.py --model runs/augmented/balatro_ppo.zip --augment-obs
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from balatro_rl.config import load_config
from balatro_rl.env.factory import make_env


def evaluate(
    model_path: str,
    n_episodes: int = 100,
    max_steps: int = 10_000,
    deterministic: bool = True,
    seed: int = 0,
    augment_obs: bool = False,
    config_path: str | None = None,
) -> dict[str, float]:
    cfg = load_config(config_path)
    cfg.env.max_steps = max_steps
    cfg.env.reward_shaping = False

    env = make_env(
        cfg,
        seed=seed,
        seed_prefix="EVAL",
        augment_obs=augment_obs,
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
    parser.add_argument("--config", type=str, default=None, help="TOML config override file")
    parser.add_argument(
        "--augment-obs", action=argparse.BooleanOptionalAction, default=False,
        help="Apply ObservationAugmentWrapper (must match training config)",
    )
    args = parser.parse_args()

    print(f"Evaluating {args.model} over {args.n_episodes} episodes...")
    if args.augment_obs:
        print("  ObservationAugmentWrapper: enabled")
    results = evaluate(
        model_path=args.model,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        seed=args.seed,
        augment_obs=args.augment_obs,
        config_path=args.config,
    )

    print("\nResults:")
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-' * 25} {'-' * 10}")
    for key, val in sorted(results.items()):
        print(f"  {key:<25} {val:>10.4f}")


if __name__ == "__main__":
    main()
