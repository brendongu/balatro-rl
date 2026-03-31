"""Collect demonstration episodes using a heuristic or trained agent.

Runs the agent for N episodes and saves transitions to disk for
behavioral cloning training.

Usage::

    uv run python scripts/collect_demos.py --n-episodes 500 --save-dir data/demos
    uv run python scripts/collect_demos.py --model runs/baseline/balatro_ppo.zip -n 100
    uv run python scripts/collect_demos.py --agent heuristic -n 50
    uv run python scripts/collect_demos.py --agent heuristic -n 3 --verbose
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv

from balatro_rl.env.wrappers import ActionInfoWrapper
from balatro_rl.imitation.collector import DemoCollector


def _build_heuristic_agent(verbose: bool = False):
    from balatro_rl.agents.blind import HeuristicBlindPolicy
    from balatro_rl.agents.dispatch import PhaseDispatchAgent
    from balatro_rl.agents.hand import HeuristicHandPolicy
    from balatro_rl.agents.shop import HeuristicShopPolicy

    agent = PhaseDispatchAgent(
        hand_policy=HeuristicHandPolicy(),
        shop_policy=HeuristicShopPolicy(),
        blind_policy=HeuristicBlindPolicy(),
    )
    if verbose:
        from balatro_rl.agents.tracer import DecisionTracer
        return DecisionTracer(agent, out=sys.stdout)
    return agent


def collect(
    n_episodes: int,
    save_dir: str,
    model_path: str | None = None,
    agent_type: str = "random",
    max_steps: int = 10_000,
    seed: int = 0,
    verbose: bool = False,
) -> None:
    env = BalatroGymnasiumEnv(
        adapter_factory=DirectAdapter,
        max_steps=max_steps,
        seed_prefix=f"DEMO_{seed}",
        reward_shaping=True,
    )

    use_action_table = agent_type == "heuristic"
    if use_action_table:
        env = ActionInfoWrapper(env)

    collector = DemoCollector(save_dir=save_dir)

    model = None
    heuristic_agent = None
    if agent_type == "heuristic":
        heuristic_agent = _build_heuristic_agent(verbose=verbose)
    elif model_path:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(model_path)

    rng = np.random.default_rng(seed)
    antes_reached: list[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        collector.begin_episode()

        if hasattr(heuristic_agent, "new_episode"):
            heuristic_agent.new_episode(ep)

        while True:
            mask = env.action_masks()

            if heuristic_agent is not None:
                action_table = env.action_table if use_action_table else None
                action = heuristic_agent.select_action(obs, mask, action_table=action_table)
            elif model is not None:
                action, _ = model.predict(obs, action_masks=mask, deterministic=False)
                action = int(action)
            else:
                legal = np.where(mask)[0]
                action = int(rng.choice(legal))

            collector.record(obs, mask, action)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        if hasattr(heuristic_agent, "end_episode"):
            heuristic_agent.end_episode(info)

        path = collector.end_episode(info)
        ante = info.get("balatro/ante_reached", 1)
        antes_reached.append(ante)

        if not verbose and (ep + 1) % 10 == 0:
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
    parser.add_argument(
        "--agent",
        choices=["random", "heuristic", "model"],
        default="random",
        help="Agent type: random, heuristic, or model (requires --model)",
    )
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print decision traces")
    args = parser.parse_args()

    agent_type = args.agent
    if args.model and agent_type == "random":
        agent_type = "model"

    print(f"Collecting {args.n_episodes} episodes...")
    print(f"  Agent: {agent_type}")
    if args.model:
        print(f"  Model: {args.model}")

    collect(
        n_episodes=args.n_episodes,
        save_dir=args.save_dir,
        model_path=args.model,
        agent_type=agent_type,
        max_steps=args.max_steps,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
