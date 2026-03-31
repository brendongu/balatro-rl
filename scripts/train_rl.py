"""Train MaskablePPO on the Balatro gymnasium environment.

Usage::

    uv run python scripts/train_rl.py --total-timesteps 50000
    uv run python scripts/train_rl.py --total-timesteps 50000 --reward-wrapper --log-dir runs/reward
    uv run python scripts/train_rl.py --config configs/default.toml --curriculum --augment-obs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from balatro_rl.config import load_config
from balatro_rl.env.factory import make_env


class BalatroMetricsCallback(BaseCallback):
    """Log Balatro-specific episode metrics to TensorBoard.

    Records balatro/* metrics from episode info dicts, and curriculum/*
    metrics when the CurriculumWrapper is active.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._antes: list[int] = []
        self._rounds: list[int] = []
        self._wins: list[bool] = []
        self._curriculum_stages: list[int] = []
        self._curriculum_success_rates: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "balatro/ante_reached" in info:
                self._antes.append(info["balatro/ante_reached"])
                self._rounds.append(info["balatro/rounds_beaten"])
                self._wins.append(info["balatro/won"])
            if "curriculum/stage" in info:
                self._curriculum_stages.append(info["curriculum/stage"])
                self._curriculum_success_rates.append(info["curriculum/success_rate"])
        return True

    def _on_rollout_end(self) -> None:
        if not self._antes:
            return
        self.logger.record("balatro/mean_ante_reached", float(np.mean(self._antes)))
        self.logger.record("balatro/max_ante_reached", int(np.max(self._antes)))
        self.logger.record("balatro/mean_rounds_beaten", float(np.mean(self._rounds)))
        self.logger.record("balatro/win_rate", float(np.mean(self._wins)))
        self.logger.record("balatro/episodes", len(self._antes))
        self._antes.clear()
        self._rounds.clear()
        self._wins.clear()

        if self._curriculum_stages:
            self.logger.record(
                "curriculum/stage", float(np.mean(self._curriculum_stages))
            )
            self.logger.record(
                "curriculum/success_rate", float(np.mean(self._curriculum_success_rates))
            )
            self._curriculum_stages.clear()
            self._curriculum_success_rates.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Balatro")
    parser.add_argument("--config", type=str, default=None, help="TOML config override file")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-reward-shaping", action="store_true",
                        help="Disable jackdaw's built-in dense reward")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)

    wrapper_group = parser.add_argument_group("wrapper stack")
    wrapper_group.add_argument(
        "--reward-wrapper", action=argparse.BooleanOptionalAction, default=None,
        help="Apply ExpertRewardWrapper (default: from config, off if no config)",
    )
    wrapper_group.add_argument(
        "--curriculum", action=argparse.BooleanOptionalAction, default=None,
        help="Apply CurriculumWrapper (default: from config, off if no config)",
    )
    wrapper_group.add_argument(
        "--augment-obs", action=argparse.BooleanOptionalAction, default=None,
        help="Apply ObservationAugmentWrapper (default: from config, off if no config)",
    )

    parser.add_argument("--checkpoint-freq", type=int, default=50_000,
                        help="Save checkpoint every N steps (0 to disable)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides for config values
    if args.total_timesteps is not None:
        cfg.training.total_timesteps = args.total_timesteps
    if args.log_dir is not None:
        cfg.training.log_dir = args.log_dir
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.max_steps is not None:
        cfg.env.max_steps = args.max_steps
    if args.no_reward_shaping:
        cfg.env.reward_shaping = False
    if args.n_steps is not None:
        cfg.training.n_steps = args.n_steps
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.ent_coef is not None:
        cfg.training.ent_coef = args.ent_coef
    if args.clip_range is not None:
        cfg.training.clip_range = args.clip_range

    # Wrapper toggles: CLI flag > config value > False
    use_reward = args.reward_wrapper if args.reward_wrapper is not None else False
    use_curriculum = (
        args.curriculum if args.curriculum is not None else cfg.curriculum.enabled
    )
    use_augment = (
        args.augment_obs if args.augment_obs is not None else cfg.observation.augment
    )

    log_path = Path(cfg.training.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    env = make_env(
        cfg,
        seed=cfg.training.seed,
        seed_prefix="PPO",
        reward_wrapper=use_reward,
        curriculum=use_curriculum,
        augment_obs=use_augment,
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=cfg.training.seed,
        tensorboard_log=str(log_path),
        ent_coef=cfg.training.ent_coef,
        learning_rate=cfg.training.learning_rate,
        n_steps=cfg.training.n_steps,
        clip_range=cfg.training.clip_range,
    )

    callbacks: list[BaseCallback] = [BalatroMetricsCallback()]
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(log_path / "checkpoints"),
                name_prefix="balatro_ppo",
            )
        )

    wrappers_desc = []
    if use_reward:
        wrappers_desc.append("ExpertReward")
    if use_curriculum:
        wrappers_desc.append("Curriculum")
    if use_augment:
        wrappers_desc.append("ObsAugment")
    wrapper_str = ", ".join(wrappers_desc) if wrappers_desc else "none"

    print(f"Training for {cfg.training.total_timesteps} timesteps...")
    print(f"  wrappers: {wrapper_str}")
    print(f"  reward_shaping={cfg.env.reward_shaping}")
    print(f"  n_steps={cfg.training.n_steps}, lr={cfg.training.learning_rate}, "
          f"ent_coef={cfg.training.ent_coef}")
    print(f"  Logging to {log_path}")

    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=CallbackList(callbacks),
    )

    save_path = log_path / "balatro_ppo"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")

    env.close()


if __name__ == "__main__":
    main()
