"""Configuration loading from TOML files.

Provides a typed config dataclass tree and a loader that reads
from TOML with defaults from configs/default.toml.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EnvConfig:
    max_steps: int = 10_000
    reward_shaping: bool = True
    back_keys: list[str] = field(default_factory=lambda: ["b_red"])
    stakes: list[int] = field(default_factory=lambda: [1])


@dataclass
class RewardConfig:
    ante_progress_exp: float = 1.5
    ante_progress_scale: float = 0.1
    discard_efficiency_bonus: float = 0.005
    hand_diversity_penalty: float = -0.01
    diversity_window: int = 20


@dataclass
class CurriculumConfig:
    enabled: bool = False
    window_size: int = 100
    start_stage: int = 0


@dataclass
class ObservationConfig:
    augment: bool = False


@dataclass
class TrainingConfig:
    algorithm: str = "MaskablePPO"
    total_timesteps: int = 500_000
    n_steps: int = 4096
    learning_rate: float = 1e-4
    ent_coef: float = 0.05
    clip_range: float = 0.15
    seed: int = 0
    log_dir: str = "runs/balatro_ppo"


@dataclass
class EvalConfig:
    n_episodes: int = 100
    deterministic: bool = True


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluate: EvalConfig = field(default_factory=EvalConfig)


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _apply_section(dc: object, data: dict[str, Any]) -> None:
    """Apply dict values to a dataclass instance."""
    for k, v in data.items():
        if hasattr(dc, k):
            setattr(dc, k, v)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a TOML file.

    Reads configs/default.toml as base, then overlays the provided file.
    """
    default_path = Path(__file__).parent.parent.parent / "configs" / "default.toml"
    base_data: dict[str, Any] = {}
    if default_path.exists():
        with open(default_path, "rb") as f:
            base_data = tomllib.load(f)

    if path is not None:
        with open(path, "rb") as f:
            override = tomllib.load(f)
        base_data = _merge(base_data, override)

    cfg = Config()
    if "env" in base_data:
        _apply_section(cfg.env, base_data["env"])
    if "reward" in base_data:
        _apply_section(cfg.reward, base_data["reward"])
    if "curriculum" in base_data:
        _apply_section(cfg.curriculum, base_data["curriculum"])
    if "observation" in base_data:
        _apply_section(cfg.observation, base_data["observation"])
    if "training" in base_data:
        section = base_data["training"]
        _apply_section(cfg.training, section)
        if "tensorboard" in section:
            cfg.training.log_dir = section["tensorboard"].get("log_dir", cfg.training.log_dir)
    if "evaluate" in base_data:
        _apply_section(cfg.evaluate, base_data["evaluate"])

    return cfg
