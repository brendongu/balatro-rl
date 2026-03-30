"""Tests for config loading."""

from balatro_rl.config import Config, load_config


def test_default_config():
    cfg = load_config()
    assert cfg.env.max_steps == 10_000
    assert cfg.training.algorithm == "MaskablePPO"
    assert cfg.reward.ante_progress_exp == 1.5


def test_config_dataclass_defaults():
    cfg = Config()
    assert cfg.env.reward_shaping is True
    assert cfg.curriculum.enabled is False
    assert cfg.observation.augment is False


def test_load_config_from_file(tmp_path):
    override = tmp_path / "test.toml"
    override.write_text('[training]\ntotal_timesteps = 1000\nseed = 99\n')
    cfg = load_config(override)
    assert cfg.training.total_timesteps == 1000
    assert cfg.training.seed == 99
    # Defaults should still be present
    assert cfg.env.max_steps == 10_000
