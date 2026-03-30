"""Tests for imitation learning data collection and dataset."""


import numpy as np
import pytest
from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter

from balatro_rl.imitation.collector import DemoCollector
from balatro_rl.imitation.dataset import DemoDataset


@pytest.fixture
def env():
    e = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=100)
    yield e
    e.close()


def test_collector_saves_episode(env, tmp_path):
    collector = DemoCollector(save_dir=tmp_path)
    obs, info = env.reset(seed=42)
    collector.begin_episode()
    rng = np.random.default_rng(42)

    while True:
        mask = env.action_masks()
        legal = np.where(mask)[0]
        action = int(rng.choice(legal))
        collector.record(obs, mask, action)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    path = collector.end_episode(info)
    assert path.exists()
    assert collector.episodes_saved == 1

    data = np.load(path)
    assert "obs_global" in data
    assert "actions" in data
    assert "phases" in data
    assert data["actions"].dtype == np.int32


def test_dataset_loads_episodes(env, tmp_path):
    collector = DemoCollector(save_dir=tmp_path)
    rng = np.random.default_rng(42)

    for seed in range(3):
        obs, info = env.reset(seed=seed)
        collector.begin_episode()
        while True:
            mask = env.action_masks()
            legal = np.where(mask)[0]
            action = int(rng.choice(legal))
            collector.record(obs, mask, action)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        collector.end_episode(info)

    dataset = DemoDataset(tmp_path)
    assert len(dataset) > 0

    item = dataset[0]
    assert "obs_global" in item
    assert "action" in item
    assert "phase" in item


def test_dataset_phase_filter(env, tmp_path):
    collector = DemoCollector(save_dir=tmp_path)
    rng = np.random.default_rng(42)

    obs, info = env.reset(seed=42)
    collector.begin_episode()
    while True:
        mask = env.action_masks()
        legal = np.where(mask)[0]
        action = int(rng.choice(legal))
        collector.record(obs, mask, action)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    collector.end_episode(info)

    full = DemoDataset(tmp_path)
    filtered = DemoDataset(tmp_path, phase_filter=1)  # SELECTING_HAND only

    assert len(filtered) <= len(full)
    if len(filtered) > 0:
        assert all(filtered.phases == 1)


def test_dataset_summary(env, tmp_path):
    collector = DemoCollector(save_dir=tmp_path)
    rng = np.random.default_rng(42)

    obs, info = env.reset(seed=42)
    collector.begin_episode()
    while True:
        mask = env.action_masks()
        legal = np.where(mask)[0]
        action = int(rng.choice(legal))
        collector.record(obs, mask, action)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    collector.end_episode(info)

    dataset = DemoDataset(tmp_path)
    summary = dataset.summary()
    assert summary["total_transitions"] > 0
    assert "phase_distribution" in summary


def test_empty_dataset(tmp_path):
    dataset = DemoDataset(tmp_path)
    assert len(dataset) == 0
    assert dataset.summary()["total_transitions"] == 0
