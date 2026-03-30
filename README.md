# balatro-rl

Reinforcement learning agent for [Balatro](https://www.playbalatro.com/), using [jackdaw-balatro](https://github.com/TylerFlar/jackdaw-balatro) as the game engine and Gymnasium environment.

## Architecture

```
jackdaw-balatro (dependency)
├── Engine: 30 modules, 150 jokers, full game loop
├── BalatroGymnasiumEnv: SB3-compatible, action masking
├── Observation: 235-dim global + AlphaStar-style entity encoding
└── Bridge: live validation via BalatroBot

balatro-rl (this repo)
├── env/wrappers.py      Reward shaping + Curriculum + Obs augmentation
├── agents/dispatch.py   Phase-aware routing to sub-policies
├── agents/{hand,shop,blind}.py   Phase-specific policies
├── features/            Hand evaluator, derived obs features
├── imitation/           Demo collection + BC training pipeline
├── config.py            TOML-based configuration
└── client.py            BalatroBot JSON-RPC API client
```

### Data Flow

```
BalatroGymnasiumEnv(DirectAdapter)
  → ExpertRewardWrapper (ante-progress, discard efficiency, hand diversity)
    → CurriculumWrapper (staged ante caps, auto-promotion)
      → ObservationAugmentWrapper (+10 derived features)
        → PhaseDispatchAgent → {HandPolicy, ShopPolicy, BlindPolicy}
```

## File Structure

```
src/balatro_rl/           Python package (src/ layout)
  config.py               TOML config loader
  client.py               BalatroBot API client
  env/
    wrappers.py           ExpertRewardWrapper, CurriculumWrapper, ObservationAugmentWrapper
  agents/
    base.py               PhasePolicy protocol
    dispatch.py           PhaseDispatchAgent (routes by game phase)
    hand.py               SELECTING_HAND phase policy
    shop.py               SHOP phase policy
    blind.py              BLIND_SELECT phase policy
  features/
    hand_evaluator.py     Hand-type detection + chip estimation
  imitation/
    collector.py          Records (obs, action, phase) tuples as .npz
    dataset.py            Numpy-backed dataset for behavioral cloning
scripts/
  train_rl.py             MaskablePPO training with wrappers + TensorBoard
  train_bc.py             Behavioral cloning from demonstrations
  evaluate.py             Load model, run N episodes, report metrics
  collect_demos.py        Record demos from agent or heuristic
configs/
  default.toml            Base hyperparameters (env, reward, curriculum, training)
tests/                    pytest suite (env, agents, features, imitation, config)
.claude/skills/           Expert domain knowledge references
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone jackdaw-balatro to sibling directory
git clone https://github.com/TylerFlar/jackdaw-balatro ../jackdaw-balatro

# Install dependencies
uv sync --native-tls

# Install training extras (PyTorch, SB3, TensorBoard)
uv sync --native-tls --extra train

# Run tests
uv run pytest
```

## Training

```bash
# Baseline MaskablePPO (default config)
uv run python scripts/train_rl.py --total-timesteps 500000

# With custom seed and log directory
uv run python scripts/train_rl.py --total-timesteps 500000 --log-dir runs/exp1 --seed 42

# Monitor with TensorBoard
uv run tensorboard --logdir runs/
```

## Evaluation

```bash
uv run python scripts/evaluate.py --model-path models/best_model.zip --n-episodes 100
```

## Imitation Learning

```bash
# Collect demonstrations
uv run python scripts/collect_demos.py --n-episodes 50 --save-dir data/demos

# Train behavioral cloning policy
uv run python scripts/train_bc.py --data-dir data/demos --epochs 20
```

## Phase-Specific Policies

Balatro has 5 player-facing game phases with different action spaces:

| Phase | Actions | Policy |
|-------|---------|--------|
| BLIND_SELECT | Select, Skip | BlindPolicy |
| SELECTING_HAND | PlayHand, Discard (combinatorial) | HandPolicy |
| ROUND_EVAL | CashOut | Fallback (trivial) |
| SHOP | Buy, Sell, Reroll, NextRound | ShopPolicy |
| PACK_OPENING | Pick, Skip | Fallback |

`PhaseDispatchAgent` reads the phase one-hot from `obs["global"][0:6]` and routes to the appropriate sub-policy. This enables:
- Training hand-play via RL (exploration effective for combinatorial search)
- Training shop via imitation learning (human expertise hard to discover via RL)
- Independently swapping/upgrading policies

## Wrapper Stack

### ExpertRewardWrapper
- Ante progress: `0.1 * (ante/8)^1.5` bonus on ante increase
- Discard efficiency: `0.005 * discards_left` on round win
- Hand diversity: `-0.01` penalty when >50% of recent plays are same type

### CurriculumWrapper
| Stage | Ante Cap | Stake | Success Threshold |
|-------|----------|-------|-------------------|
| 0 | 1 | WHITE | 80% |
| 1 | 3 | WHITE | 60% |
| 2 | 5 | WHITE | 40% |
| 3 | 8 | WHITE | 10% |
| 4 | 8 | RED | 5% |

### ObservationAugmentWrapper
Appends 10 derived features to the global vector: chips_progress, hands_urgency, flush/straight proximity, suit density (4), economy health, hand size ratio.

## Experiment Plan

| Run | Reward | Curriculum | Obs Augment | IL Warm-start |
|-----|--------|------------|-------------|---------------|
| A | jackdaw default | none | none | no (baseline) |
| B | expert | none | none | no |
| C | jackdaw default | staged | none | no |
| D | expert | staged | augmented | no |
| E | expert | staged | augmented | yes (shop) |

## Key Dependencies

- [jackdaw-balatro](https://github.com/TylerFlar/jackdaw-balatro) — 1:1 Python reimplementation of Balatro engine
- [sb3-contrib](https://github.com/Stable-Baselines-Contrib/stable-baselines3-contrib) — MaskablePPO for discrete action masking
- [gymnasium](https://gymnasium.farama.org/) — Environment interface
