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
├── env/wrappers.py      Reward shaping + Curriculum + Obs augmentation + ActionInfo
├── agents/dispatch.py   Phase-aware routing to sub-policies
├── agents/{hand,shop,blind}.py   Phase-specific policies (RL + heuristic)
├── agents/tracer.py     Decision trace logger for debugging heuristics
├── features/            Hand evaluator, discard planner, joker catalog
├── imitation/           Demo collection + BC training pipeline
├── capture/             Live expert gameplay capture via balatrobot
├── config.py            TOML-based configuration
└── client.py            BalatroBot JSON-RPC API client
```

### Data Flow

```
BalatroGymnasiumEnv(DirectAdapter)
  → ActionInfoWrapper (exposes action_table for heuristic agents)
    → ExpertRewardWrapper (ante-progress, discard efficiency, hand diversity)
      → CurriculumWrapper (staged ante caps, auto-promotion)
        → ObservationAugmentWrapper (+10 derived features)
          → PhaseDispatchAgent → {HandPolicy, ShopPolicy, BlindPolicy}
```

## Current Status

### Heuristic Agent (v3)

Hand-crafted policies that serve as evaluation baseline and demo source for imitation learning.

| Metric | Random | Heuristic |
|--------|--------|-----------|
| mean_ante | 1.00 | **1.99** |
| max_ante | 1 | **5** |

**HandPolicy** — Evaluates all 5-card combinations, discards toward flushes (preferred) or full houses (when trips exist). Uses all discards freely. Plays best available hand when it beats the blind, or weakest hand to cycle cards otherwise.

**ShopPolicy** — Buys scoring jokers (common +chips/+mult > scaling > any scoring), opens Buffoon packs, buys relevant planet cards (Flush/Straight/FH for big-hand path; Pair/HC/Two Pair for scaling path). Never rerolls early.

**BlindPolicy** — Always selects blind (never skips).

**Decision tracing** — `--verbose` flag prints human-readable logs of every decision:
```bash
uv run python scripts/collect_demos.py --agent heuristic -n 3 --verbose --seed 42
```

## File Structure

```
src/balatro_rl/           Python package (src/ layout)
  config.py               TOML config loader
  client.py               BalatroBot API client
  env/
    wrappers.py           ExpertReward, Curriculum, ObsAugment, ActionInfo wrappers
    factory.py            make_env() with CLI-toggleable wrapper stack
  agents/
    base.py               PhasePolicy protocol
    dispatch.py           PhaseDispatchAgent (routes by game phase)
    hand.py               HandPolicy (RL) + HeuristicHandPolicy
    shop.py               ShopPolicy (RL) + HeuristicShopPolicy
    blind.py              BlindPolicy (RL) + HeuristicBlindPolicy
    tracer.py             DecisionTracer (verbose logging wrapper)
  features/
    hand_evaluator.py     Hand detection, scoring, obs decoding, discard planning
    joker_catalog.py      centers.json lookup: joker categories, planets, packs
  imitation/
    collector.py          Records (obs, action, phase) tuples as .npz
    dataset.py            Numpy-backed dataset for behavioral cloning
  capture/
    state_builder.py      balatrobot JSON → engine game_state with Card objects
    recorder.py           JSONL session recorder
    observer.py           Passive polling + action inference from state diffs
    interactive.py        Terminal UI for API-driven expert play
    scenarios.py          TOML scenario loader for custom gamestates
scripts/
  train_rl.py             MaskablePPO training with wrappers + TensorBoard
  train_bc.py             Behavioral cloning from demonstrations
  evaluate.py             Load model, run N episodes, report metrics
  collect_demos.py        Record demos (random / heuristic / model, --verbose)
  capture_expert.py       Capture live expert gameplay via balatrobot
  convert_captures.py     Convert JSONL captures to NPZ demo format
configs/
  default.toml            Base hyperparameters (env, reward, curriculum, training)
tests/                    pytest suite (116 tests: env, agents, heuristics, features, imitation, capture, config)
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

## Usage

### Heuristic Demo Collection

```bash
# Collect 500 episodes with heuristic agent
uv run python scripts/collect_demos.py --agent heuristic -n 500 --save-dir data/demos

# Trace decisions for debugging
uv run python scripts/collect_demos.py --agent heuristic -n 3 --verbose --seed 42
```

### RL Training

```bash
# Baseline MaskablePPO (default config)
uv run python scripts/train_rl.py --total-timesteps 500000

# With wrapper stack
uv run python scripts/train_rl.py --reward-wrapper --curriculum --augment-obs

# Monitor with TensorBoard
uv run tensorboard --logdir runs/
```

### Evaluation

```bash
uv run python scripts/evaluate.py --model-path models/best_model.zip --n-episodes 100
```

### Imitation Learning

```bash
# Train behavioral cloning from heuristic demos
uv run python scripts/train_bc.py --data-dir data/demos --epochs 20
```

### Expert Gameplay Capture

Capture live gameplay from a running Balatro instance with the balatrobot mod.

```bash
# Passive observation — play in the Balatro UI, the harness polls and records
uv run python scripts/capture_expert.py --mode observe --save-dir data/captures

# Interactive terminal — pick actions from a menu, sent to balatrobot
uv run python scripts/capture_expert.py --mode interactive --save-dir data/captures

# Load a custom scenario before capture (targeted data collection)
uv run python scripts/capture_expert.py --mode observe --scenario scenarios/late_game.toml

# Convert captured JSONL sessions to NPZ demo format
uv run python scripts/convert_captures.py --input data/captures --output data/demos
```

**Custom Scenarios** — TOML files that configure the game to a specific state:

```toml
[game]
deck = "RED"
stake = "WHITE"
seed = "EXPERT_001"

[state]
ante = 3
money = 15

[[jokers]]
key = "j_lusty_joker"

[[jokers]]
key = "j_blueprint"
edition = "FOIL"
```

## Phase-Specific Policies

Balatro has 5 player-facing game phases with different action spaces:

| Phase | Actions | Policy |
|-------|---------|--------|
| BLIND_SELECT | Select, Skip | BlindPolicy / HeuristicBlindPolicy |
| SELECTING_HAND | PlayHand, Discard (combinatorial) | HandPolicy / HeuristicHandPolicy |
| ROUND_EVAL | CashOut | Fallback (trivial) |
| SHOP | Buy, Sell, Reroll, NextRound | ShopPolicy / HeuristicShopPolicy |
| PACK_OPENING | Pick, Skip | Fallback |

`PhaseDispatchAgent` reads the phase one-hot from `obs["global"][0:6]` and routes to the appropriate sub-policy. This enables:
- Training hand-play via RL (exploration effective for combinatorial search)
- Training shop via imitation learning (human expertise hard to discover via RL)
- Independently swapping/upgrading policies

## Wrapper Stack

### ActionInfoWrapper
Exposes jackdaw's internal `action_table` (list of `FactoredAction` with `action_type`, `card_target`, `entity_target`) through the wrapper chain. Required for heuristic agents to interpret the flat action space.

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

## Next Steps

1. **Improve heuristic agent** — Current bottleneck is ante 2-3 boss blinds (1600-4000 target). Key improvements:
   - Factor in owned jokers when evaluating hand strength (e.g., Lusty Joker makes Hearts worth +3 mult)
   - Smarter discard sequencing: stop discarding when flush/FH draw is assembled instead of burning all discards
   - Boss blind awareness: adapt strategy to debuff effects (e.g., avoid playing flush into The Club)
   - Economy management: maintain $5 interest threshold, save for bigger purchases

2. **A/B wrapper comparison** — Run baseline vs reward-wrapper vs curriculum training runs (~100k steps each) to validate that wrapper signals help learning before tuning coefficients.

3. **Imitation learning from heuristic demos** — Collect 1000+ heuristic episodes, train BC policy, use as warm-start for RL. The BC training script exists but hasn't been tested end-to-end.

4. **Reward coefficient tuning** — Once wrapper A/B shows directional signal, sweep `ante_progress_scale`, `discard_efficiency_bonus`, `hand_diversity_penalty` to find optimal coefficients.

5. **Curriculum training** — Use CurriculumWrapper with staged ante caps. Start with ante-1-only training, auto-promote as success rate meets thresholds.

6. **Joker-aware hand evaluation** — The hand evaluator currently estimates base score (chips x mult) without joker effects. Adding joker trigger simulation would significantly improve play quality for both heuristic and RL agents.

7. **Shop strategy depth** — Current shop policy is purely reactive (buy what's available). Future work: build coherence scoring (does this joker synergize with my existing jokers?), sell low-value jokers to make room, voucher evaluation.

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
