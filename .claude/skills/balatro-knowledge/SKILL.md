---
name: balatro-knowledge
description: Expert domain knowledge about Balatro gameplay. Use whenever making decisions about observation space design, reward shaping, action space design, curriculum ordering, hand selection logic, Joker valuation, scoring calculations, or any other game mechanic. Also use when writing or reviewing any code that encodes Balatro rules or strategy.
---

# Balatro Domain Knowledge

Expert knowledge from 1000+ hours of gameplay, organized for RL project use.

---

## Reference Files

Read the relevant file before answering questions in that domain.

| Topic | File | Status | Answered By |
|-------|------|--------|-------------|
| Scoring & Hand Selection | `references/scoring.md` | 🟢 Q8/Q9/Q47/Q48 filled | Game data + math from jackdaw; Q65-70 need expert |
| Observation Space | `references/observation.md` | 🟢 Jackdaw 235-dim spec + augmented features | Q100 answered; Q101-104 need expert |
| Economy & Resources | `references/economy.md` | 🟢 Mechanics filled | Game data; Q14-18 need expert strategy |
| Jokers — Principles | `references/jokers_principles.md` | 🟢 Taxonomy + Q22/Q28 filled | Game data; Q20-21/Q23-27/Q29-35 need expert |
| Jokers — Specific | `references/jokers_specific.md` | 🟢 Q40/Q41/Q44/Q49 + state table filled | Game data; Q36-39/Q42-43/Q45-46 need expert |
| Blinds & Antes | `references/blinds.md` | 🟢 Boss effects + ante ranges from blinds.json | Game data; Q10-13 + curriculum need expert |
| Deck & Cards | `references/deck.md` | 🟢 All modifier/consumable tables filled | Game data; Q50-64 need expert strategy |
| Strategy & Risk | `references/strategy.md` | 🔲 All expert-only | Q71-85 need expert |

### What's Been Answered (Game Data)

Sourced from the balatrobot API, jackdaw engine source (`jackdaw/engine/data/`), and verified game mechanics:
- Full scoring resolution order (card-by-card, joker-by-joker)
- Blind target table (White Stake base, with stake-parameterized scaling)
- +Mult vs xMult breakpoint math with worked examples
- Joker taxonomy (9 categories + 8 trigger types)
- All xMult jokers listed with conditions
- All conditional jokers mapped to required hand types / suits / ranks
- All economy jokers with gold/round values
- Joker state tracking requirements (28 jokers with mutable state)
- Joker resolution order rules
- Enhancement, edition, and seal effects with interactions
- All 22 Tarot, 12 Planet, and 18 Spectral card effects
- Boss blind effects (~30 blinds with hand-play impact classification)
- Interest formula and economy mechanics

### What Needs Expert Input

Highest priority (blocks implementation):
1. **Q8 VERIFY**: Exact White Stake blind targets and GREEN/PURPLE scaling multipliers
2. **Q65**: Hand selection decision hierarchy
3. **Q67**: Discard vs play decision rules
4. **Q100**: Observation feature ranking (reorder proposed list)

Medium priority (informs design quality):
5. **Q10-11**: Boss blind danger ranking and counter-strategies
6. **Q20-21**: Universal vs build-dependent jokers
7. **Q35**: Target joker count by ante (for curriculum)
8. **Q80**: Minimum viable build per ante

Lower priority (Phase 3+):
9. **Q71-75**: Shop strategy
10. **Q76-80**: Build coherence
11. **Q50-59**: Deck composition strategy

---

## How to Use This Skill

When Claude reads this skill, it should:

1. Identify which reference file(s) are relevant to the current task
2. Read those files before writing any code or making design decisions
3. Treat game data answers as ground truth
4. Treat `> **EXPERT**` blocks as open questions requiring human input

### Retrieval Triggers

Read **scoring.md** when:
- Implementing or reviewing the scoring engine
- Designing the action space or candidate hand ranking
- Writing reward shaping logic related to chip progress

Read **observation.md** when:
- Designing the observation encoder or Gymnasium env
- Deciding what features to include/exclude from state encoding

Read **economy.md** when:
- Implementing shop-phase reward shaping
- Designing gold/interest features in the observation

Read **jokers_principles.md** + **jokers_specific.md** when:
- Implementing joker trigger logic
- Selecting joker loadouts for curriculum
- Designing build coherence reward terms

Read **blinds.md** when:
- Designing curriculum (which Antes, which order)
- Implementing boss blind mechanics
- Setting chip requirement targets

Read **deck.md** when:
- Implementing card modifier effects
- Designing consumable use logic
- Working on deck composition features

Read **strategy.md** when:
- Designing reward functions
- Implementing shop decision logic
- Building risk management heuristics

---

## Quick Reference

### Scoring Formula
```
Score = (Base Chips + Card Chips + Chip Additions)
      x (Base Mult + Mult Additions)
      x Mult Multipliers
```
- **Base Chips/Mult**: From hand type at current level (see table below).
- **Card Chips**: Sum of rank values of scoring cards (A=11, Face/T=10, else face value).
- **Chip/Mult Additions**: From card enhancements, editions, and joker effects.
- **Mult Multipliers**: From xMult jokers, Glass enhancement, Polychrome edition.
- **Resolution**: Left-to-right on scoring cards, then held cards, then jokers.
- **Order matters**: +Mult jokers left of xMult jokers maximizes score.

### Hand Base Values and Level-Up Increments

From `jackdaw/engine/data/hands.py`. At level L: `chips = s_chips + l_chips*(L-1)`, `mult = s_mult + l_mult*(L-1)`.

| Hand | s_chips | s_mult | l_chips | l_mult |
|------|---------|--------|---------|--------|
| High Card | 5 | 1 | +10 | +1 |
| Pair | 10 | 2 | +15 | +1 |
| Two Pair | 20 | 2 | +20 | +1 |
| Three of a Kind | 30 | 3 | +20 | +2 |
| Straight | 30 | 4 | +30 | +3 |
| Flush | 35 | 4 | +15 | +2 |
| Full House | 40 | 4 | +25 | +2 |
| Four of a Kind | 60 | 7 | +30 | +3 |
| Straight Flush | 100 | 8 | +40 | +4 |
| Five of a Kind | 120 | 12 | +35 | +3 |
| Flush House | 140 | 14 | +40 | +4 |
| Flush Five | 160 | 16 | +50 | +3 |

### Card Chip Values
- Number cards: face value (2=2, ..., 9=9)
- Ten, Jack, Queen, King: 10
- Ace: 11
- Stone card: 50 (no rank/suit)

### Most Important Facts for RL (fill in)
> **Q65: When deciding which 5 cards to play from 8, what's the decision hierarchy?**
>
> *[EXPERT — highest priority question]*

> **Q67: When is discarding strictly better than playing a weak hand?**
>
> *[EXPERT]*

> **Q100: Top 5 most important features in the observation space?**
>
> *[EXPERT — see proposed ranking in observation.md]*

> **Q8: What are the chip requirements at Ante 1 (small/big/boss)?**
>
> White Stake: 300 / 450 / 600 (VERIFY exact values).
> Other stakes: same base table except GREEN and PURPLE which scale faster.

> **Q47: When does +mult beat xmult?**
>
> +N beats xK when N > current_mult x (K - 1).
> At mult=8, +4 = x1.5. At mult>8, xMult dominates.
