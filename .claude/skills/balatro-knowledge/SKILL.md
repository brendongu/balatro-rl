---
name: balatro-knowledge
description: Expert domain knowledge about Balatro gameplay. Use whenever making decisions about observation space design, reward shaping, action space design, curriculum ordering, hand selection logic, Joker valuation, scoring calculations, or any other game mechanic. Also use when writing or reviewing any code that encodes Balatro rules or strategy.
---

# Balatro Domain Knowledge

Expert knowledge from 1000+ hours of gameplay, organized for RL project use.

---

## Reference Files

Read the relevant file before answering questions in that domain.

| Topic | File | Key Questions | Status |
|-------|------|---------------|--------|
| Scoring & Hand Selection | `references/scoring.md` | 8–9, 47–48, 65–70 | 🔲 Empty |
| Observation Space | `references/observation.md` | 100–104 | 🔲 Empty |
| Economy & Resources | `references/economy.md` | 14–18 | 🔲 Empty |
| Jokers — Principles | `references/jokers_principles.md` | 19–35 | 🔲 Empty |
| Jokers — Specific | `references/jokers_specific.md` | 36–49 | 🔲 Empty |
| Blinds & Antes | `references/blinds.md` | 8–13 | 🔲 Empty |
| Deck & Cards | `references/deck.md` | 50–64 | 🔲 Empty |
| Strategy & Risk | `references/strategy.md` | 71–85 | 🔲 Empty |

Update status to ✅ once a section has substantial answers.

---

## How to Use This Skill

When Claude reads this skill, it should:

1. Identify which reference file(s) are relevant to the current task
2. Read those files with the `view` tool before writing any code or making design decisions
3. Treat the answers as ground truth — they override generic poker/RL assumptions

### Retrieval Triggers

Claude should read **scoring.md** when:
- Implementing or reviewing `hand_evaluator.py`
- Designing the action space or candidate hand ranking
- Writing reward shaping logic related to chip progress

Claude should read **observation.md** when:
- Designing `observation.py` or `env.py`
- Deciding what features to include/exclude from state encoding
- Writing the Gymnasium observation space spec

Claude should read **economy.md** when:
- Implementing shop-phase reward shaping
- Designing gold/interest features in the observation
- Writing curriculum logic around Ante progression

Claude should read **jokers_principles.md** + **jokers_specific.md** when:
- Selecting the Phase 2 fixed Joker loadout
- Implementing Joker trigger detection
- Designing build coherence reward terms (Phase 3)

Claude should read **blinds.md** when:
- Designing curriculum (which Antes, which order)
- Implementing boss blind mechanics or counter-strategies
- Setting chip requirement targets in the reward function

---

## Quick Reference (filled in by you — 1–3 line answers only)

These are kept in SKILL.md for instant access without opening a reference file.
Fill these in first — they're the highest-leverage facts for the RL implementation.

### Scoring Formula
```
Score = (Base Chips + Card Value + Chip Additions)
      × (Base Multiplier + Mult Additions)
      × Mult Multipliers
```
- **Base Chips** are the base chips associated with the hand type.
- **Cards Value** is the sum of the ranks of the cards that are used in the hand type. Ace is worth 11 and face cards are worth 10.
- **Chip Additions** originate from jokers and card enhancements.
- **Base Multiplier** is the base multiplier associated with the hand type.
- **Mult Additions** originate from jokers and enhanced cards.
- **Mult Multipliers** originate from jokers and enhanced cards.

### Hand Base Values (Level 1)

| Hand | Chips | Mult |
|------|-------|------|
| High Card | 5 | 1 |
| Pair | 10 | 2 |
| Two Pair | 20 | 2 |
| Three of a Kind | 30 | 3 |
| Straight | 30 | 4 |
| Flush | 35 | 4 |
| Full House | 40 | 4 |
| Four of a Kind | 60 | 7 |
| Straight Flush | 100 | 8 |
| Five of a Kind | 120 | 12 |
| Flush House | 140 | 14 |
| Flush Five | 160 | 16 |

### Card Chip Values
- Number cards: face value (2=2, 9=9)
- Ten, Jack, Queen, King: 10
- Ace: 11
- Stone card: 50 (no rank/suit)

### Most Important Facts for RL (fill in)
> **When deciding which 5 cards to play from 8, what's the decision hierarchy?**
> 
> *[Your answer here]*

> **Q67: When is discarding strictly better than playing a weak hand?**
> 
> *[Your answer here]*

> **Q100: Top 5 most important features in the observation space?**
> 
> *[Your answer here]*

> **Q8: What are the chip requirements at Ante 1 (small/big/boss)?**
> 
> *[Your answer here]*

> **Q47: When does +mult beat ×mult?**
> 
> *[Your answer here]*
