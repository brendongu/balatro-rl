# Observation Space

Questions that directly shape `observation.py` and the Gymnasium env spec.

---

**Q100: What information is absolutely essential in the observation space?**
*(If the agent can't see X, it will make systematically wrong decisions)*

*[Your answer here]*

---

**Q101: What information is redundant or derivable from other features?**
*(Things that look important but can be computed from simpler inputs)*

*[Your answer here]*

---

**Q102: What information do you use when making decisions that would be hard to encode?**
*(Intuitions, vibes, pattern recognition that's hard to express as a feature)*

*[Your answer here]*

---

**Q103: How far ahead do you plan?**
*(Current hand only? Rest of the round? Rest of the Ante? Full run?)*
*(Does this change by game phase — early Ante vs. late Ante?)*

*[Your answer here]*

---

**Q104: What do you track mentally that the game doesn't display explicitly?**
*(e.g., "how many face cards are left in the deck", "what's my expected gold next round")*

*[Your answer here]*

---

## Derived Features Worth Computing

> After answering above, list derived features that are worth pre-computing
> before feeding to the model (i.e., things you compute mentally that
> the raw game state doesn't directly expose).

*[Your notes here — e.g., "flush draw completeness", "chips_remaining / chips_needed ratio"]*

---

## Observation Space Spec Draft

> Fill this in after answering the questions. This becomes the spec for
> `observation.py`.

```
# Phase 2 observation (hand selection only, fixed Jokers)
Hand cards (up to 8):
  - rank (one-hot or ordinal)
  - suit (one-hot)
  - enhancement (one-hot)
  - edition (one-hot)

Game state scalars:
  - chips_needed         (normalize by ante)
  - chips_scored_so_far  (normalize by chips_needed)
  - hands_remaining
  - discards_remaining
  - ante_number
  - round_number (1=small, 2=big, 3=boss)

Candidate hand features (pre-computed by hand_evaluator):
  - [TBD based on Q100–104 answers]

Active Jokers (Phase 2 — fixed, so could be constants):
  - [TBD]
```
