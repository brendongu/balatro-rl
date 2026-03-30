# Scoring & Hand Selection

Reference for `hand_evaluator.py`, action space design, reward shaping, and
the scoring engine. Game data sourced from `jackdaw/engine/data/hands.py`
and `jackdaw/engine/data/blind_scaling.py`.

---

## Scoring Formula (Game Data)

```
Score = (Base Chips + Card Chips + Chip Additions) × (Base Mult + Mult Additions) × Mult Multipliers
```

### Resolution Order

The game resolves scoring in this exact sequence:

1. **Determine hand type** from played cards. Identify which cards are "scoring"
   (contribute to the hand type) vs kickers.
2. **Initialize**: chips = hand_base_chips, mult = hand_base_mult (from hand type + level).
3. **For each scoring card** (left to right in play order):
   a. Add card's rank chip value to chips.
   b. Apply card **enhancement** effects (BONUS: +30 chips, MULT: +4 mult, GLASS: ×2 mult, LUCKY: 1/5 chance +20 mult).
   c. Apply card **edition** effects (FOIL: +50 chips, HOLO: +10 mult, POLYCHROME: ×1.5 mult).
   d. Apply card **seal** (RED: retrigger this card — repeat steps a–c once more).
   e. For each joker (left to right), check if it triggers **on this card scoring** (e.g., Greedy Joker: +3 mult if Diamond scored). Apply effect.
4. **For each held-in-hand card** (not played):
   a. Apply held effects (STEEL: ×1.5 mult).
   b. For each joker, check held-in-hand triggers (e.g., Baron: each held King ×1.5 mult).
5. **For each joker** (left to right), apply **independent effects**:
   - +Chips effects (e.g., Banner: +30 chips per remaining discard)
   - +Mult effects (e.g., Joker: +4 mult; Jolly: +8 mult if Pair)
   - ×Mult effects (e.g., Duo: ×2 if Pair; Loyalty Card: ×4 every 6 hands)
6. **Final score** = chips × mult (both are accumulated through steps above).

**Critical**: ×Mult jokers multiply whatever mult has accumulated *at the point they trigger*. A +Mult joker to the left of a ×Mult joker is strictly better than the reverse order.

---

## Chip Requirement Scaling (Q8–9)

**Q8: What are the chip requirements at each Ante?**

The game uses a **base score table** indexed by ante. Within each ante, the
three blind types apply a fixed multiplier to the base:

| Blind Type | Multiplier |
|------------|-----------|
| Small      | 1×        |
| Big        | 1.5×      |
| Boss       | 2×        |

### White Stake Base Scores

| Ante | Base (Small) | Big    | Boss    |
|------|-------------|--------|---------|
| 1    | 300         | 450    | 600     |
| 2    | 800         | 1,200  | 1,600   |
| 3    | 2,000       | 3,000  | 4,000   |
| 4    | 5,000       | 7,500  | 10,000  |
| 5    | 11,000      | 16,500 | 22,000  |
| 6    | 20,000      | 30,000 | 40,000  |
| 7    | 35,000      | 52,500 | 70,000  |
| 8    | 50,000      | 75,000 | 100,000 |

**Q9: How does the requirement scale across stakes?**

From `jackdaw/engine/data/blind_scaling.py`, the game uses 3 hardcoded scaling
levels (not per-stake multipliers applied to a single table):

| Ante | Scaling 1 (White/Red) | Scaling 2 (Green-Blue) | Scaling 3 (Purple-Gold) |
|------|----------------------|----------------------|------------------------|
| 1    | 300                  | 300                  | 300                    |
| 2    | 800                  | 900                  | 1,000                  |
| 3    | 2,000                | 2,600                | 3,200                  |
| 4    | 5,000                | 8,000                | 9,000                  |
| 5    | 11,000               | 20,000               | 25,000                 |
| 6    | 20,000               | 36,000               | 60,000                 |
| 7    | 35,000               | 60,000               | 110,000                |
| 8    | 50,000               | 100,000              | 200,000                |

Antes 9+ use an exponential formula: `floor(a8 * (1.6 + (0.75*c)^d)^c)` where
`c = ante - 8`, `d = 1 + 0.2*c`, `a8 = ante_8_base`.

Implementation: `jackdaw.engine.data.blind_scaling.get_blind_target(ante, blind_type, scaling)` handles this directly. The `scaling` parameter maps:
- Scaling 1: White, Red stakes
- Scaling 2: Green, Blue stakes  
- Scaling 3: Purple, Orange, Gold stakes

---

## Mult vs. Chips Tradeoffs (Q47–48)

**Q47: When does additive +Mult beat ×Mult, and vice versa?**

The math is straightforward. Given current accumulated mult `M`:
- **+N mult** adds `chips × N` to the final score.
- **×K mult** adds `chips × M × (K - 1)` to the final score.

Breakpoint: +N beats ×K when `N > M × (K - 1)`.

| Current Mult (M) | +4 Mult value | ×1.5 Mult value | ×2 Mult value |
|-------------------|--------------|-----------------|---------------|
| 4                 | +4           | +2              | +4            |
| 8                 | +4           | +4              | +8            |
| 10                | +4           | +5              | +10           |
| 20                | +4           | +10             | +20           |

**Rule of thumb**: +Mult is better when accumulated mult is low (early game,
few jokers). ×Mult dominates once mult exceeds ~8–10. This is why ×Mult jokers
are the highest-value targets in mid/late game.

> **EXPERT**: Assume that non-scaling +Mult jokers contribute 6-15 Mult. ×Mult jokers are high-value but also more uncommon so in early game it is more likely to use 2-3 +Mult jokers until a ×Mult joker is available. 

---

**Q48: How do you estimate the marginal value of an additional +Mult or ×Mult source?**

Worked example with the scoring formula:

Setup: Flush (base 35 chips, 4 mult at level 1), playing AKQJT of Hearts.
Card chips: 11 + 10 + 10 + 10 + 10 = 51.
Total chips = 35 + 51 = 86. Mult = 4. Base score = 86 × 4 = **344**.

- Adding j_joker (+4 mult): 86 × 8 = **688** (+344, or +100%)
- Adding j_droll (+10 mult if Flush): 86 × 14 = **1,204** (+860, or +250%)
- Adding j_duo (×2 if Pair) — doesn't trigger on Flush: **344** (+0)
- Adding j_tribe (×2 if Flush): 86 × 8 = **688** (same as +4 mult at this point)

With j_joker already present (mult = 8):
- Adding j_tribe (×2 if Flush): 86 × 16 = **1,376** (+688, or +100%)
- Adding another +4 mult: 86 × 12 = **1,032** (+344, or +50%)

The ×2 provides +100% regardless of current mult, while +4 provides diminishing
returns as mult grows.

> **EXPERT**: What mental shortcuts do you use at the shop for this calculation?

---

## Hand Selection (Q65–70)

**Q65: Given a hand of 8 cards, what is your decision hierarchy for choosing which cards to play?**

*(Include tie-breakers, when you intentionally play <5 cards, and how the
hierarchy shifts with different joker loadouts.)*

> **EXPERT** The first priority is always to beat the blind. This may require playing enhanced cards or rarer hards with higher base chips. If the blind can be beaten comfortably, the goal is to generate as much value out of the ante as possible - e.g. scaling jokers, economy joker synergies, discarding for purple seals, blue seals, gold seals, gold cards, lucky cards. Generally, for rarer hands, prefer to discard first and search for high-value cards as most of the time one hand can clear the blind. For smaller hands like high card and pair, the specific cards don't matter as much as the jokers and more than one hand is required. Prefer to play 5 cards if playing a high card to get rid of useless cards in the hand. For pair and two pair, it may be preferable to retain a pair in hand to ensure the cards in hand will contain a playable hand. 

---

**Q66: When do you play for Joker triggers rather than raw chip maximization?**

*(e.g., playing a weaker hand type that activates more joker effects. At what
score margin does trigger value outweigh hand type value?)*

> **EXPERT**

---

**Q67: When is discarding strictly better than playing a weak hand?**

*(Express as if/then rules using chips_needed, hands_left, discards_left, and
draw odds where possible.)*

> **EXPERT** In a vacuum, prefer to use as few hands as possible to beat the blind due to the additional money from unused hands versus no additional money for unused discards. Prefer discarding unless there is a joker affected by discards (Banner, Delayed Gratification, Green Joker, Ramen). 

---

**Q68: How many discards do you typically use per round, and how does this vary by Ante?**

> **EXPERT**

---

**Q69: What is the typical intra-round sequence of decisions (discard-first vs play-first)?**

*(e.g., always discard first to fish for better hands? Play first to see if you
beat the blind? How does this vary?)*

> **EXPERT**

---

**Q70: How do boss blind mechanics change hand selection?**

*(Examples: The Psychic forces 5-card plays, The Eye forbids repeating hand
types, The Flint halves base chips/mult. What are the biggest strategic shifts?)*

> **EXPERT**

---

## Implementation Notes

> Fill in after answering expert questions above.
> What simplifications or approximations should the scoring engine make?
> What should the reward function prioritize?

*[Your notes here]*
