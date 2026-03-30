# Blinds & Antes

Reference for curriculum design, blind target scaling, and boss blind mechanics.
Game data sourced from `jackdaw/engine/data/blinds.json` and
`jackdaw/engine/data/blind_scaling.py`.

---

## Chip Requirements

See `scoring.md` for the full table. Summary:

- Deterministic lookup table indexed by (ante, blind_type, stake).
- Small = base, Big = 1.5× base, Boss = 2× base.
- WHITE/RED/BLACK/BLUE/ORANGE/GOLD stakes share the same base table.
- GREEN and PURPLE stakes apply additional scaling multipliers.
- Implementation: `blind_score(ante, blind_type, stake) -> int`

---

## Ante Structure (Game Data)

Each ante consists of 3 blinds in fixed order: Small → Big → Boss.
- Small and Big blinds can be **skipped** (grants a Tag reward instead).
- Boss blind **cannot be skipped** and has a special effect.
- Beating the Boss blind advances to the next ante.
- A standard run has 8 antes. Beating Ante 8 Boss wins the run.

Tags from skipping (examples from balatrobot API):
- Uncommon Tag: "Shop has a free Uncommon Joker"
- Various others that provide shop benefits, gold, or card generation.

> **EXPERT**: How often and under what conditions do you skip small/big blinds?
> Is there a general rule (e.g., "always skip small in Ante 1-3 if tag is good")?

---

## Boss Blinds (Q10–11)

### Known Boss Blind Effects (Game Data)

These are the boss blind effects from the game. Each boss blind appears
randomly when you reach the Boss round of an ante.

| Boss Blind | Effect | Impact on Hand Play |
|------------|--------|---------------------|
| The Psychic | Must play exactly 5 cards | Eliminates <5 card plays |
| The Eye | No hand type can repeat this round | Forces hand type diversity |
| The Mouth | Must play same hand type as first hand | Locks you into one type |
| The Needle | Only 1 hand this round | Must beat blind in a single hand |
| The Flint | Base chips and mult are halved | Raw scoring power cut in half |
| The Wall | Blind score is 4× base (instead of 2×) | Doubles effective target |
| The Club | All Club cards are debuffed | Suit removed from scoring |
| The Goad | All Spade cards are debuffed | Suit removed from scoring |
| The Head | All Heart cards are debuffed | Suit removed from scoring |
| The Window | All Diamond cards are debuffed | Suit removed from scoring |
| The Plant | All face cards are debuffed | Removes J/Q/K from scoring |
| The Hook | Discards 2 random cards from hand per hand played | Reduces hand quality |
| The Tooth | Lose $1 per card played | Economic pressure |
| The Arm | Decreases level of played hand type by 1 | Punishes leveled hands |
| The Ox | Lose all money when playing most played hand type | Economic punishment |
| The Wheel | 1 in 7 cards drawn face-down | Partial information loss |
| The Mark | All face cards drawn face-down | Can't see J/Q/K values |
| The Fish | Cards drawn face-down after play/discard | Progressive info loss |
| The House | All cards face-down on first hand | Blind first play |
| The Water | Start with 0 discards | No discards available |
| The Manacle | -1 hand size this round | 7-card hand instead of 8 |
| The Serpent | Draw 3 extra cards after play/discard | Larger hand, more options |
| The Pillar | Cards played previously this round are debuffed | Can't reuse cards |
| Amber Acorn | Flips and shuffles all jokers | Joker order disrupted |
| Cerulean Bell | Forces 1 specific card to always be selected | Constrains card choice |
| Crimson Heart | 1 random joker disabled each hand | Joker reliability reduced |
| Violet Vessel | ×6 base score (instead of ×2) | Massive score target |
| Verdant Leaf | All cards debuffed until a joker is sold | Forces joker sacrifice |

### Boss Blind Ante Ranges (from `blinds.json`)

Each boss blind has a `min` ante (earliest it can appear) and a `max` ante:

| Boss Blind | Min Ante | Mult | Debuff Config |
|------------|----------|------|---------------|
| The Hook | 1 | ×2 | — |
| The Club | 1 | ×2 | suit: Clubs |
| The Goad | 1 | ×2 | suit: Spades |
| The Window | 1 | ×2 | suit: Diamonds |
| The Manacle | 1 | ×2 | — |
| The Pillar | 1 | ×2 | — |
| The Psychic | 1 | ×2 | h_size_ge: 5 |
| The Head | 1 | ×2 | suit: Hearts |
| The Flint | 2 | ×2 | — |
| The Wall | 2 | ×4 | — |
| The House | 2 | ×2 | — |
| The Wheel | 2 | ×2 | — |
| The Water | 2 | ×2 | — |
| The Fish | 2 | ×2 | — |
| The Arm | 2 | ×2 | — |
| The Needle | 2 | ×1 | — |
| The Mouth | 2 | ×2 | — |
| The Eye | 3 | ×2 | — |
| The Tooth | 3 | ×2 | — |
| The Plant | 4 | ×2 | is_face: face |
| The Serpent | 5 | ×2 | — |
| The Ox | 6 | ×2 | — |
| Amber Acorn | 10 | ×2 | — (showdown) |
| Cerulean Bell | 10 | ×2 | — (showdown) |
| Crimson Heart | 10 | ×2 | — (showdown) |
| Verdant Leaf | 10 | ×2 | — (showdown) |
| Violet Vessel | 10 | ×6 | — (showdown) |

Note: The Needle has mult=1 (not ×2), meaning its base score is *lower* than
other bosses despite only allowing 1 hand. The Wall has mult=4 (double other bosses).

---

**Q10: Which boss blinds are most dangerous and why?**

*(Rank the top 5 most run-ending boss blinds based on your experience.)*

1. > **EXPERT**
2.
3.
4.
5.

---

**Q11: Which boss blinds require specific counter-strategies?**

*(For each: what's the correct hand selection / play pattern response?)*

| Boss Blind | Counter-Strategy |
|------------|-----------------|
| The Psychic | > **EXPERT** |
| The Eye | > **EXPERT** |
| The Needle | > **EXPERT** |
| The Flint | > **EXPERT** |
| The Plant | > **EXPERT** |
| Crimson Heart | > **EXPERT** |
| Verdant Leaf | > **EXPERT** |

---

## Run Failure Analysis (Q12–13)

**Q12: At what Ante do runs typically fail and what causes the failure?**

> **EXPERT**

---

**Q13: What's the typical "chip gap" where scaling requirements outpace average builds?**

> **EXPERT**

---

## Curriculum Design Notes

> After filling in the above, write the recommended curriculum progression.
> Format: Stage N → Antes X-Y, N Jokers, boss blinds Y/N, success criterion

```
Stage 1: ...
Stage 2: ...
Stage 3: ...
Stage 4: ...
```

> **EXPERT**

---

## Boss Blind Handling by Phase

> Which boss blinds should be enabled at each phase of environment development?

**Hand-play only (no shop/economy):**
- Safe to include: suit debuffs (Club/Diamond/Heart/Spade), The Psychic,
  The Eye, The Mouth, The Needle, The Flint, The Water, The Manacle.
  These only affect hand selection.
- Exclude: The Tooth, The Ox (require economy modeling), Verdant Leaf
  (requires sell action), Amber Acorn (requires joker rearrangement).

**Full game loop:**
- Include all boss blinds.

> **EXPERT**: Does this split make sense? Any boss blinds I've mis-categorized?
