# Jokers — General Principles

Reference for joker taxonomy, synergy reasoning, valuation, and loadout design.
Applies across all phases of the environment.

---

## Categorization (Q19–24)

**Q19: What taxonomy should we use for jokers in code?**

Proposed taxonomy derived from the 150 joker effects in the balatrobot API.
Each joker has a **primary category** and an optional **trigger condition**.

### Primary Categories

| Category | Code Enum | Description | Example Jokers |
|----------|-----------|-------------|----------------|
| Additive Chips | `CHIPS_ADD` | Flat +chips to score | j_sly (+50 if Pair), j_banner (+30/discard), j_ice_cream (+100 decaying) |
| Additive Mult | `MULT_ADD` | Flat +mult to score | j_joker (+4), j_jolly (+8 if Pair), j_fibonacci (+8 per A/2/3/5/8) |
| Multiplicative Mult | `MULT_X` | ×N mult to score | j_duo (×2 if Pair), j_tribe (×2 if Flush), j_loyalty_card (×4 every 6 hands) |
| Scaling | `SCALER` | Value changes over time | j_green_joker (+1 mult/hand, -1/discard), j_ride_the_bus, j_ice_cream |
| Retrigger | `RETRIGGER` | Re-fires card scoring | j_hack (retrigger 2/3/4/5), j_dusk (retrigger all on final hand) |
| Copy | `COPY` | Duplicates another joker's ability | j_blueprint (copies right), j_brainstorm (copies leftmost) |
| Economy | `ECONOMY` | Generates gold | j_golden ($4/round), j_delayed_grat ($2/discard if none used) |
| Generation | `GENERATOR` | Creates cards (Tarot, Planet, etc.) | j_8_ball, j_cartomancer, j_superposition |
| Utility | `UTILITY` | Modifies game rules | j_four_fingers (4-card flush/straight), j_splash (all cards score) |

### Trigger Conditions

| Trigger | Code Enum | When It Fires |
|---------|-----------|---------------|
| Always | `INDEPENDENT` | Every hand, no condition |
| On card scored | `ON_CARD_SCORED` | Per scoring card, left to right |
| On hand type | `ON_HAND_TYPE` | If played hand matches specific type |
| On held card | `ON_HELD` | Per card held in hand (not played) |
| On discard | `ON_DISCARD` | When cards are discarded |
| End of round | `END_OF_ROUND` | After final hand of round |
| On blind select | `ON_BLIND_SELECT` | When a blind is chosen |
| Passive | `PASSIVE` | Always active, modifies rules |

> **EXPERT**: Does this taxonomy capture your mental model? Any categories
> missing or that should be split differently?

---

**Q20: Which Jokers are universally strong regardless of build?**

*(The "never wrong to take" Jokers — important for loadout sampling baseline.)*

> **EXPERT**

---

**Q21: Which Jokers are build-dependent and require specific support?**

*(Jokers that are 0 or hero depending on context — important for coherence
modeling.)*

> **EXPERT**

---

**Q22: Which Jokers have exponential/compounding scaling potential?**

From game data, these jokers accumulate value over time:

| Joker | Growth Mechanic | Cap |
|-------|----------------|-----|
| j_green_joker | +1 mult per hand, -1 per discard | Uncapped (net +1/hand if no discards) |
| j_ride_the_bus | +1 mult per consecutive hand without scoring face card | Resets on face card |
| j_runner | +15 chips if played hand contains Straight | Uncapped |
| j_ice_cream | +100 chips, -5 per hand played | Decays to 0 |
| j_constellation | +×0.1 mult per Planet card used | Uncapped |
| j_vampire | +×0.1 mult per Enhanced card scored (removes enhancement) | Limited by enhanced cards |
| j_obelisk | +×0.2 mult per hand without playing most-played type | Resets when most-played type played |
| j_hologram | +×0.25 mult per card added to deck | Uncapped |
| j_lucky_cat | +×0.25 mult per Lucky card trigger | Uncapped |
| j_campfire | +×0.25 mult per card sold | Resets on Boss Blind defeat |
| j_glass | +×0.75 mult per Glass card destroyed | Uncapped |
| j_flash | +2 mult per shop reroll | Uncapped |
| j_popcorn | +20 mult, -4 per round | Decays to 0 |
| j_red_card | +3 mult per Booster Pack skipped | Uncapped |
| j_wee | +8 chips per 2 scored | Uncapped |
| j_caino | +×1 mult per face card destroyed | Uncapped |
| j_yorick | +×1 mult per 23 cards discarded | Uncapped |

> **EXPERT**: Which of these actually carry late-game runs vs being theoretical
> but impractical?

---

**Q23: Which Jokers are noob traps or RL policy traps?**

*(High variance, misleading short-term value, or anti-synergistic with common
strategies.)*

> **EXPERT**

---

**Q24: Which Jokers are early-game vs. late-game focused?**

> **EXPERT**

---

## Synergies (Q25–30)

**Q25: What are the most powerful 2-Joker synergies?**

> **EXPERT**

---

**Q26: What are the most powerful 3+ Joker combinations (archetypes)?**

> **EXPERT**

---

**Q27: Which Jokers anti-synergize or dilute each other?**

> **EXPERT**

---

**Q28: How does Joker order matter?**

From game data, joker effects resolve **left to right**. This matters because:

1. **+Mult before ×Mult**: A +Mult joker left of a ×Mult joker amplifies the
   multiplication. Reversing the order wastes the ×Mult on a lower base.

2. **Position-dependent jokers**:
   - `j_blueprint`: Copies ability of joker **to the right**.
   - `j_brainstorm`: Copies ability of **leftmost** joker.
   - `j_ceremonial`: Destroys joker **to the right** when blind is selected.

3. **Retrigger order**: Retrigger jokers fire in card-scoring order (left to
   right on scored cards), then joker order for independent effects.

4. **Scoring card order**: When multiple jokers trigger `ON_CARD_SCORED`, each
   card is fully resolved (all joker triggers) before moving to the next card.

General ordering rule:
```
[+Chips jokers] → [+Mult jokers] → [Conditional ×Mult] → [Universal ×Mult] → [Blueprint/Brainstorm last if copying ×Mult]
```

> **EXPERT**: Does this match your ordering intuition? Any nuances for specific
> joker pairs where the "standard" order is wrong?

---

**Q29: What makes a "coherent build" vs. a "pile of Jokers"?**

> **EXPERT**

---

**Q30: When do you abandon a build direction and pivot?**

> **EXPERT**

---

## Valuation (Q31–35)

**Q31: How do you value a Joker (in the shop or abstractly)?**

> **EXPERT**

---

**Q32: At what point is a Joker "good enough" vs. worth holding out for better?**

> **EXPERT**

---

**Q33: How does Ante timing (early/mid/late) change Joker valuation?**

> **EXPERT**

---

**Q34: When is it correct to sell or replace a previously acquired Joker?**

> **EXPERT**

---

**Q35: How many Joker slots should be filled by each Ante?**

| Ante | Target Joker Count | Notes |
|------|-------------------|-------|
| 1 | > **EXPERT** | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5+ | | |

---

## Starter Joker Loadout Recommendation

> After answering the above, recommend a curated set for initial training.
> Goal: simple jokers, diverse triggers, no niche build dependencies.
> Include both a minimal set (3–5) and a broader set (~25) for curriculum.

*[Your recommendation here]*
