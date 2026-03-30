# Strategy & Risk Management

Reference for reward design, shop decision logic, build coherence signals, and
risk management heuristics.

---

## Shop Decisions (Q71–75)

**Q71: How do you prioritize shop purchases?**

*(Your actual priority order and the reasoning behind it.)*

> **EXPERT**

---

**Q72: When is rerolling worth it vs. saving gold?**

> **EXPERT**

---

**Q73: How does your current gold buffer affect risk tolerance in the shop?**

> **EXPERT**

---

**Q74: When do you skip the shop entirely?**

> **EXPERT**

---

**Q75: How do you decide between buying now vs. waiting for the next shop?**

> **EXPERT**

---

## Build Coherence (Q76–80)

**Q76: At what point in a run do you "commit" to a build direction?**

*(e.g., "by Ante 2 I know if I'm going flush or pairs." What are the signals?)*

> **EXPERT**

---

**Q77: How do you recognize when a build isn't working and needs to pivot?**

> **EXPERT**

---

**Q78: What are the signs a build will scale well vs. hit a ceiling?**

*(Important for reward design: should we reward builds with ×Mult more than
+Mult builds because they scale further?)*

> **EXPERT**

---

**Q79: When do you hedge (keep build flexible) vs. specialize (all-in on one strategy)?**

> **EXPERT**

---

**Q80: What's the minimum viable build for each Ante?**

*(What does a build need to look like to survive? This defines the "floor" for
reward shaping.)*

| Ante | Minimum to Survive |
|------|-------------------|
| 1 | > **EXPERT** |
| 2–3 | |
| 4–5 | |
| 6–8 | |

---

## Risk Management (Q81–85)

**Q81: When should you play conservatively vs. aggressively?**

*(What game-state signals trigger the switch?)*

> **EXPERT**

---

**Q82: How do you evaluate high-variance strategies?**

*(e.g., Glass cards with 1/4 destruction chance, Lucky cards with 1/5 +20 mult.
When is the EV worth the variance?)*

> **EXPERT**

---

**Q83: When is it correct to take a risky Joker that might brick your run?**

*(e.g., Madness destroys a random joker for ×0.5 mult. Hex destroys all jokers
for Polychrome on one. Under what conditions are these +EV?)*

> **EXPERT**

---

**Q84: Consistent small scaling vs. boom-or-bust — when do you choose each?**

> **EXPERT**

---

**Q85: Which blinds/situations are "must not lose" where you play extra safe?**

> **EXPERT**

---

## Reward Shaping Implications

> After answering the above, note what these answers imply for reward design.

| Expert Insight | Reward Implication |
|---------------|-------------------|
| Q76 (build commitment timing) | Reward coherent joker sets more at higher antes |
| Q78 (scaling signals) | Reward ×Mult acquisition more than +Mult after ante 3 |
| Q80 (minimum viable build) | Use as threshold for curriculum advancement |
| Q81 (conservative/aggressive) | Hands_remaining < 2 should trigger risk-averse play |
| Q85 (must-not-lose blinds) | Boss blinds could have slightly higher loss penalty |

> **EXPERT**: Fill in after answering questions above.
