# Economy & Resources

Questions that shape gold-related features in the observation space and
reward shaping around shop decisions (Phase 3).

---

**Q14: What is the gold economy progression?**
*(Starting gold, gold per blind, interest mechanics, how much you typically have at each Ante)*

*[Your answer here]*

---

**Q15: When should you prioritize economy (saving hands/discards) vs. safety (using all resources)?**
*(What's the tradeoff and when does it flip?)*

*[Your answer here]*

---

**Q16: How valuable is each additional hand remaining at end of round?**
*(In terms of gold via interest, ability to skip small blind, other effects)*

*[Your answer here]*

---

**Q17: When is it ever correct to intentionally lose a blind to preserve resources?**
*(Is this a real strategy? Under what conditions?)*

*[Your answer here]*

---

**Q18: What's the typical shop reroll strategy at different Antes?**
*(When do you reroll freely vs. conserve gold?)*

*[Your answer here]*

---

## Interest Mechanics

> Fill in the exact interest formula here once you've answered Q14.
> This goes directly into the reward function.

```
Interest = min(floor(gold / 5), cap) per round
Cap at base stake = $5 (upgradeable via Seed Money / Money Tree vouchers)
```

*[Confirm or correct the above, add any stake-level changes]*

---

## Economy Features for Observation Space

> Which economy quantities should be in the observation vector?
> (e.g., current_gold, expected_interest, rounds_until_shop)

*[Your notes here]*
