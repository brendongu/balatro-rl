# Economy & Resources

Reference for gold-related features, reward shaping, and shop decision logic.

---

## Gold Economy Mechanics (Game Data)

### Starting Gold

| Deck | Starting Gold |
|------|--------------|
| Default (most decks) | $4 |
| Yellow Deck | $14 (+$10 extra) |

### Round Rewards

After beating a blind, the player earns:

| Reward Type | Amount | Notes |
|-------------|--------|-------|
| Small Blind win | $3 | $0 on RED stake and above |
| Big Blind win | $4 | Always awarded |
| Boss Blind win | $5 | Always awarded |
| Interest | floor(gold / 5), max $5 | Paid every round |
| Remaining hands | Depends on deck | GREEN deck: $2 per remaining hand |
| Remaining discards | Depends on deck | GREEN deck: $1 per remaining discard |

### Interest Formula

```
interest = min(floor(current_gold / 5), cap)
```

- **Base cap**: $5 per round (at $25+ gold, you earn max interest).
- **Seed Money voucher**: Raises cap to $10 per round.
- **Money Tree voucher**: Raises cap to $20 per round.
- **GREEN deck**: No interest. Instead earns $2/hand + $1/discard remaining.

### Gold Sinks

| Sink | Cost | Notes |
|------|------|-------|
| Shop cards (Jokers) | $2–$8 (common–rare) | Varies by rarity and edition |
| Shop consumables | $3–$6 | Tarot, Planet, Spectral |
| Booster packs | $4–$8 | Standard, Arcana, Celestial, Buffoon, Spectral |
| Vouchers | $10 (tier 1), $10 (tier 2) | One per shop visit |
| Reroll shop | $5 (base) | Reduced by Reroll Surplus/Glut vouchers |
| Rental jokers | $1/round | Ongoing cost |

> **VERIFY**: Exact shop prices vary. These are base prices before voucher
> discounts (Clearance Sale: -25%, Liquidation: -50%).

---

**Q14: What is the typical gold progression by Ante?**

*(How much gold do you typically have at the start of each Ante's shop? What's
a "rich" run vs "poor" run look like?)*

> **EXPERT**

---

**Q15: When should you prioritize economy (saving hands/discards, banking gold) vs. safety (spending all resources to beat the blind)?**

> **EXPERT**

---

**Q16: How valuable is each additional hand remaining at end of round?**

From game data: GREEN deck earns $2/hand remaining. Other decks get no direct
hand-to-gold conversion, but:
- Fewer hands used → faster round → no direct gold benefit.
- Saved hands don't carry over between rounds.

> **EXPERT**: Are there indirect benefits (tempo, tag skipping, etc.) that make
> hand conservation valuable even on non-GREEN decks?

---

**Q17: When is it ever correct to intentionally lose a blind?**

> **EXPERT**

---

**Q18: What's the typical shop reroll strategy at different Antes?**

> **EXPERT**

---

## Economy Features for Observation Space

Relevant features (all phases):

| Feature | Source | Relevance |
|---------|--------|-----------|
| `current_gold` | gamestate.money | Direct purchasing power |
| `interest_next_round` | min(floor(gold/5), cap) | Opportunity cost of spending |
| `round_reward` | blind_type + stake | Expected income |
| `reroll_cost` | gamestate.round.reroll_cost | Shop action cost |
| `joker_sell_values` | sum of joker sell prices | Emergency liquidity |

For hand-play-only env, gold is not directly relevant (no shop). Include it
only when shop phase is modeled.
