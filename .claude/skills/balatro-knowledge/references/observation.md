# Observation Space

Reference for designing the Gymnasium observation spec across all phases.

---

## Observation Features (Q100–104)

**Q100: What information is essential in the observation space?**

*(Rank by importance. For each, explain what failure mode occurs if omitted.)*

| Rank | Feature | Failure Mode if Omitted |
|------|---------|------------------------|
| 1 | Current hand cards (rank, suit, enhancements, editions, seals) | Cannot evaluate any play |
| 2 | Chips remaining to beat blind (target - scored) | Cannot assess urgency |
| 3 | Hands remaining | Cannot budget play/discard decisions |
| 4 | Discards remaining | Cannot decide discard vs play |
| 5 | Active jokers + their current state | Cannot evaluate trigger value of plays |
| 6 | Remaining deck composition | Cannot estimate draw quality after discard |
| 7 | Hand type levels (12 types) | Will mis-estimate hand scores |
| 8 | Blind type (small/big/boss) and boss effect | Cannot adapt to boss mechanics |
| 9 | Ante number | Cannot calibrate scoring expectations |
| 10 | Cards played/discarded this round | Cannot track boss blind constraints (Eye, Mouth, Pillar) |

> **EXPERT**: The most important for initial training are 1-7. 

---

**Q101: What information is redundant or derivable?**

| Candidate Feature | Derivation | Include? |
|-------------------|-----------|----------|
| Best possible hand type from current hand | Computed from hand cards | Pre-compute as derived feature |
| Flush draw completeness (e.g., 4/5 Hearts) | Computed from hand cards | Pre-compute |
| Straight draw completeness | Computed from hand cards | Pre-compute |
| Expected score of best play | Computed from hand + jokers + levels | Pre-compute (expensive but high value) |
| Probability of improving hand after discard | Computed from hand + remaining deck | Pre-compute (approximation OK) |
| Number of face cards in hand | Counted from hand cards | Redundant if NN has hand cards |
| Suit distribution in hand | Counted from hand cards | Redundant if NN has hand cards |

> **EXPERT**: Which derived features do you think are worth pre-computing vs
> letting the neural network learn to derive them? Any features you compute
> mentally that aren't on this list?

---

**Q102: What expert heuristics feel "hard to encode"?**

*(Intuitions that would benefit from a proxy feature.)*

| Heuristic | Possible Proxy Feature |
|-----------|----------------------|
| "My deck is good for flushes" | Suit concentration ratio in remaining deck |
| "I'm on pace / behind pace" | chips_scored / chips_needed vs hands_used / total_hands |
| "This joker is about to pay off" | Scaling joker accumulated value / rounds remaining |
| "I should save discards for later" | ??? |

> **EXPERT**: What other intuitions do you rely on? How would you express them
> as computable quantities?

---

**Q103: How far ahead do you plan?**

*(Current hand only? Full round? Full ante? Entire run? Does this change by
game phase?)*

> **EXPERT**

---

**Q104: What latent quantities do expert players track mentally?**

From game mechanics, these are trackable but not displayed:

| Quantity | How to Compute | Strategic Value |
|----------|---------------|-----------------|
| Cards remaining by suit | Full deck - played - discarded | Flush draw probability |
| Cards remaining by rank | Full deck - played - discarded | Pair/set probability |
| Face cards remaining in deck | Count J/Q/K in remaining | Ride the Bus, face card jokers |
| Expected cards drawn on next discard | hand_size - current_hand + discard_count | Action value estimation |
| Hands played this round (for Card Sharp) | Round state counter | ×3 mult on repeat hand types |
| Joker trigger probability | Joker conditions vs hand/deck state | Action selection |

> **EXPERT**: What else do you track mentally that isn't on this list?

---

## Derived Features Worth Pre-Computing

> List features with high signal-to-compute ratio for the observation vector.

| Feature | Formula | Relevance |
|---------|---------|-----------|
| `chips_progress` | chips_scored / chips_needed | Urgency signal |
| `hands_urgency` | (chips_needed - chips_scored) / hands_remaining | "chips per hand needed" |
| `best_hand_type` | hand_evaluator.best_play(hand).hand_type | Strongest available play |
| `best_hand_score` | hand_evaluator.best_play(hand).estimated_chips | Expected score of greedy play |
| `flush_draw_count` | max suits in hand (counting wilds) | Flush potential |
| `straight_draw_count` | longest consecutive rank run | Straight potential |
| `suit_density_remaining[4]` | count per suit in remaining deck | Draw quality by suit |
| `rank_density_remaining[13]` | count per rank in remaining deck | Draw quality by rank |

---

## Jackdaw Observation Encoding (Implemented)

Jackdaw provides a Dict observation with these keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `global` | (235,) | Fixed-size context vector |
| `hand_card` | (8, 15) | Hand cards (padded, 15 features each) |
| `joker` | (5, 15) | Active jokers (padded) |
| `consumable` | (2, 7) | Consumable cards |
| `shop_item` | (10, 9) | Shop items |
| `pack_card` | (5, 15) | Booster pack cards |
| `entity_counts` | (5,) | Actual count per entity type |

### Global Vector Layout (235 dims)

| Range | Dims | Content |
|-------|------|---------|
| [0:6] | 6 | Phase one-hot (BLIND_SELECT, SELECTING_HAND, ROUND_EVAL, SHOP, PACK_OPENING, GAME_OVER) |
| [6:10] | 4 | Blind-on-deck one-hot (None, Small, Big, Boss) |
| [10:30] | 20 | Scalar features (ante/8, round/30, log(dollars), hands_left/10, discards_left/10, hand_size/15, joker_slots/10, consumable_slots/5, log(blind_chips), log(chips), chips/blind ratio, reroll_cost/10, free_rerolls/5, interest_cap/100, discount_pct/50, skips/10, blind_key_id, log(deck_size), log(discard_pile_size), voucher flags) |
| [30:90] | 60 | Hand levels: 12 types x 5 features (level/20, log(chips), log(mult), log(played), visible) |
| [90:122] | 32 | Vouchers owned (binary) |
| [122:130] | 8 | Blind effect: boss, disabled, mult/4, debuff_suit(4), debuff_face |
| [130:133] | 3 | Round position one-hot (small/big/boss) |
| [133:135] | 2 | Round progress (hands_played/10, discards_used/10) |
| [135:159] | 24 | Awarded tags (binary) |
| [159:211] | 52 | Discard pile histogram (4 suits x 13 ranks) |
| [211:223] | 12 | Hand type indicators for current hand |
| [223:226] | 3 | Base score estimate: log(chips), log(mult), log(score) |
| [226] | 1 | Score-to-blind ratio |
| [227:229] | 2 | Draw proximity: flush, straight (continuous 0-1) |
| [229] | 1 | Interest earned |
| [230:235] | 5 | Reserved/padding |

### Our Augmented Features (+10 dims via ObservationAugmentWrapper)

| Index | Feature | Derivation |
|-------|---------|------------|
| +0 | chips_progress | chips_scored / chips_needed (0-1) |
| +1 | hands_urgency | chips_remaining / (hands_remaining × best_estimate) |
| +2 | flush_proximity | Re-scaled from jackdaw global[227] |
| +3 | straight_proximity | Re-scaled from jackdaw global[228] |
| +4:+8 | suit_density[4] | Remaining cards per suit / total remaining |
| +8 | economy_health | dollars / (ante × 10), clamped |
| +9 | hand_size_ratio | Current hand count / hand size limit |

### Previous Draft Spec (Superseded)

The draft below was written before adopting jackdaw. Retained for reference.

```
Hand cards (padded to max_hand_size):
  Per card:
    - rank: ordinal 0-12 (2..A) or one-hot(13)
    - suit: one-hot(4)
    - enhancement: one-hot(9) [none + 8 types]
    - edition: one-hot(5) [none + 4 types]
    - seal: one-hot(5) [none + 4 types]
    - debuff: binary(1)
    - present: binary(1) [padding mask]
  Total per card: ~37 features

Game state scalars:
  - chips_needed, chips_scored_so_far, hands/discards remaining
  - ante_number, blind_type (one-hot), hand levels (12 floats)

Active Jokers, Deck context, Derived features (see above)
```
