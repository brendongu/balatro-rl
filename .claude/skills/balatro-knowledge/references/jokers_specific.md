# Jokers — Specific Knowledge

Detailed reference on joker mechanics. Covers trigger encoding, scaling curves,
conditional requirements, economy valuation, and resolution order.

---

## Scoring Resolution Order (Q49)

**Q49: In what order do joker effects resolve during scoring?**

From game mechanics (see also `scoring.md` Resolution Order):

```
For each scoring card (left to right):
  1. Card base chips added
  2. Card enhancement applied (+chips, +mult, ×mult)
  3. Card edition applied (Foil, Holo, Polychrome)
  4. Card seal check (Red Seal → retrigger steps 1-3)
  5. Each joker checked left→right for ON_CARD_SCORED triggers

For each held-in-hand card:
  6. Held effects (Steel ×1.5)
  7. Each joker checked left→right for ON_HELD triggers

Independent joker effects (left→right):
  8. +Chips jokers
  9. +Mult jokers
  10. ×Mult jokers (multiply accumulated mult at this point)
```

### ×Mult Jokers (highest-value scoring targets)

These are all jokers from the API that apply multiplicative mult:

| Joker | ×Mult | Condition |
|-------|-------|-----------|
| j_duo | ×2 | Hand contains Pair |
| j_tribe | ×2 | Hand contains Flush |
| j_order | ×3 | Hand contains Straight |
| j_trio | ×3 | Hand contains Three of a Kind |
| j_family | ×4 | Hand contains Four of a Kind |
| j_loyalty_card | ×4 | Every 6 hands played |
| j_seeing_double | ×2 | Hand has scoring Club + card of other suit |
| j_card_sharp | ×3 | Hand type already played this round |
| j_photograph | ×2 | First face card scored |
| j_blackboard | ×3 | All held cards are Spades or Clubs |
| j_cavendish | ×3 | Always (1/1000 chance self-destruct/round) |
| j_flower_pot | ×3 | Hand has Diamond + Club + Heart + Spade |
| j_acrobat | ×3 | Final hand of round |
| j_idol | ×2 | Specific rank+suit card played (changes each round) |
| j_ancient | ×1.5 | Per played card of a specific suit (changes each round) |
| j_baron | ×1.5 | Per King held in hand |
| j_bloodstone | ×1.5 | 1/2 chance per Heart scored |
| j_drivers_license | ×3 | 16+ Enhanced cards in full deck |
| j_stencil | ×1 per empty slot | Per empty joker slot |
| j_glass | ×0.75 additive | Per Glass card destroyed (scaling) |
| j_steel_joker | ×0.2 additive | Per Steel card in full deck (scaling) |
| j_constellation | ×0.1 additive | Per Planet card used (scaling) |
| j_madness | ×0.5 additive | When Small/Big Blind selected (scaling, destroys random joker) |
| j_vampire | ×0.1 additive | Per Enhanced card scored (scaling, removes enhancement) |
| j_obelisk | ×0.2 additive | Per hand without most-played type (scaling, resets) |
| j_hologram | ×0.25 additive | Per card added to deck (scaling) |
| j_lucky_cat | ×0.25 additive | Per Lucky trigger (scaling) |
| j_campfire | ×0.25 additive | Per card sold (scaling, resets on Boss) |
| j_ramen | ×2 base, -×0.01 per discard | Decaying |
| j_triboulet | ×2 | Per King or Queen scored |
| j_caino | ×1 additive | Per face card destroyed (scaling) |
| j_yorick | ×1 additive | Per 23 cards discarded (scaling) |

> **EXPERT**: Which of these ×Mult jokers are the most impactful in practice?
> Which are "win more" (only good when already strong) vs "comeback" jokers?

---

## Scaling Jokers (Q36–39)

**Q36: Which Jokers have uncapped scaling and what is their growth rate?**

| Joker | Growth | Per | Net Rate (realistic) | Notes |
|-------|--------|-----|---------------------|-------|
| j_green_joker | +1 mult | hand played | ~+3-4 mult/round | -1 per discard offsets |
| j_ride_the_bus | +1 mult | consecutive hand w/o face card | Volatile, resets | Strong in face-card-light decks |
| j_runner | +15 chips | hand containing Straight | +15-30 chips/round | Only if playing Straights |
| j_ice_cream | -5 chips | hand played | Decays ~4 rounds | Early spike, dies fast |
| j_popcorn | -4 mult | round | Decays ~5 rounds | Early spike, dies fast |
| j_flash | +2 mult | shop reroll | +2-10 mult/shop | Depends on reroll frequency |
| j_wee | +8 chips | 2 scored | +8-24 chips/round | If playing 2s regularly |
| j_constellation | +×0.1 mult | Planet used | +×0.1-0.3/shop | Depends on Planet availability |
| j_hologram | +×0.25 mult | card added to deck | Slow, +×0.25/shop | Needs Standard Packs |
| j_lucky_cat | +×0.25 mult | Lucky trigger | +×0.25-0.5/round | Needs Lucky cards in deck |
| j_campfire | +×0.25 mult | card sold | Resets on Boss Blind | Sprint scaling |
| j_caino | +×1 mult | face card destroyed | Rare triggers | Legendary, hard to feed |
| j_yorick | +×1 mult | 23 cards discarded | ~1 trigger/2-3 rounds | Legendary, slow but powerful |

> **EXPERT**: What realistic growth rates do you see for these in actual runs?

---

**Q37: What conditions maximize scaling speed for each key scaling Joker?**

| Joker | Maximizing Condition |
|-------|---------------------|
| j_green_joker | Minimize discards (use all hands, avoid discarding) |
| j_ride_the_bus | Avoid face cards in scoring hands (pairs of low cards, flushes of number cards) |
| j_runner | Play Straights frequently (four_fingers helps) |
| j_constellation | Buy/use Planet cards aggressively |
| j_lucky_cat | Fill deck with Lucky-enhanced cards |
| j_campfire | Sell cheap cards frequently before Boss |

> **EXPERT**: For each, what play pattern changes are worth making to accelerate
> scaling?

---

**Q38: Which scaling Jokers pay off earliest vs. require long-term investment?**

> **EXPERT**

---

**Q39: What's the latest Ante to acquire each scaling Joker and still get value?**

> **EXPERT**

---

## Conditional Jokers (Q40–43)

**Q40: Which Jokers require specific hand types?**

| Joker | Required Hand Type | Effect |
|-------|-------------------|--------|
| j_jolly / j_sly | Pair | +8 mult / +50 chips |
| j_mad / j_clever | Two Pair | +10 mult / +80 chips |
| j_zany / j_wily | Three of a Kind | +12 mult / +100 chips |
| j_crazy / j_devious | Straight | +12 mult / +100 chips |
| j_droll / j_crafty | Flush | +10 mult / +80 chips |
| j_duo | Pair | ×2 mult |
| j_trio | Three of a Kind | ×3 mult |
| j_family | Four of a Kind | ×4 mult |
| j_order | Straight | ×3 mult |
| j_tribe | Flush | ×2 mult |
| j_card_sharp | Repeated hand type | ×3 mult (second+ time playing same type in round) |
| j_seance | Straight Flush | Creates random Spectral card |
| j_superposition | Ace + Straight | Creates Tarot card |

---

**Q41: Which Jokers require specific suits or ranks?**

| Joker | Requirement | Effect |
|-------|-------------|--------|
| j_greedy_joker | Diamond scored | +3 mult per Diamond |
| j_lusty_joker | Heart scored | +3 mult per Heart |
| j_wrathful_joker | Spade scored | +3 mult per Spade |
| j_gluttenous_joker | Club scored | +3 mult per Club |
| j_fibonacci | A, 2, 3, 5, or 8 scored | +8 mult per card |
| j_even_steven | Even rank scored (2,4,6,8,T) | +4 mult per card |
| j_odd_todd | Odd rank scored (3,5,7,9,A) | +31 chips per card |
| j_scholar | Ace scored | +20 chips, +4 mult per Ace |
| j_hack | 2, 3, 4, or 5 played | Retrigger |
| j_walkie_talkie | 10 or 4 scored | +10 chips, +4 mult |
| j_baron | King held in hand | ×1.5 mult per held King |
| j_shoot_the_moon | Queen held in hand | +13 mult per held Queen |
| j_blackboard | All held cards Spades or Clubs | ×3 mult |
| j_arrowhead | Spade scored | +50 chips per Spade |
| j_onyx_agate | Club scored | +7 mult per Club |
| j_rough_gem | Diamond scored | $1 per Diamond |
| j_bloodstone | Heart scored | 1/2 chance ×1.5 mult per Heart |
| j_ancient | Specific suit scored (changes/round) | ×1.5 mult per card |
| j_idol | Specific rank+suit scored (changes/round) | ×2 mult per card |
| j_triboulet | King or Queen scored | ×2 mult per card |
| j_scary_face | Face card scored | +30 chips per face card |
| j_smiley | Face card scored | +5 mult per face card |
| j_business | Face card scored | 1/2 chance $2 per face card |

> **EXPERT**: Which of these are strong enough to be "build-around" jokers vs
> minor bonuses you take opportunistically?

---

**Q42: How should conditional jokers alter play/discard decisions?**

> **EXPERT**

---

**Q43: What deck transformations enable conditional joker archetypes?**

> **EXPERT**

---

## Economic Jokers (Q44–46)

**Q44: Which Jokers provide economy and what is their gold value?**

| Joker | Gold/Round | Mechanism | Notes |
|-------|-----------|-----------|-------|
| j_golden | $4 | End of round | Unconditional |
| j_rocket | $1 + $2/Boss defeated | End of round | Scales: $1→$3→$5→... |
| j_cloud_9 | $1 per 9 in deck | End of round | 4× in standard deck = $4/round |
| j_to_the_moon | $1 per $5 held | End of round (extra interest) | Stacks with normal interest |
| j_egg | +$3 sell value | End of round | Only realized when sold |
| j_delayed_grat | $2 per remaining discard | End of round, if 0 discards used | $6 with 3 discards |
| j_business | $2 per face card scored | 1/2 chance on score | ~$2-4/round |
| j_faceless | $5 | If 3+ face cards discarded | Situational |
| j_ticket | $4 per Gold card scored | On score | Needs Gold-enhanced cards |
| j_rough_gem | $1 per Diamond scored | On score | ~$1-3/round |
| j_reserved_parking | $1 per face card held | 1/2 chance while held | ~$1-2/round |
| j_mail | $5 per specific rank discarded | On discard (rank changes/round) | Unreliable |
| j_credit_card | Allows -$20 debt | Passive | Not gold gen, but liquidity |

---

**Q45: What conversion heuristic maps +$X/round into equivalent scoring power?**

> **EXPERT**: How do you mentally convert "this joker earns $4/round" into
> "this is worth approximately +N mult"? At what ante does economy stop
> converting into scoring power (because you can't find upgrades fast enough)?

---

**Q46: At what Ante does each economy Joker stop being worth a slot?**

> **EXPERT**

---

## Joker Trigger State Requirements

For implementation, each joker needs specific state tracking:

| Joker | State Fields | Reset/Update Rule |
|-------|-------------|-------------------|
| j_green_joker | `accumulated_mult: int` | +1 per hand played, -1 per discard |
| j_ride_the_bus | `consecutive_hands: int` | +1 per hand without face card scored, reset to 0 on face card |
| j_runner | `accumulated_chips: int` | +15 per hand containing Straight |
| j_ice_cream | `remaining_chips: int` | Start 100, -5 per hand. Destroy at 0. |
| j_popcorn | `remaining_mult: int` | Start 20, -4 per round. Destroy at 0. |
| j_loyalty_card | `hands_played: int` | +1 per hand. Active (×4) when hands_played % 6 == 0 |
| j_flash | `accumulated_mult: int` | +2 per shop reroll |
| j_constellation | `accumulated_xmult: float` | +0.1 per Planet card used |
| j_obelisk | `accumulated_xmult: float, most_played: str` | +0.2 per hand without most_played type, reset when played |
| j_hologram | `accumulated_xmult: float` | +0.25 per card added to deck |
| j_vampire | `accumulated_xmult: float` | +0.1 per Enhanced card scored |
| j_lucky_cat | `accumulated_xmult: float` | +0.25 per Lucky trigger |
| j_campfire | `accumulated_xmult: float` | +0.25 per card sold, reset on Boss defeat |
| j_ramen | `current_xmult: float` | Start 2.0, -0.01 per card discarded. Destroy at ≤1.0 |
| j_red_card | `accumulated_mult: int` | +3 per Booster Pack skipped |
| j_wee | `accumulated_chips: int` | +8 per 2 scored |
| j_caino | `accumulated_xmult: float` | +1.0 per face card destroyed |
| j_yorick | `accumulated_xmult: float, discard_count: int` | +1.0 per 23 cards discarded |
| j_supernova | — (derived) | Reads `hand_type_play_count[hand_type]` from run state |
| j_card_sharp | — (derived) | Reads `hand_types_played_this_round` set |
| j_todo_list | `target_hand: str` | Changes each round |
| j_ancient | `active_suit: str` | Changes each round |
| j_idol | `active_rank: str, active_suit: str` | Changes each round |
| j_mail | `active_rank: str` | Changes each round |
| j_seltzer | `remaining_hands: int` | Start 10, -1 per hand. Destroy at 0. |
| j_swashbuckler | — (derived) | Sum of sell values of all other jokers |
| j_abstract | — (derived) | +3 mult × number of jokers owned |
| j_blue_joker | — (derived) | +2 chips × cards remaining in deck |
| j_bull | — (derived) | +2 chips × current gold |
| j_bootstraps | — (derived) | +2 mult per $5 held |

> **EXPERT**: Any state tracking subtleties I'm missing? For example, does
> j_ride_the_bus count face cards in kicker positions, or only scoring cards?
