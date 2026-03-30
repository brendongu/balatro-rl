# Deck Building & Card Management

Reference for card modifiers, consumable effects, deck composition strategy,
and the card schema used throughout the codebase.

---

## Card Schema (Game Data)

From the balatrobot API, every card has this structure:

```json
{
  "id": 1,
  "key": "H_A",
  "set": "DEFAULT",
  "label": "Ace of Hearts",
  "value": { "suit": "H", "rank": "A" },
  "modifier": {
    "seal": null,
    "edition": null,
    "enhancement": null,
    "eternal": false,
    "perishable": null,
    "rental": false
  },
  "state": { "debuff": false, "hidden": false, "highlight": false }
}
```

### Standard Deck

52 cards: 4 suits (H, D, C, S) × 13 ranks (2–9, T, J, Q, K, A).

Special starting decks:
- **Abandoned**: No face cards (J, Q, K removed) → 40 cards
- **Checkered**: 26 Spades + 26 Hearts (no Diamonds/Clubs)
- **Erratic**: Randomized ranks and suits

---

## Card Modifiers (Game Data)

### Enhancements

Applied to playing cards. Affect scoring when the card is played or held.

| Enhancement | Effect When Scored | Effect When Held | Special |
|-------------|-------------------|------------------|---------|
| BONUS | +30 chips | — | — |
| MULT | +4 mult | — | — |
| WILD | Counts as every suit | — | Enables any-suit flush |
| GLASS | ×2 mult | — | 1/4 chance to be destroyed after scoring |
| STEEL | — | ×1.5 mult | Only scores while in hand, not played |
| STONE | +50 chips, replaces rank chips | — | No rank or suit; cannot contribute to hand type |
| GOLD | — | $3 at end of round if held | Economy, not scoring |
| LUCKY | 1/5 chance: +20 mult; 1/15 chance: +$20 | — | Probabilistic; j_oops doubles odds |

Key interactions:
- STEEL cards want to be **held**, not played. They score ×1.5 mult passively.
- STONE cards always contribute their +50 chips but cannot form hand types.
- GLASS cards are high-risk high-reward: ×2 mult but may self-destruct.
- WILD cards make flush builds trivially achievable.

### Editions

Applied to playing cards and jokers. Always beneficial, no downside.

| Edition | Effect on Playing Card | Effect on Joker |
|---------|----------------------|-----------------|
| FOIL | +50 chips when scored | +50 chips |
| HOLO | +10 mult when scored | +10 mult |
| POLYCHROME | ×1.5 mult when scored | ×1.5 mult |
| NEGATIVE | — | +1 joker/consumable slot |

Editions stack with enhancements: a POLYCHROME GLASS card gives ×2 × ×1.5 = ×3 mult.

### Seals

Applied to playing cards. Provide meta-effects beyond scoring.

| Seal | Effect |
|------|--------|
| RED | Retrigger this card 1 time (all scoring effects fire twice) |
| BLUE | Creates Planet card for final poker hand if held in hand at end of round |
| GOLD | Earn $3 when this card is scored |
| PURPLE | Creates Tarot card when this card is discarded |

RED seal is the most scoring-relevant: it doubles a card's chip contribution,
enhancement, and edition effects. On a POLYCHROME GLASS card, RED seal means
×2 × ×1.5 × ×2 × ×1.5 = ×9 mult from a single card.

---

## Consumables (Game Data)

### Tarot Cards (22 total)

Consumables that modify playing cards. Key effects for deck building:

| Tarot | Effect | Deck Impact |
|-------|--------|-------------|
| The Magician (c_magician) | Enhance 2 cards to Lucky | Adds probabilistic mult |
| The Empress (c_empress) | Enhance 2 cards to Mult | Adds +4 mult per card |
| The Hierophant (c_heirophant) | Enhance 2 cards to Bonus | Adds +30 chips per card |
| The Lovers (c_lovers) | Enhance 1 card to Wild | Enables suit flexibility |
| The Chariot (c_chariot) | Enhance 1 card to Steel | Adds ×1.5 held mult |
| Justice (c_justice) | Enhance 1 card to Glass | Adds ×2 mult (risky) |
| The Devil (c_devil) | Enhance 1 card to Gold | Adds $3/round economy |
| The Tower (c_tower) | Enhance 1 card to Stone | Adds +50 chips, loses rank/suit |
| Strength (c_strength) | Increase rank of up to 2 cards by 1 | Raises chip value |
| The Hanged Man (c_hanged_man) | Destroy up to 2 cards | Thins deck |
| Death (c_death) | Copy right card onto left card | Duplicates strong cards |
| The Star/Moon/Sun/World | Convert up to 3 cards to D/C/H/S | Suit concentration |
| The Hermit (c_hermit) | Double money (max $20) | Economy |
| Judgement (c_judgement) | Create random Joker | Joker generation |

### Planet Cards (12 total)

Level up poker hand types. Each use: +chips and +mult to that hand's base.

| Planet | Hand Type | +Chips | +Mult |
|--------|-----------|--------|-------|
| Pluto | High Card | +10 | +1 |
| Mercury | Pair | +15 | +1 |
| Uranus | Two Pair | +20 | +1 |
| Venus | Three of a Kind | +20 | +2 |
| Saturn | Straight | +30 | +3 |
| Jupiter | Flush | +15 | +2 |
| Earth | Full House | +25 | +2 |
| Mars | Four of a Kind | +30 | +3 |
| Neptune | Straight Flush | +40 | +4 |
| Planet X | Five of a Kind | +35 | +3 |
| Ceres | Flush House | +40 | +4 |
| Eris | Flush Five | +50 | +3 |

### Spectral Cards (18 total)

Powerful effects with drawbacks. Key ones:

| Spectral | Effect | Risk |
|----------|--------|------|
| Cryptid (c_cryptid) | Create 2 copies of 1 card | Grows deck but concentrates rank |
| Familiar (c_familiar) | Destroy 1 random card, add 3 Enhanced face cards | Deck modification |
| Grim (c_grim) | Destroy 1, add 2 Enhanced Aces | High-value cards |
| Immolate (c_immolate) | Destroy 5 cards, gain $20 | Aggressive thinning + economy |
| Hex (c_hex) | Add Polychrome to random Joker, destroy all others | All-in on one joker |
| Ankh (c_ankh) | Copy 1 random Joker, destroy all others | All-in gamble |
| Black Hole (c_black_hole) | Upgrade every hand type by 1 level | Universal upgrade |

---

## Deck Composition (Q50–54)

**Q50: What is the optimal deck size at different stages of the game?**

| Stage | Optimal Size | Reasoning |
|-------|-------------|-----------|
| Ante 1–2 | > **EXPERT** | |
| Ante 3–5 | | |
| Ante 6–8 | | |

---

**Q51: When should you add cards to your deck vs. keep it lean?**

> **EXPERT**

---

**Q52: When should you remove cards, and which cards to remove first?**

> **EXPERT**

---

**Q53: How do editions change deck building decisions?**

> **EXPERT**

---

**Q54: How do enhancements change card value for deck building?**

See enhancement table above for mechanics. Strategic value:

> **EXPERT**: For each enhancement, when do you prioritize applying it?

---

## Suits & Ranks (Q55–59)

**Q55: Which suits are most valuable and why?**

From game data: no intrinsic difference between suits. Value is entirely
context-dependent:
- Suit jokers (Greedy/Lusty/Wrathful/Gluttenous) make one suit more valuable.
- j_smeared merges Hearts/Diamonds and Spades/Clubs.
- Boss blinds debuff specific suits.

> **EXPERT**: In practice, is there a default suit preference (e.g., Hearts
> for Bloodstone ×1.5)?

---

**Q56: Which ranks are most valuable and why?**

From game data, chip values vary: Ace (11) > Face cards/10 (10) > 9 > ... > 2.
Special rank interactions:
- Ace: j_scholar (+20 chips, +4 mult), j_fibonacci, high chip value.
- Face cards (J/Q/K): j_scary_face, j_smiley, j_photograph, j_triboulet, j_baron.
- Low cards (2-5): j_hack retrigger, j_fibonacci (2,3,5).
- Even (2,4,6,8,T): j_even_steven.
- Odd (3,5,7,9,A): j_odd_todd.

> **EXPERT**: Which ranks do you prioritize keeping/adding to the deck?

---

**Q57: When is unbalanced suit distribution correct?**

> **EXPERT**

---

**Q58: When is unbalanced rank distribution correct?**

> **EXPERT**

---

**Q59: How do you build for specific hand types?**

> **EXPERT**

---

## Consumable Strategy (Q60–64)

**Q60: Which Tarot cards are highest priority in which situations?**

> **EXPERT**

---

**Q61–62: Which Planet cards are worth buying, and how do you decide which hand type to level?**

> **EXPERT**

---

**Q63: Which Spectral cards are worth the risk?**

> **EXPERT**

---

**Q64: When do you use vs. save consumable slots?**

> **EXPERT**
