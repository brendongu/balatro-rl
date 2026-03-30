# Scoring & Hand Selection

Questions most critical for `hand_evaluator.py`, action space design, and reward shaping.

---

## Hand Selection (Q65–70)

**Q65: Given a hand of 8 cards, what is your decision hierarchy for which 5 to play?**

*[Your answer here]*

---

**Q66: When do you play for Joker triggers rather than raw point maximization?**

*[Your answer here]*

---

**Q67: When is discarding strictly better than playing a weak hand?**

*[Your answer here]*

---

**Q68: How many discards do you typically use per round, and how does this vary by Ante?**

*[Your answer here]*

---

**Q69: What is the typical sequence of discard → play decisions within a round?**
*(e.g., do you play first and then discard, or discard first to fish for better hands?)*

*[Your answer here]*

---

**Q70: How do you "play around" boss blind mechanics when selecting cards?**
*(Examples of specific boss blinds and the hand selection adjustments they force)*

*[Your answer here]*

---

## Mult vs. Chips Tradeoffs (Q47–48)

**Q47: When does additive +Mult become better than ×Mult, and vice versa?**
*(Think about the breakpoint: if base score is low, ×Mult on a small number is still small)*

*[Your answer here]*

---

**Q48: How do you calculate the marginal value of adding another Mult Joker to an existing build?**
*(What's the mental math you use at the shop?)*

*[Your answer here]*

---

## Chip Requirement Scaling (Q8–9)

**Q8: What are the chip requirements at each Ante (small blind / big blind / boss blind)?**

| Ante | Small | Big | Boss |
|------|-------|-----|------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |

---

**Q9: How does the requirement scale within each Ante?**
*(Is it a fixed multiplier? Does it compound? Any nonlinear jumps?)*

*[Your answer here]*

---

## Implementation Notes for RL

> Fill this in after answering the questions above.
> What simplifications or approximations should the reward function make
> based on your answers?

*[Your notes here]*
