"""Joker-aware scoring simulation for hand evaluation.

Estimates the score contribution of owned jokers for a given hand candidate,
following Balatro's resolution order:

  1. Per scoring card (left to right): suit/rank-conditional joker triggers
  2. Per held card (not played): held-card joker triggers
  3. Independent joker effects: unconditional, hand-type-conditional, state-derived

This is an *estimation* — probabilistic effects use expected value and some
edge cases (Blueprint/Brainstorm, boss blind debuffs) are not modeled.
"""

from __future__ import annotations

from dataclasses import dataclass

from balatro_rl.features.hand_evaluator import (
    RANK_CHIPS,
    HandCandidate,
    ParsedCard,
    ParsedJoker,
)

_FACE_RANKS = frozenset({"J", "Q", "K"})
_FIBONACCI_RANKS = frozenset({"A", "2", "3", "5", "8"})
_EVEN_RANKS = frozenset({"2", "4", "6", "8", "T"})
_ODD_RANKS = frozenset({"3", "5", "7", "9", "A"})
_HACK_RANKS = frozenset({"2", "3", "4", "5"})


@dataclass
class GameContext:
    """Game state needed for joker scoring that isn't in the hand candidate."""

    discards_left: int
    n_jokers: int
    deck_size: int
    hand_cards: list[ParsedCard]


# ---------------------------------------------------------------------------
# Hand-type containment check
# ---------------------------------------------------------------------------

# Maps hand type -> set of hand types it "contains" for joker trigger purposes.
# e.g. Full House contains both Pair and Three of a Kind.
_HAND_TYPE_CONTAINS: dict[str, frozenset[str]] = {
    "High Card": frozenset({"High Card"}),
    "Pair": frozenset({"Pair"}),
    "Two Pair": frozenset({"Two Pair", "Pair"}),
    "Three of a Kind": frozenset({"Three of a Kind"}),
    "Straight": frozenset({"Straight"}),
    "Flush": frozenset({"Flush"}),
    "Full House": frozenset({"Full House", "Pair", "Three of a Kind"}),
    "Four of a Kind": frozenset({"Four of a Kind", "Pair"}),
    "Straight Flush": frozenset({"Straight Flush", "Straight", "Flush"}),
    "Five of a Kind": frozenset({"Five of a Kind", "Pair", "Three of a Kind", "Four of a Kind"}),
    "Flush House": frozenset({"Flush House", "Flush", "Full House", "Pair", "Three of a Kind"}),
    "Flush Five": frozenset({"Flush Five", "Flush", "Five of a Kind", "Pair", "Three of a Kind", "Four of a Kind"}),
}


def _hand_contains(hand_type: str, required: str) -> bool:
    """True if playing ``hand_type`` satisfies a joker requiring ``required``."""
    return required in _HAND_TYPE_CONTAINS.get(hand_type, frozenset())


# ---------------------------------------------------------------------------
# Phase A: Per-scored-card joker effects
# ---------------------------------------------------------------------------

# Suit -> joker_key -> (+chips, +mult)
_SUIT_ADD: dict[str, dict[str, tuple[float, float]]] = {
    "D": {"j_greedy_joker": (0, 3)},
    "H": {"j_lusty_joker": (0, 3)},
    "S": {"j_wrathful_joker": (0, 3), "j_arrowhead": (50, 0)},
    "C": {"j_gluttenous_joker": (0, 3), "j_onyx_agate": (0, 7)},
}


def _per_card_effects(
    scoring_cards: list[ParsedCard],
    joker_keys: frozenset[str],
) -> tuple[float, float, float]:
    """Compute additional (chips, mult, xmult) from per-scored-card triggers."""
    add_chips = 0.0
    add_mult = 0.0
    x_mult = 1.0
    photograph_used = False

    for card in scoring_cards:
        if card.debuff:
            continue
        suit = card.suit
        rank = card.rank
        is_face = rank in _FACE_RANKS

        # Suit-conditional +chips/+mult
        if suit in _SUIT_ADD:
            for jk, (c, m) in _SUIT_ADD[suit].items():
                if jk in joker_keys:
                    add_chips += c
                    add_mult += m

        # j_bloodstone: 1/2 chance x1.5 per Heart scored -> EV x1.25
        if suit == "H" and "j_bloodstone" in joker_keys:
            x_mult *= 1.25

        # Rank-conditional effects
        if rank in _FIBONACCI_RANKS and "j_fibonacci" in joker_keys:
            add_mult += 8

        if rank == "A" and "j_scholar" in joker_keys:
            add_chips += 20
            add_mult += 4

        if rank in _EVEN_RANKS and "j_even_steven" in joker_keys:
            add_mult += 4

        if rank in _ODD_RANKS and "j_odd_todd" in joker_keys:
            add_chips += 31

        if rank in {"T", "4"} and "j_walkie_talkie" in joker_keys:
            add_chips += 10
            add_mult += 4

        if is_face and "j_scary_face" in joker_keys:
            add_chips += 30

        if is_face and "j_smiley" in joker_keys:
            add_mult += 5

        # j_triboulet: x2 per King or Queen scored
        if rank in {"K", "Q"} and "j_triboulet" in joker_keys:
            x_mult *= 2.0

        # j_photograph: x2 on FIRST face card scored only
        if is_face and "j_photograph" in joker_keys and not photograph_used:
            x_mult *= 2.0
            photograph_used = True

        # j_hack: retrigger 2/3/4/5 (card scores twice -> adds rank chips again)
        if rank in _HACK_RANKS and "j_hack" in joker_keys:
            add_chips += RANK_CHIPS.get(rank, 0)

    return add_chips, add_mult, x_mult


# ---------------------------------------------------------------------------
# Phase B: Per-held-card joker effects
# ---------------------------------------------------------------------------


def _held_card_effects(
    held_cards: list[ParsedCard],
    joker_keys: frozenset[str],
) -> tuple[float, float, float]:
    """Compute (chips, mult, xmult) from held-in-hand card triggers."""
    add_chips = 0.0
    add_mult = 0.0
    x_mult = 1.0

    for card in held_cards:
        if card.debuff:
            continue
        if card.rank == "K" and "j_baron" in joker_keys:
            x_mult *= 1.5
        if card.rank == "Q" and "j_shoot_the_moon" in joker_keys:
            add_mult += 13

    # j_blackboard: x3 if ALL held cards are Spades or Clubs
    if held_cards and "j_blackboard" in joker_keys:
        non_debuff_held = [c for c in held_cards if not c.debuff and not c.stone]
        if non_debuff_held and all(c.suit in {"S", "C"} for c in non_debuff_held):
            x_mult *= 3.0

    return add_chips, add_mult, x_mult


# ---------------------------------------------------------------------------
# Phase C: Independent joker effects
# ---------------------------------------------------------------------------

# Hand-type conditional: joker_key -> (required_hand_type, +chips, +mult)
_HAND_TYPE_ADD: dict[str, tuple[str, float, float]] = {
    "j_jolly": ("Pair", 0, 8),
    "j_sly": ("Pair", 50, 0),
    "j_mad": ("Two Pair", 0, 10),
    "j_clever": ("Two Pair", 80, 0),
    "j_zany": ("Three of a Kind", 0, 12),
    "j_wily": ("Three of a Kind", 100, 0),
    "j_crazy": ("Straight", 0, 12),
    "j_devious": ("Straight", 100, 0),
    "j_droll": ("Flush", 0, 10),
    "j_crafty": ("Flush", 80, 0),
}

# Hand-type conditional xmult: joker_key -> (required_hand_type, xmult)
_HAND_TYPE_XMULT: dict[str, tuple[str, float]] = {
    "j_duo": ("Pair", 2.0),
    "j_trio": ("Three of a Kind", 3.0),
    "j_family": ("Four of a Kind", 4.0),
    "j_order": ("Straight", 3.0),
    "j_tribe": ("Flush", 2.0),
}


def _independent_effects(
    hand_type: str,
    jokers: list[ParsedJoker],
    joker_keys: frozenset[str],
    ctx: GameContext,
) -> tuple[float, float, float]:
    """Compute (chips, mult, xmult) from independent joker triggers."""
    add_chips = 0.0
    add_mult = 0.0
    x_mult = 1.0

    for joker in jokers:
        if joker.debuffed:
            continue
        jk = joker.key

        # --- Unconditional flat effects ---
        if jk == "j_joker":
            add_mult += 4
        elif jk == "j_abstract":
            add_mult += 3 * ctx.n_jokers
        elif jk == "j_misprint":
            add_mult += 12  # EV of uniform 0-23
        elif jk == "j_banner":
            add_chips += 30 * ctx.discards_left
        elif jk == "j_blue_joker":
            add_chips += 2 * ctx.deck_size
        elif jk == "j_stuntman":
            add_chips += 250
        elif jk == "j_ceremonial":
            add_mult += 50
        elif jk == "j_half":
            add_mult += 20
        elif jk == "j_cavendish":
            x_mult *= 3.0

        # --- Hand-type conditional +chips/+mult ---
        elif jk in _HAND_TYPE_ADD:
            req, c, m = _HAND_TYPE_ADD[jk]
            if _hand_contains(hand_type, req):
                add_chips += c
                add_mult += m

        # --- Hand-type conditional xmult ---
        elif jk in _HAND_TYPE_XMULT:
            req, xm = _HAND_TYPE_XMULT[jk]
            if _hand_contains(hand_type, req):
                x_mult *= xm

        # --- Scaling jokers: read accumulated state from observation ---
        elif jk in {
            "j_green_joker", "j_red_card", "j_flash", "j_supernova",
            "j_ride_the_bus", "j_popcorn", "j_swashbuckler", "j_bootstraps",
        }:
            add_mult += max(joker.ability_mult, 0)
        elif jk in {
            "j_ice_cream", "j_runner", "j_wee", "j_square",
            "j_castle", "j_bull",
        }:
            add_chips += max(joker.ability_chips, 0)

        # --- Scaling xmult jokers ---
        elif jk in {
            "j_constellation", "j_obelisk", "j_hologram",
            "j_vampire", "j_lucky_cat", "j_campfire",
            "j_glass", "j_steel_joker", "j_madness",
            "j_ramen", "j_caino", "j_yorick",
        }:
            xm = joker.ability_x_mult
            if xm > 1.0:
                x_mult *= xm

        # j_loyalty_card: x4 every 6 hands -> EV x1.5
        elif jk == "j_loyalty_card":
            x_mult *= 1.5

    return add_chips, add_mult, x_mult


# ---------------------------------------------------------------------------
# Main scoring simulation
# ---------------------------------------------------------------------------


def simulate_joker_scoring(
    candidate: HandCandidate,
    jokers: list[ParsedJoker],
    ctx: GameContext,
) -> float:
    """Estimate joker-adjusted score for a hand candidate.

    Follows Balatro's resolution order:
      1. Start from candidate's total_chips and total_mult (includes enhancements)
      2. Add per-scored-card joker effects
      3. Add per-held-card joker effects
      4. Add independent joker effects
      5. Return total_chips * total_mult * xmult_product

    Returns the estimated total score including joker effects.
    """
    if not jokers:
        return candidate.estimated_chips

    joker_keys = frozenset(j.key for j in jokers if not j.debuffed)
    if not joker_keys:
        return candidate.estimated_chips

    played_indices = set(candidate.card_indices)
    held_cards = [c for c in ctx.hand_cards if c.index not in played_indices]

    pc_chips, pc_mult, pc_xmult = _per_card_effects(
        candidate.scoring_cards, joker_keys
    )
    hc_chips, hc_mult, hc_xmult = _held_card_effects(held_cards, joker_keys)
    ind_chips, ind_mult, ind_xmult = _independent_effects(
        candidate.hand_type, jokers, joker_keys, ctx
    )

    # candidate.total_chips and candidate.total_mult already include
    # base hand values, card rank chips, and card enhancement effects.
    final_chips = candidate.total_chips + pc_chips + hc_chips + ind_chips
    final_mult = candidate.total_mult + pc_mult + hc_mult + ind_mult
    final_xmult = pc_xmult * hc_xmult * ind_xmult

    return max(final_chips * final_mult * final_xmult, 0.0)
