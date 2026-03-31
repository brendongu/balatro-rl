"""
hand_evaluator.py
-----------------
Balatro hand evaluator grounded in the balatrobot API card schema.

Given a list of Card dicts (as returned by the API's `hand.cards`),
this module:
  1. Detects every valid poker hand that can be formed from 1-5 cards
  2. Estimates a chip score for each candidate (base chips × mult, no Jokers)
  3. Returns candidates ranked by estimated score, ready for the agent to pick

Card schema (from API):
  {
    "id": 1,
    "key": "H_A",              # "{Suit}_{Rank}"
    "label": "Ace of Hearts",
    "value": { "suit": "H", "rank": "A" },
    "modifier": {
      "seal": null,            # RED | BLUE | GOLD | PURPLE
      "edition": null,         # FOIL | HOLO | POLYCHROME | NEGATIVE
      "enhancement": null,     # BONUS | MULT | WILD | GLASS | STEEL | STONE | GOLD | LUCKY
      "eternal": false,
      "perishable": null,
      "rental": false
    },
    "state": { "debuff": false, "hidden": false, "highlight": false }
  }

Hand levels (from API Planet card table) - base chips and mult at level 1:
  High Card:        5  chips, 1  mult
  Pair:             10 chips, 2  mult
  Two Pair:         20 chips, 2  mult
  Three of a Kind:  30 chips, 3  mult
  Straight:         30 chips, 4  mult
  Flush:            35 chips, 4  mult
  Full House:       40 chips, 4  mult
  Four of a Kind:   60 chips, 7  mult
  Straight Flush:   100 chips, 8 mult
  Five of a Kind:   120 chips, 12 mult  (requires Joker/wild)
  Flush House:      140 chips, 14 mult  (requires Joker/wild)
  Flush Five:       160 chips, 16 mult  (requires Joker/wild)

Card chip values (base, before enhancements):
  2-9: face value, 10/J/Q/K: 10, A: 11
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from balatro_rl.features.joker_catalog import NUM_CENTER_KEYS, id_to_key

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANK_ORDER = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

RANK_CHIPS = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 11,
}

# Base (chips, mult) at level 1 for each hand type
HAND_BASE: dict[str, tuple[int, int]] = {
    "High Card": (5, 1),
    "Pair": (10, 2),
    "Two Pair": (20, 2),
    "Three of a Kind": (30, 3),
    "Straight": (30, 4),
    "Flush": (35, 4),
    "Full House": (40, 4),
    "Four of a Kind": (60, 7),
    "Straight Flush": (100, 8),
    "Five of a Kind": (120, 12),
    "Flush House": (140, 14),
    "Flush Five": (160, 16),
}

# Hand ranking for tie-breaking (higher = better)
HAND_RANK: dict[str, int] = {
    name: i
    for i, name in enumerate(
        [
            "High Card",
            "Pair",
            "Two Pair",
            "Three of a Kind",
            "Straight",
            "Flush",
            "Full House",
            "Four of a Kind",
            "Straight Flush",
            "Five of a Kind",
            "Flush House",
            "Flush Five",
        ]
    )
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ParsedCard:
    """Internal representation parsed from an API Card dict."""

    index: int  # position in hand (0-based), used to call the API
    rank: str  # e.g. "A", "K", "T", "2"
    suit: str  # e.g. "H", "D", "C", "S"
    enhancement: str | None = None
    edition: str | None = None
    seal: str | None = None
    debuff: bool = False
    stone: bool = False  # Stone cards have no rank/suit but give +50 chips


@dataclass
class ParsedJoker:
    """A joker decoded from the observation tensor."""

    key: str  # center_key string, e.g. "j_joker"
    ability_mult: float  # accumulated +mult (log-inverted from obs)
    ability_x_mult: float  # accumulated xmult (raw from obs)
    ability_chips: float  # accumulated +chips (log-inverted from obs)
    debuffed: bool = False


@dataclass
class HandCandidate:
    """A specific set of cards that forms a playable hand."""

    card_indices: list[int]  # API indices to pass to play/discard
    hand_type: str  # e.g. "Flush"
    scoring_cards: list[ParsedCard]  # cards that actually score
    kicker_cards: list[ParsedCard]  # held but not scoring
    estimated_chips: float  # chips × mult estimate (no Jokers)
    base_chips: int
    base_mult: int
    card_chip_sum: int  # sum of rank chips for scoring cards
    # Enhancement breakdown (populated by estimate_chips)
    total_chips: float = 0.0  # base_chips + card_chip_sum + enhancement chips
    total_mult: float = 0.0  # (base_mult + enhancement additive mult) * enhancement xmult

    @property
    def hand_rank(self) -> int:
        return HAND_RANK[self.hand_type]

    def __repr__(self):
        indices = self.card_indices
        names = [f"{c.rank}{c.suit}" for c in self.scoring_cards]
        return (
            f"HandCandidate({self.hand_type}, cards={indices}, "
            f"scoring={names}, est={self.estimated_chips:.0f})"
        )


# ---------------------------------------------------------------------------
# Card parsing
# ---------------------------------------------------------------------------


def parse_card(index: int, card: dict) -> ParsedCard:
    """Parse an API Card dict into a ParsedCard."""
    value = card.get("value", {})
    modifier = card.get("modifier", {})
    state = card.get("state", {})
    enhancement = modifier.get("enhancement")
    is_stone = enhancement == "STONE"
    return ParsedCard(
        index=index,
        rank=value.get("rank", ""),
        suit=value.get("suit", ""),
        enhancement=enhancement,
        edition=modifier.get("edition"),
        seal=modifier.get("seal"),
        debuff=state.get("debuff", False),
        stone=is_stone,
    )


def parse_hand(hand_cards: list[dict]) -> list[ParsedCard]:
    """Parse the full hand from the API into ParsedCard list."""
    return [parse_card(i, c) for i, c in enumerate(hand_cards)]


# ---------------------------------------------------------------------------
# Suit resolution (WILD cards count as any suit)
# ---------------------------------------------------------------------------


def effective_suits(card: ParsedCard) -> list[str]:
    """Return the suits this card can count as (WILD = all 4 suits)."""
    if card.enhancement == "WILD":
        return ["H", "D", "C", "S"]
    if card.stone or not card.suit:
        return []  # Stone cards don't contribute to flush/straight
    return [card.suit]


# ---------------------------------------------------------------------------
# Hand detection helpers
# ---------------------------------------------------------------------------


def _rank_counts(cards: list[ParsedCard]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in cards:
        if not c.stone:
            counts[c.rank] = counts.get(c.rank, 0) + 1
    return counts


def _is_flush(cards: list[ParsedCard]) -> bool:
    """True if all non-stone cards share at least one common suit (respects WILD)."""
    non_stone = [c for c in cards if not c.stone]
    if not non_stone:
        return False
    # Start with all suits as candidates; intersect with each card's possible suits
    candidate_suits = set(["H", "D", "C", "S"])
    for card in non_stone:
        suits = set(effective_suits(card))
        if suits:
            candidate_suits &= suits
    return len(candidate_suits) > 0


def _is_straight(cards: list[ParsedCard], allow_gaps: bool = False) -> bool:
    """True if cards form a straight (5 consecutive ranks, or A-low 5-4-3-2-A)."""
    non_stone = [c for c in cards if not c.stone and c.rank in RANK_ORDER]
    if len(non_stone) < 5:
        return False
    ranks = sorted(set(RANK_ORDER[c.rank] for c in non_stone))
    if len(ranks) < 5:
        return False
    # Normal straight
    if ranks[-1] - ranks[0] == 4 and len(ranks) == 5:
        return True
    # Ace-low straight: A-2-3-4-5
    if set(ranks) == {14, 2, 3, 4, 5}:
        return True
    # Four-finger / shortcut joker variants handled at agent level
    return False


# ---------------------------------------------------------------------------
# Core hand type detection for a specific 5-card (or fewer) combination
# ---------------------------------------------------------------------------


def detect_hand_type(cards: list[ParsedCard]) -> tuple[str, list[ParsedCard]]:
    """
    Detect the best hand type for this exact card combination.
    Returns (hand_type_name, scoring_cards).
    Cards that are debuffed still occupy positions but don't score.
    Stone cards score chips but don't contribute to hand type.
    """
    # Filter debuffed cards for hand detection (they still sit in the hand)
    active = [c for c in cards if not c.debuff]
    non_stone_active = [c for c in active if not c.stone]
    n = len(non_stone_active)
    counts = _rank_counts(non_stone_active)
    count_vals = sorted(counts.values(), reverse=True)
    is_fl = _is_flush(active) and len(active) == 5
    is_st = _is_straight(active)

    # Flush Five (all same rank and flush)
    if n == 5 and len(counts) == 1 and is_fl:
        return "Flush Five", active

    # Flush House (full house + flush)
    if n == 5 and count_vals[:2] == [3, 2] and is_fl:
        return "Flush House", active

    # Five of a Kind
    if n >= 1 and count_vals and count_vals[0] == 5:
        rank = max(counts, key=lambda r: counts[r])
        return "Five of a Kind", [c for c in active if c.rank == rank]

    # Straight Flush
    if is_fl and is_st:
        return "Straight Flush", active

    # Four of a Kind
    if count_vals and count_vals[0] == 4:
        rank = max(counts, key=lambda r: counts[r])
        scoring = [c for c in active if c.rank == rank]
        return "Four of a Kind", scoring

    # Full House
    if count_vals[:2] == [3, 2]:
        return "Full House", active

    # Flush
    if is_fl:
        return "Flush", active

    # Straight
    if is_st:
        return "Straight", active

    # Three of a Kind
    if count_vals and count_vals[0] == 3:
        rank = max(counts, key=lambda r: counts[r])
        scoring = [c for c in active if c.rank == rank]
        return "Three of a Kind", scoring

    # Two Pair
    pairs = [r for r, cnt in counts.items() if cnt >= 2]
    if len(pairs) >= 2:
        pairs_sorted = sorted(pairs, key=lambda r: RANK_ORDER[r], reverse=True)[:2]
        scoring = [c for c in active if c.rank in pairs_sorted]
        return "Two Pair", scoring

    # Pair
    if len(pairs) == 1:
        scoring = [c for c in active if c.rank == pairs[0]]
        return "Pair", scoring

    # High Card — only the highest card scores
    if non_stone_active:
        best = max(non_stone_active, key=lambda c: RANK_ORDER.get(c.rank, 0))
        return "High Card", [best]

    # Edge: all stones
    return "High Card", active


# ---------------------------------------------------------------------------
# Chip estimation
# ---------------------------------------------------------------------------


def estimate_chips(
    hand_type: str,
    scoring_cards: list[ParsedCard],
    hand_levels: dict[str, int] | None = None,
) -> tuple[float, int, int, int, float, float]:
    """
    Estimate chips × mult for a hand candidate.
    hand_levels: dict mapping hand type name -> level (default 1).
    Returns (estimated_score, base_chips, base_mult, card_chip_sum,
             total_chips, total_mult).
    total_chips/total_mult include enhancement effects (BONUS, MULT, GLASS, etc.)
    so that joker scoring can layer on top without re-deriving enhancements.
    Does NOT include Joker effects.
    """
    level = 1
    if hand_levels:
        level = hand_levels.get(hand_type, 1)

    base_chips_l1, base_mult_l1 = HAND_BASE[hand_type]
    # Planet card level scaling (approximate from API planet card effects)
    level_chip_bonus = {
        "High Card": 10,
        "Pair": 15,
        "Two Pair": 20,
        "Three of a Kind": 20,
        "Straight": 30,
        "Flush": 15,
        "Full House": 25,
        "Four of a Kind": 30,
        "Straight Flush": 40,
        "Five of a Kind": 35,
        "Flush House": 40,
        "Flush Five": 50,
    }
    level_mult_bonus = {
        "High Card": 1,
        "Pair": 1,
        "Two Pair": 1,
        "Three of a Kind": 2,
        "Straight": 3,
        "Flush": 2,
        "Full House": 2,
        "Four of a Kind": 3,
        "Straight Flush": 4,
        "Five of a Kind": 3,
        "Flush House": 4,
        "Flush Five": 3,
    }
    extra_levels = max(0, level - 1)
    base_chips = base_chips_l1 + extra_levels * level_chip_bonus[hand_type]
    base_mult = base_mult_l1 + extra_levels * level_mult_bonus[hand_type]

    # Sum card chips (rank value + enhancements)
    card_chip_sum = 0
    additive_mult = 0
    mult_multiplier = 1.0
    additive_chips = 0

    for card in scoring_cards:
        # Base rank chips
        if card.stone:
            chip_val = 50  # Stone card base
        else:
            chip_val = RANK_CHIPS.get(card.rank, 0)

        # Enhancement effects on chips/mult
        if card.enhancement == "BONUS":
            additive_chips += 30
        elif card.enhancement == "MULT":
            additive_mult += 4
        elif card.enhancement == "GLASS":
            mult_multiplier *= 2.0
        elif card.enhancement == "STONE":
            additive_chips += 50  # stone gives +50 chips
        # STEEL enhances cards held in hand, not scoring cards — skip here
        # LUCKY: probabilistic, use expected value (1/5 × 20 = 4 mult EV)
        elif card.enhancement == "LUCKY":
            additive_mult += 4  # EV

        # Edition effects
        if card.edition == "FOIL":
            additive_chips += 50
        elif card.edition == "HOLO":
            additive_mult += 10
        elif card.edition == "POLYCHROME":
            mult_multiplier *= 1.5

        card_chip_sum += chip_val

    total_chips = base_chips + card_chip_sum + additive_chips
    total_mult = (base_mult + additive_mult) * mult_multiplier
    estimated_score = total_chips * total_mult

    return estimated_score, base_chips, base_mult, card_chip_sum, total_chips, total_mult


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def evaluate_hand(
    hand_cards: list[dict],
    hand_levels: dict[str, int] | None = None,
    max_candidates: int = 20,
) -> list[HandCandidate]:
    """
    Given the API hand cards list, return all valid HandCandidates
    ranked by estimated score (best first).

    Args:
        hand_cards: list of Card dicts from API (hand.cards)
        hand_levels: dict of hand type -> current level (from API hands data)
        max_candidates: cap on returned candidates (avoid combinatorial explosion)

    Returns:
        Sorted list of HandCandidate, best first.
    """
    parsed = parse_hand(hand_cards)
    candidates: list[HandCandidate] = []

    # Enumerate all 1-5 card subsets
    for size in range(1, 6):
        for combo in combinations(parsed, size):
            combo_list = list(combo)
            hand_type, scoring_cards = detect_hand_type(combo_list)
            score, base_chips, base_mult, card_chip_sum, tot_c, tot_m = estimate_chips(
                hand_type, scoring_cards, hand_levels
            )
            kicker_cards = [c for c in combo_list if c not in scoring_cards]
            candidate = HandCandidate(
                card_indices=[c.index for c in combo_list],
                hand_type=hand_type,
                scoring_cards=scoring_cards,
                kicker_cards=kicker_cards,
                estimated_chips=score,
                base_chips=base_chips,
                base_mult=base_mult,
                card_chip_sum=card_chip_sum,
                total_chips=tot_c,
                total_mult=tot_m,
            )
            candidates.append(candidate)

    # Deduplicate: same hand_type + same scoring card indices → keep highest score
    seen: dict[tuple, HandCandidate] = {}
    for c in candidates:
        key = (c.hand_type, tuple(sorted(ci.index for ci in c.scoring_cards)))
        if key not in seen or c.estimated_chips > seen[key].estimated_chips:
            seen[key] = c

    unique = list(seen.values())

    # Sort: primary = estimated score, secondary = hand rank (for equal scores)
    unique.sort(key=lambda c: (c.estimated_chips, c.hand_rank), reverse=True)

    return unique[:max_candidates]


def best_play(
    hand_cards: list[dict],
    hand_levels: dict[str, int] | None = None,
) -> HandCandidate:
    """Return the single best candidate play (by estimated score)."""
    candidates = evaluate_hand(hand_cards, hand_levels)
    return candidates[0]


# ---------------------------------------------------------------------------
# Utility: parse hand levels from API gamestate
# ---------------------------------------------------------------------------


def parse_hand_levels(gamestate: dict) -> dict[str, int]:
    """
    Extract hand type levels from the API gamestate dict.
    gamestate["hands"] maps hand name -> Hand object with "level" field.

    API hand name examples: "Flush", "Straight Flush", etc.
    Returns: dict mapping our hand type names -> level int
    """
    levels: dict[str, int] = {}
    hands_data = gamestate.get("hands", {})
    if isinstance(hands_data, dict):
        cards_list = hands_data.get("cards", [])
        for hand_info in cards_list:
            # hand_info has keys matching Hand schema
            name = hand_info.get("label") or hand_info.get("name", "")
            level = hand_info.get("level", 1)
            if name:
                levels[name] = level
    return levels


# ---------------------------------------------------------------------------
# Observation tensor decoding
# ---------------------------------------------------------------------------

_RANK_FROM_ID: dict[int, str] = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}
_SUIT_FROM_ID: dict[int, str] = {0: "H", 1: "D", 2: "C", 3: "S"}


def parse_cards_from_obs(
    hand_card: np.ndarray,
    n_cards: int,
) -> list[ParsedCard]:
    """Decode ``obs["hand_card"]`` tensor into ParsedCard objects.

    Args:
        hand_card: Shape ``(max_hand, 15)`` float32 array from the observation.
        n_cards: Actual number of cards (from ``obs["entity_counts"][0]``).

    Features per card (from jackdaw ``encode_playing_card``):
        [0] rank_id / 14   (card.base.id: 2=2, ..., 14=Ace)
        [1] suit / 3       (0=Hearts, 1=Diamonds, 2=Clubs, 3=Spades)
        [2] chip_value / 11
        [3] enhancement / 8
        [4] edition / 4
        [5] seal / 4
        [6] debuff (0/1)
        [7] face_down (0/1)
    """
    cards: list[ParsedCard] = []
    for i in range(min(n_cards, hand_card.shape[0])):
        row = hand_card[i]
        face_down = row[7] > 0.5 if hand_card.shape[1] > 7 else False
        if face_down:
            continue
        rank_id = round(row[0] * 14)
        suit_id = round(row[1] * 3)
        rank = _RANK_FROM_ID.get(rank_id, "")
        suit = _SUIT_FROM_ID.get(suit_id, "")
        debuff = row[6] > 0.5
        enhancement_id = round(row[3] * 8) if hand_card.shape[1] > 3 else 0
        stone = enhancement_id == 6
        cards.append(ParsedCard(index=i, rank=rank, suit=suit, debuff=debuff, stone=stone))
    return cards


def _inv_log_scale(v: float) -> float:
    """Invert jackdaw's log_scale: sign(x) * log2(1 + |x|)."""
    if v >= 0:
        return 2.0**v - 1.0
    return -(2.0 ** (-v) - 1.0)


def parse_jokers_from_obs(
    joker_obs: np.ndarray,
    n_jokers: int,
) -> list[ParsedJoker]:
    """Decode ``obs["joker"]`` tensor into ParsedJoker objects.

    Args:
        joker_obs: Shape ``(max_jokers, 15)`` float32 array.
        n_jokers: Actual joker count (from ``obs["entity_counts"][1]``).

    Joker feature layout (from jackdaw ``encode_joker``):
        [0]  center_key_id / NUM_CENTER_KEYS
        [1]  rarity / 4
        [2]  edition / 4
        [3]  sell_value (log-scaled)
        [4]  eternal (0/1)
        [5]  perishable (0/1)
        [6]  perish_tally / 5
        [7]  rental (0/1)
        [8]  debuffed (0/1)
        [9]  position / 20
        [10] ability_mult (log-scaled)
        [11] ability_x_mult (raw, typically 1-5)
        [12] ability_chips (log-scaled)
        [13] ability_extra (log-scaled)
        [14] condition_met (0/1)
    """
    jokers: list[ParsedJoker] = []
    for i in range(min(n_jokers, joker_obs.shape[0])):
        row = joker_obs[i]
        center_id = round(row[0] * NUM_CENTER_KEYS)
        key = id_to_key(center_id)
        if not key:
            continue
        debuffed = row[8] > 0.5
        ability_mult = _inv_log_scale(row[10])
        ability_x_mult = float(row[11])
        ability_chips = _inv_log_scale(row[12])
        jokers.append(ParsedJoker(
            key=key,
            ability_mult=ability_mult,
            ability_x_mult=ability_x_mult,
            ability_chips=ability_chips,
            debuffed=debuffed,
        ))
    return jokers


# ---------------------------------------------------------------------------
# Discard planning
# ---------------------------------------------------------------------------


@dataclass
class FlushDraw:
    """Analysis of flush potential from a set of cards."""

    target_suit: str
    keep_indices: list[int]
    discard_indices: list[int]
    cards_of_suit: int
    cards_needed: int


def flush_potential(cards: list[ParsedCard]) -> FlushDraw | None:
    """Assess flush draw potential. Returns None if no suit has >= 3 cards."""
    suit_counts: Counter[str] = Counter()
    suit_cards: dict[str, list[ParsedCard]] = {}
    for c in cards:
        if c.suit and not c.debuff and not c.stone:
            suit_counts[c.suit] += 1
            suit_cards.setdefault(c.suit, []).append(c)

    if not suit_counts:
        return None
    best_suit, best_count = suit_counts.most_common(1)[0]
    if best_count < 3:
        return None

    suited = suit_cards[best_suit]
    suited.sort(key=lambda c: RANK_ORDER.get(c.rank, 0), reverse=True)
    keep = suited[:5]
    keep_idx_set = {c.index for c in keep}
    discard = [c.index for c in cards if c.index not in keep_idx_set]

    return FlushDraw(
        target_suit=best_suit,
        keep_indices=[c.index for c in keep],
        discard_indices=discard,
        cards_of_suit=best_count,
        cards_needed=max(0, 5 - best_count),
    )


@dataclass
class FullHouseDraw:
    """Analysis of full house potential from a set of cards."""

    keep_indices: list[int]
    discard_indices: list[int]
    has_triple: bool
    has_pair: bool
    pairs_count: int


def fullhouse_potential(cards: list[ParsedCard]) -> FullHouseDraw | None:
    """Assess full house draw potential. Returns None if no pair or triple exists."""
    rank_groups: dict[str, list[ParsedCard]] = {}
    for c in cards:
        if c.rank and not c.debuff and not c.stone:
            rank_groups.setdefault(c.rank, []).append(c)

    groups_by_size = sorted(rank_groups.items(), key=lambda x: len(x[1]), reverse=True)
    has_triple = any(len(g) >= 3 for _, g in groups_by_size)
    pairs = [(r, g) for r, g in groups_by_size if len(g) >= 2]

    if not pairs:
        return None

    keep: list[ParsedCard] = []
    if has_triple:
        triple_rank, triple_cards = next((r, g) for r, g in groups_by_size if len(g) >= 3)
        keep.extend(triple_cards[:3])
        remaining_pairs = [(r, g) for r, g in groups_by_size if r != triple_rank and len(g) >= 2]
        if remaining_pairs:
            keep.extend(remaining_pairs[0][1][:2])
    else:
        # Two pairs - keep both, need to convert one to a triple
        for _, g in pairs[:2]:
            keep.extend(g[:2])

    keep_idx_set = {c.index for c in keep}
    discard = [c.index for c in cards if c.index not in keep_idx_set]

    return FullHouseDraw(
        keep_indices=[c.index for c in keep],
        discard_indices=discard,
        has_triple=has_triple,
        has_pair=len(pairs) >= 1,
        pairs_count=len(pairs),
    )


@dataclass
class StraightDraw:
    """Analysis of straight potential from a set of cards."""

    keep_indices: list[int]
    discard_indices: list[int]
    cards_in_run: int
    cards_needed: int
    is_open_ended: bool  # True if draw can be completed from both ends


def straight_potential(cards: list[ParsedCard]) -> StraightDraw | None:
    """Assess straight draw potential.

    Finds the longest run of consecutive ranks (considering Ace-high wrapping)
    and returns the best draw if at least 3 consecutive ranks are present.
    Prefers open-ended draws (completable from either end) over gutshots.
    """
    active = [c for c in cards if c.rank and not c.debuff and not c.stone
              and c.rank in RANK_ORDER]
    if len(active) < 3:
        return None

    rank_to_cards: dict[int, list[ParsedCard]] = {}
    for c in active:
        r = RANK_ORDER[c.rank]
        rank_to_cards.setdefault(r, []).append(c)

    unique_ranks = sorted(rank_to_cards.keys())

    # Build candidate windows of 5 consecutive rank values.
    # Ace can appear as both 14 (high) and 1 (low for A-2-3-4-5).
    extended_ranks = set(unique_ranks)
    if 14 in extended_ranks:
        extended_ranks.add(1)

    best_window: tuple[int, ...] | None = None
    best_count = 0
    best_open = False

    for low in range(1, 11):  # windows: 1-5, 2-6, ..., 10-14
        window = range(low, low + 5)
        present = [r for r in window if r in extended_ranks]
        count = len(present)
        # open-ended if both neighbours of the filled range could extend it,
        # i.e. we have a contiguous block that doesn't touch both edges
        present_sorted = sorted(present)
        contiguous = (
            all(
                present_sorted[i + 1] - present_sorted[i] == 1
                for i in range(len(present_sorted) - 1)
            )
            if len(present_sorted) >= 2
            else True
        )
        open_ended = (
            contiguous
            and count >= 4
            and (present_sorted[0] > low or present_sorted[-1] < low + 4)
        ) if count >= 4 else False

        # Prefer: higher count > open-ended > higher window (better scoring straight)
        better = (
            count > best_count
            or (count == best_count and open_ended and not best_open)
            or (count == best_count and open_ended == best_open and low > (best_window[0] if best_window else 0))
        )
        if better:
            best_count = count
            best_window = tuple(window)
            best_open = open_ended

    if best_window is None or best_count < 3:
        return None

    # Select best card per rank to keep (highest chip value for ties)
    keep: list[ParsedCard] = []
    for r in best_window:
        actual_rank = r if r != 1 else 14  # map low-ace back to real rank
        if actual_rank in rank_to_cards:
            candidates = rank_to_cards[actual_rank]
            best_card = max(candidates, key=lambda c: RANK_ORDER.get(c.rank, 0))
            keep.append(best_card)

    keep_idx_set = {c.index for c in keep}
    discard = [c.index for c in cards if c.index not in keep_idx_set]
    cards_needed = 5 - best_count

    return StraightDraw(
        keep_indices=[c.index for c in keep],
        discard_indices=discard,
        cards_in_run=best_count,
        cards_needed=cards_needed,
        is_open_ended=best_open,
    )


def recommend_discards(
    cards: list[ParsedCard],
    blind_target: float,
    hand_levels: dict[str, int] | None = None,
) -> list[int]:
    """Recommend card indices to discard to improve the hand.

    Strategy:
    1. If any 5-card combo already beats the blind -> return [] (play it).
    2. If a three-of-a-kind exists -> prefer full house draw (especially high rank).
    3. Try flush draw (need <=2 cards) or straight draw (need 1 card, open-ended).
    4. Try moderate flush draw (need 3) or strong straight draw (need 1-2).
    5. Try full house draw (two pair basis).
    6. Weaker draws, then fallback to discarding weakest cards.

    Returns up to 5 card indices to discard.
    """
    best = _best_available_score(cards, hand_levels)
    if best >= blind_target:
        return []

    fhd = fullhouse_potential(cards)
    fd = flush_potential(cards)
    sd = straight_potential(cards)

    # Full house draw with existing trips takes priority
    if fhd is not None and fhd.has_triple:
        return fhd.discard_indices[:5]

    # 4-card flush draw (need 1 card) — highest hit rate (~23%)
    if fd is not None and fd.cards_needed <= 1:
        return fd.discard_indices[:5]

    # 4-card straight draw (need 1 card) — ~9% gutshot, ~18% open-ended
    if sd is not None and sd.cards_needed <= 1:
        return sd.discard_indices[:5]

    # 3-card flush draw (need 2 cards)
    if fd is not None and fd.cards_needed <= 2:
        return fd.discard_indices[:5]

    # Straight draw needing 2 cards
    if sd is not None and sd.cards_needed <= 2:
        return sd.discard_indices[:5]

    # Moderate flush draw (need 3)
    if fd is not None and fd.cards_needed <= 3:
        return fd.discard_indices[:5]

    # Two pair → full house draw
    if fhd is not None:
        return fhd.discard_indices[:5]

    # Weak flush draw (need 4+) still better than nothing
    if fd is not None:
        return fd.discard_indices[:5]

    # Weak straight draw (need 3)
    if sd is not None and sd.cards_needed <= 3:
        return sd.discard_indices[:5]

    # Last resort: discard lowest-value cards not in any pair
    return _discard_weakest(cards)


def _best_available_score(
    cards: list[ParsedCard],
    hand_levels: dict[str, int] | None = None,
) -> float:
    """Score the best 5-card (or fewer) combination from the given cards."""
    best = 0.0
    limit = min(len(cards), 5)
    for size in range(max(1, limit), limit + 1):
        for combo in combinations(cards, size):
            hand_type, scoring = detect_hand_type(list(combo))
            score, _, _, _, _, _ = estimate_chips(hand_type, scoring, hand_levels)
            if score > best:
                best = score
    return best


def _discard_weakest(cards: list[ParsedCard], max_discard: int = 5) -> list[int]:
    """Discard lowest-value cards that aren't part of a pair or better."""
    rank_counts = Counter(c.rank for c in cards if c.rank and not c.debuff)
    paired_ranks = {r for r, cnt in rank_counts.items() if cnt >= 2}

    unpaired = [c for c in cards if c.rank not in paired_ranks]
    unpaired.sort(key=lambda c: RANK_ORDER.get(c.rank, 0))
    return [c.index for c in unpaired[:max_discard]]


def evaluate_hand_from_parsed(
    cards: list[ParsedCard],
    hand_levels: dict[str, int] | None = None,
    max_candidates: int = 20,
) -> list[HandCandidate]:
    """Like evaluate_hand() but takes pre-parsed ParsedCard objects."""
    candidates: list[HandCandidate] = []
    for size in range(1, min(len(cards), 5) + 1):
        for combo in combinations(cards, size):
            combo_list = list(combo)
            hand_type, scoring_cards = detect_hand_type(combo_list)
            score, base_chips, base_mult, card_chip_sum, tot_c, tot_m = estimate_chips(
                hand_type, scoring_cards, hand_levels
            )
            kicker_cards = [c for c in combo_list if c not in scoring_cards]
            candidates.append(HandCandidate(
                card_indices=[c.index for c in combo_list],
                hand_type=hand_type,
                scoring_cards=scoring_cards,
                kicker_cards=kicker_cards,
                estimated_chips=score,
                base_chips=base_chips,
                base_mult=base_mult,
                card_chip_sum=card_chip_sum,
                total_chips=tot_c,
                total_mult=tot_m,
            ))

    seen: dict[tuple, HandCandidate] = {}
    for c in candidates:
        key = (c.hand_type, tuple(sorted(ci.index for ci in c.scoring_cards)))
        if key not in seen or c.estimated_chips > seen[key].estimated_chips:
            seen[key] = c

    unique = list(seen.values())
    unique.sort(key=lambda c: (c.estimated_chips, c.hand_rank), reverse=True)
    return unique[:max_candidates]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate a hand: A♥ K♥ Q♥ J♥ T♥ 9♦ 8♣ 7♠
    sample_hand = [
        {
            "id": 1,
            "key": "H_A",
            "label": "Ace of Hearts",
            "value": {"suit": "H", "rank": "A"},
            "modifier": {},
            "state": {"debuff": False},
        },
        {
            "id": 2,
            "key": "H_K",
            "label": "King of Hearts",
            "value": {"suit": "H", "rank": "K"},
            "modifier": {},
            "state": {"debuff": False},
        },
        {
            "id": 3,
            "key": "H_Q",
            "label": "Queen of Hearts",
            "value": {"suit": "H", "rank": "Q"},
            "modifier": {},
            "state": {"debuff": False},
        },
        {
            "id": 4,
            "key": "H_J",
            "label": "Jack of Hearts",
            "value": {"suit": "H", "rank": "J"},
            "modifier": {},
            "state": {"debuff": False},
        },
        {
            "id": 5,
            "key": "H_T",
            "label": "Ten of Hearts",
            "value": {"suit": "H", "rank": "T"},
            "modifier": {},
            "state": {"debuff": False},
        },
        {
            "id": 6,
            "key": "D_9",
            "label": "Nine of Diamonds",
            "value": {"suit": "D", "rank": "9"},
            "modifier": {},
            "state": {"debuff": False},
        },
        {
            "id": 7,
            "key": "C_8",
            "label": "Eight of Clubs",
            "value": {"suit": "C", "rank": "8"},
            "modifier": {},
            "state": {"debuff": False},
        },
        {
            "id": 8,
            "key": "S_7",
            "label": "Seven of Spades",
            "value": {"suit": "S", "rank": "7"},
            "modifier": {},
            "state": {"debuff": False},
        },
    ]

    print("Top 5 candidate plays:\n")
    candidates = evaluate_hand(sample_hand)
    for i, c in enumerate(candidates[:5]):
        scoring_names = [f"{sc.rank}{sc.suit}" for sc in c.scoring_cards]
        print(
            f"  {i + 1}. {c.hand_type:20s}  cards={c.card_indices}  "
            f"scoring={scoring_names}  est={c.estimated_chips:.0f} chips"
        )

    print(f"\nBest play: {best_play(sample_hand)}")
