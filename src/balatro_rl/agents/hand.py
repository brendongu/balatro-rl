"""Hand selection policy (SELECTING_HAND phase).

Responsible for PlayHand and Discard decisions. This is the most
combinatorially complex phase — the action space includes all legal
card combinations for both playing and discarding.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from balatro_rl.features.hand_evaluator import (
    HAND_BASE,
    HAND_RANK,
    RANK_ORDER,
    HandCandidate,
    ParsedCard,
    ParsedJoker,
    evaluate_hand_from_parsed,
    flush_potential,
    fullhouse_potential,
    parse_cards_from_obs,
    parse_jokers_from_obs,
    recommend_discards,
)
from balatro_rl.features.joker_scoring import GameContext, simulate_joker_scoring

# jackdaw ActionType constants
_PLAY_HAND = 0
_DISCARD = 1

# Max card chip sum for ideal cards of each hand type.
# Used to estimate the theoretical ceiling for "can this hand type one-shot?"
_MAX_CARD_CHIPS: dict[str, int] = {
    "High Card": 11,        # A
    "Pair": 22,             # AA
    "Two Pair": 42,         # AA KK
    "Three of a Kind": 33,  # AAA
    "Straight": 51,         # A K Q J T
    "Flush": 51,            # A K Q J T suited
    "Full House": 53,       # AAA KK
    "Four of a Kind": 44,   # AAAA
    "Straight Flush": 51,   # A K Q J T suited
}

# Hand types worth checking for one-shot potential (ordered by score ceiling)
_ONE_SHOT_TYPES = [
    "Straight Flush", "Four of a Kind", "Full House", "Flush", "Straight",
]


def _inv_log_scale(v: float) -> float:
    """Invert jackdaw's log_scale: sign(x) * log2(1 + |x|)."""
    if v >= 0:
        return 2.0 ** v - 1.0
    return -(2.0 ** (-v) - 1.0)


def _can_one_shot_blind(
    blind_chips: float,
    jokers: list[ParsedJoker],
    ctx: GameContext,
) -> bool:
    """Check if any major hand type could theoretically one-shot the blind.

    Builds phantom HandCandidates with ideal cards for each hand type and
    scores them through simulate_joker_scoring. Returns True if any type's
    max score exceeds blind_chips.
    """
    # Phantom scoring cards: 5 ideal cards (Hearts, face/high ranks)
    # Suit choice is approximate; per-card suit jokers add ~15 mult which
    # doesn't change the one-shot determination in most cases.
    phantom_cards = [
        ParsedCard(index=0, rank="A", suit="H"),
        ParsedCard(index=1, rank="K", suit="H"),
        ParsedCard(index=2, rank="Q", suit="H"),
        ParsedCard(index=3, rank="J", suit="H"),
        ParsedCard(index=4, rank="T", suit="H"),
    ]

    for hand_type in _ONE_SHOT_TYPES:
        base_chips, base_mult = HAND_BASE[hand_type]
        max_card = _MAX_CARD_CHIPS[hand_type]
        total_chips = float(base_chips + max_card)
        total_mult = float(base_mult)

        # Determine which phantom cards "score" for this hand type
        if hand_type == "Full House":
            scoring = phantom_cards[:5]
        elif hand_type == "Four of a Kind":
            scoring = phantom_cards[:4]
        elif hand_type in ("Flush", "Straight", "Straight Flush"):
            scoring = phantom_cards[:5]
        else:
            scoring = phantom_cards[:5]

        candidate = HandCandidate(
            card_indices=[c.index for c in scoring],
            hand_type=hand_type,
            scoring_cards=scoring,
            kicker_cards=[],
            estimated_chips=total_chips * total_mult,
            base_chips=base_chips,
            base_mult=base_mult,
            card_chip_sum=max_card,
            total_chips=total_chips,
            total_mult=total_mult,
        )

        score = simulate_joker_scoring(candidate, jokers, ctx)
        if score >= blind_chips:
            return True

    return False


class HandPolicy:
    """Policy for the hand-play phase.

    Wraps a MaskablePPO model. Falls back to random legal action
    if no model is provided.
    """

    def __init__(self, model: object | None = None) -> None:
        self._model = model

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        action_mask: np.ndarray,
        action_table: list | None = None,
    ) -> int:
        if self._model is not None:
            return self._predict_from_model(obs, action_mask)
        return self._random_legal(action_mask)

    def _predict_from_model(
        self,
        obs: dict[str, np.ndarray],
        action_mask: np.ndarray,
    ) -> int:
        action, _ = self._model.predict(obs, action_masks=action_mask, deterministic=True)
        return int(action)

    def _random_legal(self, action_mask: np.ndarray) -> int:
        legal = np.where(action_mask)[0]
        return int(np.random.choice(legal))

    def update(self, **kwargs: object) -> None:
        pass


class HeuristicHandPolicy:
    """Multi-ante heuristic hand policy.

    Strategy overview:
    - Prefer to beat the blind in as few hands as possible (unused hands = money).
    - Discards are free — use all of them if it improves the hand.
    - For big hands (flush, full house): discard aggressively toward the draw.
      Prefer full house when trips already exist. Otherwise flush.
    - For smaller hands (pair, two pair, high card): playable with any cards,
      focus on cycling through deck to reach modified cards.
    - If the hand can already beat the blind, play it immediately.
    - If it can't and no discards remain, play the best available hand.
    """

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        action_mask: np.ndarray,
        action_table: list | None = None,
    ) -> int:
        if action_table is None:
            return int(np.where(action_mask)[0][0])

        g = obs["global"]
        blind_chips = _inv_log_scale(g[18])
        discards_left = round(g[14] * 10.0)
        hands_left = round(g[13] * 10.0)
        n_cards = int(obs["entity_counts"][0])

        cards = parse_cards_from_obs(obs["hand_card"], n_cards)
        if not cards:
            return int(np.where(action_mask)[0][0])

        # Parse owned jokers for scoring simulation
        n_jokers = int(obs["entity_counts"][1])
        jokers = parse_jokers_from_obs(obs["joker"], n_jokers)
        deck_size = round(_inv_log_scale(g[27]))
        ctx = GameContext(
            discards_left=discards_left,
            n_jokers=n_jokers,
            deck_size=max(deck_size, 0),
            hand_cards=cards,
        )

        play_actions, discard_actions = _index_actions(action_table, action_mask)

        candidates = evaluate_hand_from_parsed(cards, max_candidates=20)

        # Re-score candidates with joker effects and re-sort
        if jokers:
            for cand in candidates:
                cand.estimated_chips = simulate_joker_scoring(cand, jokers, ctx)
            candidates.sort(
                key=lambda c: (c.estimated_chips, c.hand_rank), reverse=True
            )

        best = candidates[0] if candidates else None

        # 1. Can we beat the blind right now?
        if best and best.estimated_chips >= blind_chips:
            action = _find_play_action(best.card_indices, play_actions)
            if action is not None:
                return action

        # 2. Multi-hand mode: if we already have a strong hand (Straight+)
        #    but no hand type can one-shot the blind, play immediately.
        #    Save discards for weaker future hands that benefit more.
        if (
            discards_left > 0
            and best
            and best.hand_rank >= HAND_RANK["Straight"]
            and not _can_one_shot_blind(blind_chips, jokers, ctx)
        ):
            action = _find_play_action(best.card_indices, play_actions)
            if action is not None:
                return action

        # 3. Discard to improve — fish for a one-shot or upgrade a weak hand
        if discards_left > 0 and discard_actions:
            discard_indices = recommend_discards(cards, blind_chips)
            if discard_indices:
                action = _find_discard_action(discard_indices, discard_actions)
                if action is not None:
                    return action

        # 4. No discards left or nothing to discard — play best available
        if best and play_actions:
            action = _find_play_action(best.card_indices, play_actions)
            if action is not None:
                return action
            for cand in candidates[1:]:
                action = _find_play_action(cand.card_indices, play_actions)
                if action is not None:
                    return action
            return next(iter(play_actions.values()))

        return int(np.where(action_mask)[0][0])

    def update(self, **kwargs: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers shared between policies
# ---------------------------------------------------------------------------


def _index_actions(
    action_table: list,
    action_mask: np.ndarray,
) -> tuple[dict[tuple[int, ...], int], dict[tuple[int, ...], int]]:
    """Build play/discard action lookups from the action table."""
    play_actions: dict[tuple[int, ...], int] = {}
    discard_actions: dict[tuple[int, ...], int] = {}
    for idx, fa in enumerate(action_table):
        if not action_mask[idx]:
            continue
        if fa.action_type == _PLAY_HAND and fa.card_target:
            play_actions[fa.card_target] = idx
        elif fa.action_type == _DISCARD and fa.card_target:
            discard_actions[fa.card_target] = idx
    return play_actions, discard_actions


def _find_play_action(
    card_indices: list[int],
    play_actions: dict[tuple[int, ...], int],
) -> int | None:
    """Find a play action matching the given card indices."""
    target = tuple(sorted(card_indices))
    return play_actions.get(target)


def _find_discard_action(
    desired_indices: list[int],
    discard_actions: dict[tuple[int, ...], int],
) -> int | None:
    """Find the discard action best matching the desired indices."""
    target = tuple(sorted(desired_indices[:5]))
    if target in discard_actions:
        return discard_actions[target]
    # Find closest overlap
    desired_set = set(desired_indices)
    best_action = None
    best_overlap = -1
    for card_target, action_idx in discard_actions.items():
        overlap = len(desired_set & set(card_target))
        if overlap > best_overlap:
            best_overlap = overlap
            best_action = action_idx
    return best_action
