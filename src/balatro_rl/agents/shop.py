"""Shop policy (SHOP phase).

Responsible for BuyCard, SellJoker, SellConsumable, Reroll, RedeemVoucher,
OpenBooster, and NextRound decisions. This is where long-term build strategy
matters most — joker synergies, economy management, and build coherence.
"""

from __future__ import annotations

import math

import numpy as np

from balatro_rl.features.joker_catalog import (
    NUM_CENTER_KEYS,
    decode_shop_card_set,
    decode_shop_center_id,
    id_to_key,
    is_buffoon_pack,
    is_common_scoring_joker,
    is_scaling_joker,
    is_scoring_joker,
    planet_hand_type,
)

# jackdaw ActionType constants
_BUY_CARD = 8
_OPEN_BOOSTER = 13
_NEXT_ROUND = 6
_CASH_OUT = 4
_REROLL = 5

# Planets for deck-agnostic scaling path
_SCALING_PLANETS = frozenset({"Pair", "High Card", "Two Pair"})
# Planets for big-hand path
_BIG_HAND_PLANETS = frozenset({"Flush", "Straight", "Full House"})
# High-value planets worth buying in ante 1
_ANTE1_PLANETS = frozenset({"Flush", "Straight"})

# Interest thresholds: earn $1 per $5 held (up to $25 = $5/round)
_INTEREST_STEP = 5


def _inv_log_scale(v: float) -> float:
    """Invert jackdaw's log_scale: sign(x) * log2(1 + |x|)."""
    if v >= 0:
        return 2.0 ** v - 1.0
    return -(2.0 ** (-v) - 1.0)


class ShopPolicy:
    """Policy for shop-phase decisions.

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


class HeuristicShopPolicy:
    """Improved heuristic shop policy.

    Priority order:
    1. Buy any affordable scoring joker (+chips/+mult/xmult, any rarity).
       Prefer common scoring jokers, then scaling jokers, then any scoring.
    2. Open a Buffoon (Joker) pack if available.
    3. Buy relevant planet cards (ante-gated):
       - Ante 1: only Jupiter (Flush) and Saturn (Straight).
       - Ante 2+: also Earth (Full House), Mercury (Pair), Pluto (HC), Uranus (2P).
    4. Leave shop (NextRound/CashOut).

    Economy rules:
    - Jokers are always bought (scoring impact >> interest).
    - Planets are skipped if they'd drop below a $5 interest threshold.
    - Don't reroll in first two antes.
    """

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        action_mask: np.ndarray,
        action_table: list | None = None,
    ) -> int:
        if action_table is None:
            return self._leave_or_fallback(action_mask, action_table)

        g = obs["global"]
        dollars = _inv_log_scale(g[12])
        ante = max(round(g[10] * 8.0), 1)
        n_shop = int(obs["entity_counts"][3])
        n_jokers = int(obs["entity_counts"][1])
        shop_items = obs["shop_item"]

        # Parse available actions
        buy_items = _parse_buy_actions(action_table, action_mask, shop_items, n_shop)
        booster_items = _parse_booster_actions(action_table, action_mask, shop_items, n_shop)

        # Priority 1: Buy scoring joker (always worth it)
        best_joker = _pick_best_joker(buy_items)
        if best_joker is not None:
            return best_joker

        # Priority 2: Open buffoon pack (free to open, already paid on purchase)
        for action_idx, _slot, key in booster_items:
            if is_buffoon_pack(key):
                return action_idx

        # Priority 3: Buy relevant planet cards (ante-gated, interest-aware)
        target_planets = _ANTE1_PLANETS if ante <= 1 else (_BIG_HAND_PLANETS | _SCALING_PLANETS)
        for action_idx, _slot, key, card_set, affordable, cost in buy_items:
            if not affordable or card_set != "Planet":
                continue
            if _would_break_interest(dollars, cost):
                continue
            ht = planet_hand_type(key)
            if ht in target_planets:
                return action_idx

        # Nothing worth buying -> leave shop
        return self._leave_or_fallback(action_mask, action_table)

    def _leave_or_fallback(
        self,
        action_mask: np.ndarray,
        action_table: list | None,
    ) -> int:
        if action_table is not None:
            for idx, fa in enumerate(action_table):
                if action_mask[idx] and fa.action_type in (_NEXT_ROUND, _CASH_OUT):
                    return idx
        legal = np.where(action_mask)[0]
        return int(legal[0]) if len(legal) > 0 else 0

    def update(self, **kwargs: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Shop parsing helpers
# ---------------------------------------------------------------------------

_BuyItem = tuple[int, int, str, str, bool, float]  # (action_idx, slot, key, card_set, affordable, cost)
_BoosterItem = tuple[int, int, str]  # (action_idx, slot, key)


def _parse_buy_actions(
    action_table: list,
    action_mask: np.ndarray,
    shop_items: np.ndarray,
    n_shop: int,
) -> list[_BuyItem]:
    result: list[_BuyItem] = []
    for idx, fa in enumerate(action_table):
        if not action_mask[idx]:
            continue
        if fa.action_type == _BUY_CARD and fa.entity_target is not None:
            slot = fa.entity_target
            if slot < n_shop:
                row = shop_items[slot]
                center_id = decode_shop_center_id(row[0])
                card_set = decode_shop_card_set(row[1])
                affordable = row[3] > 0.5
                cost = _inv_log_scale(row[2])
                key = id_to_key(center_id)
                result.append((idx, slot, key, card_set, affordable, cost))
    return result


def _parse_booster_actions(
    action_table: list,
    action_mask: np.ndarray,
    shop_items: np.ndarray,
    n_shop: int,
) -> list[_BoosterItem]:
    result: list[_BoosterItem] = []
    for idx, fa in enumerate(action_table):
        if not action_mask[idx]:
            continue
        if fa.action_type == _OPEN_BOOSTER and fa.entity_target is not None:
            slot = fa.entity_target
            if slot < n_shop:
                row = shop_items[slot]
                center_id = decode_shop_center_id(row[0])
                key = id_to_key(center_id)
                result.append((idx, slot, key))
    return result


def _would_break_interest(dollars: float, cost: float) -> bool:
    """True if buying this item drops us below a $5 interest threshold."""
    current_tier = int(min(dollars, 25)) // _INTEREST_STEP
    after_tier = int(min(max(dollars - cost, 0), 25)) // _INTEREST_STEP
    return after_tier < current_tier


def _pick_best_joker(buy_items: list[_BuyItem]) -> int | None:
    """Pick the best affordable joker: common scoring > scaling > any scoring.

    Jokers are always worth buying — their scoring impact far exceeds
    the $1/round from interest thresholds.
    """
    common_scoring = None
    scaling = None
    any_scoring = None

    for action_idx, _slot, key, card_set, affordable, _cost in buy_items:
        if not affordable or card_set != "Joker":
            continue
        if is_common_scoring_joker(key):
            if common_scoring is None:
                common_scoring = action_idx
        elif is_scaling_joker(key):
            if scaling is None:
                scaling = action_idx
        elif is_scoring_joker(key):
            if any_scoring is None:
                any_scoring = action_idx

    for candidate in (common_scoring, scaling, any_scoring):
        if candidate is not None:
            return candidate
    return None
