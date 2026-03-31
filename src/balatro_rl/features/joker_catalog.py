"""Joker and center-key catalog for heuristic shop decisions.

Loads jackdaw's centers.json once at import time and provides:
- Reverse mapping from integer ID back to center_key string
- Joker categorization (chips / mult / xmult / economy / other)
- Planet card hand-type lookup
- Booster pack identification

Uses the same ``sorted(keys)`` ordering and ``start=1`` convention as
``jackdaw.env.observation._load_center_ids`` so integer IDs match exactly.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

# Resolve centers.json from the installed jackdaw package
_CENTERS_PATH: Path = (
    Path(str(resources.files("jackdaw"))) / "engine" / "data" / "centers.json"
)

# Built at module load time — same ordering as jackdaw's observation.py
_ID_TO_KEY: dict[int, str] = {}
_KEY_TO_ID: dict[str, int] = {}
_CENTER_DATA: dict[str, dict] = {}


def _load() -> None:
    global _ID_TO_KEY, _KEY_TO_ID, _CENTER_DATA
    if _ID_TO_KEY:
        return
    with open(_CENTERS_PATH) as f:
        _CENTER_DATA = json.load(f)
    for i, key in enumerate(sorted(_CENTER_DATA.keys()), start=1):
        _ID_TO_KEY[i] = key
        _KEY_TO_ID[key] = i


_load()

NUM_CENTER_KEYS: int = len(_ID_TO_KEY)


def id_to_key(center_id: int) -> str:
    """Map a center_key integer ID back to its string key. Returns '' for unknown."""
    return _ID_TO_KEY.get(center_id, "")


def key_to_id(key: str) -> int:
    """Map a center_key string to its integer ID. Returns 0 for unknown."""
    return _KEY_TO_ID.get(key, 0)


# ---------------------------------------------------------------------------
# Joker categorization (by effect field + rarity)
# ---------------------------------------------------------------------------

_MULT_EFFECTS = frozenset({
    "Mult", "Suit Mult", "Type Mult", "Card Mult", "Hand Size Mult",
    "Random Mult", "No Discard Mult", "Socialized Mult", "Hand played mult",
    "Set Mult", "Joker Mult", "1 in 10 mult", "Low Card double",
    "Even Card Buff", "Odd Card Buff",
})

_XMULT_EFFECTS = frozenset({
    "X1.5 Mult", "X2 Mult", "X3 Mult", "X1.5 Mult club 7",
})

_CHIPS_EFFECTS = frozenset({
    "Discard Chips", "Scary Face Cards", "Ace Buff", "Steel Card Buff",
    "Stone Card Buff",
})

_ECONOMY_EFFECTS = frozenset({
    "Face Card dollar Chance", "Credit", "Discard dollars", "Bonus dollars",
    "dollars for Gold cards",
})

_JOKER_CATEGORY: dict[str, str] = {}


def _categorize_jokers() -> dict[str, str]:
    if _JOKER_CATEGORY:
        return _JOKER_CATEGORY
    for key, entry in _CENTER_DATA.items():
        if entry.get("set") != "Joker":
            continue
        effect = entry.get("effect", "")
        if effect in _XMULT_EFFECTS:
            _JOKER_CATEGORY[key] = "xmult"
        elif effect in _MULT_EFFECTS:
            _JOKER_CATEGORY[key] = "mult"
        elif effect in _CHIPS_EFFECTS:
            _JOKER_CATEGORY[key] = "chips"
        elif effect in _ECONOMY_EFFECTS:
            _JOKER_CATEGORY[key] = "economy"
        else:
            _JOKER_CATEGORY[key] = "other"
    return _JOKER_CATEGORY


_categorize_jokers()


def joker_category(key: str) -> str:
    """Return the category for a joker key: 'chips', 'mult', 'xmult', 'economy', 'other'."""
    return _JOKER_CATEGORY.get(key, "other")


def joker_rarity(key: str) -> int:
    """Return rarity: 1=Common, 2=Uncommon, 3=Rare, 4=Legendary. 0 if unknown."""
    entry = _CENTER_DATA.get(key, {})
    return entry.get("rarity", 0)


def is_common_scoring_joker(key: str) -> bool:
    """True if the joker is Common rarity AND provides +chips or +mult."""
    return joker_rarity(key) == 1 and joker_category(key) in ("chips", "mult")


def is_scoring_joker(key: str) -> bool:
    """True if the joker provides +chips, +mult, or xmult (any rarity)."""
    return joker_category(key) in ("chips", "mult", "xmult")


# ---------------------------------------------------------------------------
# Scaling jokers — accumulate value over hands/discards/rounds
# ---------------------------------------------------------------------------

_SCALING_JOKERS: frozenset[str] = frozenset({
    "j_green_joker",      # +1 mult per hand played, -1 per discard
    "j_red_card",         # +3 mult per skip (not useful for always-select)
    "j_spare_trousers",   # +2 mult when hand type matches
    "j_ride_the_bus",     # +1 mult per consecutive hand without face card
    "j_runner",           # +15 chips when playing a Straight
    "j_ice_cream",        # starts at 100 chips, loses 5 per hand (anti-scaling)
    "j_square",           # +4 chips when playing exactly 4 cards
    "j_flash",            # +2 mult per reroll used
    "j_constellation",    # x0.01 mult per planet used
    "j_campfire",         # x0.25 mult per card sold
    "j_obelisk",          # x0.2 mult per consecutive hand of most played type
    "j_loyalty_card",     # x4 mult every 6 hands played
    "j_photograph",       # x2 mult on first face card scored
    "j_supernova",        # +mult = times hand type has been played this run
})


def is_scaling_joker(key: str) -> bool:
    """True if the joker scales its effect over time (accumulates chips/mult)."""
    return key in _SCALING_JOKERS


def joker_cost(key: str) -> int:
    """Return the base cost for a joker. 0 if unknown."""
    return _CENTER_DATA.get(key, {}).get("cost", 0)


# ---------------------------------------------------------------------------
# Planet card lookup
# ---------------------------------------------------------------------------

_PLANET_HAND_TYPE: dict[str, str] = {}


def _build_planet_map() -> None:
    if _PLANET_HAND_TYPE:
        return
    for key, entry in _CENTER_DATA.items():
        set_type = entry.get("set", "")
        if set_type != "Planet":
            continue
        config = entry.get("config", {})
        if isinstance(config, dict):
            ht = config.get("hand_type", "")
            if ht:
                _PLANET_HAND_TYPE[key] = ht


_build_planet_map()


def planet_hand_type(key: str) -> str | None:
    """Return the hand type a planet card levels up, or None."""
    return _PLANET_HAND_TYPE.get(key)


# ---------------------------------------------------------------------------
# Pack identification
# ---------------------------------------------------------------------------


def is_buffoon_pack(key: str) -> bool:
    return key.startswith("p_buffoon")


def is_booster_pack(key: str) -> bool:
    return key.startswith("p_")


# ---------------------------------------------------------------------------
# Shop item decoding from observation tensor
# ---------------------------------------------------------------------------


def decode_shop_center_id(shop_item_feature_0: float) -> int:
    """Decode center_key_id from the normalized shop_item[0] feature."""
    return round(shop_item_feature_0 * NUM_CENTER_KEYS)


def decode_shop_card_set(shop_item_feature_1: float) -> str:
    """Decode card_set from the normalized shop_item[1] feature."""
    idx = round(shop_item_feature_1 * 9)
    return _CARD_SET_FROM_IDX.get(idx, "")


_CARD_SET_FROM_IDX: dict[int, str] = {
    0: "Default",
    1: "Enhanced",
    2: "Joker",
    3: "Tarot",
    4: "Planet",
    5: "Spectral",
    6: "Voucher",
    7: "Booster",
    8: "Back",
    9: "Edition",
}
